"""Geometric scoring functions for compass routing.

Contains deflection, compass, directed, and contrastive scoring strategies.
"""

from __future__ import annotations

from collections import defaultdict


def _deflection_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    candidate_wids: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Score windows by how much they deflect the query residual.

    For each candidate window, extend the query against that window's
    1-token checkpoint KV.  Measure L2 distance between the resulting
    residual and the bare query residual.  Larger deflection = more
    relevant.

    Returns list of (window_id, deflection) sorted descending.
    """
    import mlx.core as mx

    # Bare query residual (no context)
    q_ids = mx.array(prompt_ids)[None]
    _logits, _bare_kv, bare_residual = kv_gen.prefill_with_residual(q_ids)
    mx.eval(_logits)
    bare_vec = bare_residual.reshape(-1).astype(mx.float32)

    wids = candidate_wids if candidate_wids is not None else list(range(lib.num_windows))

    scores: list[tuple[int, float]] = []
    for wid in wids:
        ckpt_kv = lib.get_checkpoint(wid)
        _logits, _ext_kv, ext_residual = kv_gen.extend_with_residual(
            q_ids, ckpt_kv, abs_start=1
        )
        mx.eval(_logits)
        ext_vec = ext_residual.reshape(-1).astype(mx.float32)

        diff = ext_vec - bare_vec
        deflection = mx.sqrt(mx.sum(diff * diff)).item()
        scores.append((wid, deflection))

    scores.sort(key=lambda x: -x[1])
    return scores


def _compass_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    query_vec_np: "np.ndarray | None" = None,
) -> list[tuple[int, float]]:
    """Score windows by cosine similarity in the compass subspace.

    Three projection modes:
      - Darkspace (frame bank): stored residuals are pre-projected. Only
        project the query through the frame bank at routing time.
      - Structural removal: remove structural PCs, match in full dark space.
      - Fixed 16D (older libraries): project into content subspace.

    The commitment layer (~75% model depth) is where query routing signal
    is maximally expressed.
    """
    import mlx.core as mx
    import numpy as np

    # Load compass data from library
    mean_vec, basis, pc_start, pc_end = lib.get_compass_basis()
    compass_layer = lib.compass_layer
    n_dims = pc_end - pc_start

    # Convert to numpy for fast linear algebra
    mean_np = np.array(mean_vec.reshape(-1).tolist(), dtype=np.float32)
    basis_np = np.array(basis.tolist(), dtype=np.float32)

    is_darkspace = lib.is_darkspace

    if is_darkspace:
        # Darkspace: stored residuals are already projected into frame bank space.
        all_vecs_raw = []
        wid_map = []
        for wid in range(lib.num_windows):
            for si, res in enumerate(lib.get_compass_residuals(wid)):
                vec = np.array(res.reshape(-1).tolist(), dtype=np.float32)
                all_vecs_raw.append(vec)
                wid_map.append((wid, si))

        all_projected = np.stack(all_vecs_raw, axis=0)  # (N_stored, D)
        all_norms = np.linalg.norm(all_projected, axis=1)  # (N_stored,)

        # Single-vector query: last position through frame bank
        if query_vec_np is not None:
            q_vec = query_vec_np
        else:
            q_ids = mx.array(prompt_ids)[None]
            q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
            q_vec = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)
        q_projected = q_vec @ basis_np.T  # (n_frame_dims,)
        q_norm = np.linalg.norm(q_projected)

    else:
        # PCA-based modes: structural removal or fixed 16D
        use_structural_removal = lib.has_structural_basis
        if use_structural_removal:
            structural_np = np.array(
                lib.get_structural_basis().tolist(), dtype=np.float32
            )

            def _clean(vec: np.ndarray) -> np.ndarray:
                v = vec - mean_np
                projections = v @ structural_np.T
                v = v - projections @ structural_np
                return v
        else:
            def _clean(vec: np.ndarray) -> np.ndarray:
                return (vec - mean_np) @ basis_np.T

        all_vecs_raw = []
        wid_map = []
        for wid in range(lib.num_windows):
            for si, res in enumerate(lib.get_compass_residuals(wid)):
                vec = np.array(res.reshape(-1).tolist(), dtype=np.float32)
                all_vecs_raw.append(_clean(vec))
                wid_map.append((wid, si))

        all_projected = np.stack(all_vecs_raw, axis=0)  # (N_stored, D)
        all_norms = np.linalg.norm(all_projected, axis=1)  # (N_stored,)

        # Single-vector query: last position
        if query_vec_np is not None:
            q_vec = query_vec_np
        else:
            q_ids = mx.array(prompt_ids)[None]
            q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
            q_vec = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)
        q_projected = _clean(q_vec)
        q_norm = np.linalg.norm(q_projected)

    if q_norm < 1e-10:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    # Cosine similarity — fully vectorised
    cosines = (all_projected @ q_projected) / (all_norms * q_norm + 1e-10)

    # Collect scores per window
    per_window_scores: dict[int, list[float]] = defaultdict(list)
    for idx, (wid, _si) in enumerate(wid_map):
        per_window_scores[wid].append(float(cosines[idx]))

    # Aggregation strategy depends on sample density:
    #   Interval (≤16 samples): max — best-matching sample wins
    #   Full (>16 samples): top-k mean — clusters of matches win
    samples_per_window = len(per_window_scores[next(iter(per_window_scores))])
    _TOP_K_AGG = 10

    per_window: dict[int, float] = {}
    if samples_per_window <= 16:
        # Sparse: max aggregation
        for wid, score_list in per_window_scores.items():
            per_window[wid] = max(score_list)
    else:
        # Dense: top-k mean aggregation
        for wid, score_list in per_window_scores.items():
            score_list.sort(reverse=True)
            top_k = score_list[:_TOP_K_AGG]
            per_window[wid] = sum(top_k) / len(top_k)

    scores = [(wid, s) for wid, s in per_window.items()]
    scores.sort(key=lambda x: -x[1])
    return scores


def _directed_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
) -> list[tuple[int, float]]:
    """Score windows by projecting stored residuals onto the query's own direction.

    The query residual at L26, minus the corpus mean, defines a single direction
    in the dark space. Each stored residual is scored by its projection onto that
    direction. Positions that deviate from the mean in the same way as the query
    are the relevant ones.

    No PCA. No frame bank. No categories. The query tells you which direction
    matters. One dimension — the query's own.
    """
    import mlx.core as mx
    import numpy as np

    compass_layer = lib.compass_layer

    # Load all compass residuals (raw L26, 2560D)
    all_vecs = []
    wid_map = []
    for wid in range(lib.num_windows):
        for si, res in enumerate(lib.get_compass_residuals(wid)):
            vec = np.array(res.reshape(-1).tolist(), dtype=np.float32)
            all_vecs.append(vec)
            wid_map.append((wid, si))

    all_stored = np.stack(all_vecs, axis=0)  # (N, 2560)
    corpus_mean = all_stored.mean(axis=0)    # (2560,)

    # Compute structural PCs to remove format/structural dominance.
    # PCA on stored residuals — top K PCs are structural.
    all_centered = all_stored - corpus_mean  # (N, 2560)
    # Use a subsample for SVD if too many vectors (speed)
    if all_centered.shape[0] > 2000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(all_centered.shape[0], 2000, replace=False)
        svd_input = all_centered[sample_idx]
    else:
        svd_input = all_centered
    _U, _S, Vt_struct = np.linalg.svd(svd_input, full_matrices=False)

    # Auto-detect structural boundary (same algorithm as compass calibration)
    explained = (_S ** 2) / (np.sum(_S ** 2) + 1e-10)
    structural_end = 0
    for si in range(min(len(explained) - 3, 50)):
        ratios = [
            explained[si + j] / max(explained[si + j + 1], 1e-10)
            for j in range(3)
        ]
        if all(r < 1.5 for r in ratios):
            structural_end = si
            break
    else:
        structural_end = 4
    structural_basis = Vt_struct[:structural_end]  # (K, 2560)

    def _remove_structural(v: np.ndarray) -> np.ndarray:
        """Remove structural PCs from a centered vector."""
        if structural_basis.shape[0] == 0:
            return v
        projections = v @ structural_basis.T  # (K,) or (N, K)
        return v - projections @ structural_basis

    # Remove structural from stored residuals, then normalise
    all_cleaned = _remove_structural(all_centered)  # (N, 2560)
    all_norms = np.linalg.norm(all_cleaned, axis=1, keepdims=True)
    all_norms = np.where(all_norms > 1e-10, all_norms, 1.0)
    all_normed = all_cleaned / all_norms

    # Extract query residual at L26
    q_ids = mx.array(prompt_ids)[None]
    q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
    q_vec = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)

    # Remove structural from query, then normalise
    q_cleaned = _remove_structural(q_vec - corpus_mean)
    q_norm = np.linalg.norm(q_cleaned)
    if q_norm < 1e-10:
        return [(wid, 0.0) for wid in range(lib.num_windows)]
    q_direction = q_cleaned / q_norm

    # Cosine similarity along structurally-cleaned query direction
    scores_raw = all_normed @ q_direction  # (N,)

    # Aggregate per window (max for sparse, top-k mean for dense)
    per_window_scores: dict[int, list[float]] = defaultdict(list)
    for idx, (wid, _si) in enumerate(wid_map):
        per_window_scores[wid].append(float(scores_raw[idx]))

    samples_per_window = len(per_window_scores[next(iter(per_window_scores))])
    _TOP_K_AGG = 10

    per_window: dict[int, float] = {}
    if samples_per_window <= 16:
        for wid, score_list in per_window_scores.items():
            per_window[wid] = max(score_list)
    else:
        for wid, score_list in per_window_scores.items():
            score_list.sort(reverse=True)
            top_k = score_list[:_TOP_K_AGG]
            per_window[wid] = sum(top_k) / len(top_k)

    scores = [(wid, s) for wid, s in per_window.items()]
    scores.sort(key=lambda x: -x[1])
    return scores


# Diverse contrast prompts — unrelated to any specific query.
# Used to discover what makes the query geometrically unique.
_CONTRAST_PROMPTS = [
    "The weather today is sunny with a high of seventy five degrees",
    "Please pass the salt and pepper from the other side of the table",
    "The quarterly earnings report showed revenue growth of twelve percent",
    "Mix the flour and sugar together before adding the eggs slowly",
    "The highway construction project is expected to finish by next summer",
    "She practiced the piano for three hours every afternoon after school",
    "The database migration requires updating all foreign key constraints first",
    "The old lighthouse has stood on that cliff for over two hundred years",
]


def _contrastive_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    tokenizer,
    n_dims: int = 8,
    query_vec_np: "np.ndarray | None" = None,
) -> list[tuple[int, float]]:
    """Score windows using query-specific contrastive subspace discovery.

    The proven Lazarus methodology applied at routing time:
      1. Extract L26 residual for the query
      2. Extract L26 residuals for 8 diverse unrelated prompts
      3. Cross-domain PCA: find the 7-8 directions where the query
         differs from everything else
      4. Project all stored compass residuals onto those directions
      5. Cosine match

    The subspace IS the address. The query defines its own coordinate
    frame — the directions the model uses to navigate to THIS content.
    Different queries produce different frames.

    Cost: ~9 forward passes to L26 (fast) + projection + matching.
    """
    import mlx.core as mx
    import numpy as np

    compass_layer = lib.compass_layer

    # 1. Extract query residual at L26
    if query_vec_np is not None:
        q_vec = query_vec_np
    else:
        q_ids = mx.array(prompt_ids)[None]
        q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
        q_vec = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)

    # 2. Contrast set: random interval residuals from the corpus itself.
    # "What makes this query different from typical transcript content?"
    rng = np.random.default_rng(42)
    n_contrast = 30
    contrast_wids = rng.choice(lib.num_windows, size=n_contrast, replace=False)
    contrast_vecs = []
    for wid in contrast_wids:
        samples = lib.get_compass_residuals(int(wid))
        # Take the middle sample from each window
        mid = len(samples) // 2
        contrast_vecs.append(np.array(samples[mid].reshape(-1).tolist(), dtype=np.float32))

    # 3. Cross-domain PCA + Fisher criterion
    # Stack query (1 vector) + contrast (8 vectors) = 9 vectors
    all_probes = np.stack([q_vec] + contrast_vecs, axis=0)  # (9, 2560)
    n_target = 1  # the query

    mean = all_probes.mean(axis=0)
    centered = all_probes - mean
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Fisher criterion: which PCs separate query from contrast?
    n_pcs = min(len(_S), Vt.shape[0])
    projections = centered @ Vt[:n_pcs].T  # (9, n_pcs)

    target_proj = projections[:n_target]   # (1, n_pcs)
    other_proj = projections[n_target:]    # (8, n_pcs)

    fisher_scores = []
    for i in range(n_pcs):
        t_mean = target_proj[:, i].mean()
        o_mean = other_proj[:, i].mean()
        t_var = target_proj[:, i].var() + 1e-10
        o_var = other_proj[:, i].var() + 1e-10
        fisher = (t_mean - o_mean) ** 2 / (t_var + o_var)
        fisher_scores.append((i, float(fisher)))

    fisher_scores.sort(key=lambda x: -x[1])
    top_pcs = fisher_scores[:n_dims]
    top_indices = [idx for idx, _ in top_pcs]

    # The query's routing frame: PCA directions at discriminative indices
    query_frame = Vt[top_indices]  # (n_dims, 2560)

    # 4. Project query into its own frame
    q_in_frame = (q_vec - mean) @ query_frame.T  # (n_dims,)
    q_norm = np.linalg.norm(q_in_frame)
    if q_norm < 1e-10:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    # 5. Project all stored compass residuals into the query frame
    all_vecs = []
    wid_map = []
    for wid in range(lib.num_windows):
        for si, res in enumerate(lib.get_compass_residuals(wid)):
            vec = np.array(res.reshape(-1).tolist(), dtype=np.float32)
            all_vecs.append(vec)
            wid_map.append((wid, si))

    all_stored = np.stack(all_vecs, axis=0)  # (N, 2560)
    all_in_frame = (all_stored - mean) @ query_frame.T  # (N, n_dims)
    all_norms = np.linalg.norm(all_in_frame, axis=1)

    # 6. Cosine similarity in the query's frame
    cosines = (all_in_frame @ q_in_frame) / (all_norms * q_norm + 1e-10)

    # Aggregate per window (max for sparse, top-k mean for dense)
    per_window_scores: dict[int, list[float]] = defaultdict(list)
    for idx, (wid, _si) in enumerate(wid_map):
        per_window_scores[wid].append(float(cosines[idx]))

    samples_per_window = len(per_window_scores[next(iter(per_window_scores))])

    per_window: dict[int, float] = {}
    if samples_per_window <= 16:
        for wid, sl in per_window_scores.items():
            per_window[wid] = max(sl)
    else:
        for wid, sl in per_window_scores.items():
            sl.sort(reverse=True)
            per_window[wid] = sum(sl[:10]) / len(sl[:10])

    scores = [(wid, s) for wid, s in per_window.items()]
    scores.sort(key=lambda x: -x[1])
    return scores
