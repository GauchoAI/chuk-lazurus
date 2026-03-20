"""Composite scoring functions for compass routing.

Contains darkspace and guided scoring strategies that combine other scorers.
"""

from __future__ import annotations

from collections import defaultdict

from ._geometric import _compass_score_windows


def _darkspace_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Dual-score routing in the 16D PCA content subspace.

    Two complementary readings of the same geometric map:
      1. Compass cosine: standard cosine in 16D (finds anomalous content)
      2. Directed cosine: query-weighted cosine in 16D (finds query-aligned content)

    Combined: alpha × compass + (1 - alpha) × directed.

    Both are pure model geometry. Same layer. Same basis. Same subspace.
    Compass finds what's geometrically distinctive. Directed finds what
    the query is looking for. Together they surface windows that are both
    distinctive AND relevant.
    """
    import mlx.core as mx
    import numpy as np

    # Load compass data
    mean_vec, basis, pc_start, pc_end = lib.get_compass_basis()
    compass_layer = lib.compass_layer
    mean_np = np.array(mean_vec.reshape(-1).tolist(), dtype=np.float32)
    basis_np = np.array(basis.tolist(), dtype=np.float32)  # (16, 2560)

    # Load and project all window residuals into 16D
    all_projected = []
    wid_map = []
    for wid in range(lib.num_windows):
        for si, res in enumerate(lib.get_compass_residuals(wid)):
            vec = np.array(res.reshape(-1).tolist(), dtype=np.float32)
            projected = (vec - mean_np) @ basis_np.T  # (16,)
            all_projected.append(projected)
            wid_map.append((wid, si))

    all_proj = np.stack(all_projected, axis=0)  # (N, 16)
    all_norms = np.linalg.norm(all_proj, axis=1)  # (N,)

    # Project query into 16D
    q_ids = mx.array(prompt_ids)[None]
    q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
    q_vec = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)
    q_proj = (q_vec - mean_np) @ basis_np.T  # (16,)
    q_norm = np.linalg.norm(q_proj)

    if q_norm < 1e-10:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    # Score 1: Compass cosine (standard — all 16 dims equal weight)
    compass_cos = (all_proj @ q_proj) / (all_norms * q_norm + 1e-10)

    # Score 2: Directed cosine (query-weighted dims)
    q_weights = np.abs(q_proj)
    q_weights = q_weights / (q_weights.sum() + 1e-10)

    all_w = all_proj * q_weights[None, :]
    q_w = q_proj * q_weights
    aw_norms = np.linalg.norm(all_w, axis=1)
    aw_norms = np.where(aw_norms > 1e-10, aw_norms, 1.0)
    qw_norm = np.linalg.norm(q_w)
    directed_cos = (all_w @ q_w) / (aw_norms * qw_norm + 1e-10)

    # Combine
    combined = alpha * compass_cos + (1.0 - alpha) * directed_cos

    # Aggregate per window
    per_window_scores: dict[int, list[float]] = defaultdict(list)
    for idx, (wid, _si) in enumerate(wid_map):
        per_window_scores[wid].append(float(combined[idx]))

    samples_per_window = len(per_window_scores[next(iter(per_window_scores))])
    _TOP_K_AGG = 10

    per_window: dict[int, float] = {}
    if samples_per_window <= 16:
        for wid, sl in per_window_scores.items():
            per_window[wid] = max(sl)
    else:
        for wid, sl in per_window_scores.items():
            sl.sort(reverse=True)
            per_window[wid] = sum(sl[:_TOP_K_AGG]) / len(sl[:_TOP_K_AGG])

    scores = list(per_window.items())
    scores.sort(key=lambda x: -x[1])
    return scores


def _guided_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    tokenizer,
) -> list[tuple[int, float]]:
    """Score windows by compass geometry × token overlap.

    Two model-internal signals:
      1. Compass: 16D PCA subspace cosine (the model's geometric state)
      2. Token overlap: query token IDs present in the window (what the model read)

    Both come from the model's own computation. No external embedding model.
    No BM25 statistics. Pure model vocabulary + pure model geometry.

    Combined score = compass_score × (1 + token_overlap_fraction)
    Windows that match geometrically AND contain query tokens rise to the top.
    """
    # Get compass scores
    compass_scores = _compass_score_windows(lib, kv_gen, prompt_ids)
    compass_map = dict(compass_scores)

    # Token overlap: which windows contain query token IDs?
    # Skip special/common tokens — only match content tokens
    query_set = set(prompt_ids)
    # Remove tokens that appear in >50% of windows (structural tokens)
    if lib.num_windows > 10:
        token_doc_freq: dict[int, int] = {}
        for wid in range(lib.num_windows):
            w_tokens = set(lib.get_window_tokens(wid))
            for t in query_set:
                if t in w_tokens:
                    token_doc_freq[t] = token_doc_freq.get(t, 0) + 1
        threshold = lib.num_windows * 0.5
        content_tokens = {t for t in query_set if token_doc_freq.get(t, 0) < threshold}
    else:
        content_tokens = query_set

    if not content_tokens:
        content_tokens = query_set  # fallback: use all

    # Count content token matches per window
    overlap_scores: dict[int, float] = {}
    max_overlap = 0
    for wid in range(lib.num_windows):
        w_tokens = set(lib.get_window_tokens(wid))
        overlap = len(content_tokens & w_tokens)
        overlap_scores[wid] = overlap
        if overlap > max_overlap:
            max_overlap = overlap

    # Normalise overlap to 0-1
    if max_overlap > 0:
        for wid in overlap_scores:
            overlap_scores[wid] /= max_overlap

    # Combine: compass × (1 + overlap)
    # Windows with token matches get boosted. Windows without are unchanged.
    combined = []
    for wid in range(lib.num_windows):
        c = compass_map.get(wid, 0.0)
        o = overlap_scores.get(wid, 0.0)
        combined.append((wid, c * (1.0 + o)))

    combined.sort(key=lambda x: -x[1])
    return combined
