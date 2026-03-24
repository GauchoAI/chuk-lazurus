"""K vector routing — the model's own fact-addressing mechanism at L29 H4.

The retrieval circuit uses L29 H4 to copy novel facts from the KV cache
at 62% attention. The K vectors at that head are what make positions
addressable. Q.K IS the model's own routing score.

Three modes of operation:
  1. Stored K-vector index (fastest, ~50ms): pre-extracted K vectors at
     fact positions. Load from kv_route_index.npz. One Q.K dot product.
  2. Compass residuals at L29 (medium): project stored L29 residuals
     through W_K at the retrieval head. No extra forward pass needed.
  3. Compass residuals at L26 (approximate): project L26 residuals
     through L29's W_K. Approximate but still captures head geometry.

Head mapping (Gemma 4B: 8 query heads, 4 KV heads, n_rep=2):
  Query head 4 -> KV head 2 (heads 4-5 share KV head 2)

Storage (mode 1):
  K at L29 KV head 2: 256D per position.
  At bf16: 512 bytes per fact position.
  3,625 facts (Apollo 11): ~1.86 MB.
  16x smaller than compass residuals (29 MB).
"""

from __future__ import annotations

import sys
from collections import defaultdict


def _kv_route_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    retrieval_layer: int | None = None,
    query_head: int | None = None,
) -> list[tuple[int, float]]:
    """Score windows by Q.K at the retrieval head — the model's own addressing.

    Uses stored K-vector index if available (1.86MB, ~50ms).
    Falls back to computing K from compass residuals.
    """
    import mlx.core as mx

    from .....inference.context.arch_config import ArchitectureNotCalibrated

    backbone = kv_gen.backbone
    num_layers = len(backbone.adapted_layers)

    # Resolve retrieval_layer and query_head from library manifest if not specified
    if retrieval_layer is None or query_head is None:
        ac = lib.arch_config
        if ac is None:
            raise ArchitectureNotCalibrated(
                "unknown (no arch_config in library manifest — re-run prefill)",
                num_layers,
            )
        retrieval_layer = retrieval_layer if retrieval_layer is not None else ac.retrieval_layer
        query_head = query_head if query_head is not None else ac.query_head

    if retrieval_layer >= num_layers:
        retrieval_layer = num_layers - 1

    # Find nearest global layer
    routing_layer_idx = retrieval_layer
    for offset in range(0, 10):
        if offset <= retrieval_layer and backbone.is_global_layer(retrieval_layer - offset):
            routing_layer_idx = retrieval_layer - offset
            break
        if retrieval_layer + offset < num_layers and backbone.is_global_layer(
            retrieval_layer + offset
        ):
            routing_layer_idx = retrieval_layer + offset
            break

    layer = backbone.adapted_layers[routing_layer_idx]
    n_rep = layer.n_rep
    nkv = layer.num_kv_heads
    kv_head_idx = min(query_head // n_rep, nkv - 1) if n_rep > 1 else min(query_head, nkv - 1)

    # ── Mode 1: Use stored K-vector index ──────────────────────
    if lib.has_kv_route_index:
        stored_layer = lib.kv_route_layer
        stored_head = lib.kv_route_kv_head
        k_matrix, wid_map = lib.get_kv_route_vectors()

        if k_matrix.shape[0] == 0:
            return [(wid, 0.0) for wid in range(lib.num_windows)]

        # k_matrix: (N, head_dim) — stored post-RoPE K vectors
        N = k_matrix.shape[0]

        # Compute Q for query
        q_retrieval = _extract_query_q(
            kv_gen,
            prompt_ids,
            routing_layer_idx,
            layer,
            query_head,
        )

        # Q . K^T scoring
        # q_retrieval: (1, head_dim), k_matrix: (N, head_dim)
        scores_flat = (q_retrieval @ k_matrix.T) * layer.attn_scale  # (1, N)
        scores_flat = scores_flat[0]  # (N,)
        mx.eval(scores_flat)

        method = f"stored index (L{stored_layer} KV{stored_head}, {N} facts)"
        return _aggregate_scores(scores_flat.tolist(), wid_map, lib.num_windows, method)

    # ── Mode 2/3: Compute K from compass residuals ─────────────
    if not lib.has_compass:
        print(
            "  Warning: no compass data or kv_route_index, falling back to empty scores",
            file=sys.stderr,
        )
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    compass_layer = lib.compass_layer
    use_exact = compass_layer == routing_layer_idx

    all_vecs = []
    wid_map = []
    for wid in range(lib.num_windows):
        for si, res in enumerate(lib.get_compass_residuals(wid)):
            all_vecs.append(res.reshape(-1))
            wid_map.append((wid, si))

    all_h = mx.stack(all_vecs, axis=0)  # (N, hidden_size)
    N = all_h.shape[0]

    if not use_exact:
        print(
            f"  KV route: compass at L{compass_layer}, routing at L{routing_layer_idx}. "
            f"Using W_K from L{routing_layer_idx} on L{compass_layer} residuals "
            f"(rebuild with --phases kvectors for stored K vectors).",
            file=sys.stderr,
        )

    # Compute K from residuals
    all_h_batch = all_h[None, :, :]  # (1, N, hidden_size)
    x_stored = layer.pre_attn_norm(all_h_batch)
    _q_stored, k_stored, _v_stored = layer.project_qkv_pre_rope(x_stored, 1, N)
    k_retrieval = k_stored[:, kv_head_idx, :, :]  # (1, N, dh)
    mx.eval(k_retrieval)

    # Compute Q for query
    q_retrieval = _extract_query_q(
        kv_gen,
        prompt_ids,
        routing_layer_idx,
        layer,
        query_head,
    )

    # Q . K^T scoring
    scores = (q_retrieval @ k_retrieval[0].T) * layer.attn_scale  # (1, N)
    scores_flat = scores[0]  # (N,)
    mx.eval(scores_flat)

    method = f"computed from L{compass_layer} residuals"
    return _aggregate_scores(scores_flat.tolist(), wid_map, lib.num_windows, method)


def _extract_query_q(kv_gen, prompt_ids, routing_layer_idx, layer, query_head):
    """Extract Q at the retrieval query head for the last query position."""
    import mlx.core as mx

    q_ids = mx.array(prompt_ids)[None]
    q_h = kv_gen.prefill_to_layer(q_ids, target_layer=routing_layer_idx)
    q_last = q_h[:, -1:, :]  # (1, 1, hidden_size)

    x_query = layer.pre_attn_norm(q_last)
    q_query, _k_q, _v_q = layer.project_qkv_pre_rope(x_query, 1, 1)
    # q_query: (1, nq, 1, head_dim)

    q_retrieval = q_query[0, query_head, :, :]  # (1, head_dim)
    mx.eval(q_retrieval)
    return q_retrieval


def _aggregate_scores(scores_list, wid_map, num_windows, method):
    """Aggregate per-position scores into per-window scores."""
    per_window_scores: dict[int, list[float]] = defaultdict(list)
    for idx, (wid, _si) in enumerate(wid_map):
        per_window_scores[wid].append(float(scores_list[idx]))

    if not per_window_scores:
        return [(wid, 0.0) for wid in range(num_windows)]

    _TOP_K_AGG = 10
    per_window: dict[int, float] = {}
    for wid, sl in per_window_scores.items():
        if len(sl) <= 16:
            per_window[wid] = max(sl)
        else:
            sl.sort(reverse=True)
            per_window[wid] = sum(sl[:_TOP_K_AGG]) / _TOP_K_AGG

    result = list(per_window.items())
    result.sort(key=lambda x: -x[1])
    return result
