"""Legacy residual cosine scoring (mean-centered)."""

from __future__ import annotations


def _residual_cosine_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
) -> list[tuple[int, float]]:
    """Legacy: mean-centered cosine similarity of boundary residuals."""
    import mlx.core as mx

    q_ids = mx.array(prompt_ids)[None]
    _logits, _kv, query_residual = kv_gen.prefill_with_residual(q_ids)
    mx.eval(_logits)
    query_vec = query_residual.reshape(-1).astype(mx.float32)

    window_vecs = []
    for wid in range(lib.num_windows):
        window_vecs.append(lib.get_residual(wid).reshape(-1).astype(mx.float32))

    stacked = mx.stack(window_vecs, axis=0)
    mean_vec = mx.mean(stacked, axis=0)
    window_vecs = [v - mean_vec for v in window_vecs]
    query_vec = query_vec - mean_vec

    query_norm = mx.sqrt(mx.sum(query_vec * query_vec))

    scores = []
    for wid in range(lib.num_windows):
        wv = window_vecs[wid]
        wn = mx.sqrt(mx.sum(wv * wv))
        cos = (mx.sum(query_vec * wv) / (query_norm * wn + 1e-8)).item()
        scores.append((wid, cos))

    scores.sort(key=lambda x: -x[1])
    return scores
