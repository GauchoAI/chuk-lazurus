"""Model-based scoring strategies: attention, preview, Q/K."""

from __future__ import annotations

import sys

_ATTENTION_SAMPLE_POSITIONS = 32  # tokens per window in routing context
_PREVIEW_TOKENS = 64  # tokens from each end of the window


def _attention_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    capture_layers: set[int] | None = None,
    positions_per_window: int = _ATTENTION_SAMPLE_POSITIONS,
) -> list[tuple[int, float]]:
    """Score windows by the model's attention over multi-position window samples.

    For each window, sample `positions_per_window` tokens (evenly spaced).
    Prefill all samples concatenated into one KV context.  Extend the query
    against it.  The attention weights tell us which window regions the
    query attends to.

    With 32 positions per window × 91 windows = 2,912 context tokens + query.
    One prefill + one extend.  The model sees real content from every window
    and the retrieval circuit can differentiate.

    Returns list of (window_id, attention_score) sorted descending.
    """
    import mlx.core as mx

    num_windows = lib.num_windows
    ppw = positions_per_window

    # Build concatenated token sequence: ppw sampled tokens per window
    all_sample_tokens: list[int] = []
    window_ranges: list[tuple[int, int]] = []  # (start, end) in the concat sequence

    # Use contiguous chunks rather than scattered positions.
    # 4 chunks of ppw//4 tokens, evenly spread through the window.
    # Contiguous text lets the model form local attention patterns.
    n_chunks = 4
    chunk_size = max(ppw // n_chunks, 1)

    for wid in range(num_windows):
        w_tokens = lib.get_window_tokens(wid)
        w_len = len(w_tokens)

        if w_len <= ppw:
            sample = w_tokens
        else:
            # Pick n_chunks evenly spaced start positions, take chunk_size from each
            sample = []
            for ci in range(n_chunks):
                start_pos = int(ci * (w_len - chunk_size) / max(n_chunks - 1, 1))
                sample.extend(w_tokens[start_pos : start_pos + chunk_size])

        start = len(all_sample_tokens)
        all_sample_tokens.extend(sample)
        window_ranges.append((start, start + len(sample)))

    total_context = len(all_sample_tokens)
    print(
        f"  Attention routing: {ppw} positions/window × {num_windows} windows = "
        f"{total_context} context tokens",
        file=sys.stderr,
    )

    # Prefill the sampled context
    ctx_ids = mx.array(all_sample_tokens)[None]  # (1, total_context)
    _logits, ctx_kv = kv_gen.prefill(ctx_ids)
    mx.eval(*[t for pair in ctx_kv for t in pair])

    # Extend query against the context, capturing attention weights
    q_ids = mx.array(prompt_ids)[None]
    _logits, _ext_kv, attn_weights = kv_gen.extend_with_attention_weights(
        q_ids,
        ctx_kv,
        abs_start=total_context,
        capture_layers=capture_layers,
    )

    # Accumulate per-window attention across captured layers
    window_scores = [0.0] * num_windows

    for layer_idx, weights in attn_weights.items():
        # weights: (1, num_heads, q_len, total_context + q_len) float32
        # Slice to context positions only
        context_weights = weights[0, :, :, :total_context]  # (num_heads, q_len, total_context)
        # Mean across heads and query positions → (total_context,)
        per_pos = mx.mean(context_weights, axis=(0, 1))
        mx.eval(per_pos)

        for wid in range(num_windows):
            start, end = window_ranges[wid]
            # Sum attention to this window's positions
            window_scores[wid] += mx.sum(per_pos[start:end]).item()

    # Normalize by number of captured layers
    n_layers = max(len(attn_weights), 1)
    scores = [(wid, window_scores[wid] / n_layers) for wid in range(num_windows)]
    scores.sort(key=lambda x: -x[1])
    return scores


def _preview_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
    candidate_wids: list[int] | None = None,
    preview_size: int = _PREVIEW_TOKENS,
) -> list[tuple[int, float]]:
    """Score windows by how well the preview helps the model predict the query.

    For each candidate window:
      1. Build a compressed preview: first N + last N tokens
      2. Prefill the preview to build a mini-KV cache
      3. Extend the query against that cache
      4. Measure mean log-probability of query tokens (query perplexity)

    Windows that make the query tokens more predictable (higher log-prob,
    lower perplexity) are more relevant — the preview content activates
    the retrieval circuit and helps the model "understand" the query.

    128 tokens × 91 windows = ~11.6K tokens of scout work, each as a
    fast independent prefill+extend.  Roughly 3-5 seconds for 91 windows.

    Returns list of (window_id, score) sorted descending.
    """
    import mlx.core as mx

    q_ids = mx.array(prompt_ids)[None]
    q_len = len(prompt_ids)
    wids = candidate_wids if candidate_wids is not None else list(range(lib.num_windows))

    scores: list[tuple[int, float]] = []
    for wid in wids:
        w_tokens = lib.get_window_tokens(wid)

        # Compressed preview: first + last N tokens
        if len(w_tokens) <= preview_size * 2:
            preview = w_tokens
        else:
            preview = w_tokens[:preview_size] + w_tokens[-preview_size:]

        # Prefill preview
        p_ids = mx.array(preview)[None]
        _logits, preview_kv = kv_gen.prefill(p_ids)

        # Extend with query
        logits, _ext_kv = kv_gen.extend(q_ids, preview_kv, abs_start=len(preview))
        mx.eval(logits)

        # Query perplexity: mean log-prob of each query token given
        # the preview context + preceding query tokens.
        # logits shape: (1, q_len, vocab_size)
        # logits[0, i, :] predicts token at position i+1
        # So logits[0, i, prompt_ids[i+1]] is the log-prob of the
        # actual next query token at each position.
        logits_f32 = logits[0, :-1, :].astype(mx.float32)  # (q_len-1, vocab)
        # log_softmax = logits - log(sum(exp(logits)))
        log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)

        # Gather log-probs of actual next tokens
        target_ids = mx.array(prompt_ids[1:])  # (q_len-1,)
        # Index into log_probs: for each position i, get log_probs[i, target_ids[i]]
        token_log_probs = log_probs[mx.arange(q_len - 1), target_ids]
        mean_log_prob = mx.mean(token_log_probs).item()

        # Higher mean_log_prob = lower perplexity = more relevant
        scores.append((wid, mean_log_prob))

    scores.sort(key=lambda x: -x[1])
    return scores


def _qk_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
) -> list[tuple[int, float]]:
    """Score windows using the model's own Q/K attention projections.

    The model's W_Q and W_K at a global layer near L26 are the trained
    routing mechanism. Q·K^T IS the dark space matching function — the
    same computation the model would do if all tokens were in context.

    No PCA. No frames. No projection. The model's own attention weights
    operating on stored L26 residuals. The full dark space, read through
    the model's own glasses.
    """
    import mlx.core as mx

    compass_layer = lib.compass_layer
    backbone = kv_gen.backbone

    # Find the global layer at or nearest to compass_layer
    routing_layer_idx = compass_layer
    # Prefer a global layer for full-context attention patterns
    for offset in range(0, 10):
        if backbone.is_global_layer(compass_layer - offset):
            routing_layer_idx = compass_layer - offset
            break
        if backbone.is_global_layer(compass_layer + offset):
            routing_layer_idx = compass_layer + offset
            break

    layer = backbone.adapted_layers[routing_layer_idx]

    # Load all stored compass residuals, batch them
    all_vecs = []
    wid_map = []
    for wid in range(lib.num_windows):
        for si, res in enumerate(lib.get_compass_residuals(wid)):
            # res shape: (1, hidden_size) — squeeze and collect
            all_vecs.append(res.reshape(-1))
            wid_map.append((wid, si))

    # Stack: (N, hidden_size)
    all_h = mx.stack(all_vecs, axis=0)  # (N, 2560)
    N = all_h.shape[0]

    # Reshape to (1, N, hidden_size) for layer projection
    all_h_batch = all_h[None, :, :]  # (1, N, 2560)

    # Compute K for all stored residuals (pre-RoPE, with norms)
    x_stored = layer.pre_attn_norm(all_h_batch)  # (1, N, 2560)
    _q_stored, k_stored, _v_stored = layer.project_qkv_pre_rope(x_stored, 1, N)
    # k_stored: (1, nkv, N, head_dim) — pre-RoPE, normed
    mx.eval(k_stored)

    # Compute Q for query (pre-RoPE, with norms)
    q_ids = mx.array(prompt_ids)[None]
    q_h = kv_gen.prefill_to_layer(q_ids, target_layer=routing_layer_idx)
    # Take last position
    q_last = q_h[:, -1:, :]  # (1, 1, 2560)

    x_query = layer.pre_attn_norm(q_last)
    q_query, _k_query, _v_query = layer.project_qkv_pre_rope(x_query, 1, 1)
    # q_query: (1, nq, 1, head_dim)
    mx.eval(q_query)

    # Attention score: Q · K^T / sqrt(head_dim)
    # q_query: (1, nq, 1, dh), k_stored: (1, nkv, N, dh)
    # GQA: repeat K to match Q heads
    n_rep = layer.n_rep
    if n_rep > 1:
        k_expanded = mx.repeat(k_stored, n_rep, axis=1)  # (1, nq, N, dh)
    else:
        k_expanded = k_stored

    # (1, nq, 1, dh) @ (1, nq, dh, N) → (1, nq, 1, N)
    scores = (q_query @ k_expanded.transpose(0, 1, 3, 2)) * layer.attn_scale
    # Average across heads → (N,)
    scores_avg = mx.mean(scores[0, :, 0, :], axis=0)  # (N,)
    mx.eval(scores_avg)

    scores_np = scores_avg.tolist()

    # Aggregate per window
    from collections import defaultdict

    per_window_scores: dict[int, list[float]] = defaultdict(list)
    for idx, (wid, _si) in enumerate(wid_map):
        per_window_scores[wid].append(float(scores_np[idx]))

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

    result = list(per_window.items())
    result.sort(key=lambda x: -x[1])
    return result
