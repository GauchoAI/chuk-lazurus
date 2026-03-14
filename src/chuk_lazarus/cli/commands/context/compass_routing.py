"""
Compass routing strategies for automatic window selection.

Four strategies, from cheapest to most expensive:

  1. BM25 (token-level)
     Decode each window's tokens, score against the query text using
     BM25.  Fast, content-aware, works on any query shape.  No model
     inference required.

  2. Residual deflection (geometric)
     Extend the query against each window's 1-token checkpoint.
     Measure how much the query residual shifts (L2 from bare query
     residual).  One cheap extend per window.

  3. Preview (model-routed)
     For each window, prefill a compressed preview (first + last N
     tokens), extend the query against it, and measure the model's
     logit entropy on the first predicted token.  Low entropy = the
     model is confident = it found relevant content.  The retrieval
     circuit fires on the preview content and routes itself.

  4. Hybrid (BM25 pre-filter → preview re-rank)
     BM25 narrows to ~10 candidates.  Preview scoring re-ranks.
     Best of both: fast keyword pre-filter + model-level routing.

Usage in context_generate_cmd:
    from .compass_routing import compass_route, RoutingStrategy

    replay_ids = compass_route(
        lib, kv_gen, prompt_ids, prompt_text, tokenizer,
        strategy=RoutingStrategy.HYBRID,
        top_k=3,
    )
"""

from __future__ import annotations

import math
import sys
import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RoutingStrategy(str, Enum):
    """Available compass routing strategies."""

    BM25 = "bm25"
    TWOPASS = "twopass"
    ATTENTION = "attention"
    DEFLECTION = "deflection"
    PREVIEW = "preview"
    HYBRID = "hybrid"
    COMPASS = "compass"
    RESIDUAL = "residual"  # legacy: mean-centered cosine similarity


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------


def _bm25_score_windows(
    lib,
    tokenizer,
    query_text: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    """Score each window against the query using BM25.

    Returns list of (window_id, score) sorted descending by score.
    """
    # Tokenize query into terms (simple whitespace + lowercased)
    query_terms = set(query_text.lower().split())
    if not query_terms:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    # Build per-window term frequency and document frequency
    num_windows = lib.num_windows
    window_term_freqs: list[dict[str, int]] = []
    window_lengths: list[int] = []

    for wid in range(num_windows):
        tokens = lib.get_window_tokens(wid)
        text = tokenizer.decode(tokens, skip_special_tokens=True).lower()

        words = text.split()
        window_lengths.append(len(words))

        tf: dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        window_term_freqs.append(tf)

    avg_dl = sum(window_lengths) / max(num_windows, 1)

    # Document frequency for query terms
    df: dict[str, int] = {}
    for term in query_terms:
        count = 0
        for tf in window_term_freqs:
            if term in tf:
                count += 1
        df[term] = count

    # Score each window
    scores: list[tuple[int, float]] = []
    for wid in range(num_windows):
        score = 0.0
        dl = window_lengths[wid]
        tf = window_term_freqs[wid]

        for term in query_terms:
            if term not in tf:
                continue
            term_tf = tf[term]
            term_df = df.get(term, 0)

            # IDF with smoothing
            idf = math.log((num_windows - term_df + 0.5) / (term_df + 0.5) + 1.0)

            # BM25 term score
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator

        scores.append((wid, score))

    scores.sort(key=lambda x: -x[1])
    return scores


# ---------------------------------------------------------------------------
# Two-pass speculative routing — the model's hallucination IS the signal
# ---------------------------------------------------------------------------


def two_pass_generate(
    lib,
    kv_gen,
    prompt_ids: list[int],
    prompt_text: str,
    tokenizer,
    engine,
    max_new_tokens: int = 200,
    speculative_tokens: int = 10,
    top_k: int = 3,
    temperature: float = 0.0,
) -> dict:
    """Two-pass: generate N tokens, take residual, compare vs checkpoints, replay.

    1. Generate speculative_tokens without context
    2. Extract residual at each generated token
    3. Compare each step-residual against 91 checkpoint residuals
    4. Print the routing table at every step so we can see where the
       compass points as the model transitions from format to content
    5. Route using the residual at the final step
    6. Replay top-k windows, regenerate with context
    """
    import mlx.core as mx

    last_wid = lib.num_windows - 1

    use_interval = lib.has_interval_residuals
    if not use_interval and not lib.has_residuals:
        print("  Error: library has no residuals for twopass routing", file=sys.stderr)
        return {"tokens": [], "speculative_text": "", "selected_windows": [],
                "residual_scores": [], "source": "error", "context_tokens": 0}

    # ── Load residuals (from npz, instant) ───────────────────
    if use_interval:
        # Interval residuals: multiple interior samples per window
        n_samples = lib.interval_samples_per_window
        # Flatten all interval residuals into a single list with window mapping
        all_vecs_raw = []  # list of (wid, sample_idx, vec)
        for wid in range(lib.num_windows):
            for si, res in enumerate(lib.get_interval_residuals(wid)):
                all_vecs_raw.append((wid, si, res.reshape(-1).astype(mx.float32)))

        all_flat = [v for _, _, v in all_vecs_raw]
        stacked = mx.stack(all_flat, axis=0)
        mean_vec = mx.mean(stacked, axis=0)
        all_centered = [v - mean_vec for v in all_flat]
        all_norms = [mx.sqrt(mx.sum(v * v)) for v in all_centered]

        print(
            f"  Loaded {len(all_flat)} interval residuals "
            f"({n_samples} per window × {lib.num_windows} windows)",
            file=sys.stderr,
        )

        def _rank_residual(res_vec):
            """Cosine similarity against all interval residuals, aggregated per window."""
            v = res_vec.reshape(-1).astype(mx.float32) - mean_vec
            vn = mx.sqrt(mx.sum(v * v))
            # Score each interval sample
            per_window_max: dict[int, float] = {}
            for idx, (wid, si, _) in enumerate(all_vecs_raw):
                cos = (mx.sum(v * all_centered[idx]) / (vn * all_norms[idx] + 1e-8)).item()
                # Take MAX across samples within a window — the best-matching interior point
                if wid not in per_window_max or cos > per_window_max[wid]:
                    per_window_max[wid] = cos
            scores = [(wid, s) for wid, s in per_window_max.items()]
            scores.sort(key=lambda x: -x[1])
            return scores
    else:
        # Boundary residuals only (fallback)
        window_vecs_raw = []
        for wid in range(lib.num_windows):
            window_vecs_raw.append(lib.get_residual(wid).reshape(-1).astype(mx.float32))

        stacked = mx.stack(window_vecs_raw, axis=0)
        mean_vec = mx.mean(stacked, axis=0)
        window_vecs = [v - mean_vec for v in window_vecs_raw]
        window_norms = [mx.sqrt(mx.sum(v * v)) for v in window_vecs]

        def _rank_residual(res_vec):
            v = res_vec.reshape(-1).astype(mx.float32) - mean_vec
            vn = mx.sqrt(mx.sum(v * v))
            scores = []
            for wid in range(lib.num_windows):
                cos = (mx.sum(v * window_vecs[wid]) / (vn * window_norms[wid] + 1e-8)).item()
                scores.append((wid, cos))
            scores.sort(key=lambda x: -x[1])
            return scores

    # ── Pass 1: Generate N tokens, capture residual at each step ─
    t0 = time.time()
    q_ids = mx.array(prompt_ids)[None]

    # Prefill query — get residual at the last query token
    logits, spec_kv, query_residual = kv_gen.prefill_with_residual(q_ids)
    mx.eval(logits, query_residual)

    # Show query residual ranking
    q_scores = _rank_residual(query_residual)
    print(f"  Step 0 (query residual): top-3 = [{q_scores[0][0]}, {q_scores[1][0]}, {q_scores[2][0]}]", file=sys.stderr)

    spec_tokens = []
    seq_len = len(prompt_ids)
    eos_id = tokenizer.eos_token_id
    step_scores = None  # will hold scores at final step

    for step in range(speculative_tokens):
        if temperature == 0.0:
            next_tok = int(mx.argmax(logits[0, -1, :]).item())
        else:
            scaled = logits[0, -1, :] / temperature
            next_tok = int(mx.random.categorical(scaled[None]).item())

        if eos_id is not None and next_tok == eos_id:
            break
        spec_tokens.append(next_tok)

        # Step and capture residual
        logits, spec_kv, step_residual = kv_gen.extend_with_residual(
            mx.array([[next_tok]]), spec_kv, abs_start=seq_len
        )
        mx.eval(logits, step_residual)
        seq_len += 1

        # Rank this step's residual
        step_scores = _rank_residual(step_residual)
        tok_text = tokenizer.decode([next_tok], skip_special_tokens=True)
        top3 = [(wid, f"{s:+.4f}") for wid, s in step_scores[:3]]
        print(
            f"  Step {step+1:>2} tok={next_tok:>6} '{tok_text}': "
            f"top-3 = {top3}",
            file=sys.stderr,
        )

    spec_text = tokenizer.decode(spec_tokens, skip_special_tokens=True)
    pass1_ms = (time.time() - t0) * 1000
    print(f"  Pass 1 ({len(spec_tokens)} tokens, {pass1_ms:.0f}ms): {spec_text}", file=sys.stderr)

    # ── Route using final step residual ──────────────────────
    if step_scores is None:
        step_scores = q_scores  # fallback if no tokens generated

    selected = [wid for wid, _ in step_scores[:top_k]]
    if last_wid not in selected:
        selected.append(last_wid)

    # Print full routing table
    print(f"  Routing (step-{len(spec_tokens)} residual):", file=sys.stderr)
    show_n = max(top_k + 2, 5)
    for i, (wid, score) in enumerate(step_scores[:show_n]):
        marker = " *" if wid in selected else ""
        w = lib.windows[wid]
        print(
            f"    window {wid:>2} (score={score:+.4f}){marker}  {w.preview[:50]}",
            file=sys.stderr,
        )
        if i == top_k - 1 and top_k < len(step_scores):
            print(f"    {'─' * 60}", file=sys.stderr)

    all_scores = [s for _, s in step_scores]
    if all_scores:
        print(
            f"  Score range: min={min(all_scores):+.4f} max={max(all_scores):+.4f} "
            f"spread={max(all_scores) - min(all_scores):.2e}",
            file=sys.stderr,
        )

    # ── Pass 2: Grounded generation ──────────────────────────
    context_kv = engine._make_empty_kv()
    seq_len = 0

    for wid in sorted(selected):
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]
        t0_w = time.time()
        if seq_len == 0:
            _logits, context_kv = kv_gen.prefill(w_ids)
        else:
            _logits, context_kv = kv_gen.extend(w_ids, context_kv, abs_start=seq_len)
        mx.eval(*[t for pair in context_kv for t in pair])
        elapsed_ms = (time.time() - t0_w) * 1000
        print(
            f"  Replayed window {wid} @ pos {seq_len}–{seq_len + len(w_tokens) - 1} ({elapsed_ms:.0f}ms)",
            file=sys.stderr,
        )
        seq_len += len(w_tokens)

    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=seq_len)
    mx.eval(logits)
    seq_len += len(prompt_ids)
    context_tokens = seq_len

    generated = []
    for _ in range(max_new_tokens):
        if temperature == 0.0:
            next_tok = int(mx.argmax(logits[0, -1, :]).item())
        else:
            scaled = logits[0, -1, :] / temperature
            next_tok = int(mx.random.categorical(scaled[None]).item())

        if eos_id is not None and next_tok == eos_id:
            break
        generated.append(next_tok)
        sys.stdout.write(tokenizer.decode([next_tok], skip_special_tokens=True))
        sys.stdout.flush()
        logits, gen_kv = kv_gen.step_uncompiled(mx.array([[next_tok]]), gen_kv, seq_len)
        mx.eval(logits)
        seq_len += 1

    print()

    return {
        "tokens": generated,
        "speculative_text": spec_text,
        "selected_windows": sorted(selected),
        "residual_scores": step_scores[:10],
        "source": "grounded",
        "context_tokens": context_tokens,
    }


# ---------------------------------------------------------------------------
# Attention scoring — the model routes itself via checkpoint KVs
# ---------------------------------------------------------------------------


_ATTENTION_SAMPLE_POSITIONS = 32  # tokens per window in routing context


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
        q_ids, ctx_kv, abs_start=total_context, capture_layers=capture_layers,
    )

    # Aggregate attention: sum attention to each window's positions
    q_len = len(prompt_ids)

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


# ---------------------------------------------------------------------------
# Preview scoring — let the model route itself
# ---------------------------------------------------------------------------

_PREVIEW_TOKENS = 64  # tokens from each end of the window


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
        logits, _ext_kv = kv_gen.extend(
            q_ids, preview_kv, abs_start=len(preview)
        )
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


# ---------------------------------------------------------------------------
# Residual deflection scoring
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Compass routing — PCA subspace projection at the commitment layer
# ---------------------------------------------------------------------------


def _compass_score_windows(
    lib,
    kv_gen,
    prompt_ids: list[int],
) -> list[tuple[int, float]]:
    """Score windows by cosine similarity in the compass subspace.

    Uses pre-computed PCA basis (stored during prefill) to project both
    query and window residuals into the content subspace, removing structural
    dominance that lives in the top PCs.

    The commitment layer (~75% model depth) is where query routing signal
    is maximally expressed. The content subspace (mid-range PCs) is where
    different queries separate geometrically.
    """
    import mlx.core as mx
    import numpy as np

    # Load compass data from library
    mean_vec, basis, pc_start, pc_end = lib.get_compass_basis()
    compass_layer = lib.compass_layer
    n_dims = pc_end - pc_start

    # Convert to numpy for fast linear algebra
    mean_np = np.array(mean_vec.reshape(-1).tolist(), dtype=np.float32)
    basis_np = np.array(basis.tolist(), dtype=np.float32)  # (n_dims, hidden_dim)

    # Load and project all window residuals
    all_vecs_raw = []  # (wid, si, projected_vector)
    wid_map = []
    for wid in range(lib.num_windows):
        for si, res in enumerate(lib.get_compass_residuals(wid)):
            vec = np.array(res.reshape(-1).tolist(), dtype=np.float32)
            projected = (vec - mean_np) @ basis_np.T  # (n_dims,)
            all_vecs_raw.append(projected)
            wid_map.append((wid, si))

    all_projected = np.stack(all_vecs_raw, axis=0)  # (N, n_dims)
    all_norms = np.linalg.norm(all_projected, axis=1)  # (N,)

    # Extract query residual at commitment layer
    q_ids = mx.array(prompt_ids)[None]
    q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
    # Take last position as the query vector
    q_vec = np.array(q_h[0, -1, :].tolist(), dtype=np.float32)
    q_projected = (q_vec - mean_np) @ basis_np.T  # (n_dims,)
    q_norm = np.linalg.norm(q_projected)

    if q_norm < 1e-10:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    # Cosine similarity in subspace — fully vectorised
    cosines = (all_projected @ q_projected) / (all_norms * q_norm + 1e-10)

    # Collect scores per window
    from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Legacy residual cosine (mean-centered)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compass_route(
    lib,
    kv_gen,
    prompt_ids: list[int],
    prompt_text: str,
    tokenizer,
    model_config=None,
    strategy: RoutingStrategy = RoutingStrategy.BM25,
    top_k: int = 3,
    bm25_shortlist: int = 10,
) -> list[int]:
    """Route a query to the most relevant windows.

    Parameters
    ----------
    lib             : CheckpointLibrary
    kv_gen          : KVDirectGenerator
    prompt_ids      : Encoded prompt token IDs
    prompt_text     : Raw prompt text (for BM25 tokenization)
    tokenizer       : Tokenizer (for decoding window tokens in BM25)
    model_config    : Model config (unused currently, reserved)
    strategy        : Which routing strategy to use
    top_k           : Number of windows to select
    bm25_shortlist  : Number of BM25 candidates for hybrid re-ranking

    Returns
    -------
    Sorted list of window IDs to replay.
    """
    t0 = time.time()
    last_wid = lib.num_windows - 1

    if strategy == RoutingStrategy.BM25:
        scores = _bm25_score_windows(lib, tokenizer, prompt_text)
        method_name = "BM25"

    elif strategy == RoutingStrategy.ATTENTION:
        scores = _attention_score_windows(lib, kv_gen, prompt_ids)
        num_captured = len([i for i in range(len(kv_gen.backbone.adapted_layers))
                           if kv_gen.backbone.is_global_layer(i)])
        method_name = f"attention ({lib.num_windows} checkpoints, {num_captured} global layers)"

    elif strategy == RoutingStrategy.PREVIEW:
        scores = _preview_score_windows(lib, kv_gen, prompt_ids)
        method_name = f"preview ({_PREVIEW_TOKENS}+{_PREVIEW_TOKENS} tok/window)"

    elif strategy == RoutingStrategy.DEFLECTION:
        scores = _deflection_score_windows(lib, kv_gen, prompt_ids)
        method_name = "residual deflection"

    elif strategy == RoutingStrategy.HYBRID:
        # Stage 1: BM25 pre-filter
        bm25_scores = _bm25_score_windows(lib, tokenizer, prompt_text)
        candidates = [
            wid for wid, s in bm25_scores[:bm25_shortlist] if s > 0.0
        ]

        if len(candidates) < top_k:
            # BM25 didn't find enough — fall back to preview on all windows
            scores = _preview_score_windows(lib, kv_gen, prompt_ids)
            method_name = f"preview (BM25 fallback, {lib.num_windows} windows)"
        else:
            # Stage 2: preview re-rank on shortlist
            scores = _preview_score_windows(
                lib, kv_gen, prompt_ids, candidate_wids=candidates
            )
            method_name = f"hybrid (BM25→{len(candidates)}→preview)"

    elif strategy == RoutingStrategy.COMPASS:
        if not lib.has_compass:
            print("  Warning: no compass data in library, falling back to BM25", file=sys.stderr)
            print("  Re-run prefill to generate compass data.", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _compass_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            pc_s, pc_e = lib.compass_pc_start, lib.compass_pc_end
            method_name = f"compass (L{layer}, PC {pc_s}-{pc_e-1}, {pc_e-pc_s}D)"

    elif strategy == RoutingStrategy.RESIDUAL:
        if not lib.has_residuals:
            print("  Warning: no residuals in library, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no residuals)"
        else:
            scores = _residual_cosine_score_windows(lib, kv_gen, prompt_ids)
            method_name = "residual cosine (legacy)"

    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")

    elapsed_ms = (time.time() - t0) * 1000

    # Select top-k
    selected = [wid for wid, _ in scores[:top_k]]

    # Always include last window for continuity
    if last_wid not in selected:
        selected.append(last_wid)

    # Print routing table
    print(f"  Compass routing ({method_name}, {elapsed_ms:.0f}ms):", file=sys.stderr)
    show_n = max(top_k + 2, 5)
    for i, (wid, score) in enumerate(scores[:show_n]):
        marker = " *" if wid in selected else ""
        w = lib.windows[wid]
        print(
            f"    window {wid:>2} (score={score:+.4f}){marker}  {w.preview[:50]}",
            file=sys.stderr,
        )
        if i == top_k - 1 and top_k < len(scores):
            print(f"    {'─' * 60}", file=sys.stderr)

    # Score range
    all_scores = [s for _, s in scores]
    if all_scores:
        print(
            f"  Score range: min={min(all_scores):+.4f} max={max(all_scores):+.4f} "
            f"spread={max(all_scores) - min(all_scores):.2e}",
            file=sys.stderr,
        )

    # Return in replay order: continuity window first (far from prompt),
    # then routed windows in ascending score, so the best match is last
    # (closest to the prompt).  This matters for sliding-window attention
    # models where non-global layers only attend to nearby positions.
    score_map = {wid: s for wid, s in scores}
    continuity = [wid for wid in selected if wid == last_wid and wid not in {w for w, _ in scores[:top_k]}]
    routed = [wid for wid in selected if wid not in continuity]
    routed.sort(key=lambda wid: score_map.get(wid, -1e9))
    return continuity + routed


__all__ = ["compass_route", "two_pass_generate", "RoutingStrategy"]
