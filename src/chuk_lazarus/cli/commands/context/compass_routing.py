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
    DIRECTED = "directed"  # query-directed projection — the query IS the basis
    GUIDED = "guided"      # compass × token overlap — both model-internal
    DARKSPACE = "darkspace" # dual-score: compass + directed in 16D PCA
    CONTRASTIVE = "contrastive"  # query-specific subspace discovery at runtime
    GEOMETRIC = "geometric"      # compass + contrastive fused — both model geometry
    QK = "qk"                    # model's own Q/K attention projections — the dark space
    ITERATIVE = "iterative"      # multi-round compass navigation — model reads, compass shifts
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
# Query-directed projection — the query IS the basis
# ---------------------------------------------------------------------------


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
    from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Q/K routing — the model's own attention mechanism, externalized
# ---------------------------------------------------------------------------


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

    result = [(wid, s) for wid, s in per_window.items()]
    result.sort(key=lambda x: -x[1])
    return result


# ---------------------------------------------------------------------------
# Contrastive routing — query-specific subspace discovery at runtime
# ---------------------------------------------------------------------------

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
    from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Two-stage dark space routing — coarse compass → fine re-rank
# ---------------------------------------------------------------------------


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
    from collections import defaultdict
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

    scores = [(wid, s) for wid, s in per_window.items()]
    scores.sort(key=lambda x: -x[1])
    return scores


# ---------------------------------------------------------------------------
# Guided routing — compass × token overlap (both model-internal)
# ---------------------------------------------------------------------------


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
    compass_map = {wid: s for wid, s in compass_scores}

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
        content_tokens = {t for t in query_set
                          if token_doc_freq.get(t, 0) < threshold}
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
    exclude: set[int] | None = None,
    query_residual: "mx.array | None" = None,
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
    exclude         : Window IDs to exclude from selection (already visited)
    query_residual  : Pre-computed L26 residual (e.g. from generation position).
                      When provided, scoring functions use this instead of
                      computing from prompt_ids. Enables generation-guided routing.

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
            if lib.is_darkspace:
                method_name = f"darkspace (L{layer}, {pc_e}D frame bank)"
            elif lib.has_structural_basis:
                method_name = f"compass (L{layer}, structural PC 0-{pc_s-1} removed, full dark space)"
            else:
                method_name = f"compass (L{layer}, PC {pc_s}-{pc_e-1}, {pc_e-pc_s}D)"

    elif strategy == RoutingStrategy.QK:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _qk_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            method_name = f"Q/K attention (L{layer}, model's own routing)"

    elif strategy == RoutingStrategy.GEOMETRIC:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            # Convert pre-computed residual to numpy if provided
            _qvec_np = None
            if query_residual is not None:
                import numpy as np
                _qvec_np = np.array(
                    query_residual.reshape(-1).tolist(), dtype=np.float32
                )

            # Both scores from model's own geometry
            compass_scores = _compass_score_windows(
                lib, kv_gen, prompt_ids, query_vec_np=_qvec_np,
            )
            contrastive_scores = _contrastive_score_windows(
                lib, kv_gen, prompt_ids, tokenizer, query_vec_np=_qvec_np,
            )

            # Reciprocal rank fusion (RRF) — each strategy votes independently.
            # RRF score = 1/(k+rank_compass) + 1/(k+rank_contrastive)
            # k=60 is standard. Windows ranked high by EITHER strategy rise.
            _RRF_K = 5
            compass_rank = {wid: rank for rank, (wid, _) in enumerate(compass_scores)}
            contrastive_rank = {wid: rank for rank, (wid, _) in enumerate(contrastive_scores)}
            scores = [
                (wid, 1.0 / (_RRF_K + compass_rank.get(wid, 999))
                     + 1.0 / (_RRF_K + contrastive_rank.get(wid, 999)))
                for wid in range(lib.num_windows)
            ]
            scores.sort(key=lambda x: -x[1])

            layer = lib.compass_layer
            method_name = f"geometric (compass + contrastive RRF, L{layer})"

    elif strategy == RoutingStrategy.CONTRASTIVE:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _contrastive_score_windows(lib, kv_gen, prompt_ids, tokenizer)
            layer = lib.compass_layer
            method_name = f"contrastive (L{layer}, query-specific 8D frame)"

    elif strategy == RoutingStrategy.DARKSPACE:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _darkspace_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            method_name = f"darkspace (coarse 16D PCA → fine L{layer} 2560D directed)"

    elif strategy == RoutingStrategy.GUIDED:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _guided_score_windows(lib, kv_gen, prompt_ids, tokenizer)
            layer = lib.compass_layer
            pc_s, pc_e = lib.compass_pc_start, lib.compass_pc_end
            method_name = f"guided (compass L{layer} PC {pc_s}-{pc_e-1} × token overlap)"

    elif strategy == RoutingStrategy.DIRECTED:
        if not lib.has_compass:
            print("  Warning: no compass data in library, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _directed_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            method_name = f"directed (L{layer}, query-defined 1D projection)"

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

    # Filter out excluded windows before selection
    if exclude:
        scores = [(wid, s) for wid, s in scores if wid not in exclude]

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
