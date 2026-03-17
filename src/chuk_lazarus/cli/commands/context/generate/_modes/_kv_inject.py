"""Mode 6 — Fact-span prefill, with auto Mode 7 escalation.

Architecture:
  1. BM25 score all windows (24ms)
  2. IF high BM25 score: Mode 6 (fact retrieval)
     → Top-5 windows by BM25, ±5 fact spans, ~500 tokens, ~475ms
  3. IF low BM25 score: Mode 7 (broad reasoning)
     → Geometric routing, 20 windows, ~3000 tokens, ~2.5s

The BM25 max score is the mode selector:
  > 8.0: strong keyword match → Mode 6
  < 5.0: no keyword match → Mode 7
  5.0-8.0: partial match → Mode 6 with more windows

Storage: 741KB token archive + sparse_index.json with fact_spans (~150KB).
No pre-computed KV. No replay of full 512-token windows.
"""

from __future__ import annotations

import sys
import time

# BM25 score thresholds for auto mode selection
_THRESHOLD_HIGH = 8.0   # Strong keyword match → Mode 6 (fact)
_THRESHOLD_LOW = 5.0    # Below this → Mode 7 (broad reasoning)


def run_kv_inject(
    lib,
    kv_gen,
    pipeline,
    tokenizer,
    prompt_ids: list[int],
    prompt_text: str,
    config,
    args,
    mx,
):
    """Route → auto mode select → fact spans or broad context → decode."""
    from ..._types import GenerateResult
    from ...compass_routing import RoutingStrategy, compass_route

    # ── Mode selection ───────────────────────────────────────────────
    mode_override = getattr(args, "mode", None)  # auto|fact|broad

    # Always compute BM25 for mode selection (fast, ~24ms)
    bm25_scores = None
    if lib.has_sparse_index:
        from ...compass_routing._sparse import _sparse_score_windows
        bm25_scores = _sparse_score_windows(lib, prompt_text)

    max_bm25 = bm25_scores[0][1] if bm25_scores else 0.0

    if mode_override == "broad" or (mode_override != "fact" and max_bm25 < _THRESHOLD_LOW):
        # Mode 7: broad reasoning
        print(
            f"  Mode 7 (broad reasoning): BM25 max={max_bm25:.1f} < {_THRESHOLD_LOW}",
            file=sys.stderr,
        )
        from ._broad import run_broad
        return run_broad(
            lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text,
            config, args, mx, bm25_scores=bm25_scores,
        )

    if max_bm25 < _THRESHOLD_HIGH:
        print(
            f"  Mode 6 (fact, partial match): BM25 max={max_bm25:.1f}",
            file=sys.stderr,
        )
    else:
        print(
            f"  Mode 6 (fact, strong match): BM25 max={max_bm25:.1f}",
            file=sys.stderr,
        )

    # ── Mode 6: fact retrieval ───────────────────────────────────────
    strategy_arg = getattr(args, "strategy", None)
    top_k_override = getattr(args, "top_k", None)
    top_k = top_k_override if top_k_override is not None else 5

    inject_wids = None

    if strategy_arg:
        strategy = RoutingStrategy(strategy_arg)
        inject_wids = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=pipeline.config,
            strategy=strategy,
            top_k=top_k,
        )
    elif bm25_scores and max_bm25 >= _THRESHOLD_LOW:
        # BM25 already scored — use those scores directly
        last_wid = lib.num_windows - 1
        inject_wids = [wid for wid, _ in bm25_scores[:top_k]]
        if last_wid not in inject_wids:
            inject_wids.append(last_wid)
        # Print routing table
        elapsed_ms = 0  # already computed during mode selection
        print(f"  Compass routing (sparse keyword BM25, auto):", file=sys.stderr)
        for i, (wid, score) in enumerate(bm25_scores[:top_k + 2]):
            marker = " *" if wid in inject_wids else ""
            w = lib.windows[wid]
            print(
                f"    window {wid:>2} (score={score:+.4f}){marker}  {w.preview[:50]}",
                file=sys.stderr,
            )
            if i == top_k - 1:
                print(f"    {'─' * 60}", file=sys.stderr)
    else:
        # Default: L26 attention routing over sparse index
        sparse_path = lib._path / "sparse_index.json"
        if sparse_path.exists():
            inject_wids = _route_l26_attention(
                lib, kv_gen, tokenizer, prompt_text, top_k, mx,
            )

        # Compass fallback
        if inject_wids is None:
            t0 = time.time()
            routed = compass_route(
                lib, kv_gen, prompt_ids, prompt_text, tokenizer,
                model_config=pipeline.config,
                strategy=RoutingStrategy.COMPASS,
                top_k=top_k,
            )
            inject_wids = routed
            routing_ms = (time.time() - t0) * 1000
            print(
                f"  Compass fallback: {len(inject_wids)} windows ({routing_ms:.0f}ms)",
                file=sys.stderr,
            )

    if not inject_wids:
        print("Error: no windows selected for injection.", file=sys.stderr)
        return

    inject_wids = sorted(inject_wids)
    print(f"  KV inject windows: {inject_wids}", file=sys.stderr)

    # ── Build preamble + postamble ────────────────────────────────────
    no_chat = getattr(args, "no_chat_template", False)
    no_framing = getattr(args, "no_framing", False)
    system_prompt = getattr(args, "system_prompt", None)
    span_radius = getattr(args, "span_radius", None)

    if no_framing:
        # Clean architecture: [span tokens][chat-wrapped query]
        # The spans provide facts. The query provides generation context.
        # No preamble. No postamble. No instructions.
        preamble_ids = []
        if not no_chat and hasattr(tokenizer, "apply_chat_template"):
            query_text = (
                f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            postamble_ids = tokenizer.encode(query_text, add_special_tokens=True)
        else:
            postamble_ids = tokenizer.encode(
                f"Question: {prompt_text}\nAnswer:",
                add_special_tokens=False,
            )
    elif not no_chat and hasattr(tokenizer, "apply_chat_template"):
        sys_content = system_prompt or (
            "You are answering questions based on a document. "
            "Answer using only information from the document. "
            "Quote exact text when possible."
        )
        preamble_text = (
            f"<start_of_turn>user\n{sys_content}\n\n"
            f"Here is the relevant text:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)
        postamble_text = (
            f"\n\n---\nBased on the text above, "
            f"{prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
    else:
        preamble_ids = tokenizer.encode("Document:\n\n", add_special_tokens=False)
        postamble_ids = tokenizer.encode(
            f"\n\n---\nQuestion: {prompt_text}\nAnswer:",
            add_special_tokens=False,
        )

    # ── Single prefill: preamble + fact spans + postamble ─────────────
    t0 = time.time()
    span_tokens, n_spans = _collect_span_tokens(
        lib, inject_wids, radius_override=span_radius,
    )
    combined = preamble_ids + span_tokens + postamble_ids
    logits, gen_kv = kv_gen.prefill(mx.array(combined)[None])
    mx.eval(logits)
    seq_len = len(combined)
    prefill_ms = (time.time() - t0) * 1000

    mode = f"{n_spans} spans" if n_spans > 0 else "full tokens"
    framing_label = "no framing" if no_framing else "framed"
    print(
        f"  Prefill: {len(span_tokens)} span tokens ({mode}) + "
        f"{len(preamble_ids) + len(postamble_ids)} query = "
        f"{len(combined)} total ({prefill_ms:.0f}ms, {framing_label})",
        file=sys.stderr,
    )

    # ── Autoregressive decode ─────────────────────────────────────────
    context_tokens = seq_len
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    generated_tokens: list[int] = []
    for _ in range(config.max_tokens):
        last_logits = logits[0, -1]
        if config.temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / config.temperature
            next_token = int(mx.random.categorical(scaled[None]).item())

        if next_token in stop_ids:
            break
        generated_tokens.append(next_token)
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()

        logits, gen_kv = kv_gen.step_uncompiled(
            mx.array([[next_token]]), gen_kv, seq_len=seq_len,
        )
        seq_len += 1

    print()
    result = GenerateResult(
        response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        tokens_generated=len(generated_tokens),
        context_tokens=context_tokens,
    )
    print(result.to_display())


# ── Helpers ───────────────────────────────────────────────────────────


def _route_l26_attention(lib, kv_gen, tokenizer, prompt_text, top_k, mx):
    """Route by L26 attention over the full sparse index.

    Prefill all keyword entries + query in one pass (~5s for 725 entries).
    Read L26 attention at generation position. The model's own attention
    concentrates on the correct entry — 5/5 at 10, 4/4 at 20, 2/2 at 50.
    L26 attention is sparse: only ~4 entries get attention regardless of count.
    """
    from chuk_lazarus.inference.context.sparse_index import SparseSemanticIndex

    sparse_path = lib._path / "sparse_index.json"
    index = SparseSemanticIndex.load(sparse_path)

    # Build index prompt with ALL entries that have keywords
    lines = []
    for entry in index.entries:
        if entry.keywords:
            lines.append(f"W{entry.window_id}: {', '.join(entry.keywords[:3])}")

    if not lines:
        return None

    index_text = "\n".join(lines)
    prompt = (
        f"<start_of_turn>user\n"
        f"Below is a keyword index. Which entry answers the question?\n\n"
        f"Index:\n{index_text}\n\n"
        f"Question: {prompt_text}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Map each token position to its window ID (or -1 for framing)
    # Re-tokenize each line to find boundaries
    header = (
        f"<start_of_turn>user\n"
        f"Below is a keyword index. Which entry answers the question?\n\n"
        f"Index:\n"
    )
    header_ids = tokenizer.encode(header, add_special_tokens=True)
    header_len = len(header_ids)

    # Tokenize each entry line to find its token span
    entry_spans: list[tuple[int, int, int]] = []  # (start_pos, end_pos, window_id)
    pos = header_len
    for entry in index.entries:
        if not entry.keywords:
            continue
        line = f"W{entry.window_id}: {', '.join(entry.keywords[:3])}\n"
        line_ids = tokenizer.encode(line, add_special_tokens=False)
        entry_spans.append((pos, pos + len(line_ids), entry.window_id))
        pos += len(line_ids)

    # Determine L26 (commitment layer ≈ 77% depth)
    num_layers = lib.manifest.num_layers
    l26 = int(num_layers * 0.77)

    # Prefill with L26 attention capture
    t0 = time.time()
    _logits, _kv, attn_weights = kv_gen.prefill_with_attention(
        mx.array(prompt_ids)[None],
        capture_layers={l26},
    )
    routing_ms = (time.time() - t0) * 1000

    # Read attention at the last position (generation position)
    # Shape: (1, num_heads, S, S) → take last query position, sum over heads
    weights_l26 = attn_weights[l26]  # (1, num_heads, S, S)
    last_pos = len(prompt_ids) - 1
    # Attention from last position to all positions, summed over heads
    attn_to_all = weights_l26[0, :, last_pos, :].sum(axis=0)  # (S,)
    mx.eval(attn_to_all)

    # Score each window by summing attention over its token positions
    window_scores: dict[int, float] = {}
    for start, end, wid in entry_spans:
        if end <= len(prompt_ids):
            score = float(attn_to_all[start:end].sum().item())
            window_scores[wid] = score

    # Rank by attention score
    ranked = sorted(window_scores.items(), key=lambda x: -x[1])
    inject_wids = [wid for wid, sc in ranked[:top_k] if sc > 0]

    if inject_wids:
        top_scores = [(wid, window_scores[wid]) for wid in inject_wids[:5]]
        scores_str = ", ".join(f"W{wid}={sc:.3f}" for wid, sc in top_scores)
        print(
            f"  L26 attention routed to {len(inject_wids)} windows "
            f"({routing_ms:.0f}ms, {len(lines)} entries): {scores_str}",
            file=sys.stderr,
        )
        return inject_wids

    return None


def _collect_span_tokens(lib, inject_wids, radius_override=None):
    """Collect fact-span tokens from selected windows.

    Uses fact_spans from sparse index if available (±5 tokens per fact).
    Falls back to full window tokens.

    Args:
        radius_override: if set, override the stored span radius (e.g. 15
            for sentence-boundary spans instead of ±5 token spans).

    Returns (span_tokens, n_spans).
    """
    spans_by_window = _load_fact_spans(lib, inject_wids)
    use_spans = spans_by_window is not None

    all_tokens = []
    total_spans = 0
    for wid in inject_wids:
        w_tokens = lib.get_window_tokens(wid)
        if use_spans and wid in spans_by_window:
            spans = spans_by_window[wid]
            if radius_override is not None:
                from chuk_lazarus.inference.context.sparse_index import FactSpan
                spans = [FactSpan(position=s.position, radius=radius_override)
                         for s in spans]
            all_tokens.extend(_extract_span_tokens(w_tokens, spans))
            total_spans += len(spans)
        else:
            all_tokens.extend(w_tokens)

    return all_tokens, total_spans


def _load_fact_spans(lib, wids):
    """Load fact spans from sparse index if available."""
    sparse_path = lib._path / "sparse_index.json"
    if not sparse_path.exists():
        return None

    from chuk_lazarus.inference.context.sparse_index import SparseSemanticIndex
    index = SparseSemanticIndex.load(sparse_path)

    spans = {}
    for entry in index.entries:
        if entry.window_id in wids and entry.fact_spans:
            spans[entry.window_id] = entry.fact_spans

    return spans if spans else None


def _extract_span_tokens(window_tokens, fact_spans):
    """Extract and merge overlapping fact spans into a contiguous token list.

    Always includes the first 11 tokens (window header) as a baseline span.
    Content near the window start often contains the most accessible facts.
    """
    n = len(window_tokens)
    # Always include window header
    ranges = [(0, min(11, n))]
    for span in fact_spans:
        start = max(0, span.position - span.radius)
        end = min(n, span.position + span.radius + 1)
        ranges.append((start, end))

    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    tokens = []
    for start, end in merged:
        tokens.extend(window_tokens[start:end])
    return tokens
