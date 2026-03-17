"""Mode 7 — Broad-context reasoning.

For queries without keyword overlap (BM25 < threshold), use geometric
routing to find semantically relevant windows and load broader context
chunks for reasoning over tone, patterns, and relationships.

Architecture:
  1. Geometric routing (contrastive/compass/geometric, ~1s)
  2. Load contiguous chunks from top-K windows (token budget)
  3. Single prefill with chat-template framing
  4. Generate

Mode 6 (fact): 5 windows, ±5 spans, ~500 tokens, ~475ms
Mode 7 (broad): 20 windows, 150 tok/window, ~3000 tokens, ~2.5s
"""

from __future__ import annotations

import sys
import time


def run_broad(
    lib,
    kv_gen,
    pipeline,
    tokenizer,
    prompt_ids: list[int],
    prompt_text: str,
    config,
    args,
    mx,
    bm25_scores: list[tuple[int, float]] | None = None,
):
    """Geometric route → broad context chunks → single prefill → decode."""
    from ..._types import GenerateResult
    from ...compass_routing import RoutingStrategy, compass_route

    # ── Parameters ────────────────────────────────────────────
    top_k = getattr(args, "broad_windows", None) or 20
    token_budget = getattr(args, "token_budget", None) or 3000
    broad_strategy = getattr(args, "broad_strategy", None) or "contrastive"
    no_chat = getattr(args, "no_chat_template", False)
    system_prompt = getattr(args, "system_prompt", None)

    # ── Route geometrically ──────────────────────────────────
    strategy_map = {
        "contrastive": RoutingStrategy.CONTRASTIVE,
        "compass": RoutingStrategy.COMPASS,
        "geometric": RoutingStrategy.GEOMETRIC,
    }
    strategy = strategy_map.get(broad_strategy, RoutingStrategy.CONTRASTIVE)

    if lib.has_compass:
        window_ids = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=pipeline.config,
            strategy=strategy,
            top_k=top_k,
        )
    elif bm25_scores is not None:
        # Fallback: use BM25 candidates if no compass data
        window_ids = [wid for wid, _ in bm25_scores[:top_k]]
        print(
            f"  No compass data — using BM25 top-{top_k} for broad context",
            file=sys.stderr,
        )
    else:
        print("Error: no compass data and no BM25 scores.", file=sys.stderr)
        return

    if not window_ids:
        print("Error: no windows selected.", file=sys.stderr)
        return

    window_ids = sorted(window_ids)
    print(f"  Broad context windows: {window_ids}", file=sys.stderr)

    # ── Load broad context chunks ────────────────────────────
    t0 = time.time()
    context_tokens = _load_broad_context(lib, window_ids, token_budget)

    # ── Build framed prompt ──────────────────────────────────
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        sys_content = system_prompt or (
            "The following are excerpts from a document. "
            "Read them carefully and answer the question."
        )
        preamble_text = (
            f"<start_of_turn>user\n{sys_content}\n\n"
            f"Excerpts:\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)
        postamble_text = (
            f"\n\nQuestion: {prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
    else:
        preamble_ids = tokenizer.encode("Document excerpts:\n\n", add_special_tokens=False)
        postamble_ids = tokenizer.encode(
            f"\n\nQuestion: {prompt_text}\nAnswer:",
            add_special_tokens=False,
        )

    # ── Single prefill ───────────────────────────────────────
    combined = preamble_ids + context_tokens + postamble_ids
    logits, gen_kv = kv_gen.prefill(mx.array(combined)[None])
    mx.eval(logits)
    seq_len = len(combined)
    prefill_ms = (time.time() - t0) * 1000

    print(
        f"  Broad prefill: {len(context_tokens)} context tokens from "
        f"{len(window_ids)} windows + {len(preamble_ids) + len(postamble_ids)} "
        f"framing = {len(combined)} total ({prefill_ms:.0f}ms)",
        file=sys.stderr,
    )

    # ── Autoregressive decode ────────────────────────────────
    context_len = seq_len
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
        context_tokens=context_len,
    )
    print(result.to_display())


# ── Helpers ───────────────────────────────────────────────────


def _load_broad_context(
    lib,
    window_ids: list[int],
    token_budget: int = 3000,
) -> list[int]:
    """Load contiguous token chunks from selected windows within a budget.

    Takes tokens from the middle of each window (most content-dense area,
    away from boundary artifacts).
    """
    tokens_per_window = token_budget // max(len(window_ids), 1)
    all_tokens: list[int] = []

    for wid in window_ids:
        w_tokens = lib.get_window_tokens(wid)
        n = len(w_tokens)

        if n <= tokens_per_window:
            all_tokens.extend(w_tokens)
        else:
            # Take from the middle
            start = max(0, n // 2 - tokens_per_window // 2)
            end = min(n, start + tokens_per_window)
            all_tokens.extend(w_tokens[start:end])

        if len(all_tokens) >= token_budget:
            break

    return all_tokens[:token_budget]
