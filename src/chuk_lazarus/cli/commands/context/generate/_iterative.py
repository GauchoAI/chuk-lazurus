"""Iterative compass navigation — generation-guided exploration.

Each round: compass routes to a window, generates a short response,
extracts the L26 generation residual, and uses it to shift the compass
for the next round. No grounding probe. No early stopping. The model
explores for all rounds, discovering content across the document.

After exploration, replays the discovered windows together and generates
a final answer from combined context.
"""

from __future__ import annotations

import sys
import time

from .._types import GenerateConfig, GenerateResult
from ..compass_routing import RoutingStrategy, compass_route


# Tokens to generate per exploration round (just enough to shift L26).
_EXPLORE_TOKENS = 50


def _iterative_generate(
    lib,
    kv_gen,
    engine,
    tokenizer,
    model_config,
    prompt_ids: list[int],
    prompt_text: str,
    config,
    top_k: int = 1,
    max_rounds: int = 3,
    no_chat: bool = False,
    system_prompt: str | None = None,
) -> GenerateResult:
    """Generation-guided iterative navigation.

    Explores the document across multiple rounds. Each round generates
    a short response from one window, then uses the generation residual
    to steer the compass to a new region. After all rounds, replays
    discovered windows and generates the final answer.

    This strategy is designed for exploration queries ("find amusing
    moments", "summarize the key events") where the answer spans
    multiple regions of the document.
    """
    import mlx.core as mx
    import numpy as np

    compass_layer = lib.compass_layer
    visited: set[int] = set()
    gen_residual = None
    discovered: list[int] = []  # windows found during exploration

    sys_content = system_prompt or (
        "You are answering questions based on a document transcript. "
        "Answer using only information from the transcript. "
        "Quote exact text when possible."
    )

    print(
        f"  Iterative navigation: {max_rounds} exploration rounds "
        f"({_EXPLORE_TOKENS} tokens each), then combined replay",
        file=sys.stderr,
    )

    # Build frame templates
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        preamble_text = (
            f"<start_of_turn>user\n{sys_content}\n\n"
            f"Here is the relevant transcript:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)
        postamble_text = (
            f"\n\n---\nBased on the transcript above, "
            f"{prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
    else:
        preamble_ids = tokenizer.encode("Transcript:\n\n", add_special_tokens=False)
        postamble_ids = tokenizer.encode(
            f"\n\n---\nQuestion: {prompt_text}\nAnswer:",
            add_special_tokens=False,
        )

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # ── Exploration phase ────────────────────────────────────────────
    for round_idx in range(max_rounds):
        # Navigate: compass route with generation residual shift
        t0 = time.time()
        routed = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=model_config,
            strategy=RoutingStrategy.GEOMETRIC,
            top_k=top_k,
            exclude=visited,
            query_residual=gen_residual,
        )
        route_ms = (time.time() - t0) * 1000

        if not routed:
            print(f"  Round {round_idx + 1}: exhausted", file=sys.stderr)
            break

        best_wid = routed[-1]
        visited.add(best_wid)
        discovered.append(best_wid)

        # Replay window
        w_tokens = lib.get_window_tokens(best_wid)

        seq_len = 0
        _l, round_kv = kv_gen.prefill(mx.array(preamble_ids)[None])
        mx.eval(*[t for p in round_kv for t in p])
        seq_len += len(preamble_ids)

        _l, round_kv = kv_gen.extend(
            mx.array(w_tokens)[None], round_kv, abs_start=seq_len,
        )
        mx.eval(*[t for p in round_kv for t in p])
        seq_len += len(w_tokens)

        logits, round_kv = kv_gen.extend(
            mx.array(postamble_ids)[None], round_kv, abs_start=seq_len,
        )
        seq_len += len(postamble_ids)

        # Generate exploration tokens
        round_tokens: list[int] = []
        for _ in range(_EXPLORE_TOKENS):
            last_logits = logits[0, -1]
            if config.temperature == 0.0:
                next_token = int(mx.argmax(last_logits).item())
            else:
                scaled = last_logits / config.temperature
                next_token = int(mx.random.categorical(scaled[None]).item())
            if next_token in stop_ids:
                break
            round_tokens.append(next_token)
            logits, round_kv = kv_gen.step_uncompiled(
                mx.array([[next_token]]), round_kv, seq_len=seq_len,
            )
            seq_len += 1

        round_text = tokenizer.decode(round_tokens, skip_special_tokens=True)
        preview = round_text[:80].replace('\n', ' ')

        # Extract generation residual for compass shift
        full_seq = list(preamble_ids) + list(w_tokens) + list(postamble_ids) + round_tokens
        gen_h = kv_gen.prefill_to_layer(
            mx.array(full_seq)[None], target_layer=compass_layer,
        )
        gen_residual = gen_h[0, -1:, :]
        mx.eval(gen_residual)

        print(
            f"  Round {round_idx + 1}: W{best_wid} ({route_ms:.0f}ms) "
            f"→ {preview}...",
            file=sys.stderr,
        )

    if not discovered:
        return GenerateResult(
            response="(No windows discovered)",
            tokens_generated=0,
            context_tokens=0,
        )

    # ── Replay phase: last 3 discovered windows ───────────────────────
    # Take the last 3 discovered (most recent compass hits) in doc order.
    # All 5 dilutes attention; last 3 keeps context focused.
    replay_wids = sorted(discovered[-3:])
    print(f"  Replaying last {len(replay_wids)} discovered: {replay_wids}", file=sys.stderr)

    seq_len = 0
    _l, gen_kv = kv_gen.prefill(mx.array(preamble_ids)[None])
    mx.eval(*[t for p in gen_kv for t in p])
    seq_len += len(preamble_ids)

    for wid in replay_wids:
        w_tokens = lib.get_window_tokens(wid)
        t0 = time.time()
        _l, gen_kv = kv_gen.extend(
            mx.array(w_tokens)[None], gen_kv, abs_start=seq_len,
        )
        mx.eval(*[t for p in gen_kv for t in p])
        replay_ms = (time.time() - t0) * 1000
        seq_len += len(w_tokens)
        print(
            f"  Replayed W{wid} @ pos {seq_len - len(w_tokens)}–{seq_len} "
            f"({replay_ms:.0f}ms)",
            file=sys.stderr,
        )

    # Postamble
    logits, gen_kv = kv_gen.extend(
        mx.array(postamble_ids)[None], gen_kv, abs_start=seq_len,
    )
    seq_len += len(postamble_ids)

    # ── Final generation from combined context ───────────────────────
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
        logits, gen_kv = kv_gen.step_uncompiled(
            mx.array([[next_token]]), gen_kv, seq_len=seq_len,
        )
        seq_len += 1

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    preview = response[:120].replace('\n', ' ')
    print(f"    {preview}...", file=sys.stderr)

    print()
    sys.stdout.write(response)
    sys.stdout.flush()
    print()

    return GenerateResult(
        response=response,
        tokens_generated=len(generated_tokens),
        context_tokens=seq_len,
    )
