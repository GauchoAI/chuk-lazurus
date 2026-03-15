"""Iterative compass navigation — grounding-driven generation."""

from __future__ import annotations

import sys
import time

from .._types import GenerateConfig, GenerateResult
from ..compass_routing import RoutingStrategy, compass_route
from ._grounding import _calibrate_grounding


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
    """Iterative compass navigation — grounding-driven.

    The model's L26 dark space signals whether it's grounding on context
    or reaching into parametric memory. One dimension. First token.
    Universal across query types.

    Navigation:
      GROUNDED -> model has what it needs -> generate full answer
      PARTIAL  -> model has some content -> turn the page
      REACHING -> model is inventing -> compass re-routes to new region

    No EOS detection. No string matching. No round token budgets.
    One geometric test at the first generated token drives everything.
    """
    import mlx.core as mx
    import numpy as np

    compass_layer = lib.compass_layer
    visited: set[int] = set()
    gen_residual = None
    last_response = ""
    last_tokens: list[int] = []

    sys_content = system_prompt or (
        "You are answering questions based on a document transcript. "
        "Answer using only information from the transcript. Quote exact text when possible."
    )

    # -- Calibrate grounding detector --
    pc1, cal_mean, ground_thresh, partial_thresh = _calibrate_grounding(
        kv_gen, lib, tokenizer, compass_layer, sys_content,
    )

    print(f"  Iterative navigation: up to {max_rounds} rounds (grounding-driven)", file=sys.stderr)

    sequential_next: int | None = None

    for round_idx in range(max_rounds):
        # -- Navigate: sequential (page turn) or compass (new region) --
        if sequential_next is not None and sequential_next not in visited \
                and 0 <= sequential_next < lib.num_windows:
            best_wid = sequential_next
            sequential_next = None
            visited.add(best_wid)
            route_ms = 0.0
            nav_mode = "seq"
        else:
            sequential_next = None
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
            nav_mode = "compass"

        # -- Replay window --
        w_tokens = lib.get_window_tokens(best_wid)

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

        seq_len = 0
        p_ids = mx.array(preamble_ids)[None]
        _logits, round_kv = kv_gen.prefill(p_ids)
        mx.eval(*[t for pair in round_kv for t in pair])
        seq_len += len(preamble_ids)

        t0 = time.time()
        w_ids = mx.array(w_tokens)[None]
        _logits, round_kv = kv_gen.extend(w_ids, round_kv, abs_start=seq_len)
        mx.eval(*[t for pair in round_kv for t in pair])
        replay_ms = (time.time() - t0) * 1000
        seq_len += len(w_tokens)

        q_ids = mx.array(postamble_ids)[None]
        logits, round_kv = kv_gen.extend(q_ids, round_kv, abs_start=seq_len)
        seq_len += len(postamble_ids)

        # -- Generate first token + grounding test --
        stop_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)

        if config.temperature == 0.0:
            first_tok = int(mx.argmax(logits[0, -1]).item())
        else:
            first_tok = int(mx.random.categorical(logits[0, -1:] / config.temperature).item())

        # Extract L26 at first generated token
        full_probe = list(preamble_ids) + list(w_tokens) + list(postamble_ids) + [first_tok]
        probe_h = kv_gen.prefill_to_layer(mx.array(full_probe)[None], target_layer=compass_layer)
        mx.eval(probe_h)
        probe_vec = np.array(probe_h[0, -1, :].tolist(), dtype=np.float32)
        proj = float((probe_vec - cal_mean) @ pc1)

        if proj > ground_thresh:
            state = "GROUNDED"
        elif proj > partial_thresh:
            state = "PARTIAL"
        else:
            state = "REACHING"

        first_text = tokenizer.decode([first_tok], skip_special_tokens=True)

        # -- Navigation decision based on grounding state --
        if state == "REACHING":
            # Wrong region. Don't generate. Compass re-routes.
            print(
                f"  Round {round_idx + 1}: W{best_wid} [{nav_mode}] "
                f"REACHING (proj={proj:+.0f}) → re-route",
                file=sys.stderr,
            )
            # Extract generation residual for compass shift
            gen_residual = probe_h[0, -1:, :]
            continue

        if state == "PARTIAL":
            # Right region, need more. Queue next page.
            next_wid = best_wid + 1
            if next_wid < lib.num_windows and next_wid not in visited:
                sequential_next = next_wid
            print(
                f"  Round {round_idx + 1}: W{best_wid} [{nav_mode}] "
                f"PARTIAL (proj={proj:+.0f}) → page turn",
                file=sys.stderr,
            )
            gen_residual = probe_h[0, -1:, :]
            continue

        # -- GROUNDED: let the model generate its full answer --
        print(
            f"  Round {round_idx + 1}: W{best_wid} [{nav_mode}] "
            f"GROUNDED (proj={proj:+.0f}) → generating",
            file=sys.stderr,
        )

        # Continue from first token
        round_tokens = [first_tok]
        logits, round_kv = kv_gen.step_uncompiled(
            mx.array([[first_tok]]), round_kv, seq_len=seq_len,
        )
        seq_len += 1

        for _ in range(config.max_tokens - 1):
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

        round_response = tokenizer.decode(round_tokens, skip_special_tokens=True)
        last_response = round_response
        last_tokens = round_tokens

        preview = round_response[:100].replace('\n', ' ')
        print(f"    {preview}...", file=sys.stderr)

        # Extract generation residual for potential next round
        full_seq = list(preamble_ids) + list(w_tokens) + list(postamble_ids) + round_tokens
        gen_h = kv_gen.prefill_to_layer(mx.array(full_seq)[None], target_layer=compass_layer)
        gen_residual = gen_h[0, -1:, :]
        mx.eval(gen_residual)

        # Grounded generation complete. Done.
        break

    if not last_response:
        last_response = "(No grounded answer found within round limit)"

    print()
    sys.stdout.write(last_response)
    sys.stdout.flush()
    print()

    return GenerateResult(
        response=last_response,
        tokens_generated=len(last_tokens),
        context_tokens=seq_len if last_tokens else 0,
    )
