"""Generation probe re-ranking — the precision layer.

For each candidate window: replay 256 tokens, generate a short assessment,
extract L26 at the last generated token, project onto a probe direction.

This is the expensive path (~300ms per window). Only used for engagement
and tension queries where BM25/stride provides candidates but can't
distinguish quality.

Validated:
  - Engagement probe: 100% train, 80% val accuracy
  - Tension probe: 100% train, 100% val accuracy
  - Judgment ⊥ tonal: 91.16° (generation irreplaceable by reading)
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

# Max tokens to replay per window for assessment
_REPLAY_TOKENS = 256
# Tokens to generate for assessment
_ASSESS_TOKENS = 20

_ENGAGEMENT_PROMPT = (
    "\n\nIs there anything amusing, surprising, or notable about this excerpt? Rate from 1-5."
)

_TENSION_PROMPT = (
    "\n\nHow tense, dangerous, or critical is the situation described "
    "in this excerpt? Rate from 1-5."
)


def _probe_rerank_windows(
    lib,
    kv_gen,
    tokenizer,
    candidate_wids: list[int],
    probe_direction: mx.array,
    probe_mean: mx.array,
    compass_layer: int,
    probe_type: str = "engagement",
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Re-rank candidate windows using generation-mode judgment probe.

    For each candidate:
      1. Replay up to 256 clean tokens as context
      2. Append assessment prompt
      3. Generate 20 tokens (greedy)
      4. Extract L26 at last generated token
      5. Project onto probe direction

    Returns (window_id, score) sorted descending.
    """
    assessment_prompt = _TENSION_PROMPT if probe_type == "tension" else _ENGAGEMENT_PROMPT

    scores: list[tuple[int, float]] = []
    t0 = time.time()

    for i, wid in enumerate(candidate_wids):
        w_tokens = lib.get_window_tokens(wid)

        # Take center slice up to _REPLAY_TOKENS
        n = len(w_tokens)
        if n > _REPLAY_TOKENS:
            start = max(0, n // 2 - _REPLAY_TOKENS // 2)
            w_tokens = w_tokens[start : start + _REPLAY_TOKENS]

        # Build assessment context
        pre_text = "<start_of_turn>user\nHere is a text excerpt:\n\n"
        pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
        post_text = f"{assessment_prompt}<end_of_turn>\n<start_of_turn>model\n"
        post_ids = tokenizer.encode(post_text, add_special_tokens=False)

        # Prefill: preamble + window tokens + assessment prompt
        sl = 0
        _l, kv = kv_gen.prefill(mx.array(pre_ids)[None])
        mx.eval(*[t for p in kv for t in p])
        sl += len(pre_ids)

        _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=sl)
        mx.eval(*[t for p in kv for t in p])
        sl += len(w_tokens)

        logits, kv = kv_gen.extend(mx.array(post_ids)[None], kv, abs_start=sl)
        sl += len(post_ids)

        # Generate assessment tokens
        gen_tokens: list[int] = []
        eos = tokenizer.eos_token_id
        for _ in range(_ASSESS_TOKENS):
            tok = int(mx.argmax(logits[0, -1]).item())
            if eos is not None and tok == eos:
                break
            gen_tokens.append(tok)
            logits, kv = kv_gen.step_uncompiled(mx.array([[tok]]), kv, seq_len=sl)
            sl += 1

        # Extract L26 at last generated position
        full_seq = list(pre_ids) + list(w_tokens) + list(post_ids) + gen_tokens
        h = kv_gen.prefill_to_layer(
            mx.array(full_seq)[None],
            target_layer=compass_layer,
            sample_positions=[len(full_seq) - 1],
        )
        mx.eval(h)
        vec = h[0, 0, :].astype(mx.float32)

        # Project onto probe direction
        proj = float(mx.sum((vec - probe_mean) * probe_direction).item())
        scores.append((wid, proj))

        if (i + 1) % 10 == 0:
            elapsed = (time.time() - t0) * 1000
            print(
                f"    Re-ranked {i + 1}/{len(candidate_wids)} windows ({elapsed:.0f}ms)",
                file=sys.stderr,
            )

    elapsed_ms = (time.time() - t0) * 1000
    scores.sort(key=lambda x: -x[1])

    print(
        f"    Probe re-rank ({probe_type}): {len(candidate_wids)} candidates "
        f"→ top-{top_k} in {elapsed_ms:.0f}ms",
        file=sys.stderr,
    )

    return scores[:top_k]
