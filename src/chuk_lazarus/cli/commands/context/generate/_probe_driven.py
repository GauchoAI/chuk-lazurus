"""Probe-driven navigation — compass casts the net, probe ranks by grounding.

No thresholds. No PARTIAL/GROUNDED/REACHING classification.
The probe is a ranker: which compass candidates ground best for this query?

  1. Compass → wide candidate set (top_k)
  2. Probe scores each candidate at first generated token
  3. Replay top-N by probe score (most grounded windows)
  4. Generate from combined context
  5. Mid-generation probe monitors for reaching transitions

The geometry decides relative relevance. The model reads the best candidates.
"""

from __future__ import annotations

import sys
import time

from .._types import GenerateResult
from ..compass_routing import RoutingStrategy, compass_route
from ._grounding import _calibrate_grounding

# How often (in tokens) to probe grounding during generation.
_PROBE_INTERVAL = 40

# How many probe-ranked windows to replay together.
_REPLAY_COUNT = 3


def _probe_score(
    kv_gen,
    preamble_ids,
    w_tokens,
    postamble_ids,
    compass_layer,
    pc1,
    cal_mean,
    temperature,
    mx,
):
    """Score a single window: prefill, generate first token, probe L26.

    Returns (projection_value, first_token_id).
    Higher projection = more grounded for this query.
    """
    import numpy as np

    # Prefill
    seq_len = 0
    _l, kv = kv_gen.prefill(mx.array(preamble_ids)[None])
    mx.eval(*[t for p in kv for t in p])
    seq_len += len(preamble_ids)

    _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=seq_len)
    mx.eval(*[t for p in kv for t in p])
    seq_len += len(w_tokens)

    logits, kv = kv_gen.extend(mx.array(postamble_ids)[None], kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

    # First token
    if temperature == 0.0:
        first_tok = int(mx.argmax(logits[0, -1]).item())
    else:
        first_tok = int(mx.random.categorical(logits[0, -1:] / temperature).item())

    # Probe L26 at first generated token position
    full_ids = list(preamble_ids) + list(w_tokens) + list(postamble_ids) + [first_tok]
    h = kv_gen.prefill_to_layer(
        mx.array(full_ids)[None],
        target_layer=compass_layer,
        sample_positions=[len(full_ids) - 1],
    )
    mx.eval(h)
    vec = np.array(h[0, 0, :].tolist(), dtype=np.float32)
    proj = float((vec - cal_mean) @ pc1)

    return proj, first_tok


def _build_frame(tokenizer, sys_content, w_tokens, prompt_text, no_chat):
    """Build preamble + postamble token sequences for a window."""
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        preamble_text = (
            f"<start_of_turn>user\n{sys_content}\n\nHere is the relevant transcript:\n\n"
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
    return preamble_ids, postamble_ids


def _probe_driven_generate(
    lib,
    kv_gen,
    engine,
    tokenizer,
    model_config,
    prompt_ids: list[int],
    prompt_text: str,
    config,
    top_k: int = 3,
    max_rounds: int = 8,
    no_chat: bool = False,
    system_prompt: str | None = None,
) -> GenerateResult:
    """Probe-ranked generation: compass candidates ranked by grounding projection.

    No thresholds. The probe ranks all compass candidates by how grounded
    the model would be on each one for this specific query. Top-N by probe
    score are replayed together. The model generates from combined context.
    """
    import mlx.core as mx
    import numpy as np

    compass_layer = lib.compass_layer
    visited: set[int] = set()
    gen_residual = None

    sys_content = system_prompt or (
        "You are answering questions based on a document transcript. "
        "Answer using only information from the transcript. "
        "Quote exact text when possible."
    )

    # -- Calibrate grounding direction (PC1 only, no threshold used for gating) --
    pc1, cal_mean, ground_thresh, partial_thresh = _calibrate_grounding(
        kv_gen,
        lib,
        tokenizer,
        compass_layer,
        sys_content,
    )

    replay_count = min(_REPLAY_COUNT, top_k)

    print(
        f"  Probe-ranked navigation: top-{top_k} compass → probe rank → replay best {replay_count}",
        file=sys.stderr,
    )

    # ── Phase 1: Compass candidates ──────────────────────────────────
    routed = compass_route(
        lib,
        kv_gen,
        prompt_ids,
        prompt_text,
        tokenizer,
        model_config=model_config,
        strategy=RoutingStrategy.GEOMETRIC,
        top_k=top_k,
        exclude=visited,
        query_residual=gen_residual,
    )

    if not routed:
        print("  No compass candidates", file=sys.stderr)
        return GenerateResult(
            response="(No candidates found)",
            tokens_generated=0,
            context_tokens=0,
        )

    # ── Phase 2: Probe-rank each candidate ───────────────────────────
    print(f"  Probing {len(routed)} candidates...", file=sys.stderr)

    scored: list[tuple[int, float]] = []
    # Build frame once (preamble/postamble are the same for all windows)
    sample_w_tokens = lib.get_window_tokens(routed[0])
    preamble_ids, postamble_ids = _build_frame(
        tokenizer,
        sys_content,
        sample_w_tokens,
        prompt_text,
        no_chat,
    )

    for wid in reversed(routed):  # best compass score first
        visited.add(wid)
        w_tokens = lib.get_window_tokens(wid)

        proj, first_tok = _probe_score(
            kv_gen,
            preamble_ids,
            w_tokens,
            postamble_ids,
            compass_layer,
            pc1,
            cal_mean,
            config.temperature,
            mx,
        )

        first_text = tokenizer.decode([first_tok], skip_special_tokens=True)
        w_preview = tokenizer.decode(
            list(w_tokens)[:20],
            skip_special_tokens=True,
        ).replace("\n", " ")[:60]
        print(
            f'    W{wid:>3} probe={proj:+.0f}  first="{first_text}"  {w_preview}...',
            file=sys.stderr,
        )
        scored.append((wid, proj))

    # Rank by probe score (highest = most grounded)
    scored.sort(key=lambda x: x[1], reverse=True)
    best_wids = [wid for wid, _ in scored[:replay_count]]

    print(f"  Replay by probe rank: {best_wids}", file=sys.stderr)
    for wid, proj in scored[:replay_count]:
        print(f"    W{wid}: proj={proj:+.0f}", file=sys.stderr)

    # ── Phase 3: Replay best windows and generate ────────────────────
    # Prefill: preamble → [window tokens concatenated] → postamble
    seq_len = 0
    p_ids = mx.array(preamble_ids)[None]
    _l, gen_kv = kv_gen.prefill(p_ids)
    mx.eval(*[t for p in gen_kv for t in p])
    seq_len += len(preamble_ids)

    # Replay windows in document order (sorted by window ID)
    replay_wids = sorted(best_wids)
    all_w_tokens = []
    for wid in replay_wids:
        w_tokens = lib.get_window_tokens(wid)
        all_w_tokens.extend(list(w_tokens))

        t0 = time.time()
        w_ids = mx.array(w_tokens)[None]
        _l, gen_kv = kv_gen.extend(w_ids, gen_kv, abs_start=seq_len)
        mx.eval(*[t for p in gen_kv for t in p])
        replay_ms = (time.time() - t0) * 1000
        seq_len += len(w_tokens)
        print(
            f"  Replayed W{wid} @ pos {seq_len - len(w_tokens)}–{seq_len} ({replay_ms:.0f}ms)",
            file=sys.stderr,
        )

    # Postamble
    q_ids = mx.array(postamble_ids)[None]
    logits, gen_kv = kv_gen.extend(q_ids, gen_kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

    # ── Phase 4: Generate with optional mid-generation monitoring ────
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    generated_tokens: list[int] = []
    tokens_since_probe = 0

    for step in range(config.max_tokens):
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
            mx.array([[next_token]]),
            gen_kv,
            seq_len=seq_len,
        )
        seq_len += 1
        tokens_since_probe += 1

        # Mid-generation probe (informational — log but don't interrupt)
        if tokens_since_probe >= _PROBE_INTERVAL and len(generated_tokens) > 10:
            tokens_since_probe = 0
            mid_ids = list(preamble_ids) + all_w_tokens + list(postamble_ids) + generated_tokens
            h = kv_gen.prefill_to_layer(
                mx.array(mid_ids)[None],
                target_layer=compass_layer,
                sample_positions=[len(mid_ids) - 1],
            )
            mx.eval(h)
            vec = np.array(h[0, 0, :].tolist(), dtype=np.float32)
            mid_proj = float((vec - cal_mean) @ pc1)

            if mid_proj > ground_thresh:
                label = "GROUNDED"
            elif mid_proj > partial_thresh:
                label = "partial"
            else:
                label = "REACHING"
            print(
                f"    ✓ token {len(generated_tokens)}: {label} (proj={mid_proj:+.0f})",
                file=sys.stderr,
            )

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    preview = response[:120].replace("\n", " ")
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
