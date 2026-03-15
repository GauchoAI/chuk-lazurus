"""Unified navigation — the dark space decides the path.

One command. Two paths. Both automatic. No flags needed:

  FACTUAL:     compass geometric top-3 → replay directly (~2s)
  EXPLORATION: iterative compass shifting with note-taking (5 rounds) → replay last-3

The query-type probe (L26, generic examples, cached) routes automatically.
FACTUAL is fast — one compass call, direct replay.
EXPLORATION is thorough — generation-guided compass shifting discovers
content across the document that single-shot compass never reaches.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from .._types import GenerateConfig, GenerateResult
from ..compass_routing import RoutingStrategy, compass_route


# ── Constants ────────────────────────────────────────────────────────
_COMPASS_TOP_K = 10          # compass candidates for factual (direct replay)
_ITER_ROUNDS = 3             # exploration rounds (matches --strategy iterative default)
_ITER_TOP_K = 1              # windows per compass call in iterative


# ── Unified Generate ─────────────────────────────────────────────────
def _unified_generate(
    lib,
    kv_gen,
    engine,
    tokenizer,
    model_config,
    prompt_ids: list[int],
    prompt_text: str,
    config,
    top_k: int = 10,
    max_rounds: int = 8,
    no_chat: bool = False,
    system_prompt: str | None = None,
) -> GenerateResult:
    """One command. Two paths. Both automatic.

    FACTUAL:     compass geometric top-3 → replay → generate (~2s)
    EXPLORATION: iterative note-taking (5 rounds) → replay last-3 → generate

    The query-type probe at L26 decides. No flags. No domain-specific probes.
    """
    from ._probes import load_or_calibrate
    from ._iterative import _iterative_generate

    compass_layer = lib.compass_layer

    sys_content = system_prompt or (
        "You are answering questions based on a document. "
        "Answer using only information from the document. "
        "Quote exact text when possible."
    )

    # ── Load or calibrate probes ──────────────────────────────────────
    checkpoint_dir = str(lib._path) if hasattr(lib, '_path') else "."
    model_name = lib.manifest.model_id
    probes = load_or_calibrate(
        kv_gen, tokenizer, compass_layer, lib,
        checkpoint_dir, model_name,
    )

    # ── Classify query ────────────────────────────────────────────────
    query_prompt = (
        f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    q_ids = tokenizer.encode(query_prompt, add_special_tokens=True)
    q_h = kv_gen.prefill_to_layer(
        mx.array(q_ids)[None], target_layer=compass_layer,
        sample_positions=[len(q_ids) - 1],
    )
    mx.eval(q_h)
    query_vec = q_h[0, 0, :].astype(mx.float32)

    qt_proj = mx.sum((query_vec - probes.qt_mean) * probes.qt_direction)
    mx.eval(qt_proj)
    qt_proj_val = float(qt_proj.item())
    is_exploration = qt_proj_val > probes.qt_threshold
    query_type = "EXPLORATION" if is_exploration else "FACTUAL"

    print(
        f"  Query type: {query_type} (proj={qt_proj_val:+.0f}, "
        f"thresh={probes.qt_threshold:+.0f})",
        file=sys.stderr,
    )

    # ── EXPLORATION: iterative compass shifting with note-taking ──────
    if is_exploration:
        return _iterative_generate(
            lib=lib,
            kv_gen=kv_gen,
            engine=engine,
            tokenizer=tokenizer,
            model_config=model_config,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            config=config,
            top_k=_ITER_TOP_K,
            max_rounds=_ITER_ROUNDS,
            no_chat=no_chat,
            system_prompt=system_prompt,
        )

    # ── FACTUAL: compass geometric → replay directly ──────────────────
    t0 = time.time()
    routed = compass_route(
        lib, kv_gen, prompt_ids, prompt_text, tokenizer,
        model_config=model_config,
        strategy=RoutingStrategy.COMPASS,
        top_k=_COMPASS_TOP_K,
    )
    route_ms = (time.time() - t0) * 1000

    if not routed:
        return GenerateResult(
            response="(No candidates found)",
            tokens_generated=0,
            context_tokens=0,
        )

    print(
        f"  Compass geometric: {len(routed)} candidates ({route_ms:.0f}ms)",
        file=sys.stderr,
    )

    # Replay top-3 by compass score (compass_route returns ascending)
    replay_count = min(3, len(routed))
    best_wids = routed[-replay_count:]
    print(f"  Replay: {best_wids}", file=sys.stderr)

    # ── Replay and generate ───────────────────────────────────────────
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
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

    seq_len = 0
    _l, gen_kv = kv_gen.prefill(mx.array(preamble_ids)[None])
    mx.eval(*[t for p in gen_kv for t in p])
    seq_len += len(preamble_ids)

    replay_wids = sorted(best_wids)
    for wid in replay_wids:
        w_tokens = lib.get_window_tokens(wid)
        t0 = time.time()
        _l, gen_kv = kv_gen.extend(
            mx.array(w_tokens)[None], gen_kv, abs_start=seq_len,
        )
        mx.eval(*[t for p in gen_kv for t in p])
        ms = (time.time() - t0) * 1000
        seq_len += len(w_tokens)
        print(
            f"  Replayed W{wid} @ pos {seq_len - len(w_tokens)}-{seq_len} "
            f"({ms:.0f}ms)",
            file=sys.stderr,
        )

    logits, gen_kv = kv_gen.extend(
        mx.array(postamble_ids)[None], gen_kv, abs_start=seq_len,
    )
    seq_len += len(postamble_ids)

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
