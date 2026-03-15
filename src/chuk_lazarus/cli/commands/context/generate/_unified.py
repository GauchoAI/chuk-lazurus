"""Unified three-probe navigation — the dark space decides everything.

No strategy flags. No user decisions. Three probes at L26:

  1. query_type:  exploration or factual? (query's own L26 geometry)
  2. tonal:       amusing or routine?     (model's generation judgment)
  3. grounding:   grounded or reaching?   (context-query intersection)

The query arrives. The engine classifies it. Routes to the right probe.
Ranks candidates. Replays the best. Generates. One entry point.
"""

from __future__ import annotations

import sys
import time

from .._types import GenerateConfig, GenerateResult
from ..compass_routing import RoutingStrategy, compass_route


# ── Constants ────────────────────────────────────────────────────────
_TONAL_ASSESS_TOKENS = 20   # tokens to generate for tonal assessment
_REPLAY_COUNT = 3            # windows to replay for final generation


# ── PCA helper ───────────────────────────────────────────────────────
def _pca_direction(vecs, labels, positive_label=True):
    """Compute PC1 from contrastive vectors, orient so positive_label = positive.

    Returns (pc1, mean, positive_mean_proj, negative_mean_proj, variance_pct).
    """
    import numpy as np

    arr = np.stack(vecs, axis=0)
    mean = arr.mean(axis=0)
    centered = arr - mean
    _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]

    projs = centered @ pc1
    pos_mean = float(np.mean([p for p, lab in zip(projs, labels) if lab == positive_label]))
    neg_mean = float(np.mean([p for p, lab in zip(projs, labels) if lab != positive_label]))

    if neg_mean > pos_mean:
        pc1 = -pc1
        pos_mean, neg_mean = -neg_mean, -pos_mean

    variance_pct = (S[0] ** 2 / np.sum(S ** 2)) * 100
    return pc1, mean, pos_mean, neg_mean, variance_pct


# ── Probe 1: Query Type ─────────────────────────────────────────────
def _calibrate_query_type(kv_gen, tokenizer, compass_layer):
    """Calibrate exploration-vs-factual from query L26 residuals.

    No windows needed. Just query geometry.
    """
    import mlx.core as mx
    import numpy as np

    queries = [
        ("Find the most amusing moments in the transcript", True),
        ("What were the funniest or most entertaining parts?", True),
        ("What surprising or human-interest moments happened?", True),
        ("What sports scores were mentioned?", False),
        ("What were the fuel pressure readings?", False),
        ("What was the spacecraft attitude at midcourse correction?", False),
    ]

    vecs = []
    labels = []
    for query_text, is_exploration in queries:
        # Chat-wrap the query
        prompt = (
            f"<start_of_turn>user\n{query_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        h = kv_gen.prefill_to_layer(
            mx.array(ids)[None], target_layer=compass_layer,
            sample_positions=[len(ids) - 1],
        )
        mx.eval(h)
        vec = np.array(h[0, 0, :].tolist(), dtype=np.float32)
        vecs.append(vec)
        labels.append(is_exploration)

    pc1, mean, expl_mean, fact_mean, var_pct = _pca_direction(
        vecs, labels, positive_label=True,
    )
    threshold = (expl_mean + fact_mean) / 2

    print(
        f"  Query-type calibrated: PC1={var_pct:.0f}% variance, "
        f"E={expl_mean:+.0f} F={fact_mean:+.0f} sep={abs(expl_mean - fact_mean):.0f}",
        file=sys.stderr,
    )

    return pc1, mean, threshold


# ── Probe 2: Tonal Generation ───────────────────────────────────────
def _calibrate_tonal(kv_gen, lib, tokenizer, compass_layer):
    """Calibrate amusing-vs-routine from generation-mode L26 residuals.

    For each calibration window, the model reads content and generates a
    20-token assessment. L26 at the last generated token encodes the
    model's tonal judgment.
    """
    import mlx.core as mx
    import numpy as np

    assess_question = (
        "Is there anything amusing, surprising, or human-interest "
        "in this excerpt? Rate from 1-5."
    )

    calibration = [
        # (window_id, is_amusing)
        (170, True),    # Porridge eating contest + baseball + Buzz joke
        (76, True),     # Morning news, VP Agnew, Mariner launch
        (238, True),    # Sports double header + Hornet crew bet
        (382, False),   # Fuel cell purge procedures
        (23, False),    # S-band radio check
        (453, False),   # GET timestamps, P52 alignment
    ]

    vecs = []
    labels = []

    for wid, is_amusing in calibration:
        w_tokens = lib.get_window_tokens(wid)

        # Build: [window content] + assessment question
        pre_text = "<start_of_turn>user\nHere is a transcript excerpt:\n\n"
        pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
        post_text = (
            f"\n\n{assess_question}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        post_ids = tokenizer.encode(post_text, add_special_tokens=False)

        # Prefill
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
        gen_tokens = []
        for _ in range(_TONAL_ASSESS_TOKENS):
            tok = int(mx.argmax(logits[0, -1]).item())
            eos = tokenizer.eos_token_id
            if eos is not None and tok == eos:
                break
            gen_tokens.append(tok)
            logits, kv = kv_gen.step_uncompiled(
                mx.array([[tok]]), kv, seq_len=sl,
            )
            sl += 1

        # Extract L26 at last generated token
        full = list(pre_ids) + list(w_tokens) + list(post_ids) + gen_tokens
        h = kv_gen.prefill_to_layer(
            mx.array(full)[None], target_layer=compass_layer,
            sample_positions=[len(full) - 1],
        )
        mx.eval(h)
        vec = np.array(h[0, 0, :].tolist(), dtype=np.float32)

        vecs.append(vec)
        labels.append(is_amusing)

    pc1, mean, amus_mean, rout_mean, var_pct = _pca_direction(
        vecs, labels, positive_label=True,
    )
    threshold = (amus_mean + rout_mean) / 2

    print(
        f"  Tonal calibrated: PC1={var_pct:.0f}% variance, "
        f"A={amus_mean:+.0f} R={rout_mean:+.0f} sep={abs(amus_mean - rout_mean):.0f}",
        file=sys.stderr,
    )

    return pc1, mean, threshold


# ── Probe 3: Grounding (reuse existing) ─────────────────────────────
def _calibrate_grounding_for_unified(kv_gen, lib, tokenizer, compass_layer, sys_content):
    """Wrapper around existing grounding calibration."""
    from ._grounding import _calibrate_grounding
    return _calibrate_grounding(kv_gen, lib, tokenizer, compass_layer, sys_content)


# ── Tonal scoring for one window ─────────────────────────────────────
def _tonal_score_window(
    kv_gen, tokenizer, w_tokens, compass_layer, tonal_pc1, tonal_mean, mx,
):
    """Generate a 20-token assessment of a window and return tonal projection."""
    import numpy as np

    assess_question = (
        "Is there anything amusing, surprising, or human-interest "
        "in this excerpt? Rate from 1-5."
    )
    pre_text = "<start_of_turn>user\nHere is a transcript excerpt:\n\n"
    pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
    post_text = (
        f"\n\n{assess_question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    post_ids = tokenizer.encode(post_text, add_special_tokens=False)

    # Prefill
    sl = 0
    _l, kv = kv_gen.prefill(mx.array(pre_ids)[None])
    mx.eval(*[t for p in kv for t in p])
    sl += len(pre_ids)
    _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=sl)
    mx.eval(*[t for p in kv for t in p])
    sl += len(w_tokens)
    logits, kv = kv_gen.extend(mx.array(post_ids)[None], kv, abs_start=sl)
    sl += len(post_ids)

    # Generate assessment
    gen_tokens = []
    eos = tokenizer.eos_token_id
    for _ in range(_TONAL_ASSESS_TOKENS):
        tok = int(mx.argmax(logits[0, -1]).item())
        if eos is not None and tok == eos:
            break
        gen_tokens.append(tok)
        logits, kv = kv_gen.step_uncompiled(mx.array([[tok]]), kv, seq_len=sl)
        sl += 1

    # Extract L26 at last generated token
    full = list(pre_ids) + list(w_tokens) + list(post_ids) + gen_tokens
    h = kv_gen.prefill_to_layer(
        mx.array(full)[None], target_layer=compass_layer,
        sample_positions=[len(full) - 1],
    )
    mx.eval(h)
    vec = np.array(h[0, 0, :].tolist(), dtype=np.float32)
    proj = float((vec - tonal_mean) @ tonal_pc1)
    return proj


# ── Grounding scoring for one window ────────────────────────────────
def _grounding_score_window(
    kv_gen, tokenizer, w_tokens, prompt_text, compass_layer,
    ground_pc1, ground_mean, sys_content, no_chat, mx,
):
    """Prefill window + query, generate first token, return grounding projection."""
    import numpy as np

    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        pre_text = (
            f"<start_of_turn>user\n{sys_content}\n\n"
            f"Here is the relevant transcript:\n\n"
        )
        pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
        post_text = (
            f"\n\n---\nBased on the transcript above, "
            f"{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        )
        post_ids = tokenizer.encode(post_text, add_special_tokens=False)
    else:
        pre_ids = tokenizer.encode("Transcript:\n\n", add_special_tokens=False)
        post_ids = tokenizer.encode(
            f"\n\n---\nQuestion: {prompt_text}\nAnswer:",
            add_special_tokens=False,
        )

    # Prefill
    sl = 0
    _l, kv = kv_gen.prefill(mx.array(pre_ids)[None])
    mx.eval(*[t for p in kv for t in p])
    sl += len(pre_ids)
    _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=sl)
    mx.eval(*[t for p in kv for t in p])
    sl += len(w_tokens)
    logits, kv = kv_gen.extend(mx.array(post_ids)[None], kv, abs_start=sl)
    sl += len(post_ids)

    # First token
    first_tok = int(mx.argmax(logits[0, -1]).item())

    # Extract L26
    full = list(pre_ids) + list(w_tokens) + list(post_ids) + [first_tok]
    h = kv_gen.prefill_to_layer(
        mx.array(full)[None], target_layer=compass_layer,
        sample_positions=[len(full) - 1],
    )
    mx.eval(h)
    vec = np.array(h[0, 0, :].tolist(), dtype=np.float32)
    proj = float((vec - ground_mean) @ ground_pc1)
    return proj


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
    """Unified three-probe navigation. No strategy flags.

    1. Query-type probe classifies the query (exploration vs factual)
    2. Compass routes to top-k candidates
    3. Appropriate probe ranks each candidate
    4. Top-3 replayed together
    5. Generate final answer
    """
    import mlx.core as mx
    import numpy as np

    compass_layer = lib.compass_layer

    sys_content = system_prompt or (
        "You are answering questions based on a document transcript. "
        "Answer using only information from the transcript. "
        "Quote exact text when possible."
    )

    # ── Calibrate all three probes ───────────────────────────────────
    print("  Calibrating probes...", file=sys.stderr)

    qt_pc1, qt_mean, qt_thresh = _calibrate_query_type(
        kv_gen, tokenizer, compass_layer,
    )
    tonal_pc1, tonal_mean, tonal_thresh = _calibrate_tonal(
        kv_gen, lib, tokenizer, compass_layer,
    )
    ground_pc1, ground_mean, ground_thresh, partial_thresh = \
        _calibrate_grounding_for_unified(
            kv_gen, lib, tokenizer, compass_layer, sys_content,
        )

    # ── Classify the query ───────────────────────────────────────────
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
    q_vec = np.array(q_h[0, 0, :].tolist(), dtype=np.float32)
    qt_proj = float((q_vec - qt_mean) @ qt_pc1)

    is_exploration = qt_proj > qt_thresh
    query_type = "exploration" if is_exploration else "factual"

    print(
        f"  Query type: {query_type.upper()} (proj={qt_proj:+.0f}, "
        f"thresh={qt_thresh:+.0f})",
        file=sys.stderr,
    )

    # ── Compass candidates ───────────────────────────────────────────
    t0 = time.time()
    routed = compass_route(
        lib, kv_gen, prompt_ids, prompt_text, tokenizer,
        model_config=model_config,
        strategy=RoutingStrategy.GEOMETRIC,
        top_k=top_k,
    )
    route_ms = (time.time() - t0) * 1000

    if not routed:
        return GenerateResult(
            response="(No candidates found)",
            tokens_generated=0,
            context_tokens=0,
        )

    # ── Probe-rank candidates ────────────────────────────────────────
    probe_name = "tonal" if is_exploration else "grounding"
    print(
        f"  Ranking {len(routed)} candidates with {probe_name} probe...",
        file=sys.stderr,
    )

    scored: list[tuple[int, float]] = []

    for wid in reversed(routed):  # best compass score first
        w_tokens = lib.get_window_tokens(wid)

        if is_exploration:
            proj = _tonal_score_window(
                kv_gen, tokenizer, w_tokens, compass_layer,
                tonal_pc1, tonal_mean, mx,
            )
        else:
            proj = _grounding_score_window(
                kv_gen, tokenizer, w_tokens, prompt_text, compass_layer,
                ground_pc1, ground_mean, sys_content, no_chat, mx,
            )

        w_preview = tokenizer.decode(
            list(w_tokens)[:20], skip_special_tokens=True,
        ).replace('\n', ' ')[:60]
        print(
            f"    W{wid:>3} {probe_name}={proj:+.0f}  {w_preview}...",
            file=sys.stderr,
        )
        scored.append((wid, proj))

    # Rank by probe score (highest = best match)
    scored.sort(key=lambda x: x[1], reverse=True)
    replay_count = min(_REPLAY_COUNT, len(scored))
    best_wids = [wid for wid, _ in scored[:replay_count]]

    print(f"  Replay by {probe_name} rank: {best_wids}", file=sys.stderr)
    for wid, proj in scored[:replay_count]:
        print(f"    W{wid}: {probe_name}={proj:+.0f}", file=sys.stderr)

    # ── Replay and generate ──────────────────────────────────────────
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
    _l, gen_kv = kv_gen.prefill(mx.array(preamble_ids)[None])
    mx.eval(*[t for p in gen_kv for t in p])
    seq_len += len(preamble_ids)

    # Replay in document order
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
            f"  Replayed W{wid} @ pos {seq_len - len(w_tokens)}–{seq_len} "
            f"({ms:.0f}ms)",
            file=sys.stderr,
        )

    # Postamble
    logits, gen_kv = kv_gen.extend(
        mx.array(postamble_ids)[None], gen_kv, abs_start=seq_len,
    )
    seq_len += len(postamble_ids)

    # Generate
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
