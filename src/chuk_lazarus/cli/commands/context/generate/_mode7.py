"""Mode 7 — Unified Dark Space Router.

One command. Any query. The engine reads the query, classifies it via
a 5-class L26 probe, routes through the appropriate mechanism, selects
windows, replays them, and generates.

Query types and their routing strategies:
  FACTUAL:     geometric RRF (compass + contrastive), top-3 windows
  ENGAGEMENT:  compass top-20 → tonal probe re-rank, top-5
  TENSION:     compass top-20 + temporal stride → tension probe re-rank, top-5
  GLOBAL:      temporal stride, 10 evenly spaced windows
  TONE:        geometric RRF with more windows, top-7

Every routing decision is grounded in measured results from the
Lazarus experiments. See spec §2 for validation data.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from .._types import GenerateConfig, GenerateResult
from ..compass_routing import RoutingStrategy, compass_route

# ── Constants ────────────────────────────────────────────────────────
_FACTUAL_TOP_K = 3
_ENGAGEMENT_TOP_K = 5
_TENSION_TOP_K = 5
_GLOBAL_TOP_K = 10
_TONE_TOP_K = 7
_COARSE_CANDIDATES = 20  # compass pre-filter size before probe re-ranking


async def _mode7_generate(
    lib,
    kv_gen,
    model,
    engine,
    tokenizer,
    model_config,
    prompt_ids: list[int],
    prompt_text: str,
    config: GenerateConfig,
    top_k: int | None = None,
    no_chat: bool = False,
    system_prompt: str | None = None,
) -> GenerateResult:
    """Mode 7: unified dark space router.

    Classifies → routes → replays → generates. No strategy flag needed.
    """
    from ._probes import (
        QueryType,
        classify_query_m7,
        load_or_calibrate,
    )

    compass_layer = lib.compass_layer
    checkpoint_dir = str(lib._path) if hasattr(lib, "_path") else "."
    model_name = lib.manifest.model_id

    sys_content = system_prompt or (
        "You are answering questions based on a document. "
        "Answer using only information from the document. "
        "Quote exact text when possible."
    )

    # ── Step 1: Load or calibrate probes ──────────────────────────────
    probes = load_or_calibrate(
        kv_gen,
        tokenizer,
        compass_layer,
        lib,
        checkpoint_dir,
        model_name,
    )

    # ── Step 2: Classify query ────────────────────────────────────────
    t0 = time.time()
    if compass_layer is None:
        # No compass data — default to FACTUAL (BM25 fallback inside routing)
        query_type = QueryType.FACTUAL
        confidence = 0.0
        classify_ms = 0.0
    else:
        query_type, confidence = classify_query_m7(
            kv_gen,
            tokenizer,
            prompt_text,
            compass_layer,
            probes,
        )
        classify_ms = (time.time() - t0) * 1000

    print(
        f"  Query classification: {query_type.value} "
        f"(confidence={confidence:.1f}, {classify_ms:.0f}ms)",
        file=sys.stderr,
    )

    # ── Step 3: Route based on classification ─────────────────────────
    t0 = time.time()

    if query_type == QueryType.FACTUAL:
        k = top_k or _FACTUAL_TOP_K
        replay_wids = _route_factual(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config,
            k,
        )

    elif query_type == QueryType.ENGAGEMENT:
        k = top_k or _ENGAGEMENT_TOP_K
        replay_wids = _route_engagement(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config,
            probes,
            compass_layer,
            k,
        )

    elif query_type == QueryType.TENSION:
        k = top_k or _TENSION_TOP_K
        replay_wids = _route_tension(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config,
            probes,
            compass_layer,
            k,
        )

    elif query_type == QueryType.GLOBAL:
        k = top_k or _GLOBAL_TOP_K
        replay_wids = _route_global(lib, k)

    elif query_type == QueryType.TONE:
        k = top_k or _TONE_TOP_K
        replay_wids = _route_tone(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config,
            k,
        )

    else:
        # Should not happen — defensive fallback
        k = top_k or _FACTUAL_TOP_K
        replay_wids = _route_factual(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config,
            k,
        )

    route_ms = (time.time() - t0) * 1000
    print(
        f"  Routing: {query_type.value} → {len(replay_wids)} windows "
        f"({route_ms:.0f}ms): {replay_wids}",
        file=sys.stderr,
    )

    if not replay_wids:
        return GenerateResult(
            response="(No windows selected)",
            tokens_generated=0,
            context_tokens=0,
        )

    # ── Step 4: Replay and generate ───────────────────────────────────
    return _replay_and_generate(
        lib,
        kv_gen,
        tokenizer,
        replay_wids,
        prompt_text,
        config,
        sys_content,
        no_chat,
        query_type=query_type,
    )


# ── Routing functions ────────────────────────────────────────────────


def _route_factual(lib, kv_gen, prompt_ids, prompt_text, tokenizer, model_config, top_k):
    """Geometric RRF — best general-purpose factual router."""
    if not lib.has_compass:
        from ..compass_routing._bm25 import _bm25_score_windows

        scores = _bm25_score_windows(lib, tokenizer, prompt_text)
        return [wid for wid, _ in scores[:top_k]]

    return compass_route(
        lib,
        kv_gen,
        prompt_ids,
        prompt_text,
        tokenizer,
        model_config=model_config,
        strategy=RoutingStrategy.GEOMETRIC,
        top_k=top_k,
    )


def _route_engagement(
    lib,
    kv_gen,
    prompt_ids,
    prompt_text,
    tokenizer,
    model_config,
    probes,
    compass_layer,
    top_k,
):
    """Compass coarse-filter → probe re-rank for engagement content.

    Replaces BM25 indicator pre-filtering with compass geometric routing,
    which captures content-type geometry without keyword blindness.
    BM25 indicators miss keyword-free engagement content (e.g. W37 birthday);
    the compass navigates to the same subspace without vocabulary dependence.
    """
    # Compass coarse-filter: top-N windows without keyword bias
    if lib.has_compass:
        coarse_wids = compass_route(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config=model_config,
            strategy=RoutingStrategy.GEOMETRIC,
            top_k=_COARSE_CANDIDATES,
        )
        print(
            f"    Compass pre-filter: {len(coarse_wids)} candidates",
            file=sys.stderr,
        )
    else:
        # No compass — fall back to BM25 indicators (degraded)
        from ..compass_routing._indicator_bm25 import (
            ENGAGEMENT_INDICATORS,
            _indicator_bm25_score_windows,
        )

        bm25_scores = _indicator_bm25_score_windows(lib, tokenizer, ENGAGEMENT_INDICATORS)
        coarse_wids = [wid for wid, s in bm25_scores[:_COARSE_CANDIDATES] if s > 0.0]
        if not coarse_wids:
            coarse_wids = list(range(min(_COARSE_CANDIDATES, lib.num_windows)))
        print(
            f"    BM25 fallback (no compass): {len(coarse_wids)} candidates",
            file=sys.stderr,
        )

    # Probe re-rank if tonal probe is available
    if probes.tonal_available:
        from ._probe_rerank import _probe_rerank_windows

        reranked = _probe_rerank_windows(
            lib,
            kv_gen,
            tokenizer,
            coarse_wids,
            probes.tonal_direction,
            probes.tonal_mean,
            compass_layer,
            probe_type="engagement",
            top_k=top_k,
        )
        return [wid for wid, _ in reranked]

    # No probe — use compass ordering
    return coarse_wids[:top_k]


def _route_tension(
    lib,
    kv_gen,
    prompt_ids,
    prompt_text,
    tokenizer,
    model_config,
    probes,
    compass_layer,
    top_k,
):
    """Compass coarse-filter + temporal stride → tension probe re-rank.

    Compass replaces BM25 indicators as the primary candidate filter.
    Temporal stride is retained because tension is sometimes position-dependent
    (mission-critical phases cluster in specific parts of the document).
    """
    from ..compass_routing._temporal import _temporal_stride_windows

    # Compass coarse-filter
    if lib.has_compass:
        compass_wids = compass_route(
            lib,
            kv_gen,
            prompt_ids,
            prompt_text,
            tokenizer,
            model_config=model_config,
            strategy=RoutingStrategy.GEOMETRIC,
            top_k=_COARSE_CANDIDATES,
        )
    else:
        # No compass — fall back to BM25 indicators
        from ..compass_routing._indicator_bm25 import (
            TENSION_INDICATORS,
            _indicator_bm25_score_windows,
        )

        bm25_scores = _indicator_bm25_score_windows(lib, tokenizer, TENSION_INDICATORS)
        compass_wids = [wid for wid, s in bm25_scores[:_COARSE_CANDIDATES] if s > 0.0]

    # Temporal stride — tension can cluster at mission-critical positions
    stride_scores = _temporal_stride_windows(lib, k=10)
    stride_wids = [wid for wid, _ in stride_scores]

    # Union: compass first (content-based), then stride (position-based coverage)
    all_candidates = list(dict.fromkeys(compass_wids + stride_wids))
    print(
        f"    Tension candidates: {len(compass_wids)} compass + "
        f"{len(stride_wids)} stride = {len(all_candidates)} unique",
        file=sys.stderr,
    )

    if not all_candidates:
        return stride_wids[:top_k]

    # Probe re-rank if tension probe is available
    if probes.tension_available and probes.tension_direction is not None:
        from ._probe_rerank import _probe_rerank_windows

        reranked = _probe_rerank_windows(
            lib,
            kv_gen,
            tokenizer,
            all_candidates,
            probes.tension_direction,
            probes.tension_mean,
            compass_layer,
            probe_type="tension",
            top_k=top_k,
        )
        return [wid for wid, _ in reranked]

    # No probe — compass ordering first
    return all_candidates[:top_k]


def _route_global(lib, top_k):
    """Temporal stride — evenly spaced windows for global/timeline queries."""
    from ..compass_routing._temporal import _temporal_stride_windows

    scores = _temporal_stride_windows(lib, k=top_k)
    return [wid for wid, _ in scores]


def _route_tone(lib, kv_gen, prompt_ids, prompt_text, tokenizer, model_config, top_k):
    """Geometric routing with more windows — tone needs broader context."""
    if not lib.has_compass:
        from ..compass_routing._bm25 import _bm25_score_windows

        scores = _bm25_score_windows(lib, tokenizer, prompt_text)
        return [wid for wid, _ in scores[:top_k]]

    return compass_route(
        lib,
        kv_gen,
        prompt_ids,
        prompt_text,
        tokenizer,
        model_config=model_config,
        strategy=RoutingStrategy.GEOMETRIC,
        top_k=top_k,
    )


# ── Replay + generate ───────────────────────────────────────────────


def _replay_and_generate(
    lib,
    kv_gen,
    tokenizer,
    replay_wids,
    prompt_text,
    config,
    sys_content,
    no_chat,
    query_type=None,
):
    """Common replay path: preamble → windows → postamble → decode."""
    from ._probes import QueryType

    # For global/timeline queries, use a template scaffold that forces
    # the model to distribute attention across all excerpts rather than
    # narrating the first few in exhaustive detail.
    if query_type == QueryType.GLOBAL:
        postamble_prompt = (
            f"\n\n---\nBased on the excerpts above, {prompt_text}\n\n"
            "Cover the full arc from beginning to end. "
            "Mention each major phase or event briefly (1-2 sentences each). "
            "Do not spend more than 2 sentences on any single excerpt."
        )
    else:
        postamble_prompt = f"\n\n---\nBased on the text above, {prompt_text}"

    # Build chat framing
    if not no_chat and hasattr(tokenizer, "apply_chat_template"):
        preamble_text = f"<start_of_turn>user\n{sys_content}\n\nHere is the relevant text:\n\n"
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)
        postamble_text = f"{postamble_prompt}<end_of_turn>\n<start_of_turn>model\n"
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)
    else:
        preamble_ids = tokenizer.encode("Document:\n\n", add_special_tokens=False)
        postamble_ids = tokenizer.encode(
            f"{postamble_prompt}\nAnswer:",
            add_special_tokens=False,
        )

    # Prefill preamble
    seq_len = 0
    _l, gen_kv = kv_gen.prefill(mx.array(preamble_ids)[None])
    mx.eval(*[t for p in gen_kv for t in p])
    seq_len += len(preamble_ids)

    # Replay windows in document order (ascending wid)
    for wid in sorted(replay_wids):
        w_tokens = lib.get_window_tokens(wid)
        t0 = time.time()
        _l, gen_kv = kv_gen.extend(
            mx.array(w_tokens)[None],
            gen_kv,
            abs_start=seq_len,
        )
        mx.eval(*[t for p in gen_kv for t in p])
        ms = (time.time() - t0) * 1000
        seq_len += len(w_tokens)
        print(
            f"  Replayed W{wid} @ pos {seq_len - len(w_tokens)}-{seq_len} ({ms:.0f}ms)",
            file=sys.stderr,
        )

    # Postamble
    logits, gen_kv = kv_gen.extend(
        mx.array(postamble_ids)[None],
        gen_kv,
        abs_start=seq_len,
    )
    seq_len += len(postamble_ids)

    # Autoregressive decode
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
        sys.stdout.write(tokenizer.decode([next_token], skip_special_tokens=True))
        sys.stdout.flush()
        logits, gen_kv = kv_gen.step_uncompiled(
            mx.array([[next_token]]),
            gen_kv,
            seq_len=seq_len,
        )
        seq_len += 1

    print()

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return GenerateResult(
        response=response,
        tokens_generated=len(generated_tokens),
        context_tokens=seq_len,
    )
