"""Vec inject generate mode — 1D subspace fact injection at L30.

Flow
----
1. Load LocalVecInjectProvider from checkpoint.
2. Route bare query via Q·K at L29 H4 — selects ONE matching fact.
3. If routing_confident and fact is distinctive:
     a. Run full prompt through L0→29 → residual h.
     b. Inject c × (e/‖e‖²) into h at the last position.
     c. Continue L30→33 → logits + KV store.
     d. Autoregressive decode.
4. If routing confidence is too low or no distinctive facts:
     Fall back to geometric window replay.

The 0.05% injected signal at L30 dominates the first answer token.
Wrong injection = wrong answer at >99% confidence — the confidence
gate is mandatory before calling vec_inject_all().
"""

from __future__ import annotations

import asyncio
import sys
import time

from ..._types import GenerateConfig, GenerateResult


def run_vec_inject(
    lib, kv_gen, pipeline, tokenizer,
    prompt_ids: list[int],
    prompt_text: str,
    config: GenerateConfig,
    args,
    mx,
) -> None:
    """Synchronous entry point — runs the async inject pipeline."""
    asyncio.run(
        _run_async(
            lib, kv_gen, pipeline, tokenizer,
            prompt_ids, prompt_text, config, args, mx,
        )
    )


async def _run_async(
    lib, kv_gen, pipeline, tokenizer,
    prompt_ids: list[int],
    prompt_text: str,
    config: GenerateConfig,
    args,
    mx,
) -> None:
    from ......inference.context.vec_inject import LocalVecInjectProvider, vec_inject_all

    checkpoint_dir = lib.path
    top_k = getattr(args, "top_k", None) or 1
    confidence_threshold = float(getattr(args, "confidence_threshold", None) or 0.15)

    # ── Load provider ─────────────────────────────────────────────────
    print(f"  Loading vec_inject index from {checkpoint_dir} ...", file=sys.stderr)
    t0 = time.monotonic()
    try:
        provider = await LocalVecInjectProvider.load(
            checkpoint_dir, kv_gen,
            confidence_threshold=confidence_threshold,
        )
    except FileNotFoundError as e:
        print(f"  {e}", file=sys.stderr)
        _fallback(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx)
        return

    provider.log_stats()
    print(f"  Load: {(time.monotonic() - t0) * 1000:.0f} ms", file=sys.stderr)

    # ── Route — bare query tokens only (chat wrap adds structural noise) ──
    bare_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    result = await provider.retrieve(bare_ids, prompt_text, top_k=top_k)

    conf_str = "CONFIDENT" if result.routing_confident else "LOW CONFIDENCE"
    print(
        f"  Routing [{conf_str}]: top_score={result.top_score:.4f}  "
        f"({result.retrieval_ms:.1f} ms)",
        file=sys.stderr,
    )

    if not result.routing_confident or not result.matches:
        print(
            "  Confidence below threshold — falling back to window replay.",
            file=sys.stderr,
        )
        _fallback(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx)
        return

    # Filter: only distinctive answer tokens are safe for 1D injection
    matches = [m for m in result.matches if m.distinctive]
    if not matches:
        print(
            "  Top matches have non-distinctive tokens — falling back to window replay.",
            file=sys.stderr,
        )
        _fallback(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx)
        return

    inject_layer = result.injection_layer
    for i, m in enumerate(matches):
        tok = tokenizer.decode([m.token_id], skip_special_tokens=True).strip()
        print(
            f"  Inject rank={i + 1}  W{m.window_id}[{m.position}]  "
            f"score={m.score:.4f}  c={m.coefficient:+.4f}  tok={tok!r}",
            file=sys.stderr,
        )

    # ── Forward pass L0→inject_layer-1 ───────────────────────────────
    q_ids = mx.array(prompt_ids)[None]
    t0 = time.monotonic()
    h = kv_gen.prefill_to_layer(q_ids, target_layer=inject_layer - 1)
    # h: (1, S, hidden_size)  residuals entering inject_layer

    # ── Inject into last position ─────────────────────────────────────
    embed_matrix = pipeline.model.model.embed_tokens.weight
    h_last = h[:, -1:, :]                                     # (1, 1, hidden_size)
    h_injected = vec_inject_all(h_last, matches, embed_matrix) # (1, 1, hidden_size)
    # Reconstruct full-sequence h with modified last position
    if h.shape[1] > 1:
        h = mx.concatenate([h[:, :-1, :], h_injected], axis=1)
    else:
        h = h_injected

    # ── Continue L{inject_layer}→end — builds real KV for upper layers ──
    logits, gen_kv = kv_gen.prefill_from_layer(h, start_layer=inject_layer)
    seq_len = q_ids.shape[1]
    inject_ms = (time.monotonic() - t0) * 1000
    print(f"  Inject + L{inject_layer}→end forward: {inject_ms:.0f} ms", file=sys.stderr)

    # ── Autoregressive decode ─────────────────────────────────────────
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
            mx.array([[next_token]]), gen_kv, seq_len=seq_len,
        )
        seq_len += 1

    print()
    gen_result = GenerateResult(
        response=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        tokens_generated=len(generated_tokens),
        context_tokens=seq_len,
    )
    print(gen_result.to_display())


def _fallback(lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx):
    """Fall back to geometric window replay when routing confidence is too low."""
    from ...compass_routing import RoutingStrategy, compass_route
    from .._cmd import _decode_loop
    from ._standard import run_standard

    top_k = getattr(args, "top_k", None) or 3
    replay_ids = compass_route(
        lib, kv_gen, prompt_ids, prompt_text, tokenizer,
        model_config=pipeline.config,
        strategy=RoutingStrategy.GEOMETRIC,
        top_k=top_k,
    )
    print(f"  Fallback replay windows: {replay_ids}", file=sys.stderr)

    preamble_ids = tokenizer.encode("Transcript:\n\n", add_special_tokens=False)
    context_kv, seq_len = run_standard(lib, kv_gen, replay_ids, preamble_ids, mx)

    postamble_ids = tokenizer.encode(
        f"\n\n---\nQuestion: {prompt_text}\nAnswer:", add_special_tokens=False,
    )
    q_ids = mx.array(postamble_ids)[None]
    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

    result = _decode_loop(logits, gen_kv, kv_gen, tokenizer, config, seq_len, mx)
    print(result.to_display())
