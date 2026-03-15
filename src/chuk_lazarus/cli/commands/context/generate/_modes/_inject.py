"""Dark space injection mode — the model reads its own L26 residuals."""

from __future__ import annotations

import sys
import time

from ..._types import GenerateResult
from ...compass_routing import RoutingStrategy, compass_route


def run_inject(
    lib, kv_gen, pipeline, tokenizer, prompt_ids, prompt_text, config, args, mx,
):
    """Dark space injection — the model reads its own L26 residuals.

    1. Load ALL stored L26 residuals (5,800 positions from prefill)
    2. Process query through L0-L26 → query residual at L26
    3. Concatenate: [stored residuals] + [query residual] at L26
    4. Run layers 27-34 on the combined sequence
    5. The model's own attention reads ALL dark space states
       in full 2560D — content, tone, register, entity

    No tokens. No projection. No compass selection.
    The model reads its own computation through its own layers.

    Handles its own generation and returns directly.
    """
    compass_layer = lib.compass_layer

    # Select windows: compass-routed if strategy given, otherwise all
    strategy_arg = getattr(args, "strategy", None)
    top_k_override = getattr(args, "top_k", None)

    if strategy_arg:
        strategy = RoutingStrategy(strategy_arg)
        top_k = top_k_override if top_k_override is not None else 3
        inject_wids = compass_route(
            lib, kv_gen, prompt_ids, prompt_text, tokenizer,
            model_config=pipeline.config,
            strategy=strategy,
            top_k=top_k,
        )
    else:
        inject_wids = list(range(lib.num_windows))

    # Load L26 residuals for selected windows
    inject_vecs = []
    for wid in inject_wids:
        for res in lib.get_compass_residuals(wid):
            inject_vecs.append(res.reshape(-1))

    n_stored = len(inject_vecs)
    print(
        f"  Loading {n_stored} L26 residuals from {len(inject_wids)} windows",
        file=sys.stderr,
    )

    # Process query through L0-L26 to get its residual
    q_ids = mx.array(prompt_ids)[None]
    q_h = kv_gen.prefill_to_layer(q_ids, target_layer=compass_layer)
    # q_h: (1, q_len, 2560) — all query positions at L26
    q_len = q_h.shape[1]

    # Concatenate: [stored dark space] + [query] at L26
    stored_h = mx.stack(inject_vecs, axis=0)[None, :, :]  # (1, N, 2560)
    combined_h = mx.concatenate([stored_h, q_h], axis=1)  # (1, N+q_len, 2560)
    total_seq = n_stored + q_len

    print(
        f"  Combined sequence: {n_stored} stored + {q_len} query = "
        f"{total_seq} positions at L{compass_layer}",
        file=sys.stderr,
    )

    # Run layers 27-34 on the combined sequence
    t0 = time.time()
    logits, context_kv = kv_gen.prefill_from_layer(
        combined_h, start_layer=compass_layer,
    )
    elapsed_ms = (time.time() - t0) * 1000
    seq_len = total_seq
    print(
        f"  Dark space processed: {total_seq} positions through "
        f"L{compass_layer}-L33 ({elapsed_ms:.0f}ms)",
        file=sys.stderr,
    )

    # Skip the normal postamble — logits already include query positions.
    # Generate directly from the last position's logits.
    context_tokens = seq_len
    gen_kv = context_kv

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
