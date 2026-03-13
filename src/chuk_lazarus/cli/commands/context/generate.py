"""Generate command — extend a saved KV checkpoint with a prompt and generate."""

from __future__ import annotations

import sys
from argparse import Namespace

from ._types import GenerateConfig, GenerateResult


async def context_generate_cmd(args: Namespace) -> None:
    """CLI entry point: load a KV checkpoint, extend with prompt, generate."""
    import mlx.core as mx

    from ....inference import UnifiedPipeline
    from ....inference.context import make_kv_generator
    from ....inference.context.kv_checkpoint import KVCheckpoint

    config = GenerateConfig.from_args(args)

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    if not config.checkpoint.exists():
        print(f"Error: checkpoint not found: {config.checkpoint}", file=sys.stderr)
        return

    print(f"Loading checkpoint: {config.checkpoint}", file=sys.stderr)
    kv_store, token_ids, meta = KVCheckpoint.load(config.checkpoint)
    print(
        f"  {meta.seq_len} tokens in context  |  model: {meta.model_id}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {config.model}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(config.model, verbose=False)

    if meta.model_id != config.model:
        print(
            f"Warning: checkpoint model_id={meta.model_id!r} but loading model={config.model!r}",
            file=sys.stderr,
        )

    tokenizer = pipeline.tokenizer
    kv_gen = make_kv_generator(pipeline.model)

    # ------------------------------------------------------------------
    # 3. Extend with prompt (if provided) or seed from last prefill token
    # ------------------------------------------------------------------
    prompt_text = config.prompt_text
    seq_len = meta.seq_len

    if prompt_text:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        print(f"Extending with {len(prompt_ids)} prompt tokens...", file=sys.stderr)
        prompt_array = mx.array([prompt_ids])
        logits, kv_store = kv_gen.extend(prompt_array, kv_store, abs_start=seq_len)
        seq_len += len(prompt_ids)
    else:
        # Seed with the last prefill token to obtain current logits
        seed_token = token_ids[-1] if token_ids else (tokenizer.bos_token_id or 1)
        logits, kv_store = kv_gen.step(mx.array([[seed_token]]), kv_store, seq_len=seq_len - 1)

    context_tokens = seq_len

    # ------------------------------------------------------------------
    # 4. Greedy / temperature sampling loop
    # ------------------------------------------------------------------
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    generated_tokens: list[int] = []

    for _ in range(config.max_tokens):
        last_logits = logits[0, -1]  # (vocab_size,)

        if config.temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / config.temperature
            probs = mx.softmax(scaled, axis=-1)
            next_token = int(mx.random.categorical(probs[None]).item())

        if next_token in stop_ids:
            break

        generated_tokens.append(next_token)
        logits, kv_store = kv_gen.step(mx.array([[next_token]]), kv_store, seq_len=seq_len)
        seq_len += 1

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    result = GenerateResult(
        response=response,
        tokens_generated=len(generated_tokens),
        context_tokens=context_tokens,
    )
    print(result.to_display())


__all__ = ["context_generate_cmd"]
