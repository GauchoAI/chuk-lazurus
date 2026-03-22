"""Shared helpers for knowledge CLI commands."""

from __future__ import annotations

import sys

import mlx.core as mx


def load_model(model_id: str):
    """Load model, create KV generator, warm compute graph.

    Returns (pipeline, kv_gen, tokenizer).
    """
    from ....inference import UnifiedPipeline
    from ....inference.context.kv_generator import make_kv_generator

    print(f"Loading model: {model_id}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(model_id, verbose=False)
    kv_gen = make_kv_generator(pipeline.model, pipeline.config)
    tokenizer = pipeline.tokenizer

    # Warm compute graph
    _ = kv_gen.prefill(mx.array([[1, 2, 3]]))

    return pipeline, kv_gen, tokenizer


def prepare_prompt(tokenizer, prompt_text: str) -> list[int]:
    """Wrap prompt in chat template if tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return tokenizer.encode(prompt_text, add_special_tokens=True)


def stop_token_ids(tokenizer) -> set[int]:
    """Return the set of stop token IDs for the tokenizer."""
    if tokenizer.eos_token_id is not None:
        return {tokenizer.eos_token_id}
    return set()


def generate_plain(
    kv_gen,
    prompt_ids: list[int],
    max_tokens: int,
    stop_ids: set[int],
    *,
    stream: bool = False,
    tokenizer=None,
) -> list[int]:
    """Plain generation without injection. Returns generated token IDs.

    When stream=True, writes each token to stdout as it's generated
    (requires tokenizer).
    """
    q_ids = mx.array(prompt_ids)[None]
    logits, kv_store = kv_gen.prefill(q_ids)
    mx.eval(logits)

    generated: list[int] = []
    seq_len = q_ids.shape[1]

    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        if token in stop_ids:
            break
        generated.append(token)
        if stream and tokenizer is not None:
            sys.stdout.write(tokenizer.decode([token], skip_special_tokens=True))
            sys.stdout.flush()
        logits, kv_store = kv_gen.step_uncompiled(
            mx.array([[token]]), kv_store, seq_len=seq_len
        )
        seq_len += 1

    return generated
