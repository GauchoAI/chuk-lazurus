"""Prefill command — tokenize a document and save a KV checkpoint with progress."""

from __future__ import annotations

import sys
import time
from argparse import Namespace

from ._types import PrefillConfig, PrefillResult


def _progress(tokens_done: int, total: int, elapsed: float, width: int = 40) -> str:
    """Return an in-place progress line."""
    pct = tokens_done / total if total else 0.0
    filled = int(width * pct)
    bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled)
    rate = tokens_done / elapsed if elapsed > 0 else 0.0
    eta = (total - tokens_done) / rate if rate > 0 else 0.0
    return f"\r  [{bar}] {tokens_done:>6}/{total} tokens  {rate:>6.0f} tok/s  ETA {eta:>4.0f}s"


async def context_prefill_cmd(args: Namespace) -> None:
    """CLI entry point: prefill a text file and save a KV checkpoint."""
    import mlx.core as mx

    from ....inference import UnifiedPipeline
    from ....inference.context import make_kv_generator
    from ....inference.context.kv_checkpoint import (
        CheckpointMeta,
        ContextCheckpointStatus,
        KVCheckpoint,
    )

    config = PrefillConfig.from_args(args)

    # ------------------------------------------------------------------
    # 1. Read source
    # ------------------------------------------------------------------
    if not config.input_file.exists():
        print(f"Error: input file not found: {config.input_file}", file=sys.stderr)
        return

    source_bytes = config.input_file.read_bytes()
    source_hash = KVCheckpoint.source_hash(source_bytes)
    source_text = source_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 2. Load model and tokenizer
    # ------------------------------------------------------------------
    print(f"Loading model: {config.model}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(config.model, verbose=False)
    tokenizer = pipeline.tokenizer

    # ------------------------------------------------------------------
    # 3. Tokenize
    # ------------------------------------------------------------------
    token_ids: list[int] = tokenizer.encode(source_text, add_special_tokens=False)
    if config.max_tokens is not None:
        token_ids = token_ids[: config.max_tokens]

    total_tokens = len(token_ids)
    print(
        f"Source: {config.input_file.name}  |  {total_tokens} tokens  |  "
        f"chunk_size={config.chunk_size}",
        file=sys.stderr,
    )

    input_ids = mx.array([token_ids])  # (1, S)

    # ------------------------------------------------------------------
    # 4. Check for resumable checkpoint
    # ------------------------------------------------------------------
    kv_gen = make_kv_generator(pipeline.model)
    resume_offset = 0
    kv_store: list = []
    created_at = KVCheckpoint._now()

    if config.resume and KVCheckpoint.is_resumable(config.checkpoint, source_hash):
        existing_kv, _, existing_meta = KVCheckpoint.load(config.checkpoint)
        resume_offset = existing_meta.seq_len
        kv_store = existing_kv
        created_at = existing_meta.created_at
        print(
            f"Resuming from token {resume_offset}/{total_tokens} "
            f"({resume_offset / total_tokens:.0%})",
            file=sys.stderr,
        )
        # Slice the remaining tokens
        input_ids = input_ids[:, resume_offset:]
        if input_ids.shape[1] == 0:
            print("Already fully prefilled. Nothing to do.", file=sys.stderr)
            return

    # ------------------------------------------------------------------
    # 5. Progressive prefill
    # ------------------------------------------------------------------
    tokens_done = resume_offset
    start_wall = time.monotonic()
    interrupted = False

    try:
        for chunk_done, chunk_total, _logits, kv_store in kv_gen.prefill_chunked(
            input_ids,
            chunk_size=config.chunk_size,
            abs_start=resume_offset,
            kv_store=kv_store if resume_offset > 0 else None,
        ):
            tokens_done = resume_offset + chunk_done
            elapsed = time.monotonic() - start_wall
            print(
                _progress(tokens_done, total_tokens, elapsed),
                end="",
                file=sys.stderr,
                flush=True,
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\n  Interrupted — saving partial checkpoint...", file=sys.stderr)

    finally:
        elapsed = time.monotonic() - start_wall
        status = (
            ContextCheckpointStatus.PARTIAL
            if interrupted or tokens_done < total_tokens
            else ContextCheckpointStatus.COMPLETE
        )

        if kv_store:
            layer0_k = kv_store[0][0]
            meta = CheckpointMeta(
                model_id=config.model,
                seq_len=tokens_done,
                total_tokens=total_tokens,
                num_layers=len(kv_store),
                num_kv_heads=int(layer0_k.shape[1]),
                head_dim=int(layer0_k.shape[3]),
                chunk_size=config.chunk_size,
                source_hash=source_hash,
                status=status,
                created_at=created_at,
                updated_at=KVCheckpoint._now(),
            )
            KVCheckpoint.save(config.checkpoint, kv_store, token_ids, meta)

    print(file=sys.stderr)  # newline after progress bar

    if interrupted:
        print(
            f"\nPartial checkpoint saved ({tokens_done}/{total_tokens} tokens).",
            file=sys.stderr,
        )
        print(
            f"Resume with:\n  lazarus context prefill "
            f"--model {config.model} --input {config.input_file} "
            f"--checkpoint {config.checkpoint}",
            file=sys.stderr,
        )
        return

    result = PrefillResult(
        checkpoint=str(config.checkpoint),
        tokens_prefilled=tokens_done,
        status=status.value,
        elapsed_seconds=elapsed,
    )
    print(result.to_display())


__all__ = ["context_prefill_cmd"]
