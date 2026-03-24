"""knowledge build — Build a knowledge store from a document."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path


async def knowledge_build_cmd(args: Namespace) -> None:
    """Build knowledge store from a text file."""
    from ....inference.context.knowledge import ArchitectureConfig, streaming_prefill
    from ._common import load_model

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────
    pipeline, kv_gen, tokenizer = load_model(args.model)

    # ── Architecture config ────────────────────────────────────────────
    ac = ArchitectureConfig.from_model_config(pipeline.config)
    if args.window_size != 512:
        object.__setattr__(ac, "window_size", args.window_size)
    if args.entries_per_window != 8:
        object.__setattr__(ac, "entries_per_window", args.entries_per_window)
    print(
        f"  Architecture: crystal_layer=L{ac.crystal_layer}, "
        f"window={ac.window_size}, entries_per_window={ac.entries_per_window}",
        file=sys.stderr,
    )

    # ── Tokenize document ─────────────────────────────────────────────
    text = input_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]

    print(f"  Document: {len(tokens)} tokens ({len(text)} chars)", file=sys.stderr)

    # ── Build knowledge store ─────────────────────────────────────────
    t0 = time.monotonic()

    def progress(wid: int, total: int) -> None:
        elapsed = time.monotonic() - t0
        rate = (wid + 1) / elapsed if elapsed > 0 else 0
        eta = (total - wid - 1) / rate if rate > 0 else 0
        print(
            f"\r  Window {wid + 1}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
            end="",
            file=sys.stderr,
        )

    store = streaming_prefill(
        kv_gen=kv_gen,
        document_tokens=tokens,
        config=ac,
        tokenizer=tokenizer,
        progress_fn=progress,
    )
    elapsed = time.monotonic() - t0
    print(file=sys.stderr)

    # ── Save ──────────────────────────────────────────────────────────
    store.save(output_path)
    store.log_stats()

    print(
        f"  Built in {elapsed:.1f}s → {output_path}",
        file=sys.stderr,
    )
