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
    from ._metrics import JsonLogger, MetricsState, start_metrics_server

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Optional observability ────────────────────────────────────────
    json_log = JsonLogger(args.json_log) if args.json_log else None
    metrics = None
    metrics_server = None

    if args.metrics_port:
        metrics = MetricsState()
        metrics_server = start_metrics_server(metrics, args.metrics_port)

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

    num_tokens = len(tokens)
    num_windows = -(-num_tokens // ac.window_size)  # ceil division

    print(f"  Document: {num_tokens} tokens ({len(text)} chars)", file=sys.stderr)

    if json_log:
        json_log.event("build_start",
                       model=args.model,
                       input=str(input_path),
                       tokens=num_tokens,
                       windows=num_windows,
                       window_size=ac.window_size,
                       crystal_layer=ac.crystal_layer)

    if metrics:
        metrics.windows_total = num_windows
        metrics.document_tokens = num_tokens
        metrics.window_size = ac.window_size
        metrics.phase = "pass2_boundaries"

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

        if metrics:
            metrics.windows_processed = wid + 1
            metrics.elapsed_s = elapsed
            metrics.eta_s = eta
            metrics.rate_windows_per_s = rate

        if json_log and (wid + 1) % 10 == 0:
            json_log.event("window", phase="pass2_boundaries", window=wid + 1,
                           total=total, rate=round(rate, 3), eta_s=round(eta, 1))

    t1 = [None]  # mutable container for pass3 start time

    def progress_pass3(wid: int, total: int) -> None:
        if t1[0] is None:
            t1[0] = time.monotonic()
            if metrics:
                metrics.phase = "pass3_keywords"
                metrics.windows_processed = 0
            print(file=sys.stderr)  # newline after pass2 progress

        elapsed_p3 = time.monotonic() - t1[0]
        rate = (wid + 1) / elapsed_p3 if elapsed_p3 > 0 else 0
        eta = (total - wid - 1) / rate if rate > 0 else 0
        print(
            f"\r  Keywords {wid + 1}/{total}  ({elapsed_p3:.0f}s elapsed, ~{eta:.0f}s remaining)",
            end="",
            file=sys.stderr,
        )

        if metrics:
            metrics.windows_processed = wid + 1
            metrics.elapsed_s = time.monotonic() - t0
            metrics.eta_s = eta
            metrics.rate_windows_per_s = rate

        if json_log and (wid + 1) % 10 == 0:
            json_log.event("window", phase="pass3_keywords", window=wid + 1,
                           total=total, rate=round(rate, 3), eta_s=round(eta, 1))

    store = streaming_prefill(
        kv_gen=kv_gen,
        document_tokens=tokens,
        config=ac,
        tokenizer=tokenizer,
        progress_fn=progress,
        progress_pass3_fn=progress_pass3,
    )
    elapsed = time.monotonic() - t0
    print(file=sys.stderr)

    if metrics:
        metrics.phase = "saving"

    # ── Save ──────────────────────────────────────────────────────────
    store.save(output_path)
    store.log_stats()

    print(
        f"  Built in {elapsed:.1f}s → {output_path}",
        file=sys.stderr,
    )

    if metrics:
        metrics.phase = "done"
        metrics.elapsed_s = time.monotonic() - t0

    if json_log:
        json_log.event("build_done",
                       elapsed_s=round(elapsed, 2),
                       entries=len(store.entries),
                       windows=store.num_windows,
                       output=str(output_path))
        json_log.close()

    if metrics_server:
        metrics_server.shutdown()
