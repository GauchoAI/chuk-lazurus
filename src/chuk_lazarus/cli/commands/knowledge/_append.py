"""knowledge append — Append a single skill/document to a knowledge store."""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path


async def knowledge_append_cmd(args: Namespace) -> None:
    """Append a new skill (Markdown file) to an existing knowledge store."""
    from ....inference.context.knowledge import ArchitectureConfig
    from ....inference.context.knowledge.append import append_skill
    from ._common import load_model
    from ._metrics import JsonLogger, MetricsState, start_metrics_server

    input_path = Path(args.input)
    store_path = Path(args.store)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not store_path.exists():
        print(f"Error: store not found: {store_path}. Run 'lazarus knowledge init' first.",
              file=sys.stderr)
        sys.exit(1)

    json_log = JsonLogger(args.json_log) if getattr(args, "json_log", None) else None
    metrics = None
    metrics_server = None

    if getattr(args, "metrics_port", None):
        metrics = MetricsState()
        metrics_server = start_metrics_server(metrics, args.metrics_port)

    pipeline, kv_gen, tokenizer = load_model(args.model)
    ac = ArchitectureConfig.from_model_config(pipeline.config)

    if json_log:
        json_log.event("append_start", model=args.model,
                       input=str(input_path), store=str(store_path))

    t0 = time.monotonic()

    def progress_fn(stage: str, pct: float):
        if metrics:
            metrics.phase = f"append_{stage}"
            metrics.elapsed_s = time.monotonic() - t0
        if json_log:
            json_log.event("append_progress", stage=stage, progress=round(pct, 3))

    result = append_skill(
        kv_gen=kv_gen,
        tokenizer=tokenizer,
        store_path=store_path,
        new_doc_path=input_path,
        config=ac,
        progress_fn=progress_fn,
    )

    if json_log:
        json_log.event("append_done", **result)
        json_log.close()

    if metrics:
        metrics.phase = "done"

    if metrics_server:
        metrics_server.shutdown()
