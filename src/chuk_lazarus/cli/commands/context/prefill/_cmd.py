"""Prefill CLI command — tokenize a document and save a windowed checkpoint library.

This is the entry point. Serialization logic lives in sibling modules.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone

from .._types import PrefillConfig, PrefillResult, ResidualMode
from ._progress import progress_line
from ._restore import restore_engine
from ._save import save_library


async def context_prefill_cmd(args: Namespace) -> None:
    """CLI entry point: prefill a text file into a windowed checkpoint library."""
    import mlx.core as mx

    from .....inference import UnifiedPipeline
    from .....inference.context import (
        LibraryFile,
    )
    from .....inference.context.unlimited_engine import UnlimitedContextEngine
    from .....inference.context.sparse_engine import SparseIndexEngine

    config = PrefillConfig.from_args(args)

    frame_bank_data = None
    if config.residual_mode == ResidualMode.DARKSPACE and config.frame_bank is not None:
        if not config.frame_bank.exists():
            print(f"Error: frame bank not found: {config.frame_bank}", file=sys.stderr)
            return
        frame_bank_data = dict(mx.load(str(config.frame_bank)))
        fb_shape = frame_bank_data["frame_bank"].shape
        print(
            f"Frame bank: {config.frame_bank.name}  |  {fb_shape[0]}D × {fb_shape[1]} hidden",
            file=sys.stderr,
        )
    elif config.residual_mode == ResidualMode.DARKSPACE:
        print(
            "Darkspace mode: frame bank will be calibrated from corpus",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 1. Read source
    # ------------------------------------------------------------------
    if not config.input_file.exists():
        print(f"Error: input file not found: {config.input_file}", file=sys.stderr)
        return

    source_text = config.input_file.read_text(errors="replace")

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
    total_windows = (total_tokens + config.window_size - 1) // config.window_size
    lib_name = config.name or config.input_file.stem.replace("_", " ").title()

    phases_str = ",".join(sorted(config.phases))
    print(
        f"Source: {config.input_file.name}  |  {total_tokens:,} tokens  |  "
        f"window_size={config.window_size}  |  ~{total_windows} windows  |  "
        f"residuals={config.residual_mode.value}  |  phases={phases_str}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 4. Build engine
    # ------------------------------------------------------------------
    if config.run_sparse:
        engine = SparseIndexEngine(
            pipeline.model, pipeline.config, window_size=config.window_size
        )
        engine.set_tokenizer(tokenizer)
    else:
        engine = UnlimitedContextEngine(
            pipeline.model, pipeline.config, window_size=config.window_size
        )

    # Enable full KV save for Mode 6 (prefix caching)
    if config.store_kv_full:
        engine.enable_kv_full_save(str(config.checkpoint))
        print("  Full KV save enabled (Mode 6)", file=sys.stderr)

    # Warm up compute graph
    _warm = mx.array([[1, 2, 3]])
    _, _kv = engine.kv_gen.prefill(_warm)
    mx.eval()

    created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 5. Resume from existing partial library
    # ------------------------------------------------------------------
    resume_tokens = 0
    output_path = config.checkpoint

    if config.resume and (output_path / LibraryFile.WINDOWS).exists():
        resume_tokens = restore_engine(engine, output_path, pipeline.config)
        if resume_tokens > 0:
            manifest_path = output_path / LibraryFile.MANIFEST
            if manifest_path.exists():
                saved = json.loads(manifest_path.read_text())
                created_at = saved.get("created_at", created_at)
            pct = resume_tokens / total_tokens
            pct_str = f"{pct:.0%}" if resume_tokens == total_tokens else f"{pct:.1%}"
            print(
                f"Resuming from token {resume_tokens}/{total_tokens} ({pct_str}, "
                f"{engine.current_window_id} windows already done)",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # 5b. Extraction-only mode: skip prefill loop if windows not requested
    #     and the library already has windows on disk.
    # ------------------------------------------------------------------
    if not config.run_windows and resume_tokens > 0:
        print(
            f"Skipping window prefill (--phases does not include 'windows'). "
            f"Running extraction on {engine.stats().archived_windows} existing windows.",
            file=sys.stderr,
        )
        start_wall = time.monotonic()
        save_library(
            engine, output_path, token_ids, lib_name,
            config.model, pipeline.config, config.window_size,
            tokenizer, created_at, is_complete=True,
            quick=False, residual_mode=config.residual_mode,
            frame_bank_data=frame_bank_data,
            frame_bank_path=config.frame_bank,
            store_pages=config.store_pages,
            append_from=engine.stats().archived_windows,
            run_interval=config.run_interval,
            run_compass=config.run_compass,
            run_darkspace=config.run_darkspace,
            run_pages=config.run_pages,
            run_surprise=config.run_surprise,
            run_sparse=config.run_sparse,
        )
        elapsed = time.monotonic() - start_wall
        s = engine.stats()
        result = PrefillResult(
            checkpoint=str(config.checkpoint),
            tokens_prefilled=resume_tokens,
            num_windows=s.archived_windows,
            status="complete",
            elapsed_seconds=elapsed,
        )
        print(result.to_display())
        return

    if not config.run_windows and resume_tokens == 0:
        print(
            "Error: --phases does not include 'windows' but no existing library found. "
            "Run with --phases windows (or all) first.",
            file=sys.stderr,
        )
        return

    # ------------------------------------------------------------------
    # 6. Full prefill: windows + requested extraction phases
    # ------------------------------------------------------------------
    if resume_tokens >= total_tokens:
        if config.store_pages and not (output_path / "pages.npz").exists():
            print("Already prefilled. Extracting pages...", file=sys.stderr)
            save_library(
                engine, output_path, token_ids, lib_name,
                config.model, pipeline.config, config.window_size,
                tokenizer, created_at, is_complete=True,
                quick=True, residual_mode=config.residual_mode,
                frame_bank_data=frame_bank_data,
                frame_bank_path=config.frame_bank,
                store_pages=True,
                run_pages=True,
                run_surprise=False,
            )
            return
        print("Already fully prefilled. Nothing to do.", file=sys.stderr)
        return

    # SIGINT handler
    _sigint_received = False
    _original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):
        nonlocal _sigint_received
        if not _sigint_received:
            print(
                "\n  Ctrl-C received — finishing current window, then saving...",
                file=sys.stderr,
                flush=True,
            )
            _sigint_received = True
            signal.signal(signal.SIGINT, lambda *_: os._exit(130))

    signal.signal(signal.SIGINT, _handle_sigint)

    # ------------------------------------------------------------------
    # 7. Process window-by-window
    # ------------------------------------------------------------------
    remaining = token_ids[resume_tokens:]
    start_wall = time.monotonic()
    interrupted = False
    tokens_done = resume_tokens
    _SAVE_INTERVAL_SECS = 300  # save every 5 minutes of wall time
    last_save_time = time.monotonic()

    _last_saved_window = engine.stats().archived_windows  # for incremental saves

    def _do_save(is_complete: bool = False, quick: bool = False) -> None:
        nonlocal _last_saved_window
        current_archived = engine.stats().archived_windows
        save_library(
            engine, output_path, token_ids, lib_name,
            config.model, pipeline.config, config.window_size,
            tokenizer, created_at, is_complete=is_complete,
            quick=quick, residual_mode=config.residual_mode,
            frame_bank_data=frame_bank_data,
            frame_bank_path=config.frame_bank,
            store_pages=config.store_pages,
            append_from=_last_saved_window,
            run_interval=config.run_interval,
            run_compass=config.run_compass,
            run_darkspace=config.run_darkspace,
            run_pages=config.run_pages,
            run_surprise=config.run_surprise,
            run_sparse=config.run_sparse,
        )
        if quick and current_archived > 1:
            evict_ids = list(range(_last_saved_window, current_archived - 1))
            engine.checkpoints.evict(evict_ids)
            engine.residuals.evict(evict_ids)
        _last_saved_window = current_archived

    try:
        for i in range(0, len(remaining), config.window_size):
            chunk = remaining[i : i + config.window_size]
            engine.process(chunk)
            tokens_done = resume_tokens + i + len(chunk)

            elapsed = time.monotonic() - start_wall
            archived = engine.stats().archived_windows
            print(
                progress_line(tokens_done, total_tokens, archived, total_windows, elapsed),
                end="",
                file=sys.stderr,
                flush=True,
            )

            # Periodic save based on wall time
            now = time.monotonic()
            if archived > 0 and (now - last_save_time) >= _SAVE_INTERVAL_SECS:
                _do_save(quick=True)
                last_save_time = now

            if _sigint_received:
                interrupted = True
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\n  Interrupted — saving library...", file=sys.stderr)

    finally:
        signal.signal(signal.SIGINT, _original_sigint)
        elapsed = time.monotonic() - start_wall

        if not interrupted:
            print(
                f"\n  flushing final window...",
                file=sys.stderr,
                flush=True,
            )
            engine.flush()

        # Final save: append remaining unsaved windows, then run extraction passes.
        _do_save(is_complete=not interrupted, quick=False)

    print(file=sys.stderr)

    if interrupted:
        s = engine.stats()
        print(
            f"\nPartial library saved ({tokens_done}/{total_tokens} tokens, "
            f"{s.archived_windows} windows).",
            file=sys.stderr,
        )
        print(
            f"Resume with:\n  lazarus context prefill "
            f"--model {config.model} --input {config.input_file} "
            f"--checkpoint {config.checkpoint} "
            f"--window-size {config.window_size}",
            file=sys.stderr,
        )
        return

    s = engine.stats()
    result = PrefillResult(
        checkpoint=str(config.checkpoint),
        tokens_prefilled=tokens_done,
        num_windows=s.archived_windows,
        status="complete",
        elapsed_seconds=elapsed,
    )
    print(result.to_display())
