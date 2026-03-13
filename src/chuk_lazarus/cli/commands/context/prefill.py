"""Prefill command — tokenize a document and save a windowed checkpoint library.

Writes the compact library format (manifest.json, checkpoints.npz, tokens.bin,
windows.json) using UnlimitedContextEngine internally.  Supports resume from a
partial library and Ctrl-C safe saving.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import struct
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from ._types import PrefillConfig, PrefillResult


def _progress(tokens_done: int, total: int, windows_done: int, total_windows: int, elapsed: float, width: int = 40) -> str:
    """Return an in-place progress line."""
    pct = tokens_done / total if total else 0.0
    filled = int(width * pct)
    bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled)
    rate = tokens_done / elapsed if elapsed > 0 else 0.0
    eta = (total - tokens_done) / rate if rate > 0 else 0.0
    line = (
        f"  [{bar}] {tokens_done:>6}/{total} tokens  "
        f"{windows_done}/{total_windows} windows  "
        f"{rate:>6.0f} tok/s  ETA {eta:>4.0f}s"
    )
    return f"\r{line}\033[K"


def _compute_config_hash(config) -> str:
    """Stable hash of the model config key fields."""
    data = {
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
    }
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return f"sha256:{digest}"


async def context_prefill_cmd(args: Namespace) -> None:
    """CLI entry point: prefill a text file into a windowed checkpoint library."""
    import mlx.core as mx

    from ....inference import UnifiedPipeline
    from ....inference.context import (
        LibraryFile,
        LibraryFormatVersion,
        LibraryManifest,
        WindowMeta,
    )
    from ....inference.context.unlimited_engine import UnlimitedContextEngine

    config = PrefillConfig.from_args(args)

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

    print(
        f"Source: {config.input_file.name}  |  {total_tokens:,} tokens  |  "
        f"window_size={config.window_size}  |  ~{total_windows} windows",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 4. Build engine
    # ------------------------------------------------------------------
    engine = UnlimitedContextEngine(
        pipeline.model, pipeline.config, window_size=config.window_size
    )

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
        resume_tokens = _restore_engine(engine, output_path, pipeline.config)
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
            if resume_tokens >= total_tokens:
                print("Already fully prefilled. Nothing to do.", file=sys.stderr)
                return

    # ------------------------------------------------------------------
    # 6. SIGINT handler
    # ------------------------------------------------------------------
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
    _SAVE_EVERY_WINDOWS = 10  # save every N windows to avoid excessive I/O
    last_saved_windows = engine.stats().archived_windows

    def _do_save(is_complete: bool = False) -> None:
        """Write library files from engine's current state."""
        _save_library(
            engine, output_path, token_ids, lib_name,
            config.model, pipeline.config, config.window_size,
            tokenizer, created_at, is_complete=is_complete,
        )

    try:
        for i in range(0, len(remaining), config.window_size):
            chunk = remaining[i : i + config.window_size]
            engine.process(chunk)
            tokens_done = resume_tokens + i + len(chunk)

            elapsed = time.monotonic() - start_wall
            archived = engine.stats().archived_windows
            print(
                _progress(tokens_done, total_tokens, archived, total_windows, elapsed),
                end="",
                file=sys.stderr,
                flush=True,
            )

            # Periodic save every N new windows
            if archived > 0 and (archived - last_saved_windows) >= _SAVE_EVERY_WINDOWS:
                _do_save()
                last_saved_windows = archived

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

        print(
            f"  saving {'partial ' if interrupted else ''}library...",
            file=sys.stderr,
            flush=True,
        )
        _do_save(is_complete=not interrupted)

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


# ---------------------------------------------------------------------------
# Library save / restore helpers
# ---------------------------------------------------------------------------


def _save_library(
    engine,
    output_path: Path,
    all_token_ids: list[int],
    name: str,
    model_id: str,
    config,
    window_size: int,
    tokenizer,
    created_at: str,
    is_complete: bool = False,
) -> None:
    """Write all four library files from the engine's current archived state."""
    import mlx.core as mx

    from ....inference.context import (
        LibraryFile,
        LibraryFormatVersion,
        LibraryManifest,
        WindowMeta,
    )

    s = engine.stats()
    num_archived = s.archived_windows
    if num_archived == 0:
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Collect per-window data
    windows: list[WindowMeta] = []
    ckpt_dict: dict[str, mx.array] = {}
    token_offset = 0

    for wid in range(num_archived):
        w_tokens, w_abs = engine.archive.retrieve(wid)
        kv_last, _ = engine.checkpoints.load(wid)
        preview = tokenizer.decode(w_tokens[:30], skip_special_tokens=True)
        windows.append(
            WindowMeta(
                window_id=wid,
                token_offset=token_offset,
                token_count=len(w_tokens),
                abs_offset=w_abs,
                preview=preview.replace("\n", " ")[:80],
            )
        )
        for li, (k, v) in enumerate(kv_last):
            ckpt_dict[f"w{wid}_l{li}_k"] = k
            ckpt_dict[f"w{wid}_l{li}_v"] = v
        token_offset += len(w_tokens)

    total_tokens_to_report = len(all_token_ids) if is_complete else token_offset

    # 1. checkpoints.npz — save per-window to stay under mx.savez 1024 kwarg limit
    import tempfile
    import zipfile

    npz_path = output_path / LibraryFile.CHECKPOINTS
    with zipfile.ZipFile(str(npz_path), "w", zipfile.ZIP_STORED) as zf:
        for wid in range(num_archived):
            # Collect this window's arrays (num_layers × 2 — well under 1024)
            w_keys = {k: v for k, v in ckpt_dict.items() if k.startswith(f"w{wid}_")}
            # Save to a temp npz, then copy entries into the combined archive
            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                mx.savez(tmp.name, **w_keys)
                with zipfile.ZipFile(tmp.name, "r") as src:
                    for name in src.namelist():
                        zf.writestr(name, src.read(name))

    # 2. tokens.bin (uint32)
    with open(output_path / LibraryFile.TOKENS, "wb") as f:
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            for tid in w_tokens:
                f.write(struct.pack("<I", tid))

    # 3. windows.json
    import json as _json
    (output_path / LibraryFile.WINDOWS).write_text(
        _json.dumps([w.model_dump() for w in windows], indent=2, ensure_ascii=False)
    )

    # 4. manifest.json (written last — the "committed" marker)
    manifest = LibraryManifest(
        name=name,
        model_id=model_id,
        model_config_hash=_compute_config_hash(config),
        num_layers=config.num_hidden_layers,
        window_size=window_size,
        total_tokens=total_tokens_to_report,
        num_windows=num_archived,
        checkpoint_bytes=s.checkpoint_bytes,
        archive_bytes=s.archive_bytes,
        created_at=created_at,
        format_version=LibraryFormatVersion.V1,
    )
    (output_path / LibraryFile.MANIFEST).write_text(manifest.model_dump_json(indent=2))


def _restore_engine(engine, output_path: Path, config) -> int:
    """Reload archived windows from a partial library into a fresh engine.

    Returns the number of tokens already processed (= tokens to skip on resume).
    """
    import json as _json
    import struct as _struct

    import mlx.core as mx

    from ....inference.context import LibraryFile

    windows_path = output_path / LibraryFile.WINDOWS
    ckpt_path = output_path / LibraryFile.CHECKPOINTS
    tokens_path = output_path / LibraryFile.TOKENS

    if not (windows_path.exists() and ckpt_path.exists() and tokens_path.exists()):
        return 0

    raw_windows: list[dict] = _json.loads(windows_path.read_text())
    if not raw_windows:
        return 0

    try:
        raw_ckpts: dict[str, mx.array] = dict(mx.load(str(ckpt_path)))
    except Exception:
        print("  Warning: corrupt checkpoints.npz — starting fresh", file=sys.stderr)
        return 0

    num_layers = config.num_hidden_layers

    # Validate that the first window's keys exist before proceeding
    first_wid = raw_windows[0]["window_id"]
    if f"w{first_wid}_l0_k" not in raw_ckpts:
        print("  Warning: incompatible checkpoint format — starting fresh", file=sys.stderr)
        return 0

    token_bytes = tokens_path.read_bytes()
    n = len(token_bytes) // 4
    all_saved_tokens = list(_struct.unpack(f"<{n}I", token_bytes[: n * 4]))

    token_offset = 0
    for w in raw_windows:
        wid = w["window_id"]
        w_tokens = all_saved_tokens[token_offset : token_offset + w["token_count"]]
        w_abs = w["abs_offset"]

        engine.archive.archive(wid, w_tokens, w_abs)

        kv_last = [
            (raw_ckpts[f"w{wid}_l{li}_k"], raw_ckpts[f"w{wid}_l{li}_v"])
            for li in range(num_layers)
        ]
        abs_last = w_abs + w["token_count"] - 1
        engine.checkpoints.save(wid, kv_last, abs_last)
        token_offset += w["token_count"]

    engine.current_window_id = len(raw_windows)
    engine.abs_offset = token_offset
    engine.kv_store = None
    engine.hot_len = 0
    engine.current_window_tokens = []

    return token_offset


__all__ = ["context_prefill_cmd"]
