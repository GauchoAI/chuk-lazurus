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

from ._types import PrefillConfig, PrefillResult, ResidualMode


def _progress(tokens_done: int, total: int, windows_done: int, total_windows: int, elapsed: float) -> str:
    """Return an in-place progress line (fits 80 columns)."""
    pct = tokens_done / total if total else 0.0
    rate = tokens_done / elapsed if elapsed > 0 else 0.0
    eta = (total - tokens_done) / rate if rate > 0 else 0.0
    # Compact: "  42/182 windows  56%  1435 tok/s  ETA 230s"
    return (
        f"\r  {windows_done}/{total_windows} windows  "
        f"{pct:>4.0%}  {rate:.0f} tok/s  ETA {eta:.0f}s\033[K"
    )


def _phase_progress(phase: str, done: int, total: int, t0: float) -> None:
    """Print in-place progress for a post-prefill phase."""
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(
        f"\r  {phase}: {done}/{total} windows  "
        f"{rate:.1f} w/s  ETA {eta:.0f}s\033[K",
        end="", file=sys.stderr, flush=True,
    )


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
        f"window_size={config.window_size}  |  ~{total_windows} windows  |  "
        f"residuals={config.residual_mode.value}",
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

    def _do_save(is_complete: bool = False, quick: bool = False) -> None:
        """Write library files from engine's current state.

        quick=True skips expensive interval residual and compass extraction
        (used for periodic checkpoint saves during prefill).
        """
        _save_library(
            engine, output_path, token_ids, lib_name,
            config.model, pipeline.config, config.window_size,
            tokenizer, created_at, is_complete=is_complete,
            quick=quick, residual_mode=config.residual_mode,
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

            # Periodic save every N new windows (quick — checkpoints only, no extraction)
            if archived > 0 and (archived - last_saved_windows) >= _SAVE_EVERY_WINDOWS:
                _do_save(quick=True)
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


def _savez_chunked(path: str, arrays: dict[str, "mx.array"], chunk_size: int = 512) -> None:
    """Save arrays to npz, chunking to stay under mx.savez's 1024 kwarg limit."""
    import tempfile
    import zipfile

    import mlx.core as mx

    keys = list(arrays.keys())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i : i + chunk_size]
            chunk = {k: arrays[k] for k in chunk_keys}
            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                mx.savez(tmp.name, **chunk)
                with zipfile.ZipFile(tmp.name, "r") as src:
                    for name in src.namelist():
                        zf.writestr(name, src.read(name))


def _calibrate_compass(
    engine,
    output_path: Path,
    num_archived: int,
    config,
    n_samples: int | None = 8,
) -> None:
    """Extract commitment-layer residuals and compute PCA basis for compass routing.

    Auto-calibrates:
      - Commitment layer at ~75% model depth
      - Structural/content boundary from the explained variance knee
      - Content subspace width (16 PCs after structural boundary)

    Parameters
    ----------
    n_samples : Positions to sample per window. None = all positions (full mode).

    Saves:
      - compass_residuals.npz — per-window residuals at the commitment layer
      - compass_basis.npz — PCA mean, basis vectors, layer and PC range metadata
    """
    import mlx.core as mx
    import numpy as np

    num_layers = config.num_hidden_layers
    compass_layer = round(num_layers * 0.77)

    full_mode = n_samples is None
    phase_label = f"compass L{compass_layer}" + (" (full)" if full_mode else "")

    # Extract commitment-layer residuals for all windows
    compass_dict: dict[str, mx.array] = {}
    all_vecs: list[np.ndarray] = []

    t_compass = time.monotonic()
    for wid in range(num_archived):
        w_tokens, _ = engine.archive.retrieve(wid)
        w_ids = mx.array(w_tokens)[None]
        S = len(w_tokens)

        # Full mode: every position. Interval mode: n_samples evenly spaced.
        w_n_samples = S if full_mode else n_samples
        if w_n_samples >= S:
            # All positions — no subsampling needed
            positions = list(range(S))
            w_n_samples = S
        else:
            positions = [int(i * (S - 1) / max(w_n_samples - 1, 1)) for i in range(w_n_samples)]

        h = engine.kv_gen.prefill_to_layer(
            w_ids, target_layer=compass_layer, sample_positions=positions,
        )
        # h shape: (1, w_n_samples, hidden_size)
        for si in range(w_n_samples):
            vec = h[0, si:si+1, :]  # (1, hidden_size)
            compass_dict[f"w{wid}_s{si}"] = vec
            all_vecs.append(np.array(vec.reshape(-1).tolist(), dtype=np.float32))
        _phase_progress(phase_label, wid + 1, num_archived, t_compass)
    print(file=sys.stderr)

    _savez_chunked(str(output_path / "compass_residuals.npz"), compass_dict)

    # PCA on all compass residuals
    X = np.stack(all_vecs, axis=0)  # (num_windows * n_samples, hidden_dim)
    mean = X.mean(axis=0)
    X_centered = X - mean

    _U, S_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained = (S_vals ** 2) / np.sum(S_vals ** 2)

    # Auto-detect structural/content boundary:
    # Find where the spectrum flattens — 3 consecutive PCs with ratio < 1.5.
    # Structural PCs have rapidly decaying variance; content PCs are near-uniform.
    structural_end = 0
    for i in range(min(len(explained) - 3, 50)):
        ratios = [
            explained[i + j] / max(explained[i + j + 1], 1e-10)
            for j in range(3)
        ]
        if all(r < 1.5 for r in ratios):
            structural_end = i
            break
    else:
        structural_end = 8  # safe default

    # Content subspace: skip structural, take next 16 PCs
    pc_start = structural_end
    pc_end = min(structural_end + 16, len(explained))

    # Report calibration
    structural_var = sum(explained[:pc_start]) * 100
    content_var = sum(explained[pc_start:pc_end]) * 100
    print(
        f"  compass calibrated: layer={compass_layer}, "
        f"structural=PC 0-{pc_start-1} ({structural_var:.1f}%), "
        f"content=PC {pc_start}-{pc_end-1} ({content_var:.1f}%)",
        file=sys.stderr, flush=True,
    )

    # Save basis: mean vector + projection matrix + metadata
    basis = Vt[pc_start:pc_end]  # (pc_end - pc_start, hidden_dim)
    mx.savez(
        str(output_path / "compass_basis.npz"),
        mean=mx.array(mean),
        basis=mx.array(basis),
        compass_layer=mx.array(compass_layer),
        pc_start=mx.array(pc_start),
        pc_end=mx.array(pc_end),
    )


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
    quick: bool = False,
    residual_mode: ResidualMode = ResidualMode.INTERVAL,
) -> None:
    """Write all library files from the engine's current archived state.

    quick=True writes only checkpoints/tokens/windows/manifest (for periodic saves).
    quick=False also extracts interval/full residuals and calibrates compass routing,
    unless residual_mode is NONE.
    """
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
    residual_dict: dict[str, mx.array] = {}
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
        # Save residual (Markov state) if available
        if wid in engine.residuals:
            residual_dict[f"w{wid}_residual"] = engine.residuals.load(wid)
        token_offset += len(w_tokens)

    total_tokens_to_report = len(all_token_ids) if is_complete else token_offset

    # 1. checkpoints.npz — save per-window to stay under mx.savez 1024 kwarg limit
    import tempfile
    import zipfile

    if not quick:
        print(f"  saving checkpoints ({num_archived} windows)...", file=sys.stderr, flush=True)
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

    # 2. residuals.npz — per-window Markov state vectors
    if residual_dict:
        mx.savez(str(output_path / LibraryFile.RESIDUALS), **residual_dict)

    if not quick and residual_mode != ResidualMode.NONE:
        # 2b. interval_residuals.npz — interior residuals for compass routing
        #     Re-prefill each window to extract residuals at sampled positions.
        #     interval mode: 8 evenly-spaced samples per window (~40 KB/window)
        #     full mode: every position (~5 MB/window for 512-token windows)
        if residual_mode == ResidualMode.FULL:
            # Full: extract at every position within each window
            n_samples_per_window = None  # determined per-window from token count
            phase_label = "full residuals"
        else:
            n_samples_per_window = 8
            phase_label = "interval residuals"

        interval_dict: dict[str, mx.array] = {}
        t_ir = time.monotonic()
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            w_ids = mx.array(w_tokens)[None]
            n_samples = len(w_tokens) if residual_mode == ResidualMode.FULL else n_samples_per_window
            _logits, _kv, interval_res = engine.kv_gen.prefill_interval_residuals(
                w_ids, n_samples=n_samples,
            )
            for si in range(n_samples):
                interval_dict[f"w{wid}_s{si}"] = interval_res[0, si:si+1, :]
            _phase_progress(phase_label, wid + 1, num_archived, t_ir)
        print(file=sys.stderr)
        _savez_chunked(str(output_path / "interval_residuals.npz"), interval_dict)

        # 2c. Compass routing data — commitment-layer residuals + PCA basis
        #     Full mode: every position per window (371K vectors for 725 × 512).
        #     Interval mode: 8 samples per window (5,800 vectors).
        compass_n_samples = None if residual_mode == ResidualMode.FULL else 8
        _calibrate_compass(
            engine, output_path, num_archived, config,
            n_samples=compass_n_samples,
        )

    # 3. tokens.bin (uint32)
    with open(output_path / LibraryFile.TOKENS, "wb") as f:
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            for tid in w_tokens:
                f.write(struct.pack("<I", tid))

    # 4. windows.json
    import json as _json
    (output_path / LibraryFile.WINDOWS).write_text(
        _json.dumps([w.model_dump() for w in windows], indent=2, ensure_ascii=False)
    )

    # 5. manifest.json (written last — the "committed" marker)
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

    # Load residuals if available
    residuals_path = output_path / LibraryFile.RESIDUALS
    raw_residuals: dict[str, mx.array] = {}
    if residuals_path.exists():
        try:
            raw_residuals = dict(mx.load(str(residuals_path)))
        except Exception:
            pass  # older library without residuals — fine

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

        # Restore residual if available
        res_key = f"w{wid}_residual"
        if res_key in raw_residuals:
            engine.residuals.save(wid, raw_residuals[res_key])

        token_offset += w["token_count"]

    engine.current_window_id = len(raw_windows)
    engine.abs_offset = token_offset
    engine.kv_store = None
    engine.hot_len = 0
    engine.current_window_tokens = []

    return token_offset


__all__ = ["context_prefill_cmd"]
