"""Restore engine state from a partial library on disk (resume support)."""

from __future__ import annotations

import sys
from pathlib import Path


def restore_engine(engine, output_path: Path, config) -> int:
    """Reload archived windows from a partial library into a fresh engine.

    Returns the number of tokens already processed (= tokens to skip on resume).
    """
    import json as _json
    import struct as _struct

    import mlx.core as mx

    from .....inference.context import LibraryFile

    windows_path = output_path / LibraryFile.WINDOWS
    ckpt_path = output_path / LibraryFile.CHECKPOINTS
    tokens_path = output_path / LibraryFile.TOKENS

    if not (windows_path.exists() and ckpt_path.exists() and tokens_path.exists()):
        return 0

    raw_windows: list[dict] = _json.loads(windows_path.read_text())
    if not raw_windows:
        return 0

    try:
        raw_ckpts = mx.load(str(ckpt_path))  # lazy dict — arrays loaded on access
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

    # Load residuals if available (lazy — materialized per-window below)
    residuals_path = output_path / LibraryFile.RESIDUALS
    raw_residuals = {}
    if residuals_path.exists():
        try:
            raw_residuals = mx.load(str(residuals_path))
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
