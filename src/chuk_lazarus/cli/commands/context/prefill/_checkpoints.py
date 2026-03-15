"""Checkpoint and residual serialization — incremental zip append logic."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path


def save_checkpoints(
    engine,
    output_path: Path,
    num_archived: int,
    append_from: int,
    quiet: bool = True,
) -> None:
    """Write checkpoints.npz — per-window KV tensors.

    Incremental: only serialize windows [append_from, num_archived) and
    append to the existing zip file.
    """
    import mlx.core as mx

    from .....inference.context import LibraryFile

    ckpt_start = append_from if append_from > 0 else 0
    if not quiet:
        print(f"  saving checkpoints ({num_archived} windows)...", file=sys.stderr, flush=True)

    npz_path = output_path / LibraryFile.CHECKPOINTS
    zip_mode = "a" if (ckpt_start > 0 and npz_path.exists()) else "w"
    with zipfile.ZipFile(str(npz_path), zip_mode, zipfile.ZIP_STORED) as zf:
        for wid in range(ckpt_start, num_archived):
            kv_last, _ = engine.checkpoints.load(wid)
            w_keys: dict[str, mx.array] = {}
            for li, (k, v) in enumerate(kv_last):
                w_keys[f"w{wid}_l{li}_k"] = k
                w_keys[f"w{wid}_l{li}_v"] = v
            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                mx.savez(tmp.name, **w_keys)
                with zipfile.ZipFile(tmp.name, "r") as src:
                    for name in src.namelist():
                        zf.writestr(name, src.read(name))


def save_residuals(
    engine,
    output_path: Path,
    num_archived: int,
    append_from: int,
) -> None:
    """Write residuals.npz — per-window Markov state vectors.

    Incremental: only collect new residuals and append to existing zip.
    Handles the mx.savez 1024 kwarg limit via chunked writes.
    """
    import mlx.core as mx

    from .....inference.context import LibraryFile

    ckpt_start = append_from if append_from > 0 else 0
    new_residuals: dict[str, mx.array] = {}
    for wid in range(ckpt_start, num_archived):
        if wid in engine.residuals:
            new_residuals[f"w{wid}_residual"] = engine.residuals.load(wid)
    if not new_residuals:
        return

    res_path = output_path / LibraryFile.RESIDUALS
    res_zip_mode = "a" if (ckpt_start > 0 and res_path.exists()) else "w"
    if len(new_residuals) <= 1024 and res_zip_mode == "w":
        mx.savez(str(res_path), **new_residuals)
    else:
        with zipfile.ZipFile(str(res_path), res_zip_mode, zipfile.ZIP_STORED) as zf:
            items = list(new_residuals.items())
            for i in range(0, len(items), 1024):
                chunk = dict(items[i : i + 1024])
                with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                    mx.savez(tmp.name, **chunk)
                    with zipfile.ZipFile(tmp.name, "r") as src:
                        for name in src.namelist():
                            zf.writestr(name, src.read(name))
