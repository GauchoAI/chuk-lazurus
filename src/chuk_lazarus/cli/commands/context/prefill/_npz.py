"""Chunked npz save utility — works around mx.savez's 1024 kwarg limit."""

from __future__ import annotations

import tempfile
import zipfile


def savez_chunked(path: str, arrays: dict, chunk_size: int = 512) -> None:
    """Save arrays to npz, chunking to stay under mx.savez's 1024 kwarg limit."""
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
