"""Pre-RoPE page extraction for instant KV injection at generate time."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

from ._progress import phase_progress


def extract_pages(
    engine,
    output_path: Path,
    num_archived: int,
    n_pages: int = 8,
) -> None:
    """Extract pre-RoPE K,V pages for each window.

    Full forward pass per window, store pre-RoPE K,V at n_pages sampled
    positions.  ~1 GB on disk for 725 windows × 8 pages × 34 layers.
    """
    import time

    import mlx.core as mx

    pages_path = output_path / "pages.npz"
    t_pages = time.monotonic()
    print(f"  extracting pre-RoPE pages ({n_pages} per window)...", file=sys.stderr, flush=True)

    with zipfile.ZipFile(str(pages_path), "w", zipfile.ZIP_STORED) as zf:
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            w_ids = mx.array(w_tokens)[None]
            _logits, _kv, pages = engine.kv_gen.prefill_pages(w_ids, n_pages=n_pages)

            # Save per-window: w{wid}_p{page}_l{layer}_k and _v
            w_arrays: dict[str, mx.array] = {}
            for pi, page in enumerate(pages):
                for li, (k_pre, v) in enumerate(page):
                    w_arrays[f"w{wid}_p{pi}_l{li}_k"] = k_pre
                    w_arrays[f"w{wid}_p{pi}_l{li}_v"] = v

            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                mx.savez(tmp.name, **w_arrays)
                with zipfile.ZipFile(tmp.name, "r") as src:
                    for name in src.namelist():
                        zf.writestr(name, src.read(name))

            phase_progress("pages", wid + 1, num_archived, t_pages)
    print(file=sys.stderr)

    pages_size = pages_path.stat().st_size / (1024 * 1024)
    print(
        f"  pages: {num_archived} windows × {n_pages} pages = "
        f"{num_archived * n_pages} entries ({pages_size:.0f} MB)",
        file=sys.stderr, flush=True,
    )
