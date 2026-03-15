"""Interval/full residual extraction for fine-grained retrieval."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .._types import ResidualMode
from ._npz import savez_chunked
from ._progress import phase_progress


def extract_interval_residuals(
    engine,
    output_path: Path,
    num_archived: int,
    residual_mode: ResidualMode,
) -> None:
    """Extract interval/full residuals.

    interval mode: 8 evenly-spaced samples per window (~40 KB/window)
    full mode: every position (~5 MB/window for 512-token windows)

    Saves interval_residuals.npz.
    """
    import mlx.core as mx

    if residual_mode == ResidualMode.FULL:
        n_samples_per_window = None
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
        phase_progress(phase_label, wid + 1, num_archived, t_ir)
    print(file=sys.stderr)
    savez_chunked(str(output_path / "interval_residuals.npz"), interval_dict)
