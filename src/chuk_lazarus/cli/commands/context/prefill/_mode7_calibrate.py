"""Mode 7 probe calibration during prefill.

Calibrates the query classifier (5-class) and engagement/tension probes
so they're ready at generate time without a ~60s first-run penalty.

Usage:
  lazarus context prefill --model ... --input ... --checkpoint ... --phases mode7
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def calibrate_mode7_probes(
    engine,
    output_path: Path,
    num_archived: int,
    config,
    tokenizer,
    model_id: str,
    compass_layer: int | None = None,
) -> None:
    """Run Mode 7 probe calibration and save to checkpoint directory.

    Loads the library from disk (just written by save_library), then
    delegates to the existing load_or_calibrate machinery in _probes.py.
    Any existing probe cache is deleted first to force recalibration.
    """
    from .....inference.context import CheckpointLibrary

    # Resolve compass layer (same logic as generate)
    if compass_layer is None:
        compass_layer = int(config.num_hidden_layers * 0.77)

    # Delete any stale probe cache so we always recalibrate
    for old_cache in output_path.glob(".probe_cache_v*.npz"):
        old_cache.unlink()

    # Load the library we just wrote
    lib = CheckpointLibrary(output_path)
    if lib.num_windows == 0:
        print("  Mode 7: no windows to calibrate from, skipping", file=sys.stderr)
        return

    print(f"  Mode 7: calibrating probes ({lib.num_windows} windows)...", file=sys.stderr)
    t0 = time.time()

    from ..generate._probes import load_or_calibrate

    _probes = load_or_calibrate(
        engine.kv_gen,
        tokenizer,
        compass_layer,
        lib,
        str(output_path),
        model_id,
    )

    elapsed = time.time() - t0
    m7_ok = "yes" if _probes.m7_available else "no"
    tension_ok = "yes" if _probes.tension_available else "no"
    print(
        f"  Mode 7: probes calibrated in {elapsed:.1f}s (classifier={m7_ok}, tension={tension_ok})",
        file=sys.stderr,
    )
