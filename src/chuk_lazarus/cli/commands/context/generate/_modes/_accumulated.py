"""Accumulated mode — inject checkpoint KVs from evenly-spaced windows."""

from __future__ import annotations

import sys


def run_accumulated(lib, kv_gen, mx):
    """Inject accumulated checkpoint KVs from evenly-spaced windows.

    Each checkpoint is the Markov state at that window's boundary —
    the accumulated understanding after all tokens up to that point.
    Concatenating N checkpoints gives N positions spanning the full
    document for the model to attend to.

    Returns (context_kv, seq_len).
    """
    n_positions = min(72, lib.num_windows)
    step = max(1, lib.num_windows // n_positions)
    sample_wids = list(range(0, lib.num_windows, step))
    # Always include the last window
    if sample_wids[-1] != lib.num_windows - 1:
        sample_wids.append(lib.num_windows - 1)

    # Concatenate checkpoint KVs: per-layer K,V along seq dimension
    first_kv = lib.get_checkpoint(sample_wids[0])
    num_layers = len(first_kv)
    concat_kv = [
        (
            mx.concatenate([lib.get_checkpoint(wid)[li][0] for wid in sample_wids], axis=2),
            mx.concatenate([lib.get_checkpoint(wid)[li][1] for wid in sample_wids], axis=2),
        )
        for li in range(num_layers)
    ]
    context_kv = concat_kv
    seq_len = len(sample_wids)
    mx.eval(*[t for pair in context_kv for t in pair])
    print(
        f"  Injected {len(sample_wids)} accumulated states "
        f"(every {step} windows, {lib.total_tokens} tokens compressed to "
        f"{seq_len} positions)",
        file=sys.stderr,
    )

    return context_kv, seq_len
