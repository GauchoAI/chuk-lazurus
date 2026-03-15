"""Progress display helpers for prefill phases."""

from __future__ import annotations

import sys
import time


def progress_line(
    tokens_done: int, total: int, windows_done: int, total_windows: int, elapsed: float,
) -> str:
    """Return an in-place progress line (fits 80 columns)."""
    pct = tokens_done / total if total else 0.0
    rate = tokens_done / elapsed if elapsed > 0 else 0.0
    eta = (total - tokens_done) / rate if rate > 0 else 0.0
    return (
        f"\r  {windows_done}/{total_windows} windows  "
        f"{pct:>4.0%}  {rate:.0f} tok/s  ETA {eta:.0f}s\033[K"
    )


def phase_progress(phase: str, done: int, total: int, t0: float) -> None:
    """Print in-place progress for a post-prefill phase."""
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(
        f"\r  {phase}: {done}/{total} windows  "
        f"{rate:.1f} w/s  ETA {eta:.0f}s\033[K",
        end="", file=sys.stderr, flush=True,
    )
