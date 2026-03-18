"""Temporal stride routing — evenly spaced windows for global/timeline queries."""

from __future__ import annotations


def _temporal_stride_windows(
    lib,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Select evenly spaced windows across the document.

    Returns list of (window_id, score) where score is 1.0 - (rank / k)
    so the ordering is preserved but scores are uniform.
    """
    n = lib.num_windows
    if n == 0:
        return []

    k = min(k, n)
    stride = max(1, n // k)
    selected = [min(i * stride, n - 1) for i in range(k)]

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for wid in selected:
        if wid not in seen:
            seen.add(wid)
            unique.append(wid)

    return [(wid, 1.0 - i / max(len(unique), 1)) for i, wid in enumerate(unique)]
