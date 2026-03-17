"""Compass routing calibration — commitment-layer PCA basis extraction."""

from __future__ import annotations

import time
from pathlib import Path

from ._npz import savez_chunked
from ._progress import phase_progress


def calibrate_compass(
    engine,
    output_path: Path,
    num_archived: int,
    config,
    n_samples: int | None = 8,
    compass_layer: int | None = None,
) -> None:
    """Extract commitment-layer residuals and compute PCA basis for compass routing.

    Auto-calibrates:
      - Commitment layer at ~75% model depth (or explicit layer if provided)
      - Structural/content boundary from the explained variance knee
      - Content subspace width (16 PCs after structural boundary)

    Parameters
    ----------
    n_samples : Positions to sample per window. None = all positions (full mode).
    compass_layer : Explicit layer index for residual extraction. None = auto
                    (~77% depth). Use 29 for novel-fact routing geometry.

    Saves:
      - compass_residuals.npz — per-window residuals at the commitment layer
      - compass_basis.npz — PCA mean, basis vectors, layer and PC range metadata
    """
    import sys

    import mlx.core as mx
    import numpy as np

    num_layers = config.num_hidden_layers
    if compass_layer is None:
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
            _flat = vec.reshape(-1)
            mx.eval(_flat)
            all_vecs.append(np.array(memoryview(_flat.astype(mx.float32)), copy=False))
        phase_progress(phase_label, wid + 1, num_archived, t_compass)
    print(file=sys.stderr)

    savez_chunked(str(output_path / "compass_residuals.npz"), compass_dict)

    # PCA on all compass residuals
    X = np.stack(all_vecs, axis=0)  # (num_windows * n_samples, hidden_dim)
    mean = X.mean(axis=0)
    X_centered = X - mean

    from sklearn.utils.extmath import randomized_svd
    # Max PCs needed: structural_end (up to 50) + 16 content PCs + 16 buffer
    n_components = min(82, min(X_centered.shape) - 1)
    _U, S_vals, Vt = randomized_svd(X_centered, n_components=n_components, random_state=42)
    explained = (S_vals ** 2) / np.sum(S_vals ** 2)

    # Auto-detect structural/content boundary:
    # Find where the spectrum flattens — 3 consecutive PCs with ratio < 1.5.
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
    structural_basis = Vt[:pc_start]  # (pc_start, hidden_dim) — PCs to remove
    mx.savez(
        str(output_path / "compass_basis.npz"),
        mean=mx.array(mean),
        basis=mx.array(basis),
        structural_basis=mx.array(structural_basis),
        compass_layer=mx.array(compass_layer),
        pc_start=mx.array(pc_start),
        pc_end=mx.array(pc_end),
    )
