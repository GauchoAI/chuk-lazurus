"""Darkspace extraction — whitened frame bank projections for compass routing."""

from __future__ import annotations

import sys
import tempfile
import time
import zipfile
from pathlib import Path

from ._progress import phase_progress


def extract_darkspace(
    engine,
    output_path: Path,
    num_archived: int,
    config,
    frame_bank_data: dict | None,
    frame_bank_path: Path | None,
) -> None:
    """Extract darkspace projections for compass routing.

    Two paths:
      A) frame_bank_data provided — project through pre-computed bank
      B) no frame bank — calibrate from the corpus itself (whitening)

    Writes compass_residuals.npz (per-window, streamed to disk) and
    compass_basis.npz.  Optionally copies the external frame bank.
    """
    import mlx.core as mx
    import numpy as np

    num_layers = config.num_hidden_layers
    compass_layer = round(num_layers * 0.77)
    _N_DARKSPACE_DIMS = 64  # dimensions in the whitened frame bank

    if frame_bank_data is not None:
        # Path A: pre-computed frame bank
        _fb = frame_bank_data["frame_bank"]
        mx.eval(_fb)
        frame_bank = np.array(_fb, copy=False).astype(np.float32)
        compass_layer = int(frame_bank_data["compass_layer"].item())
        print(
            f"  using pre-computed frame bank: {frame_bank.shape[0]}D",
            file=sys.stderr, flush=True,
        )
    else:
        # Path B: calibrate from corpus — sample residuals from
        # a random subset of windows, PCA, remove structural PCs, whiten.
        frame_bank, compass_layer = _calibrate_from_corpus(
            engine, num_archived, compass_layer,
            _N_DARKSPACE_DIMS,
        )

    n_frame_dims = frame_bank.shape[0]

    # Extract residuals at every position, project through frame bank.
    # Write per-window to disk to avoid accumulating all vectors in memory.
    compass_path = str(output_path / "compass_residuals.npz")
    t_ds = time.monotonic()
    with zipfile.ZipFile(compass_path, "w", zipfile.ZIP_STORED) as zf:
        for wid in range(num_archived):
            w_tokens, _ = engine.archive.retrieve(wid)
            w_ids = mx.array(w_tokens)[None]
            S = len(w_tokens)

            h = engine.kv_gen.prefill_to_layer(
                w_ids, target_layer=compass_layer,
            )
            _h0 = h[0]
            mx.eval(_h0)
            h_np = np.array(_h0, copy=False).astype(np.float32)  # (S, hidden_dim)
            projected = h_np @ frame_bank.T  # (S, n_frame_dims)

            # Build this window's vectors and flush to zip immediately
            w_dict: dict[str, mx.array] = {}
            for si in range(S):
                w_dict[f"w{wid}_s{si}"] = mx.array(projected[si:si+1])

            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                mx.savez(tmp.name, **w_dict)
                with zipfile.ZipFile(tmp.name, "r") as src:
                    for name in src.namelist():
                        zf.writestr(name, src.read(name))

            phase_progress(f"darkspace L{compass_layer}", wid + 1, num_archived, t_ds)
    print(file=sys.stderr)

    # Save frame bank as compass basis
    mx.savez(
        str(output_path / "compass_basis.npz"),
        mean=mx.zeros(frame_bank.shape[1]),
        basis=mx.array(frame_bank),
        compass_layer=mx.array(compass_layer),
        pc_start=mx.array(0),
        pc_end=mx.array(n_frame_dims),
        mode=mx.array([ord(c) for c in "darkspace"]),
    )

    # Copy external frame bank if provided
    if frame_bank_data is not None and frame_bank_path is not None:
        import shutil
        shutil.copy2(str(frame_bank_path), str(output_path / "frame_bank.npz"))

    print(
        f"  darkspace: {n_frame_dims}D projections at {num_archived} windows × all positions",
        file=sys.stderr, flush=True,
    )


def _calibrate_from_corpus(
    engine,
    num_archived: int,
    compass_layer: int,
    n_darkspace_dims: int,
) -> tuple["np.ndarray", int]:
    """Calibrate a whitened frame bank from the corpus itself.

    Returns (frame_bank, compass_layer).
    """
    import mlx.core as mx
    import numpy as np

    n_calibration_windows = min(50, num_archived)
    n_positions_per_window = 8

    cal_vecs: list[np.ndarray] = []
    t_cal = time.monotonic()
    cal_wids = [
        int(i * (num_archived - 1) / max(n_calibration_windows - 1, 1))
        for i in range(n_calibration_windows)
    ]
    for wid in cal_wids:
        w_tokens, _ = engine.archive.retrieve(wid)
        w_ids = mx.array(w_tokens)[None]
        S = len(w_tokens)
        positions = [int(j * (S - 1) / max(n_positions_per_window - 1, 1))
                     for j in range(n_positions_per_window)]
        h = engine.kv_gen.prefill_to_layer(
            w_ids, target_layer=compass_layer, sample_positions=positions,
        )
        for si in range(n_positions_per_window):
            _cv = h[0, si, :]
            mx.eval(_cv)
            cal_vecs.append(np.array(_cv, copy=False).astype(np.float32))
        phase_progress("calibrating", len(cal_vecs) // n_positions_per_window,
                        n_calibration_windows, t_cal)
    print(file=sys.stderr)

    # PCA + whitening on calibration samples
    X_cal = np.stack(cal_vecs, axis=0)
    cal_mean = X_cal.mean(axis=0)
    X_centered = X_cal - cal_mean

    from sklearn.utils.extmath import randomized_svd
    n_components = min(130, min(X_centered.shape) - 1)
    _U, S_vals, Vt = randomized_svd(X_centered, n_components=n_components, random_state=42)
    eigenvalues = (S_vals ** 2) / max(X_cal.shape[0] - 1, 1)
    explained = eigenvalues / (eigenvalues.sum() + 1e-10)

    # Auto-detect structural boundary
    structural_end = 0
    for si in range(min(len(explained) - 3, 50)):
        ratios = [
            explained[si + j] / max(explained[si + j + 1], 1e-10)
            for j in range(3)
        ]
        if all(r < 1.5 for r in ratios):
            structural_end = si
            break
    else:
        structural_end = 4

    pc_start = structural_end
    pc_end = min(pc_start + n_darkspace_dims, len(S_vals))

    # Whitening transform
    whitening_scales = 1.0 / np.sqrt(eigenvalues[pc_start:pc_end] + 1e-10)
    frame_bank = Vt[pc_start:pc_end] * whitening_scales[:, None]

    structural_var = explained[:pc_start].sum() * 100
    content_var = explained[pc_start:pc_end].sum() * 100
    cal_elapsed = time.monotonic() - t_cal
    print(
        f"  corpus-calibrated: {len(cal_vecs)} samples from {n_calibration_windows} windows "
        f"({cal_elapsed:.1f}s)",
        file=sys.stderr, flush=True,
    )
    print(
        f"  structural PCs 0-{pc_start-1} ({structural_var:.1f}%) removed, "
        f"content PCs {pc_start}-{pc_end-1} ({content_var:.1f}%) whitened",
        file=sys.stderr, flush=True,
    )

    return frame_bank, compass_layer
