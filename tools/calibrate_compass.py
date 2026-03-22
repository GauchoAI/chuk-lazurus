"""Calibrate compass routing for an existing checkpoint library.

Extracts commitment-layer residuals and computes PCA basis.
Run once after prefill to enable --strategy compass.

Usage:
    python calibrate_compass.py --model google/gemma-3-4b-it \
        --library ./apollo11_ctx_4k --samples 8
"""

import argparse
import sys
import time

import mlx.core as mx
import numpy as np


def _savez_chunked(path: str, arrays: dict[str, mx.array], chunk_size: int = 512) -> None:
    """Save arrays to npz, chunking to stay under mx.savez's 1024 kwarg limit."""
    import tempfile
    import zipfile

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    from chuk_lazarus.inference import UnifiedPipeline
    from chuk_lazarus.inference.context import CheckpointLibrary
    from chuk_lazarus.inference.context.kv_generator import make_kv_generator

    print(f"Loading library: {args.library}", file=sys.stderr)
    lib = CheckpointLibrary(args.library)
    print(
        f"  {lib.name}  |  {lib.num_windows} windows  |  {lib.total_tokens:,} tokens",
        file=sys.stderr,
    )

    print(f"Loading model: {args.model}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(args.model, verbose=False)
    kv_gen = make_kv_generator(pipeline.model)

    num_layers = pipeline.config.num_hidden_layers
    compass_layer = round(num_layers * 0.77)
    n_samples = args.samples

    print(
        f"Calibrating compass: layer {compass_layer}/{num_layers}, "
        f"{n_samples} samples/window",
        file=sys.stderr,
    )

    # Extract commitment-layer residuals
    compass_dict: dict[str, mx.array] = {}
    all_vecs: list[np.ndarray] = []

    t0 = time.time()
    for wid in range(lib.num_windows):
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]
        S = len(w_tokens)
        positions = [int(i * (S - 1) / max(n_samples - 1, 1)) for i in range(n_samples)]

        h = kv_gen.prefill_to_layer(
            w_ids, target_layer=compass_layer, sample_positions=positions,
        )
        for si in range(n_samples):
            vec = h[0, si : si + 1, :]
            compass_dict[f"w{wid}_s{si}"] = vec
            all_vecs.append(np.array(vec.reshape(-1).tolist(), dtype=np.float32))

        elapsed = time.time() - t0
        rate = (wid + 1) / elapsed if elapsed > 0 else 0
        eta = (lib.num_windows - wid - 1) / rate if rate > 0 else 0
        print(
            f"\r  Window {wid+1}/{lib.num_windows}  {rate:.1f} w/s  ETA {eta:.0f}s",
            end="", file=sys.stderr, flush=True,
        )

    print(file=sys.stderr)
    _savez_chunked(f"{args.library}/compass_residuals.npz", compass_dict)
    print(
        f"Saved {len(compass_dict)} compass residuals to compass_residuals.npz",
        file=sys.stderr,
    )

    # PCA
    X = np.stack(all_vecs, axis=0)
    mean = X.mean(axis=0)
    X_centered = X - mean

    print("Computing PCA (SVD)...", file=sys.stderr)
    _U, S_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained = (S_vals ** 2) / np.sum(S_vals ** 2)

    # Print explained variance
    cum_var = np.cumsum(explained)
    for i in range(min(30, len(explained))):
        print(
            f"  PC{i:>2}: {explained[i]*100:5.2f}%  (cum: {cum_var[i]*100:5.1f}%)",
            file=sys.stderr,
        )

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

    pc_start = structural_end
    pc_end = min(structural_end + 16, len(explained))

    structural_var = sum(explained[:pc_start]) * 100
    content_var = sum(explained[pc_start:pc_end]) * 100
    print(
        f"\nCalibration result:\n"
        f"  Commitment layer: {compass_layer}\n"
        f"  Structural PCs: 0-{pc_start-1} ({structural_var:.1f}% variance)\n"
        f"  Content PCs: {pc_start}-{pc_end-1} ({content_var:.1f}% variance)\n"
        f"  Subspace dims: {pc_end - pc_start}",
        file=sys.stderr,
    )

    # Save basis
    basis = Vt[pc_start:pc_end]
    mx.savez(
        f"{args.library}/compass_basis.npz",
        mean=mx.array(mean),
        basis=mx.array(basis),
        compass_layer=mx.array(compass_layer),
        pc_start=mx.array(pc_start),
        pc_end=mx.array(pc_end),
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
