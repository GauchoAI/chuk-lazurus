"""Add interval residuals to an existing checkpoint library.

Usage:
    python add_interval_residuals.py --model google/gemma-3-4b-it \
        --library ./apollo11_ctx_4k --samples 8
"""

import argparse
import sys
import time

import mlx.core as mx


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

    n_samples = args.samples
    interval_dict: dict[str, mx.array] = {}

    t0 = time.time()
    for wid in range(lib.num_windows):
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]

        _logits, _kv, interval_res = kv_gen.prefill_interval_residuals(
            w_ids, n_samples=n_samples,
        )
        # interval_res shape: (1, n_samples, hidden_size)
        for si in range(n_samples):
            interval_dict[f"w{wid}_s{si}"] = interval_res[0, si : si + 1, :]

        elapsed = time.time() - t0
        rate = (wid + 1) / elapsed if elapsed > 0 else 0
        eta = (lib.num_windows - wid - 1) / rate if rate > 0 else 0
        print(
            f"\r  Window {wid + 1}/{lib.num_windows}  "
            f"{rate:.1f} windows/s  ETA {eta:.0f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )

    print(file=sys.stderr)
    out_path = f"{args.library}/interval_residuals.npz"
    print(f"Saving {len(interval_dict)} interval residuals to {out_path}", file=sys.stderr)
    mx.savez(out_path, **interval_dict)

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.1f}s  |  {lib.num_windows * n_samples} residuals  |  "
        f"{n_samples} samples/window",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
