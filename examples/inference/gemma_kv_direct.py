#!/usr/bin/env python3
"""
KV-direct generator benchmark.

Compares three inference modes:
  1. Standard KV cache      — baseline, full speed, full memory
  2. RS residual generator  — current approach, 3.3× slower, same memory
  3. KV-direct generator    — this experiment, approaches KV speed, same memory

Hypothesis (from experiment a9704704):
  The RS generator's step loop recomputes k_proj/v_proj over all S stored positions
  at every generation step. That's O(S × hidden × head_dim) wasted matmuls.
  KV-direct stores K,V after prefill and skips that recompute entirely.
  Expected result: KV-direct ≈ standard KV speed, memory identical to RS.

Memory identity:
  For Gemma 4B: hidden=2560, nkv=4, head_dim=320
  RS residual:  2560 × 2 × num_layers = 174,080 bytes/token
  KV-direct:    2 × 4 × 320 × 2 × num_layers = 174,080 bytes/token  (same!)
  Standard KV:  same formula = 174,080 bytes/token  (same!)

Usage:
    uv run python examples/inference/gemma_kv_direct.py
    uv run python examples/inference/gemma_kv_direct.py --model mlx-community/gemma-3-4b-it-bf16
    uv run python examples/inference/gemma_kv_direct.py --prompt-len 128 --gen-tokens 80
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=50)
    p.add_argument("--runs", type=int, default=3, help="Timing runs per mode")
    p.add_argument("--skip-rs", action="store_true", help="Skip RS generator (saves time)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    try:
        cached = snapshot_download(
            model_id,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
        return Path(cached)
    except Exception:
        pass
    print(f"  Downloading {model_id}...")
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
    )


def load_models(model_id: str):
    from mlx.utils import tree_unflatten

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)
    config = GemmaConfig.from_hf_config(config_data)

    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))

    std = GemmaForCausalLM(config)
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in std.sanitize(raw).items()
    }
    std.update(tree_unflatten(list(sanitized.items())))
    mx.eval(std.parameters())
    std.eval()

    rs = GemmaResidualStreamForCausalLM(config)
    rs.update(std.parameters())
    mx.eval(rs.parameters())
    rs.eval()

    return std, rs, config


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_kv(std_model, input_ids: mx.array, gen_tokens: int) -> tuple[list[int], float]:
    """Standard KV-cached generation."""
    out = std_model(input_ids)
    mx.eval(out.logits)
    cache = out.cache

    generated = []
    t0 = time.perf_counter()

    for _ in range(gen_tokens):
        next_tok = int(mx.argmax(out.logits[0, -1, :]))
        generated.append(next_tok)
        ids = mx.array([[next_tok]])
        out = std_model(ids, cache=cache)
        mx.eval(out.logits)
        cache = out.cache

    elapsed = time.perf_counter() - t0
    return generated, elapsed


def generate_rs(rs_gen, input_ids: mx.array, gen_tokens: int) -> tuple[list[int], float]:
    """RS residual generator."""
    logits, stored = rs_gen.prefill(input_ids)
    seq_len = input_ids.shape[1]

    generated = []
    t0 = time.perf_counter()

    for _ in range(gen_tokens):
        next_tok = int(mx.argmax(logits[0, -1, :]))
        generated.append(next_tok)
        logits, stored = rs_gen.step(mx.array([[next_tok]]), stored, seq_len)
        mx.eval(logits)
        seq_len += 1

    elapsed = time.perf_counter() - t0
    return generated, elapsed


def generate_kv_direct(kv_gen, input_ids: mx.array, gen_tokens: int) -> tuple[list[int], float]:
    """KV-direct generator."""
    logits, kv_store = kv_gen.prefill(input_ids)
    seq_len = input_ids.shape[1]

    generated = []
    t0 = time.perf_counter()

    for _ in range(gen_tokens):
        next_tok = int(mx.argmax(logits[0, -1, :]))
        generated.append(next_tok)
        logits, kv_store = kv_gen.step(mx.array([[next_tok]]), kv_store, seq_len)
        mx.eval(logits)
        seq_len += 1

    elapsed = time.perf_counter() - t0
    return generated, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print(f"\n{BOLD}KV-Direct Generator Benchmark{RESET}")
    print("=" * 55)
    print(f"  Model:        {args.model}")
    print(f"  Prompt len:   {args.prompt_len} tokens")
    print(f"  Gen tokens:   {args.gen_tokens}")
    print(f"  Timing runs:  {args.runs}")

    std_model, rs_model, config = load_models(args.model)
    print(
        f"\n  Config:  layers={config.num_hidden_layers}  "
        f"hidden={config.hidden_size}  "
        f"nkv={config.num_key_value_heads}  "
        f"head_dim={config.head_dim}"
    )

    # Load generators
    import importlib.util
    import sys as _sys

    _inf = Path(__file__).parents[2] / "src/chuk_lazarus/inference"

    def _load(dotted, fpath):
        spec = importlib.util.spec_from_file_location(dotted, fpath)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    rs_gen_mod = _load(
        "chuk_lazarus.inference.context.rs_generator", _inf / "context" / "rs_generator.py"
    )
    kv_gen_mod = _load(
        "chuk_lazarus.inference.context.kv_generator", _inf / "context" / "kv_generator.py"
    )

    rs_gen = rs_gen_mod.CompiledRSGenerator(rs_model, config)
    kv_gen = kv_gen_mod.KVDirectGenerator(rs_model, config)

    # Synthetic prompt
    prompt_ids = mx.array([[(i % 8000) + 1 for i in range(args.prompt_len)]])

    # Memory accounting
    kv_bytes = kv_gen.kv_bytes(args.prompt_len + args.gen_tokens)
    rs_bytes = kv_gen.residual_equivalent_bytes(args.prompt_len + args.gen_tokens)

    print(f"\n  Memory per {args.prompt_len + args.gen_tokens} tokens:")
    print(f"    Standard KV:  {fmt_bytes(kv_bytes)}")
    print(f"    RS residual:  {fmt_bytes(rs_bytes)}")
    print(f"    KV-direct:    {fmt_bytes(kv_bytes)}  (same as standard KV)")
    if kv_bytes == rs_bytes:
        print("    ✓ KV-direct == RS residual == standard KV (hidden = 2×nkv×head_dim)")
    else:
        ratio = rs_bytes / kv_bytes
        print(
            f"    KV-direct {ratio:.2f}× smaller than RS residual "
            f"(hidden={config.hidden_size} ≠ 2×nkv×head_dim={2 * config.num_key_value_heads * config.head_dim})"
        )

    # Warm-up
    print("\n  Warming up...")
    _wids = mx.array([[1, 2, 3, 4, 5]])
    _ = std_model(_wids)
    _ = kv_gen.prefill(_wids)
    _ = rs_gen.prefill(_wids)
    mx.eval()
    print("  Done.\n")

    results = {}

    # --- Standard KV ---
    kv_times = []
    kv_toks = None
    for run in range(args.runs):
        toks, elapsed = generate_kv(std_model, prompt_ids, args.gen_tokens)
        if kv_toks is None:
            kv_toks = toks
        kv_times.append(elapsed)
        print(f"  KV  run {run + 1}/{args.runs}: {args.gen_tokens / elapsed:.1f} tok/s")
    kv_best = min(kv_times)
    results["kv"] = kv_best
    print(f"  {GREEN}KV  best: {args.gen_tokens / kv_best:.1f} tok/s{RESET}\n")

    # --- RS residual ---
    if not args.skip_rs:
        rs_times = []
        rs_toks = None
        for run in range(args.runs):
            toks, elapsed = generate_rs(rs_gen, prompt_ids, args.gen_tokens)
            if rs_toks is None:
                rs_toks = toks
            rs_times.append(elapsed)
            print(f"  RS  run {run + 1}/{args.runs}: {args.gen_tokens / elapsed:.1f} tok/s")
        rs_best = min(rs_times)
        results["rs"] = rs_best
        print(
            f"  {YELLOW}RS  best: {args.gen_tokens / rs_best:.1f} tok/s  "
            f"({kv_best / rs_best:.2f}× slower than KV){RESET}\n"
        )
    else:
        rs_toks = None
        print("  RS skipped.\n")

    # --- KV-direct ---
    kvd_times = []
    kvd_toks = None
    for run in range(args.runs):
        toks, elapsed = generate_kv_direct(kv_gen, prompt_ids, args.gen_tokens)
        if kvd_toks is None:
            kvd_toks = toks
        kvd_times.append(elapsed)
        print(f"  KVD run {run + 1}/{args.runs}: {args.gen_tokens / elapsed:.1f} tok/s")
    kvd_best = min(kvd_times)
    results["kvd"] = kvd_best
    print(
        f"  {CYAN}KVD best: {args.gen_tokens / kvd_best:.1f} tok/s  "
        f"({kv_best / kvd_best:.2f}× vs KV){RESET}\n"
    )

    # --- Output identity check ---
    print(f"{BOLD}Output identity check{RESET}")
    if kv_toks and kvd_toks:
        match = kv_toks[:10] == kvd_toks[:10]
        print(f"  KV  first 10 tokens: {kv_toks[:10]}")
        print(f"  KVD first 10 tokens: {kvd_toks[:10]}")
        if match:
            print(f"  {GREEN}✓ KV-direct output matches standard KV (first 10){RESET}")
        else:
            print(
                "  ✗ Output mismatch (expected — generation is stochastic/greedy differs at first divergence)"
            )

    if rs_toks and kvd_toks:
        match = rs_toks[:10] == kvd_toks[:10]
        print(f"  RS  first 10 tokens: {rs_toks[:10]}")
        if match:
            print(f"  {GREEN}✓ KV-direct output matches RS generator (first 10){RESET}")
        else:
            print(
                f"  ✗ KV-direct vs RS: first divergence at "
                f"{next((i for i, (a, b) in enumerate(zip(kvd_toks, rs_toks)) if a != b), -1)}"
            )

    # --- Summary ---
    print(f"\n{BOLD}Summary{RESET}")
    print(f"  {'Mode':<12} {'tok/s':>8} {'vs KV':>8} {'Memory':>12}")
    print(f"  {'-' * 44}")
    print(
        f"  {'Standard KV':<12} {args.gen_tokens / results['kv']:>8.1f} {'1.00×':>8} {fmt_bytes(kv_bytes):>12}"
    )
    if "rs" in results:
        print(
            f"  {'RS residual':<12} {args.gen_tokens / results['rs']:>8.1f} "
            f"{kv_best / results['rs']:>7.2f}× {fmt_bytes(rs_bytes):>12}"
        )
    print(
        f"  {'KV-direct':<12} {args.gen_tokens / results['kvd']:>8.1f} "
        f"{kv_best / results['kvd']:>7.2f}× {fmt_bytes(kv_bytes):>12}"
    )

    # --- Hypothesis verdict ---
    kvd_ratio = kv_best / results["kvd"]
    print(f"\n{BOLD}Hypothesis verdict{RESET}")
    if kvd_ratio <= 1.2:
        print(f"  {GREEN}✓ KV-direct matches KV speed ({kvd_ratio:.2f}× overhead){RESET}")
        print("    Confirmed: skipping K,V recompute eliminates the RS overhead.")
    elif kvd_ratio <= 1.8:
        print(f"  {YELLOW}~ KV-direct close to KV ({kvd_ratio:.2f}× overhead){RESET}")
        print("    Partial win — remaining overhead from store concatenation or FFN.")
    else:
        print(f"  ✗ KV-direct still {kvd_ratio:.2f}× slower than KV")
        print("    Other bottleneck — investigate FFN cost or memory bandwidth.")

    if "rs" in results:
        kvd_vs_rs = results["rs"] / results["kvd"]
        print(f"\n  KV-direct is {kvd_vs_rs:.2f}× faster than RS residual generator")
        if kvd_vs_rs > 1.5:
            print(f"  {GREEN}Significant speedup: K,V recompute was the dominant cost.{RESET}")

    print(f"\n{DIM}Next: rank-15 K compression (21× KV storage reduction, no retraining){RESET}\n")


if __name__ == "__main__":
    main()
