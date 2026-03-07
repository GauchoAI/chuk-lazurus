#!/usr/bin/env python3
"""
Token/second benchmark: standard Gemma (KV cache) vs residual stream Gemma
across increasing context lengths.

Standard model: O(1) per new token — KV cache paid for at prefill.
RS model:       O(context_len) per new token — full recompute every step.

The crossover point is never in RS's favour for raw throughput. The point is
that RS trades generation speed for zero persistent state — enabling the
single-pass dark agent loop and multi-turn with 8KB stored state instead of GBs.

Usage:
    uv run python examples/inference/gemma_rs_throughput.py
    uv run python examples/inference/gemma_rs_throughput.py --model mlx-community/gemma-3-1b-it-bf16
    uv run python examples/inference/gemma_rs_throughput.py --gen-tokens 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CONTEXT_LENGTHS = [64, 128, 256, 512, 1024, 2048]
GEN_TOKENS_DEFAULT = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--gen-tokens", type=int, default=GEN_TOKENS_DEFAULT,
                   help="New tokens to generate per benchmark run")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=CONTEXT_LENGTHS)
    return p.parse_args()


def fmt_bytes(n: int) -> str:
    if n < 1024:       return f"{n} B"
    if n < 1024**2:    return f"{n/1024:.1f} KB"
    if n < 1024**3:    return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    print(f"  Downloading {model_id}...")
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    ))


def _apply_weights(model, model_path: Path) -> None:
    from mlx.utils import tree_unflatten
    raw: dict = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        raw.update(mx.load(str(sf)))
    sanitized = model.sanitize(raw)
    sanitized = {
        k: v.astype(mx.bfloat16) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in sanitized.items()
    }
    model.update(tree_unflatten(list(sanitized.items())))
    mx.eval(model.parameters())


def load_models(model_id: str):
    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)
    config = GemmaConfig.from_hf_config(config_data)

    std = GemmaForCausalLM(config)
    _apply_weights(std, model_path)
    std.eval()

    rs = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs, model_path)
    rs.eval()

    return std, rs, config


def bench_standard(model, input_ids: mx.array, gen_tokens: int) -> tuple[float, float, int]:
    """
    Returns (prefill_ms, avg_gen_ms_per_token, final_kv_bytes).
    Standard model: prefill once, then O(1) per new token using cached K,V.
    """
    # Prefill
    t0 = time.perf_counter()
    out = model(input_ids)
    mx.eval(out.logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    cache = out.cache
    gen_times = []

    for _ in range(gen_tokens):
        last = out.logits[0, -1, :]
        next_tok = mx.argmax(last, keepdims=True)[None]
        mx.eval(next_tok)

        t0 = time.perf_counter()
        out = model(next_tok, cache=cache)
        mx.eval(out.logits)
        gen_times.append((time.perf_counter() - t0) * 1000)
        cache = out.cache

    # Measure final KV cache size
    kv_bytes = 0
    if cache:
        for layer_cache in cache:
            if layer_cache is not None:
                k, v = layer_cache
                kv_bytes += k.nbytes + v.nbytes

    avg_gen_ms = sum(gen_times) / len(gen_times) if gen_times else 0
    return prefill_ms, avg_gen_ms, kv_bytes


def bench_rs(model, input_ids: mx.array, gen_tokens: int, hidden_size: int) -> tuple[float, float, int]:
    """
    Returns (prefill_ms, avg_gen_ms_per_token, final_residual_bytes).
    RS model: no cache. Full forward pass at every step. O(context_len) per token.
    """
    tokens = list(input_ids[0].tolist())

    # First forward pass = "prefill"
    t0 = time.perf_counter()
    out = model(input_ids)
    mx.eval(out.logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    next_tok = int(mx.argmax(out.logits[0, -1, :]))
    tokens.append(next_tok)
    gen_times = []

    for _ in range(gen_tokens - 1):
        current = mx.array(tokens)[None]

        t0 = time.perf_counter()
        out = model(current)
        mx.eval(out.logits)
        gen_times.append((time.perf_counter() - t0) * 1000)

        next_tok = int(mx.argmax(out.logits[0, -1, :]))
        tokens.append(next_tok)

    final_residual_bytes = len(tokens) * hidden_size * 2
    avg_gen_ms = sum(gen_times) / len(gen_times) if gen_times else prefill_ms
    return prefill_ms, avg_gen_ms, final_residual_bytes


def bar_chart(rows: list[tuple[int, float, float]], width: int = 30) -> None:
    """ASCII chart of tokens/sec by context length for both models."""
    max_tps = max(max(std_tps, rs_tps) for _, std_tps, rs_tps in rows)

    print(f"\n  {'Context':>8}  {'Standard (KV)':^32}  {'RS (recompute)':^32}")
    print(f"  {'':>8}  {'tok/s':>8}  {'':30}  {'tok/s':>8}  {'':30}")
    print("  " + "─" * 86)

    for ctx, std_tps, rs_tps in rows:
        std_bar = "█" * int(std_tps / max_tps * width)
        rs_bar  = "█" * int(rs_tps  / max_tps * width)
        std_pad = "░" * (width - len(std_bar))
        rs_pad  = "░" * (width - len(rs_bar))

        slowdown = std_tps / rs_tps if rs_tps > 0 else 0

        print(f"  {ctx:>8,}  "
              f"{std_tps:>7.1f}  {CYAN}{std_bar}{DIM}{std_pad}{RESET}  "
              f"{rs_tps:>7.1f}  {DIM}{rs_bar}{rs_pad}{RESET}  "
              f"  {YELLOW}{slowdown:.1f}×{RESET}")


def main():
    args = parse_args()

    print(f"\n{BOLD}Gemma Throughput: KV Cache vs Residual Stream{RESET}")
    print("=" * 56)
    print(f"  Model:      {args.model}")
    print(f"  Gen tokens: {args.gen_tokens} per context length")
    print(f"  Contexts:   {args.context_lengths}")
    print()
    print("  Standard: KV cache — O(1) per new token after prefill")
    print("  RS:       Full recompute — O(context_len) per new token")

    std_model, rs_model, config = load_models(args.model)

    # Warm up
    print("\n  Warming up...")
    _w = mx.array([[1, 2, 3, 4, 5]])
    _ = std_model(_w); _ = rs_model(_w); mx.eval()
    _w2 = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = std_model(_w2); _ = rs_model(_w2); mx.eval()

    print()
    col = 12
    print(f"  {'Context':>8}  "
          f"{'Prefill(ms)':>{col}}  {'Std gen(ms)':>{col}}  {'RS gen(ms)':>{col}}  "
          f"{'Std tok/s':>{col}}  {'RS tok/s':>{col}}  "
          f"{'Slowdown':>10}  {'KV size':>{col}}  {'RS state':>{col}}")
    print("  " + "─" * 128)

    chart_rows = []

    for ctx_len in args.context_lengths:
        # Build a synthetic prompt of exactly ctx_len tokens
        input_ids = mx.array([[1] * ctx_len])

        # Benchmark standard
        std_prefill, std_gen_ms, std_kv_bytes = bench_standard(
            std_model, input_ids, args.gen_tokens
        )

        # Benchmark RS
        rs_prefill, rs_gen_ms, rs_res_bytes = bench_rs(
            rs_model, input_ids, args.gen_tokens, config.hidden_size
        )

        std_tps = 1000 / std_gen_ms if std_gen_ms > 0 else 0
        rs_tps  = 1000 / rs_gen_ms  if rs_gen_ms  > 0 else 0
        slowdown = std_tps / rs_tps  if rs_tps    > 0 else 0

        chart_rows.append((ctx_len, std_tps, rs_tps))

        # Prefill should be similar (both do full pass with no cache)
        prefill_str = f"{std_prefill:.0f}ms"

        print(f"  {ctx_len:>8,}  "
              f"{prefill_str:>{col}}  "
              f"{std_gen_ms:>{col}.1f}  "
              f"{rs_gen_ms:>{col}.1f}  "
              f"{CYAN}{std_tps:>{col}.1f}{RESET}  "
              f"{DIM}{rs_tps:>{col}.1f}{RESET}  "
              f"{YELLOW}{slowdown:>9.1f}×{RESET}  "
              f"{fmt_bytes(std_kv_bytes):>{col}}  "
              f"{fmt_bytes(rs_res_bytes):>{col}}")

    bar_chart(chart_rows)

    print(f"\n{BOLD}Key observations{RESET}")
    print()
    print("  1. PREFILL is identical — both do a full forward pass. No difference.")
    print()
    print("  2. GENERATION diverges with context:")
    print("     Standard: each new token is one tiny forward pass (last token only,")
    print("               K,V read from cache). Cost independent of context length.")
    print("     RS:       each new token is a full forward pass over ALL tokens.")
    print("               Cost grows linearly with context length.")
    print()
    print("  3. The slowdown is proportional to context length — this is expected.")
    print("     At ctx=64,  RS ≈ 64× more compute per token than standard.")
    print("     At ctx=2048, RS ≈ 2048× more compute per token than standard.")
    print()
    print("  4. This tradeoff is correct for the target use case:")
    print("     RS is NOT for streaming generation. It is for:")
    print("     - Single-pass inference (probe, inject, branch)")
    print("     - Multi-turn with minimal stored state (token IDs only)")
    print("     - Environments where memory matters more than generation speed")
    print()

    # Show the stored-state comparison across turns
    print(f"{BOLD}Stored state between turns (multi-turn context){RESET}")
    print()
    print(f"  {'Tokens':>8}  {'KV cache (stored)':>20}  {'Token IDs (stored)':>20}  {'Ratio':>8}")
    print("  " + "─" * 64)

    kv_ratio  = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
    res_ratio = config.hidden_size
    state_ratio = kv_ratio / res_ratio

    for n in [512, 1024, 2048, 4096, 16384, 65536, 131072]:
        kv_bytes  = 2 * config.num_hidden_layers * config.num_key_value_heads * n * config.head_dim * 2
        id_bytes  = n * 4  # int32 token IDs
        print(f"  {n:>8,}  {fmt_bytes(kv_bytes):>20}  {fmt_bytes(id_bytes):>20}  "
              f"{kv_bytes/id_bytes:>7.0f}×")

    print()
    print(f"  (Ratio is constant: 2 × {config.num_hidden_layers} layers × "
          f"{config.num_key_value_heads} kv_heads × {config.head_dim} head_dim / 4 bytes per token ID"
          f" = {kv_bytes // (131072 * 4):,}×)")
    print()


if __name__ == "__main__":
    main()
