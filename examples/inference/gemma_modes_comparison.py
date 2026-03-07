#!/usr/bin/env python3
"""
Multi-mode inference comparison across a simulated multi-turn conversation.

Three modes, same conversation, same device budget:

  unbounded-kv  Standard KV cache. No budget limit. Memory grows linearly.
  bounded-kv    Bounded KV cache + residual stream backing store.
                Hot/warm/cold tiers. Window slides at budget limit.
                Full generation speed. Dark residuals via extra RS pass.
  bounded-rs    Bounded residual stream. Per-layer residuals as hot tier.
                2-2.6× slower generation. Dark residuals free (in hot state).
                1.25× more memory per token vs KV for Gemma 270M/4B.

The comparison shows:
  - Memory growth trajectory across turns (does it bound?)
  - Generation speed per turn and per mode
  - Path taken (cold / hot / warm) per turn
  - Dark residual availability

Usage:
    uv run python examples/inference/gemma_modes_comparison.py
    uv run python examples/inference/gemma_modes_comparison.py --budget-mb 8
    uv run python examples/inference/gemma_modes_comparison.py --turns 10 --tokens-per-turn 50
"""

from __future__ import annotations

import argparse
import json
import sys
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


def fmt_bytes(n: int) -> str:
    if n < 1024:      return f"{n} B"
    if n < 1024**2:   return f"{n/1024:.1f} KB"
    if n < 1024**3:   return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--budget-mb", type=float, default=4.0)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--tokens-per-turn", type=int, default=30)
    p.add_argument("--gen-tokens", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=50)
    p.add_argument("--dark-layers", nargs="+", type=int, default=[6, 9, 14])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    # Check if already cached
    try:
        cached = snapshot_download(model_id, local_files_only=True,
                                   allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])
        print(f"  Using cached {model_id}")
        return Path(cached)
    except Exception:
        pass
    print(f"  Downloading {model_id} (this may take 30+ minutes for large models)...")
    print(f"  Each .safetensors shard downloads one at a time — progress updates per file.")
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    ))


def load_models(model_id: str):
    from mlx.utils import tree_unflatten
    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)
    config = GemmaConfig.from_hf_config(config_data)

    # Load safetensors once into std model
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

    # Share weights — no second safetensors load, no extra memory
    rs = GemmaResidualStreamForCausalLM(config)
    rs.update(std.parameters())
    mx.eval(rs.parameters())
    rs.eval()

    return std, rs, config


def load_engine_class():
    import importlib.util
    inf = Path(__file__).parents[2] / "src/chuk_lazarus/inference"

    def _load(dotted, fpath):
        spec = importlib.util.spec_from_file_location(dotted, fpath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("chuk_lazarus.inference.rs_generator", inf / "rs_generator.py")
    mod = _load("chuk_lazarus.inference.bounded_cache", inf / "bounded_cache.py")
    return mod.BoundedKVEngine


# ---------------------------------------------------------------------------
# Unbounded KV baseline
# ---------------------------------------------------------------------------

def run_unbounded_kv(std_model, input_ids_per_turn: list[list[int]], gen_tokens: int):
    """
    Standard KV cache, no budget. Memory grows linearly.
    Returns list of (hot_bytes, tok_per_sec) per turn.
    """
    cache    = None
    all_ids  = []
    results  = []

    for turn_ids in input_ids_per_turn:
        all_ids.extend(turn_ids)

        # Feed new tokens through existing cache
        if cache is None:
            ids = mx.array(all_ids)[None]
            out = std_model(ids)
        else:
            # Feed new turn tokens one at a time (mask compatibility)
            # Update cache after each token so extensions accumulate correctly
            for tok in turn_ids:
                ids = mx.array([[tok]])
                out = std_model(ids, cache=cache)
                mx.eval(out.logits)
                cache = out.cache

        mx.eval(out.logits)
        cache = out.cache

        # Generate
        t0 = time.perf_counter()
        for _ in range(gen_tokens):
            next_tok = int(mx.argmax(out.logits[0, -1, :]))
            all_ids.append(next_tok)
            ids = mx.array([[next_tok]])
            out = std_model(ids, cache=cache)
            mx.eval(out.logits)
            cache = out.cache
        gen_ms = (time.perf_counter() - t0) * 1000

        kv_bytes = sum(
            k.nbytes + v.nbytes
            for k, v in cache
            if k is not None
        )
        results.append({
            "hot_bytes":    kv_bytes,
            "tok_per_sec":  gen_tokens / (gen_ms / 1000),
            "total_tokens": len(all_ids),
            "path":         "hot",
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_turn_table(label: str, color: str, rows: list[dict], evict_marker: str = "⚠"):
    print(f"\n  {color}{BOLD}{label}{RESET}")
    print(f"  {'Turn':>4}  {'Path':>5}  {'Window':>7}  {'Hot':>9}  "
          f"{'Warm':>9}  {'Cold':>8}  {'Budget%':>7}  {'tok/s':>6}")
    print("  " + "─" * 72)

    path_color = {"hot": CYAN, "warm": YELLOW, "cold": RED}

    for i, r in enumerate(rows, 1):
        evict = evict_marker if r.get("window_start", 0) > 0 else " "
        pc    = path_color.get(r.get("path", "cold"), RESET)
        warm  = r.get("checkpoint_bytes", 0) + r.get("dark_bytes", 0)
        bpct  = r.get("budget_used_pct", 0)
        bpct_color = RED if bpct > 100 else (YELLOW if bpct > 85 else RESET)

        hot_str = fmt_bytes(r.get("hot_bytes", r.get("kv_bytes", 0)))

        print(f"  {i:>4}  "
              f"{pc}{r.get('path','?'):>5}{RESET}  "
              f"{r.get('window_size', r.get('total_tokens', 0)):>7,}  "
              f"{hot_str:>9}  "
              f"{fmt_bytes(warm):>9}  "
              f"{fmt_bytes(r.get('token_id_bytes', r.get('total_tokens', 0)*4)):>8}  "
              f"{evict}{bpct_color}{bpct:>5.1f}%{RESET}  "
              f"{r.get('tok_per_sec', 0):>6.1f}")


def main():
    args = parse_args()
    budget_bytes = int(args.budget_mb * 1024 * 1024)

    print(f"\n{BOLD}Inference Mode Comparison: Multi-Turn Conversation{RESET}")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Budget:    {fmt_bytes(budget_bytes)}")
    print(f"  Turns:     {args.turns}")
    print(f"  Input/turn:{args.tokens_per_turn} base tokens (varies ±20)")
    print(f"  Gen/turn:  {args.gen_tokens} tokens")
    print(f"  Ckpt int:  every {args.checkpoint_interval} tokens")
    print(f"  Dark:      layers {args.dark_layers}")

    std_model, rs_model, config = load_models(args.model)
    BoundedKVEngine = load_engine_class()

    kv_bytes_per_tok = (
        2 * config.num_key_value_heads * config.head_dim * 2
        * config.num_hidden_layers
    )
    rs_bytes_per_tok = config.hidden_size * 2 * config.num_hidden_layers
    res_ratio = rs_bytes_per_tok / kv_bytes_per_tok

    print(f"\n  Bytes/token — KV: {fmt_bytes(kv_bytes_per_tok)}  "
          f"RS: {fmt_bytes(rs_bytes_per_tok)}  "
          f"(RS is {res_ratio:.2f}× {'larger' if res_ratio > 1 else 'smaller'})")
    print(f"  Max tokens in budget — KV: {budget_bytes // kv_bytes_per_tok}  "
          f"RS: {budget_bytes // rs_bytes_per_tok}")

    # Build engines
    engine_kv = BoundedKVEngine(
        std_model=std_model, rs_model=rs_model, config=config,
        budget_bytes=budget_bytes, generation_mode="kv",
        checkpoint_interval=args.checkpoint_interval,
        dark_layers=args.dark_layers,
    )
    engine_rs = BoundedKVEngine(
        std_model=std_model, rs_model=rs_model, config=config,
        budget_bytes=budget_bytes, generation_mode="rs",
        checkpoint_interval=args.checkpoint_interval,
        dark_layers=args.dark_layers,
    )

    # Synthetic conversation turns
    turns = [
        [((t * 7 + i * 13) % 8000) + 1
         for i in range(args.tokens_per_turn + (t % 3) * 10)]
        for t in range(1, args.turns + 1)
    ]

    # Warm up
    print("\n  Warming up / compiling...")
    _w = mx.array([[1, 2, 3, 4, 5]])
    _ = std_model(_w); _ = rs_model(_w); mx.eval()
    _l, _s = engine_rs._rs_gen.prefill(_w)
    mx.eval(_l)
    _l2, _s2 = engine_rs._rs_gen.step(mx.array([[6]]), _s, 5)
    mx.eval(_l2)
    print("  Done.\n")

    # Run all three modes
    print(f"{BOLD}Running unbounded KV (no budget)...{RESET}")
    unbounded_rows = run_unbounded_kv(std_model, turns, args.gen_tokens)

    print(f"{BOLD}Running bounded KV (mode=kv, budget={fmt_bytes(budget_bytes)})...{RESET}")
    state_kv = engine_kv.new_conversation()
    bounded_kv_rows = []
    for turn_ids in turns:
        _, state_kv, stats = engine_kv.process_turn(
            state_kv, turn_ids, max_new_tokens=args.gen_tokens
        )
        bounded_kv_rows.append(stats)

    print(f"{BOLD}Running bounded RS (mode=rs, budget={fmt_bytes(budget_bytes)})...{RESET}")
    state_rs = engine_rs.new_conversation()
    bounded_rs_rows = []
    for turn_ids in turns:
        _, state_rs, stats = engine_rs.process_turn(
            state_rs, turn_ids, max_new_tokens=args.gen_tokens
        )
        bounded_rs_rows.append(stats)

    # Display results
    print_turn_table(
        f"Unbounded KV  (no budget — grows to {fmt_bytes(unbounded_rows[-1]['hot_bytes'])})",
        CYAN, unbounded_rows
    )
    print_turn_table(
        f"Bounded KV    (budget={fmt_bytes(budget_bytes)}, window slides, full speed)",
        GREEN, bounded_kv_rows
    )
    print_turn_table(
        f"Bounded RS    (budget={fmt_bytes(budget_bytes)}, RS compiled, dark-native)",
        YELLOW, bounded_rs_rows
    )

    # Summary
    def avg_tps(rows):
        return sum(r.get("tok_per_sec", 0) for r in rows) / len(rows)

    peak_unbounded = unbounded_rows[-1]["hot_bytes"]
    peak_kv        = max(r.get("hot_bytes", r.get("kv_bytes", 0)) for r in bounded_kv_rows)
    peak_rs        = max(r.get("hot_bytes", 0) for r in bounded_rs_rows)

    total_toks     = unbounded_rows[-1]["total_tokens"]
    cold_bytes     = total_toks * 4

    print(f"""
{BOLD}Summary{RESET}

  {'Mode':>14}  {'Peak hot':>10}  {'Cold':>8}  {'Avg tok/s':>10}  {'Budget':>8}  {'Dark cost'}
  {'─'*72}
  {'unbounded-kv':>14}  {fmt_bytes(peak_unbounded):>10}  {fmt_bytes(cold_bytes):>8}  {avg_tps(unbounded_rows):>9.1f}  {'none':>8}  extra RS pass
  {'bounded-kv':>14}  {fmt_bytes(peak_kv):>10}  {fmt_bytes(cold_bytes):>8}  {avg_tps(bounded_kv_rows):>9.1f}  {fmt_bytes(budget_bytes):>8}  extra RS pass
  {'bounded-rs':>14}  {fmt_bytes(peak_rs):>10}  {fmt_bytes(cold_bytes):>8}  {avg_tps(bounded_rs_rows):>9.1f}  {fmt_bytes(budget_bytes):>8}  zero (in hot state)

  Total conversation: {total_toks} tokens across {args.turns} turns.
  Cold tier (token IDs) identical for all modes: {fmt_bytes(cold_bytes)} — {peak_unbounded // max(cold_bytes,1):,}× smaller than KV.

  Unbounded KV would be: {fmt_bytes(peak_unbounded)}
  Bounded hot tier:      {fmt_bytes(peak_kv)} (KV) / {fmt_bytes(peak_rs)} (RS)

{BOLD}When to use each mode{RESET}

  unbounded-kv  : GPU server. Memory not a constraint. Max speed always.
  bounded-kv    : Edge device with memory budget. Full generation speed.
                  Dark inference costs one extra RS pass per turn.
  bounded-rs    : Ultra-constrained device. Dark inference is the primary workload.
                  Generation speed trades for dark residuals always in memory.
                  Also: any device where context >> budget (RS per-token cost
                  amortises better than repeated cold rebuilds of the KV window).

{BOLD}The Markov principle{RESET}

  The residual tensor at each layer is the complete forward state.
  Between turns: token IDs only. 4 bytes per token.
  {total_toks} tokens of conversation history = {fmt_bytes(cold_bytes)}.
  Same conversation in KV cache = {fmt_bytes(kv_bytes_per_tok * total_toks)}.
  Ratio: {kv_bytes_per_tok * total_toks // max(cold_bytes, 1):,}× — constant, model-independent.
""")

    # Dark residuals
    dark_state = state_rs if state_rs.dark_residuals else state_kv
    if dark_state.dark_residuals:
        print(f"{BOLD}Dark residuals (bounded-rs mode — in hot state, zero extra cost){RESET}\n")
        for layer_idx, res in sorted(dark_state.dark_residuals.items()):
            print(f"  L{layer_idx:>2}: shape={tuple(res.shape)}  {fmt_bytes(res.nbytes)}")
        print()


if __name__ == "__main__":
    main()
