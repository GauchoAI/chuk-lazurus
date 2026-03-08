#!/usr/bin/env python3
"""
Two-mode inference comparison across a simulated multi-turn conversation.

  unbounded-kv  Standard KV cache. No budget limit. Memory grows linearly.
                The baseline every production system runs today.
  bounded-rs    Bounded residual stream. Per-layer residuals as hot tier.
                Hard memory budget. Window slides when budget is hit.
                For 12B: residuals are 0.94× the size of KV per token.

The comparison shows:
  - Memory growth trajectory across turns (does it bound?)
  - Generation speed per turn and per mode
  - Path taken (cold / hot / warm) per turn

Usage:
    uv run python examples/inference/gemma_modes_comparison.py
    uv run python examples/inference/gemma_modes_comparison.py --model mlx-community/gemma-3-12b-it-bf16
    uv run python examples/inference/gemma_modes_comparison.py --budget-mb 128 --turns 20
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
    p.add_argument("--budget-mb", type=float, default=None,
                   help="Hot-tier budget in MB. Default: auto (holds ~300 KV tokens for the loaded model).")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--tokens-per-turn", type=int, default=30)
    p.add_argument("--gen-tokens", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=0,
                   help="Store residual checkpoint every N tokens. 0=disabled (default).")
    p.add_argument("--dark-layers", nargs="+", type=int, default=[])
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

    print(f"  {'Turn':>4}  {'Context':>8}  {'Hot':>9}  {'tok/s':>6}  {'t(s)':>5}")
    print("  " + "─" * 44)

    for i, turn_ids in enumerate(input_ids_per_turn, 1):
        t_turn = time.perf_counter()
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
        total_s = time.perf_counter() - t_turn

        kv_bytes = sum(
            k.nbytes + v.nbytes
            for k, v in cache
            if k is not None
        )
        tps = gen_tokens / (gen_ms / 1000)
        print(f"  {i:>4}  {len(all_ids):>8,}  {fmt_bytes(kv_bytes):>9}  {tps:>6.1f}  {total_s:>4.1f}s")

        results.append({
            "hot_bytes":    kv_bytes,
            "tok_per_sec":  tps,
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

    path_color = {"hot": CYAN, "warm": YELLOW, "cold": RED, "grow": GREEN, "full": CYAN}

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

    print(f"\n{BOLD}Inference Mode Comparison: Multi-Turn Conversation{RESET}")
    print("=" * 60)
    print(f"  Model:     {args.model}")

    std_model, rs_model, config = load_models(args.model)
    BoundedKVEngine = load_engine_class()

    # Auto-budget: hold ~300 KV tokens if not specified
    if args.budget_mb is None:
        kv_bytes_per_tok_auto = (
            2 * config.num_key_value_heads * config.head_dim * 2
            * config.num_hidden_layers
        )
        budget_bytes = 300 * kv_bytes_per_tok_auto
        print(f"  Budget:    {fmt_bytes(budget_bytes)}  (auto: 300 KV tokens)")
    else:
        budget_bytes = int(args.budget_mb * 1024 * 1024)
        print(f"  Budget:    {fmt_bytes(budget_bytes)}")

    print(f"  Turns:     {args.turns}")
    print(f"  Input/turn:{args.tokens_per_turn} base tokens (varies ±20)")
    print(f"  Gen/turn:  {args.gen_tokens} tokens")

    kv_bytes_per_tok = (
        2 * config.num_key_value_heads * config.head_dim * 2
        * config.num_hidden_layers
    )
    rs_bytes_per_tok = config.hidden_size * 2 * config.num_hidden_layers
    res_ratio = rs_bytes_per_tok / kv_bytes_per_tok

    print(f"\n  Bytes/token — KV: {fmt_bytes(kv_bytes_per_tok)}  "
          f"RS: {fmt_bytes(rs_bytes_per_tok)}  "
          f"(RS is {res_ratio:.2f}× {'larger' if res_ratio > 1 else 'smaller'})")
    print(f"  Max tokens in budget — RS: {budget_bytes // rs_bytes_per_tok}")

    # Build engine
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

    def _live_header():
        print(f"  {'Turn':>4}  {'Path':>5}  {'Context':>8}  {'Window':>7}  {'Hot':>9}  {'Budget%':>7}  {'tok/s':>6}  {'t(s)':>5}")
        print("  " + "─" * 66)

    def _live_row(i, stats):
        path = stats.get("path", "?")
        pc   = {"hot": CYAN, "warm": YELLOW, "cold": RED, "grow": GREEN, "full": CYAN}.get(path, RESET)
        bpct = stats.get("budget_used_pct", 0)
        bc   = RED if bpct > 100 else (YELLOW if bpct > 85 else RESET)
        evict = "⚠" if stats.get("window_start", 0) > 0 else " "
        total_s = stats.get("total_ms", 0) / 1000
        print(f"  {i:>4}  {pc}{path:>5}{RESET}  "
              f"{stats.get('total_tokens', 0):>8,}  "
              f"{stats.get('window_size', 0):>7,}  "
              f"{fmt_bytes(stats.get('hot_bytes', 0)):>9}  "
              f"{evict}{bc}{bpct:>5.1f}%{RESET}  "
              f"{stats.get('tok_per_sec', 0):>6.1f}  "
              f"{total_s:>4.1f}s")

    # ── Mode 1: Unbounded KV ────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}Mode 1 — Unbounded KV cache{RESET}")
    print(f"  Standard transformer inference. Every token ever processed is kept")
    print(f"  as K,V tensors in GPU memory. No eviction. Memory grows without bound.")
    print(f"  The baseline every production system runs today.\n")
    unbounded_rows = run_unbounded_kv(std_model, turns, args.gen_tokens)

    # ── Mode 2: Bounded RS ──────────────────────────────────────────────
    print(f"\n{BOLD}{YELLOW}Mode 2 — Bounded Residual Stream  (budget: {fmt_bytes(budget_bytes)}){RESET}")
    print(f"  No KV cache. Stores the raw residual tensor at each layer instead.")
    print(f"  K and V are recomputed on-the-fly from stored residuals each step.")
    print(f"  Incremental: new turns extend stored residuals — no full re-prefill.")
    print(f"  For 12B: residuals are 0.94× the size of KV — fits more in the same budget.")
    print(f"  Window slides when budget is hit. Token IDs kept forever in cold tier.\n")
    _live_header()
    state_rs = engine_rs.new_conversation()
    bounded_rs_rows = []
    for i, turn_ids in enumerate(turns, 1):
        _, state_rs, stats = engine_rs.process_turn(
            state_rs, turn_ids, max_new_tokens=args.gen_tokens
        )
        bounded_rs_rows.append(stats)
        _live_row(i, stats)

    # Display results
    print_turn_table(
        f"Unbounded KV  (no budget — grows to {fmt_bytes(unbounded_rows[-1]['hot_bytes'])})",
        CYAN, unbounded_rows
    )
    print_turn_table(
        f"Bounded RS    (budget={fmt_bytes(budget_bytes)}, window slides)",
        YELLOW, bounded_rs_rows
    )

    # Summary
    def avg_tps(rows):
        return sum(r.get("tok_per_sec", 0) for r in rows) / len(rows)

    peak_unbounded = unbounded_rows[-1]["hot_bytes"]
    peak_rs        = max(r.get("hot_bytes", 0) for r in bounded_rs_rows)

    total_toks     = unbounded_rows[-1]["total_tokens"]
    cold_bytes     = total_toks * 4

    print(f"""
{BOLD}Summary{RESET}

  {'Mode':>14}  {'Peak hot':>10}  {'Cold':>8}  {'Avg tok/s':>10}  {'Budget'}
  {'─'*60}
  {'unbounded-kv':>14}  {fmt_bytes(peak_unbounded):>10}  {fmt_bytes(cold_bytes):>8}  {avg_tps(unbounded_rows):>9.1f}  none
  {'bounded-rs':>14}  {fmt_bytes(peak_rs):>10}  {fmt_bytes(cold_bytes):>8}  {avg_tps(bounded_rs_rows):>9.1f}  {fmt_bytes(budget_bytes)}

  Total conversation: {total_toks} tokens across {args.turns} turns.
  Cold tier (token IDs): {fmt_bytes(cold_bytes)} — {peak_unbounded // max(cold_bytes,1):,}× smaller than unbounded KV.

{BOLD}The Markov principle{RESET}

  The residual tensor at each layer is the complete forward state.
  Between turns: token IDs only. 4 bytes per token.
  {total_toks} tokens of conversation history = {fmt_bytes(cold_bytes)}.
  Same conversation in KV cache = {fmt_bytes(kv_bytes_per_tok * total_toks)}.
  Ratio: {kv_bytes_per_tok * total_toks // max(cold_bytes, 1):,}× — constant, model-independent.
""")



if __name__ == "__main__":
    main()
