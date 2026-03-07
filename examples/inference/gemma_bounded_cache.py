#!/usr/bin/env python3
"""
Bounded KV cache with residual stream backing store — live demo.

Simulates a multi-turn conversation on a memory-constrained device.

The engine manages:
  Hot   : KV cache, bounded by device budget
  Warm  : Residual checkpoints (for faster KV rebuild on cold start)
          Dark residuals at L7, L10, L14 (always available for probe/inject)
  Cold  : Token IDs in Redis (simulated as in-process list, 4 bytes/token)

When the conversation grows beyond the budget:
  - Oldest tokens slide out of the active window
  - KV cache is rebuilt from the trimmed window
  - Token IDs are never evicted

Usage:
    uv run python examples/inference/gemma_bounded_cache.py
    uv run python examples/inference/gemma_bounded_cache.py --budget-mb 2
    uv run python examples/inference/gemma_bounded_cache.py --turns 8 --tokens-per-turn 40
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


def fmt_bytes(n: int) -> str:
    if n < 1024:      return f"{n} B"
    if n < 1024**2:   return f"{n/1024:.1f} KB"
    if n < 1024**3:   return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--budget-mb", type=float, default=4.0,
                   help="Device memory budget for KV cache in MB")
    p.add_argument("--turns", type=int, default=6,
                   help="Number of conversation turns to simulate")
    p.add_argument("--tokens-per-turn", type=int, default=30,
                   help="Synthetic input tokens per turn")
    p.add_argument("--gen-tokens", type=int, default=20,
                   help="Max tokens to generate per turn")
    p.add_argument("--checkpoint-interval", type=int, default=50,
                   help="Store a residual checkpoint every N tokens")
    p.add_argument("--dark-layers", nargs="+", type=int, default=[6, 9, 14],
                   help="Layers to always capture dark residuals")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    budget_bytes = int(args.budget_mb * 1024 * 1024)

    print(f"\n{BOLD}Bounded KV Cache + Residual Stream Backing Store{RESET}")
    print("=" * 58)
    print(f"  Model:               {args.model}")
    print(f"  Device budget:       {fmt_bytes(budget_bytes)}")
    print(f"  Turns:               {args.turns}")
    print(f"  Input tokens/turn:   {args.tokens_per_turn}")
    print(f"  Generated tokens:    {args.gen_tokens}")
    print(f"  Checkpoint interval: every {args.checkpoint_interval} tokens")
    print(f"  Dark layers:         {args.dark_layers}")

    std_model, rs_model, config = load_models(args.model)

    import importlib.util, sys as _sys
    _inf = Path(__file__).parents[2] / "src/chuk_lazarus/inference"
    def _load(dotted, fpath):
        spec = importlib.util.spec_from_file_location(dotted, fpath)
        mod  = importlib.util.module_from_spec(spec)
        _sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod
    _load("chuk_lazarus.inference.rs_generator", _inf / "rs_generator.py")
    BoundedKVEngine = _load("chuk_lazarus.inference.bounded_cache", _inf / "bounded_cache.py").BoundedKVEngine

    engine = BoundedKVEngine(
        std_model=std_model,
        rs_model=rs_model,
        config=config,
        budget_bytes=budget_bytes,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_layer=config.num_hidden_layers // 2,
        dark_layers=args.dark_layers,
    )

    bytes_per_token = (
        2 * config.num_key_value_heads * config.head_dim * 2
        * config.num_hidden_layers
    )
    max_kv_tokens = budget_bytes // bytes_per_token

    print(f"\n  KV bytes/token:      {fmt_bytes(bytes_per_token)}")
    print(f"  Max KV tokens:       {max_kv_tokens:,}")
    print(f"\n  Memory layout:")
    print(f"  ┌─{'─'*40}─┐")
    print(f"  │  HOT  : KV cache (up to {fmt_bytes(budget_bytes)})         │")
    print(f"  │  WARM : Checkpoints + dark residuals     │")
    print(f"  │  COLD : Token IDs (Redis, 4 bytes/tok)  │")
    print(f"  └─{'─'*40}─┘")

    # Warm up
    print("\n  Warming up...")
    _w = mx.array([[1, 2, 3, 4, 5]])
    _ = std_model(_w)
    _ = rs_model(_w)
    mx.eval()

    state = engine.new_conversation()

    print(f"\n  {'Turn':>4}  {'Path':>5}  {'Input':>6}  {'Out':>4}  "
          f"{'Window':>7}  {'KV hot':>9}  {'Warm':>9}  {'Cold':>9}  "
          f"{'Budget%':>7}  {'tok/s':>6}")
    print("  " + "─" * 98)

    for turn in range(1, args.turns + 1):
        # Synthetic input: vary length slightly each turn
        n_input = args.tokens_per_turn + (turn % 3) * 10
        input_ids = [((turn * 7 + i * 13) % 8000) + 1 for i in range(n_input)]

        generated, state, stats = engine.process_turn(
            state, input_ids, max_new_tokens=args.gen_tokens
        )

        path_color = {
            "hot":  CYAN,
            "warm": YELLOW,
            "cold": RED,
        }.get(stats["path"], RESET)

        warm_bytes = stats["checkpoint_bytes"] + stats["dark_bytes"]
        evicted    = stats["window_start"] > 0

        print(f"  {turn:>4}  "
              f"{path_color}{stats['path']:>5}{RESET}  "
              f"{n_input:>6}  "
              f"{stats['generated_tokens']:>4}  "
              f"{stats['window_size']:>7,}  "
              f"{fmt_bytes(stats['kv_bytes']):>9}  "
              f"{fmt_bytes(warm_bytes):>9}  "
              f"{fmt_bytes(stats['token_id_bytes']):>9}  "
              f"{'⚠ ' if evicted else ''}"
              f"{stats['budget_used_pct']:>5.1f}%  "
              f"{stats['tok_per_sec']:>6.1f}")

    # Final state summary
    print(f"\n{BOLD}Final conversation state{RESET}")
    report = engine.memory_report(state)
    print(f"""
  Total tokens (full history) : {report['total_tokens']:>6,}   stored in cold tier (Redis)
  Active window start         : {report['window_start']:>6,}   tokens before this are forgotten
  Active window size          : {report['window_size']:>6,}   tokens visible to the model

  ┌──────────────────────────────────────────────┐
  │  HOT  (KV cache)                             │
  │    {report['kv_token_count']:>6,} tokens    {fmt_bytes(report['kv_bytes']):>10}              │
  │    Budget: {report['budget_used_pct']:.1f}% used                          │
  ├──────────────────────────────────────────────┤
  │  WARM (checkpoints + dark residuals)         │
  │    {report['checkpoint_count']:>6,} checkpoints  {fmt_bytes(report['checkpoint_bytes']):>10}        │
  │    {report['dark_layer_count']:>6,} dark layers  {fmt_bytes(report['dark_bytes']):>10}        │
  │    Total warm:              {fmt_bytes(report['checkpoint_bytes'] + report['dark_bytes']):>10}        │
  ├──────────────────────────────────────────────┤
  │  COLD (token IDs)                            │
  │    {report['total_tokens']:>6,} token IDs   {fmt_bytes(report['token_id_bytes']):>10}              │
  └──────────────────────────────────────────────┘
""")

    print(f"{BOLD}Memory ratios{RESET}")
    kv_at_full = bytes_per_token * report['total_tokens']
    id_bytes   = report['token_id_bytes']
    print(f"""
  If we kept full KV for all {report['total_tokens']} tokens:
    KV cache (full)  : {fmt_bytes(kv_at_full)}
    Token IDs (cold) : {fmt_bytes(id_bytes)}
    Ratio            : {kv_at_full // max(id_bytes, 1):,}×

  With bounded architecture:
    HOT  : {fmt_bytes(report['kv_bytes'])} (bounded by {fmt_bytes(budget_bytes)} budget)
    COLD : {fmt_bytes(id_bytes)} (always)
    HOT + COLD vs unbounded KV: {fmt_bytes(report['kv_bytes'] + id_bytes)} vs {fmt_bytes(kv_at_full)}
""")

    if state.dark_residuals:
        print(f"{BOLD}Dark residuals (always available for probe/inject){RESET}")
        print()
        for layer_idx, residual in sorted(state.dark_residuals.items()):
            print(f"  Layer {layer_idx:>2} residual: shape={tuple(residual.shape)}  "
                  f"size={fmt_bytes(residual.nbytes)}")
        print(f"""
  These tensors are in device memory right now.
  A probe weight can read the last-token representation:
    dark_residuals[{list(state.dark_residuals.keys())[0]}][0, -1, :]  →  (hidden_dim,)
  An injection can add a steering vector at any layer before continuing.
  Both operations are zero-cost — no re-run of the model.
""")


if __name__ == "__main__":
    main()
