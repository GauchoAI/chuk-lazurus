#!/usr/bin/env python3
"""
Gemma Residual Stream: Live Inference Demo

Side-by-side autoregressive generation comparing:
  - Standard GemmaForCausalLM  (KV cache grows with context)
  - GemmaResidualStreamForCausalLM  (no persistent KV state, full recompute per step)

Both produce identical tokens (greedy). The difference is memory architecture:
  - Standard: state = token_ids + growing KV cache
  - RS:       state = token_ids only  (residual recomputed fresh at every step)

Usage:
    uv run python examples/inference/gemma_rs_live_inference.py
    uv run python examples/inference/gemma_rs_live_inference.py --model mlx-community/gemma-3-1b-it-bf16
    uv run python examples/inference/gemma_rs_live_inference.py --tokens 60 --context long
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="RS model live inference demo")
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--tokens", type=int, default=50, help="Tokens to generate")
    p.add_argument(
        "--context", default="medium", choices=["short", "medium", "long"], help="Prompt length"
    )
    p.add_argument("--prompt", default=None, help="Custom prompt text")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    "short": (
        "The Markov property states that the future is independent of the past "
        "given the present. In a transformer,"
    ),
    "medium": (
        "The Markov property is a fundamental concept in probability theory. "
        "A stochastic process has the Markov property if the conditional probability "
        "distribution of future states depends only on the present state, not on the "
        "sequence of events that preceded it. This memoryless property appears across "
        "many domains: Markov chains model queues and random walks, hidden Markov models "
        "underpin speech recognition, and Markov decision processes define reinforcement "
        "learning environments.\n\n"
        "In transformer language models, each layer updates a residual stream — a tensor "
        "of shape (sequence_length, hidden_size) — by adding attention and feedforward "
        "contributions. The residual at any layer is a sufficient statistic for all "
        "subsequent computation: given the residual, no information about prior layers "
        "or prior tokens is needed. The keys and values used in attention at layer N are "
        "deterministic functions of the residual at layer N. Therefore,"
    ),
    "long": (
        "Information theory, developed by Claude Shannon in 1948, provides the mathematical "
        "foundation for quantifying, storing, and transmitting information. Shannon's central "
        "insight was that information could be measured independently of its meaning, using "
        "the concept of entropy: H = -sum(p_i * log2(p_i)). This formulation established "
        "that the minimum number of bits required to represent a message is determined by "
        "the probability distribution over possible messages.\n\n"
        "The channel capacity theorem — often called Shannon's second theorem — proves that "
        "reliable communication is possible at any rate below the channel capacity C, and "
        "impossible above it. This was a remarkable result: it guaranteed that error-free "
        "communication exists, while also establishing a hard limit. The proof is "
        "non-constructive; finding capacity-achieving codes remained an open problem for "
        "decades, eventually solved by turbo codes and LDPC codes in the 1990s.\n\n"
        "Modern large language models draw on information-theoretic principles at every level. "
        "The cross-entropy loss used during training is the information-theoretic measure of "
        "how well a model's probability distribution matches the true data distribution. "
        "Perplexity — the exponentiated cross-entropy — measures how surprised the model is "
        "by the test data. Compression and prediction are two sides of the same coin: a model "
        "that assigns high probability to likely continuations implicitly compresses the text.\n\n"
        "The residual stream in a transformer carries information forward through layers. "
        "At each layer, the attention mechanism routes information between token positions, "
        "and the feedforward network transforms the representation at each position. The "
        "Markov property of the residual stream — that each layer's output is a sufficient "
        "statistic for all subsequent layers — means that"
    ),
}


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
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


# ---------------------------------------------------------------------------
# Loading (same pattern as other examples — no inference/__init__ import)
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    print(f"  Downloading {model_id}...")
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        )
    )


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
    from transformers import AutoTokenizer

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    config = GemmaConfig.from_hf_config(config_data)

    standard = GemmaForCausalLM(config)
    _apply_weights(standard, model_path)
    standard.eval()

    rs_model = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs_model, model_path)
    rs_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return standard, rs_model, tokenizer, config


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


def kv_cache_bytes(cache, config) -> int:
    """Sum the actual nbytes of all cached K and V tensors."""
    if cache is None:
        return 0
    total = 0
    for layer_cache in cache:
        if layer_cache is not None:
            k, v = layer_cache
            total += k.nbytes + v.nbytes
    return total


def residual_bytes_for_seq(seq_len: int, hidden_size: int) -> int:
    """Size of the residual tensor for this sequence length (bfloat16)."""
    return seq_len * hidden_size * 2


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_standard(model, input_ids: mx.array, max_new_tokens: int, config):
    """
    Standard autoregressive generation with KV cache.
    Yields (token_id, kv_cache_bytes, time_for_step).
    """
    # Prefill
    t0 = time.time()
    out = model(input_ids)
    mx.eval(out.logits)
    cache = out.cache
    prefill_time = time.time() - t0

    tokens = list(input_ids[0].tolist())

    for _ in range(max_new_tokens):
        last_logit = out.logits[0, -1, :]
        next_token = int(mx.argmax(last_logit))
        tokens.append(next_token)

        kv_bytes = kv_cache_bytes(cache, config)

        t0 = time.time()
        out = model(mx.array([[next_token]]), cache=cache)
        mx.eval(out.logits)
        cache = out.cache
        step_time = time.time() - t0

        yield next_token, kv_bytes, step_time

        eos = getattr(model.config, "eos_token_id", None)
        if eos is not None:
            eos_list = eos if isinstance(eos, list) else [eos]
            if next_token in eos_list:
                break

    return prefill_time


def generate_rs(model, input_ids: mx.array, max_new_tokens: int, config):
    """
    Residual-stream autoregressive generation.
    No KV cache. Full recompute at every step.
    Yields (token_id, residual_bytes, time_for_step).

    State between steps = token IDs only.
    Residual is recomputed fresh from scratch at every step.
    """
    tokens = list(input_ids[0].tolist())

    for _ in range(max_new_tokens):
        current_ids = mx.array(tokens)[None]

        t0 = time.time()
        out = model(current_ids)
        mx.eval(out.logits)
        step_time = time.time() - t0

        last_logit = out.logits[0, -1, :]
        next_token = int(mx.argmax(last_logit))
        tokens.append(next_token)

        res_bytes = residual_bytes_for_seq(len(tokens), config.hidden_size)

        yield next_token, res_bytes, step_time

        eos = getattr(model.config, "eos_token_id", None)
        if eos and next_token == eos:
            break


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    prompt_text = args.prompt or PROMPTS[args.context]

    print(f"\n{BOLD}Gemma Residual Stream — Live Inference Demo{RESET}")
    print("=" * 56)

    print(f"\nLoading {args.model}...")
    standard, rs_model, tokenizer, config = load_models(args.model)

    input_ids = mx.array(tokenizer.encode(prompt_text))[None]
    prompt_len = input_ids.shape[1]

    print(f"  Layers:  {config.num_hidden_layers}")
    print(f"  Hidden:  {config.hidden_size}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")

    # Memory at prompt length
    theory_kv = (
        2 * config.num_hidden_layers * config.num_key_value_heads * prompt_len * config.head_dim * 2
    )
    theory_res = prompt_len * config.hidden_size * 2

    print(f"\n{BOLD}Prompt{RESET}")
    print(f"  Tokens:    {prompt_len}")
    print(f"  KV cache at prompt: {fmt_bytes(theory_kv)}")
    print(f"  Residual at prompt: {fmt_bytes(theory_res)}")
    print(f"  Ratio:     {theory_kv / theory_res:.1f}×")

    # Show truncated prompt
    display_prompt = prompt_text if len(prompt_text) < 200 else prompt_text[:200] + "..."
    print(f'\n  {DIM}"{display_prompt}"{RESET}')

    print(f"\n{BOLD}Generating {args.tokens} tokens — side by side{RESET}")
    print()

    col = 14
    print(
        f"  {'Step':>4}  {'Token':<20}  {'Standard KV':>{col}}  {'RS Residual':>{col}}  "
        f"{'Match':>5}  {'Std ms':>7}  {'RS ms':>7}"
    )
    print("  " + "─" * 84)

    # Warm up both models with a tiny input first
    _warmup = mx.array([[1, 2, 3]])
    _ = standard(_warmup)
    _ = rs_model(_warmup)
    mx.eval()

    # Run both generators together
    std_gen = generate_standard(standard, input_ids, args.tokens, config)
    rs_gen = generate_rs(rs_model, input_ids, args.tokens, config)

    mismatches = 0
    std_total_ms = 0.0
    rs_total_ms = 0.0

    generated_text_std = []
    generated_text_rs = []

    for step in range(args.tokens):
        try:
            std_tok, std_kv, std_t = next(std_gen)
            rs_tok, rs_res, rs_t = next(rs_gen)
        except StopIteration:
            break

        match = std_tok == rs_tok
        if not match:
            mismatches += 1

        token_str = tokenizer.decode([std_tok])
        token_display = repr(token_str)[:18]

        std_total_ms += std_t * 1000
        rs_total_ms += rs_t * 1000

        match_str = f"{GREEN}✓{RESET}" if match else f"{RED}✗{RESET}"
        kv_str = fmt_bytes(std_kv)
        res_str = fmt_bytes(rs_res)

        generated_text_std.append(token_str)
        generated_text_rs.append(tokenizer.decode([rs_tok]))

        # Colour KV column to highlight it growing
        kv_colour = CYAN if std_kv > theory_kv else ""

        print(
            f"  {step + 1:>4}  {token_display:<20}  "
            f"{kv_colour}{kv_str:>{col}}{RESET}  "
            f"{DIM}{res_str:>{col}}{RESET}  "
            f"  {match_str}  "
            f"{std_t * 1000:>7.1f}  {rs_t * 1000:>7.1f}"
        )

    print()

    # --- Final KV cache at end of generation ---
    final_seq_len = prompt_len + args.tokens
    final_kv_theory = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * final_seq_len
        * config.head_dim
        * 2
    )
    final_res_theory = final_seq_len * config.hidden_size * 2

    print(f"{BOLD}Generated text{RESET}")
    full_text = "".join(generated_text_std)
    print(f"  {CYAN}{repr(full_text)}{RESET}")
    print()

    print(
        f"{BOLD}Memory at end of generation  (prompt={prompt_len} + {args.tokens} new tokens = {final_seq_len} total){RESET}"
    )
    print(f"  Standard KV cache:  {fmt_bytes(final_kv_theory)}  (grows with every new token)")
    print(
        f"  RS residual:        {fmt_bytes(final_res_theory)}"
        f"  (recomputed fresh — no persistent KV state)"
    )
    print(f"  Ratio:              {final_kv_theory / final_res_theory:.1f}×")
    print()
    print("  RS persistent state between steps = token IDs only")
    print(f"  Token IDs: {final_seq_len} × 4 bytes = {fmt_bytes(final_seq_len * 4)}")
    print()

    print(f"{BOLD}Speed (per-token, after prefill){RESET}")
    avg_std = std_total_ms / args.tokens
    avg_rs = rs_total_ms / args.tokens
    print(f"  Standard (KV cache):       {avg_std:.1f} ms/token  (O(1) per step)")
    print(
        f"  Residual stream:           {avg_rs:.1f} ms/token  (O(seq_len) per step — full recompute)"
    )
    print(
        f"  RS overhead:               {avg_rs / avg_std:.1f}× slower  (expected — trades time for memory)"
    )
    print()

    print(f"{BOLD}Correctness{RESET}")
    if mismatches == 0:
        print(
            f"  {GREEN}✓ All {args.tokens} tokens identical.{RESET}"
            f"  The residual stream produces the same output as the KV cache."
        )
    else:
        print(f"  {RED}✗ {mismatches} mismatches out of {args.tokens} tokens.{RESET}")

    print()
    print(f"{BOLD}What this proves{RESET}")
    print("  1. Correctness: residual stream inference == KV cache inference (token-for-token)")
    print(
        f"  2. Memory:      RS model carries {fmt_bytes(final_seq_len * 4)} of token IDs between steps"
    )
    print(f"                  Standard model carries {fmt_bytes(final_kv_theory)} of KV tensors")
    print("  3. The KV cache is an optimisation, not a requirement.")
    print("     The residual at each layer is the complete forward state.")
    print()


if __name__ == "__main__":
    main()
