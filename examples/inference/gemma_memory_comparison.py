#!/usr/bin/env python3
"""
Memory growth comparison: standard Gemma (KV cache) vs residual stream Gemma.

Shows how memory scales with context length for both architectures across
all Gemma model sizes.

Two sections:
  1. Theoretical — exact formula-based calculation, no model download required
  2. Measured   — actually allocates tensors and reads metal memory (270M default)

Usage:
    uv run python examples/inference/gemma_memory_comparison.py
    uv run python examples/inference/gemma_memory_comparison.py --measure
    uv run python examples/inference/gemma_memory_comparison.py --measure --model mlx-community/gemma-3-1b-it-bf16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Gemma memory growth comparison")
    p.add_argument(
        "--measure",
        action="store_true",
        help="Also measure real allocations (requires model download)",
    )
    p.add_argument(
        "--model", default="mlx-community/gemma-3-270m-it-bf16", help="Model ID for --measure mode"
    )
    p.add_argument("--max-seq", type=int, default=4096, help="Maximum sequence length to plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gemma model configurations
# ---------------------------------------------------------------------------


@dataclass
class GemmaSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # bfloat16


GEMMA_SPECS = [
    GemmaSpec("270M", num_layers=18, hidden_size=640, num_kv_heads=1, head_dim=256),
    GemmaSpec("1B", num_layers=26, hidden_size=1152, num_kv_heads=1, head_dim=256),
    GemmaSpec("4B", num_layers=34, hidden_size=2560, num_kv_heads=4, head_dim=256),
    GemmaSpec("12B", num_layers=48, hidden_size=3840, num_kv_heads=8, head_dim=256),
    GemmaSpec("27B", num_layers=62, hidden_size=5120, num_kv_heads=8, head_dim=256),
]

CONTEXT_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


# ---------------------------------------------------------------------------
# Memory formulae
# ---------------------------------------------------------------------------


def kv_cache_bytes(spec: GemmaSpec, seq_len: int) -> int:
    """
    KV cache size for standard Gemma after a single forward pass of seq_len tokens.

    Standard inference stores K and V for every token at every layer:
      2 (K+V) × num_layers × num_kv_heads × seq_len × head_dim × dtype_bytes
    """
    return 2 * spec.num_layers * spec.num_kv_heads * seq_len * spec.head_dim * spec.dtype_bytes


def residual_bytes(spec: GemmaSpec, seq_len: int) -> int:
    """
    Residual tensor size for RS Gemma.

    The residual stream is the ONLY persistent state:
      seq_len × hidden_size × dtype_bytes
    """
    return seq_len * spec.hidden_size * spec.dtype_bytes


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

BAR_CHARS = "▏▎▍▌▋▊▉█"


def bar(value: float, max_value: float, width: int = 40) -> str:
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


def print_theoretical(spec: GemmaSpec, context_lengths: list[int]) -> None:
    print(
        f"\n{BOLD}Gemma {spec.name}  "
        f"({spec.num_layers} layers · hidden={spec.hidden_size} · "
        f"kv_heads={spec.num_kv_heads} · head_dim={spec.head_dim}){RESET}"
    )
    print(
        f"  {'Seq len':>8}  {'KV cache':>12}  {'Residual':>12}  {'Ratio':>8}  "
        f"  KV cache bar              Residual bar"
    )
    print("  " + "─" * 90)

    max_kv = kv_cache_bytes(spec, context_lengths[-1])

    for seq in context_lengths:
        kv = kv_cache_bytes(spec, seq)
        res = residual_bytes(spec, seq)
        ratio = kv / res

        kv_bar = bar(kv, max_kv, width=30)
        res_bar = bar(res, max_kv, width=30)

        print(
            f"  {seq:>8,}  {fmt_bytes(kv):>12}  {fmt_bytes(res):>12}  {ratio:>7.1f}×"
            f"  {CYAN}{kv_bar}{RESET}  {DIM}{res_bar}{RESET}"
        )


def print_summary_table(context_lengths: list[int]) -> None:
    print(f"\n{BOLD}Ratio: KV cache / residual  (how many × larger the KV cache is){RESET}")

    col_width = 9
    header = f"{'Seq len':>8}  " + "  ".join(f"{s.name:>{col_width}}" for s in GEMMA_SPECS)
    print("  " + header)
    print("  " + "─" * len(header))

    for seq in context_lengths:
        row = f"  {seq:>8,}  "
        parts = []
        for spec in GEMMA_SPECS:
            kv = kv_cache_bytes(spec, seq)
            res = residual_bytes(spec, seq)
            parts.append(f"{kv / res:>{col_width}.1f}×")
        print(row + "  ".join(parts))

    print()
    print("  Formula:")
    print("    KV cache  = 2 × layers × kv_heads × seq_len × head_dim × 2 bytes")
    print("    Residual  = seq_len × hidden_size × 2 bytes")
    print("    Ratio     = 2 × layers × kv_heads × head_dim / hidden_size  (constant!)")
    print()
    for spec in GEMMA_SPECS:
        const_ratio = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim / spec.hidden_size
        print(
            f"    Gemma {spec.name:<4}: ratio = {const_ratio:.1f}× (independent of sequence length)"
        )


# ---------------------------------------------------------------------------
# Measured section
# ---------------------------------------------------------------------------


def _download(model_id: str) -> Path:
    local = Path(model_id)
    if local.exists() and local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    print(f"  Downloading {model_id} ...")
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


def measure_memory(model_id: str, context_lengths: list[int]) -> None:
    """
    Actually allocate KV cache and residual tensors for each context length
    and read MLX metal memory to get real numbers.
    """
    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    print(f"\n{BOLD}=== MEASURED: real MLX metal allocations ==={RESET}")
    print(f"Model: {model_id}")

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    config = GemmaConfig.from_hf_config(config_data)
    spec = GemmaSpec(
        name="measured",
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_kv_heads=config.num_key_value_heads or config.num_attention_heads,
        head_dim=config.head_dim,
    )
    print(
        f"  {config.num_hidden_layers} layers · hidden={config.hidden_size} · "
        f"kv_heads={spec.num_kv_heads} · head_dim={spec.head_dim}"
    )

    print("\n  Loading standard GemmaForCausalLM ...")
    standard = GemmaForCausalLM(config)
    _apply_weights(standard, model_path)
    standard.eval()

    print("  Loading GemmaResidualStreamForCausalLM ...")
    rs_model = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs_model, model_path)
    rs_model.eval()

    print()
    print(
        f"  {'Seq len':>8}  {'KV measured':>14}  {'RS measured':>14}  "
        f"{'KV theory':>12}  {'RS theory':>12}  {'Ratio (meas)':>14}"
    )
    print("  " + "─" * 92)

    for seq_len in context_lengths:
        if seq_len > 2048:
            # Skip very long sequences for measured mode (slow)
            continue

        input_ids = mx.array([[1] * seq_len])

        # --- Standard: run forward, collect cache, measure cache tensors ---
        mx.metal.clear_cache()

        std_out = standard(input_ids)
        mx.eval(std_out.logits)

        # Cache is a list of (K, V) tuples, one per layer
        cache_tensors = []
        if std_out.cache:
            for layer_cache in std_out.cache:
                if layer_cache is not None:
                    k, v = layer_cache
                    cache_tensors.extend([k, v])
        mx.eval(*cache_tensors) if cache_tensors else None

        cache_bytes_measured = sum(t.nbytes for t in cache_tensors)

        # --- RS: run forward, measure residual ---
        mx.metal.clear_cache()
        rs_out = rs_model(input_ids)
        mx.eval(rs_out.residual)

        residual_bytes_measured = rs_out.residual.nbytes

        kv_theory = kv_cache_bytes(spec, seq_len)
        res_theory = residual_bytes(spec, seq_len)
        ratio = cache_bytes_measured / residual_bytes_measured if residual_bytes_measured > 0 else 0

        print(
            f"  {seq_len:>8,}  "
            f"{fmt_bytes(cache_bytes_measured):>14}  "
            f"{fmt_bytes(residual_bytes_measured):>14}  "
            f"{fmt_bytes(kv_theory):>12}  "
            f"{fmt_bytes(res_theory):>12}  "
            f"{ratio:>13.1f}×"
        )


# ---------------------------------------------------------------------------
# Zero KV loss proof
# ---------------------------------------------------------------------------


def recompute_kv_from_residual(
    rs_model,
    layer_idx: int,
    pre_layer_residual: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Given the residual just BEFORE layer_idx, recompute K and V from scratch
    using only that residual and the layer's weights.

    This is exactly what GemmaAttention does internally — except here we
    do it explicitly to prove the residual encodes everything.

    Steps mirror GemmaAttention.__call__ with cache=None:
      1. input_layernorm(residual)
      2. k_proj  →  reshape  →  k_norm  →  RoPE
      3. v_proj  →  reshape
    """
    block = rs_model.layers[layer_idx]
    attn = block.self_attn

    batch_size, seq_len, _ = pre_layer_residual.shape

    # Pre-attention norm (Gemma's input_layernorm)
    normed = block.input_layernorm(pre_layer_residual)

    # Project K and V
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    # Reshape: (batch, seq, kv_heads * head_dim) → (batch, kv_heads, seq, head_dim)
    keys = keys.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim).transpose(
        0, 2, 1, 3
    )

    # Q/K normalization (Gemma-specific)
    keys = attn.k_norm(keys)

    # RoPE with offset=0 (no prior cache)
    keys = attn.rope(keys)

    return keys, values


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def prove_zero_kv_loss(model_id: str, prompt: str = "The Markov property states that") -> None:
    """
    Prove that the residual stream contains exactly the information
    that the KV cache stores — with zero loss.

    Method:
      1. Run standard GemmaForCausalLM. Capture KV cache for every layer.
      2. For each layer L, use RS model to get residual just BEFORE layer L.
      3. Recompute K and V from that residual using layer L's weights.
      4. Compare recomputed K,V to cached K,V.

    If K,V are recoverable from the residual with diff=0, the cache is
    storing nothing that the residual doesn't already encode.
    """
    from transformers import AutoTokenizer

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
    from chuk_lazarus.models_v2.families.gemma_rs import GemmaResidualStreamForCausalLM

    print(f"\n{BOLD}=== PROOF: Zero KV information loss ==={RESET}")
    print(f"Prompt: {repr(prompt)}")
    print()
    print("Claim: the cached K,V at every layer are a deterministic function")
    print("of the residual at that layer. The cache stores nothing extra.")
    print()

    model_path = _download(model_id)
    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    config = GemmaConfig.from_hf_config(config_data)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    input_ids = mx.array(tokenizer.encode(prompt))[None]
    seq_len = input_ids.shape[1]
    print(f"  Model: {config.num_hidden_layers} layers · hidden={config.hidden_size}")
    print(f"  Prompt tokens: {seq_len}")
    print()

    standard = GemmaForCausalLM(config)
    _apply_weights(standard, model_path)
    standard.eval()

    rs_model = GemmaResidualStreamForCausalLM(config)
    _apply_weights(rs_model, model_path)
    rs_model.eval()

    # Step 1: standard forward pass — get the full KV cache
    std_out = standard(input_ids)
    mx.eval(std_out.logits)
    if std_out.cache:
        for c in std_out.cache:
            if c is not None:
                mx.eval(c[0], c[1])

    num_layers = config.num_hidden_layers

    # Step 2+3+4: for each layer, recover K,V from the residual, compare to cache
    print(
        f"  {'Layer':>6}  {'Type':>8}  {'Cached K (bytes)':>18}  "
        f"{'K diff':>12}  {'V diff':>12}  {'Result':>8}"
    )
    print("  " + "─" * 78)

    all_pass = True
    max_k_diff_global = 0.0
    max_v_diff_global = 0.0

    for layer_idx in range(num_layers):
        # Get residual just before this layer
        partial = rs_model.forward_to_layer(input_ids, stop_layer=layer_idx)
        mx.eval(partial.residual)

        # Recompute K,V from that residual
        recomp_k, recomp_v = recompute_kv_from_residual(rs_model, layer_idx, partial.residual)
        mx.eval(recomp_k, recomp_v)

        # Compare to what the standard model cached
        cached_k, cached_v = std_out.cache[layer_idx]
        mx.eval(cached_k, cached_v)

        k_diff = max_abs_diff(recomp_k, cached_k)
        v_diff = max_abs_diff(recomp_v, cached_v)

        max_k_diff_global = max(max_k_diff_global, k_diff)
        max_v_diff_global = max(max_v_diff_global, v_diff)

        layer_type = "global" if config.is_global_layer(layer_idx) else "sliding"
        k_bytes = cached_k.nbytes
        passed = k_diff == 0.0 and v_diff == 0.0
        if not passed:
            all_pass = False

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(
            f"  {layer_idx:>6}  {layer_type:>8}  {fmt_bytes(k_bytes):>18}  "
            f"{k_diff:>12.2e}  {v_diff:>12.2e}  {status}"
        )

    print()
    print(f"  Global max K diff across all {num_layers} layers: {max_k_diff_global:.2e}")
    print(f"  Global max V diff across all {num_layers} layers: {max_v_diff_global:.2e}")
    print()

    if all_pass:
        print(
            f"  {GREEN}{BOLD}All layers: diff = 0.0  — the KV cache is exactly recoverable from the residual.{RESET}"
        )
        print()
        print("  Conclusion:")
        print("    K_layer_N = f(residual_before_layer_N)  — deterministic, lossless")
        print("    V_layer_N = g(residual_before_layer_N)  — deterministic, lossless")
        print()
        print("    The KV cache stores redundant information.")
        print("    The residual stream is the complete state.")
    else:
        print(f"  {RED}Some layers showed non-zero diff — investigate.{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    context_lengths = [ctx for ctx in CONTEXT_LENGTHS if ctx <= args.max_seq]
    if not context_lengths:
        context_lengths = CONTEXT_LENGTHS

    print(f"{BOLD}Gemma Memory Growth: KV Cache vs Residual Stream{RESET}")
    print("=" * 60)
    print()
    print("Standard Gemma stores K and V for every token at every layer.")
    print("Residual stream Gemma stores only the current residual tensor.")
    print("Both scale linearly with sequence length — but the constants differ.")

    # --- Theoretical section ---
    print(f"\n{BOLD}=== THEORETICAL (formula-based, no model required) ==={RESET}")
    for spec in GEMMA_SPECS:
        print_theoretical(spec, context_lengths)

    print_summary_table(context_lengths)

    # --- Measured + proof sections ---
    if args.measure:
        measure_memory(args.model, context_lengths)
        prove_zero_kv_loss(args.model)
    else:
        print(f"{DIM}Run with --measure to also measure real MLX metal allocations{RESET}")
        print(f"{DIM}and prove zero KV information loss across all layers.{RESET}")
        print(
            f"{DIM}Example: uv run python examples/inference/gemma_memory_comparison.py --measure{RESET}"
        )
        print()


if __name__ == "__main__":
    main()
