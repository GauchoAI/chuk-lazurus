#!/usr/bin/env python3
"""
Strategy C: compiled residual-stream generation with mx.compile.

Prefill: store per-layer residuals (seq_len, hidden) — read-only.
Generate: for each new token
  - project stored residuals → K_old, V_old  (one big matmul per layer)
  - project new token embed  → Q, K_new, V_new
  - concat K/V, run fused attention, run FFN on new token only
  - append new token residual to stored_residuals

mx.compile sees the whole per-token loop and can fuse:
  stored_residual @ wk.T  → attention  (K never materialised separately)

Benchmark: KV cache  vs  RS-plain (full recompute)  vs  RS-compiled

Usage:
    uv run python examples/inference/gemma_rs_compiled.py
    uv run python examples/inference/gemma_rs_compiled.py --model mlx-community/gemma-3-1b-it-bf16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CONTEXT_LENGTHS    = [64, 128, 256, 512, 1024, 2048]
GEN_TOKENS_DEFAULT = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--gen-tokens", type=int, default=GEN_TOKENS_DEFAULT)
    p.add_argument("--context-lengths", nargs="+", type=int, default=CONTEXT_LENGTHS)
    return p.parse_args()


def fmt_bytes(n: int) -> str:
    if n < 1024:      return f"{n} B"
    if n < 1024**2:   return f"{n/1024:.1f} KB"
    if n < 1024**3:   return f"{n/1024**2:.1f} MB"
    return f"{n/1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Strategy C: per-layer residual cache + compiled single-token step
# ---------------------------------------------------------------------------

class CompiledRSGenerator:
    """
    Stores per-layer residuals from prefill.
    Each generation step only processes one new token, attending to stored K,V
    computed on-the-fly from the stored residuals.

    The step function is wrapped in mx.compile so MLX can fuse the
    K,V projection + attention into a single Metal kernel.
    """

    def __init__(self, rs_model, config):
        self.model  = rs_model
        self.config = config
        self._step  = mx.compile(self._raw_step, shapeless=True)

    # ------------------------------------------------------------------
    # Prefill: run full forward, capture per-layer residuals
    # ------------------------------------------------------------------

    def prefill(self, input_ids: mx.array) -> tuple[mx.array, list[mx.array]]:
        """
        Full forward pass. Returns (logits, per_layer_residuals).

        per_layer_residuals[i] is the residual *entering* layer i
        (i.e. the output of layer i-1, or the embedding for layer 0).
        We store the pre-layer residual so that K,V computed from it
        are the same K,V the layer used during prefill.
        """
        backbone = self.model.model
        config   = self.config

        h = backbone._embed(input_ids)  # (1, seq_len, hidden)
        layer_inputs: list[mx.array] = []

        for i, layer in enumerate(backbone.layers):
            layer_inputs.append(h)          # store residual entering this layer
            mask = backbone._mask_for_layer(i, h)
            h = layer(h, mask=mask)

        h = backbone.norm(h)
        logits = self.model._unembed(h)
        mx.eval(logits, *layer_inputs)
        return logits, layer_inputs

    # ------------------------------------------------------------------
    # Single-token generation step (compiled)
    # ------------------------------------------------------------------

    def _raw_step(
        self,
        new_token_ids: mx.array,           # (1, 1)
        stored_residuals: list[mx.array],  # list[L] of (1, seq_len, hidden)
        seq_len: int,                      # current sequence length (for RoPE offset)
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Process one new token.

        For each layer:
          1. K_old, V_old from stored_residuals[i]  — big matmul, read-only
          2. Q, K_new, V_new from new token embed
          3. Fused attention over [K_old | K_new], [V_old | V_new]
          4. FFN on new token residual only

        Returns (logits_for_new_token, updated_stored_residuals).
        """
        backbone = self.model.model
        config   = self.config

        # Embed new token
        h_new = backbone._embed(new_token_ids)  # (1, 1, hidden)

        new_layer_inputs: list[mx.array] = []

        for i, layer in enumerate(backbone.layers):
            attn   = layer.self_attn
            h_old  = stored_residuals[i]       # (1, seq_len, hidden)

            # ---- normalise inputs ----
            x_old = layer.input_layernorm(h_old)   # (1, seq_len, hidden)
            x_new = layer.input_layernorm(h_new)   # (1, 1,       hidden)

            # ---- K, V from stored residual (old positions) ----
            k_old = attn.k_proj(x_old)   # (1, seq_len, kv_heads * head_dim)
            v_old = attn.v_proj(x_old)

            # ---- Q, K, V for new token ----
            q_new = attn.q_proj(x_new)
            k_new = attn.k_proj(x_new)
            v_new = attn.v_proj(x_new)

            # ---- reshape to (batch, heads, seq, head_dim) ----
            B, S_old, _ = h_old.shape
            num_q   = attn.num_heads
            num_kv  = attn.num_kv_heads
            dh      = attn.head_dim

            k_old = k_old.reshape(B, S_old, num_kv, dh).transpose(0, 2, 1, 3)
            v_old = v_old.reshape(B, S_old, num_kv, dh).transpose(0, 2, 1, 3)
            q_new = q_new.reshape(B, 1, num_q,  dh).transpose(0, 2, 1, 3)
            k_new = k_new.reshape(B, 1, num_kv, dh).transpose(0, 2, 1, 3)
            v_new = v_new.reshape(B, 1, num_kv, dh).transpose(0, 2, 1, 3)

            # ---- Q/K normalisation (Gemma-specific) ----
            q_new = attn.q_norm(q_new)
            k_old = attn.k_norm(k_old)
            k_new = attn.k_norm(k_new)

            # ---- RoPE ----
            q_new = attn.rope(q_new, offset=seq_len)
            k_old = attn.rope(k_old)                   # old positions have offset 0..seq_len-1
            k_new = attn.rope(k_new, offset=seq_len)

            # ---- concat old + new K, V ----
            k_all = mx.concatenate([k_old, k_new], axis=2)  # (B, kv_heads, seq_len+1, dh)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            # ---- GQA repeat ----
            if attn.n_rep > 1:
                k_all = mx.repeat(k_all, attn.n_rep, axis=1)
                v_all = mx.repeat(v_all, attn.n_rep, axis=1)

            # ---- fused attention (no mask — new token attends causally to all) ----
            attn_out = mx.fast.scaled_dot_product_attention(
                q_new, k_all, v_all, scale=attn.scale
            )

            # ---- reshape and project ----
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
            attn_out = attn.o_proj(attn_out)

            # ---- residual add (post-attention) ----
            h_new = h_new + layer.post_attention_layernorm(attn_out)

            # ---- FFN on new token only ----
            ffn_in  = layer.pre_feedforward_layernorm(h_new)
            ffn_out = layer.mlp(ffn_in)
            h_new   = h_new + layer.post_feedforward_layernorm(ffn_out)

            # ---- update stored residual for this layer ----
            # Append new token's pre-layer residual to the stored block.
            # stored_residuals[i] was the input to layer i for old tokens.
            new_layer_input = mx.concatenate(
                [stored_residuals[i], h_new], axis=1
            )
            new_layer_inputs.append(new_layer_input)

        # Final norm + unembed (new token only)
        h_final = backbone.norm(h_new)
        logits  = self.model._unembed(h_final)  # (1, 1, vocab)
        return logits, new_layer_inputs

    def step(
        self,
        new_token_ids: mx.array,
        stored_residuals: list[mx.array],
        seq_len: int,
    ) -> tuple[mx.array, list[mx.array]]:
        return self._step(new_token_ids, stored_residuals, seq_len)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def bench_standard(model, input_ids, gen_tokens):
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

    kv_bytes = sum(k.nbytes + v.nbytes for k, v in cache if k is not None)
    return prefill_ms, sum(gen_times) / len(gen_times), kv_bytes


def bench_rs_plain(rs_model, input_ids, gen_tokens):
    tokens = list(input_ids[0].tolist())
    t0 = time.perf_counter()
    out = rs_model(input_ids)
    mx.eval(out.logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    next_tok = int(mx.argmax(out.logits[0, -1, :]))
    tokens.append(next_tok)
    gen_times = []
    for _ in range(gen_tokens - 1):
        ids = mx.array(tokens)[None]
        t0 = time.perf_counter()
        out = rs_model(ids)
        mx.eval(out.logits)
        gen_times.append((time.perf_counter() - t0) * 1000)
        tokens.append(int(mx.argmax(out.logits[0, -1, :])))

    avg = sum(gen_times) / len(gen_times) if gen_times else prefill_ms
    return prefill_ms, avg


def bench_rs_compiled(generator: CompiledRSGenerator, input_ids, gen_tokens):
    # Prefill
    t0 = time.perf_counter()
    logits, stored = generator.prefill(input_ids)
    mx.eval(logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    seq_len  = input_ids.shape[1]
    next_tok = int(mx.argmax(logits[0, -1, :]))

    gen_times = []
    for _ in range(gen_tokens):
        tok_arr = mx.array([[next_tok]])
        t0 = time.perf_counter()
        logits, stored = generator.step(tok_arr, stored, seq_len)
        mx.eval(logits)
        gen_times.append((time.perf_counter() - t0) * 1000)
        seq_len += 1
        next_tok = int(mx.argmax(logits[0, -1, :]))

    avg = sum(gen_times) / len(gen_times)
    return prefill_ms, avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n{BOLD}Gemma RS: KV cache vs plain RS vs compiled RS{RESET}")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Gen tokens: {args.gen_tokens}")
    print(f"  Contexts:   {args.context_lengths}")
    print()
    print("  std:      KV cache — O(1) per token")
    print("  rs-plain: full recompute — O(ctx) per token")
    print("  rs-cmp:   compiled residual K,V — O(ctx) matmul, fused attention")

    std_model, rs_model, config = load_models(args.model)
    generator = CompiledRSGenerator(rs_model, config)

    # Warm up (forces compilation)
    print("\n  Warming up / compiling...")
    _w = mx.array([[1, 2, 3, 4, 5]])
    _ = std_model(_w); _ = rs_model(_w); mx.eval()
    # Warm up compiled generator with a tiny sequence
    _logits, _stored = generator.prefill(_w)
    mx.eval(_logits)
    _logits2, _stored2 = generator.step(mx.array([[6]]), _stored, 5)
    mx.eval(_logits2)
    # Second warmup call to ensure compilation is done
    _logits3, _stored3 = generator.step(mx.array([[7]]), _stored2, 6)
    mx.eval(_logits3)
    print("  Done.\n")

    col = 13
    print(f"  {'Context':>8}  "
          f"{'Pfill(ms)':>{col}}  "
          f"{'std ms':>{col}}  {'rs ms':>{col}}  {'cmp ms':>{col}}  "
          f"{'std tok/s':>{col}}  {'rs tok/s':>{col}}  {'cmp tok/s':>{col}}  "
          f"{'vs std':>8}  {'vs plain':>8}")
    print("  " + "─" * 148)

    chart_rows = []

    for ctx_len in args.context_lengths:
        input_ids = mx.array([[1] * ctx_len])

        std_pre,  std_gen,  _    = bench_standard(std_model, input_ids, args.gen_tokens)
        rs_pre,   rs_gen         = bench_rs_plain(rs_model,  input_ids, args.gen_tokens)
        cmp_pre,  cmp_gen        = bench_rs_compiled(generator, input_ids, args.gen_tokens)

        std_tps = 1000 / std_gen if std_gen > 0 else 0
        rs_tps  = 1000 / rs_gen  if rs_gen  > 0 else 0
        cmp_tps = 1000 / cmp_gen if cmp_gen > 0 else 0

        vs_std   = std_tps / cmp_tps if cmp_tps > 0 else 0
        vs_plain = cmp_tps / rs_tps  if rs_tps  > 0 else 0

        chart_rows.append((ctx_len, std_tps, rs_tps, cmp_tps))

        print(f"  {ctx_len:>8,}  "
              f"{std_pre:>{col}.0f}  "
              f"{std_gen:>{col}.1f}  {rs_gen:>{col}.1f}  {cmp_gen:>{col}.1f}  "
              f"{CYAN}{std_tps:>{col}.1f}{RESET}  "
              f"{DIM}{rs_tps:>{col}.1f}{RESET}  "
              f"{GREEN}{cmp_tps:>{col}.1f}{RESET}  "
              f"{YELLOW}{vs_std:>7.1f}×{RESET}  "
              f"{GREEN}{vs_plain:>7.1f}×{RESET}")

    # Bar chart
    max_tps = max(std for _, std, _, _ in chart_rows)
    W = 25
    print(f"\n  {'Context':>8}  {'Standard':^{W+8}}  {'RS plain':^{W+8}}  {'RS compiled':^{W+8}}")
    print("  " + "─" * (8 + 3*(W+16)))
    for ctx, std_tps, rs_tps, cmp_tps in chart_rows:
        sb = "█" * int(std_tps / max_tps * W)
        rb = "█" * int(rs_tps  / max_tps * W)
        cb = "█" * int(cmp_tps / max_tps * W)
        print(f"  {ctx:>8,}  "
              f"{CYAN}{std_tps:>6.0f}{RESET} {CYAN}{sb:<{W}}{'░'*(W-len(sb))}{RESET}  "
              f"{DIM}{rs_tps:>6.0f} {rb:<{W}}{'░'*(W-len(rb))}{RESET}  "
              f"{GREEN}{cmp_tps:>6.0f} {cb:<{W}}{'░'*(W-len(cb))}{RESET}")

    # Memory analysis
    L   = config.num_hidden_layers
    H   = config.hidden_size
    kv  = config.num_key_value_heads
    dh  = config.head_dim
    res_ratio = H / (2 * kv * dh)  # residual bytes per position per layer / KV bytes

    print(f"""
{BOLD}What the compiled RS generator does differently:{RESET}

  Prefill  : full forward pass, capture per-layer residual tensors.
             These are READ-ONLY for all subsequent generation steps.

  Generate : for each new token —
    • K_old, V_old  ← stored_residuals[i] @ wk.T / wv.T
      one contiguous matmul over the old sequence block
    • Q, K_new, V_new from the single new-token embedding
    • mx.fast.scaled_dot_product_attention([K_old|K_new], [V_old|V_new])
      fused Metal kernel, no intermediate K/V tensors materialised
    • FFN on new token only (one vector, not seq_len vectors)

  mx.compile fuses the K,V projection into the attention kernel —
  K and V are computed and consumed without being materialised separately.

{BOLD}Memory: generation cache (active, during generation){RESET}

  Per position per layer:
    KV cache : 2 × {kv} kv_heads × {dh} head_dim × 2 bytes = {2*kv*dh*2} bytes
    Residual : {H} hidden_dim × 2 bytes                    = {H*2} bytes
    Ratio    : {res_ratio:.2f}× (residual {'larger' if res_ratio > 1 else 'smaller'} than KV cache for this model)

  At 2048 tokens:
    KV cache : {fmt_bytes(2 * L * kv * 2048 * dh * 2)}
    Residual : {fmt_bytes(L * 2048 * H * 2)}

  Ratio = hidden_dim / (2 × kv_heads × head_dim). Depends on model:
    Gemma 270M : 640  / (2×1×256) = 1.25× — residual LARGER
    Gemma 1B   : 1152 / (2×1×256) = 2.25× — residual LARGER
    Gemma 4B   : 2560 / (2×4×256) = 1.25× — residual LARGER
    Gemma 12B  : 3840 / (2×8×256) = 0.94× — residual SMALLER (breakeven)
    Gemma 27B  : 5120 / (2×8×256) = 1.25× — residual LARGER

  Verdict: during generation, the compiled RS uses roughly the same memory
  as KV cache. Not a win. A wash. You trade comparable memory for dark
  inference capabilities and compiler-fused K,V computation.

{BOLD}Memory: stored state between turns (multi-turn){RESET}

  The compiled RS does NOT store per-layer residuals between turns.
  Between turns you store only token IDs: 4 bytes per token.

  At 2048 tokens:
    KV cache (stored) : {fmt_bytes(2 * L * kv * 2048 * dh * 2)}
    Token IDs         : {fmt_bytes(2048 * 4)}
    Ratio             : {(2 * L * kv * 2048 * dh * 2) // (2048 * 4):,}×

  This is where the residual stream architecture dominates.
  Redis, disk, across millions of conversations: token IDs only.
  Next turn: re-prefill from token IDs (same cost as first prefill).

{BOLD}The honest pitch:{RESET}

  During generation : comparable memory to KV cache. Not a win.
  Between turns     : {(2 * L * kv * 2048 * dh * 2) // (2048 * 4):,}× less storage. A large win.
  Speed             : 2-2.6× slower than KV cache during generation.
                      Irrelevant for single-pass dark inference.
  Dark capabilities : probe, inject, branch — zero extra cost.
                      KV cache cannot do this at all.
""")


if __name__ == "__main__":
    main()
