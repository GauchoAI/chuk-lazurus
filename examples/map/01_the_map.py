#!/usr/bin/env python3
"""
Section 1 — The Map

Swap the internal map from France into an Australia prompt.
The model says Paris. Every token says Australia. The map wins.
"""

import time
import mlx.core as mx
from chuk_lazarus.inference import UnifiedPipeline


def extract_residual_at_layer(kv_gen, prompt_ids, target_layer):
    """Extract the residual stream at a specific layer."""
    residual = kv_gen.prefill_to_layer(
        mx.array(prompt_ids)[None],
        target_layer=target_layer,
    )
    mx.eval(residual)
    return residual


def main():
    MODEL = "google/gemma-3-4b-it"
    PATCH_LAYER = 26

    print()
    print("Loading model...", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    tokenizer = pipeline.tokenizer
    kv_gen = pipeline.make_engine()
    print("done.")

    france_prompt = "The capital city of France is"
    australia_prompt = "The capital city of Australia is"

    france_ids = tokenizer.encode(france_prompt, add_special_tokens=True)
    australia_ids = tokenizer.encode(australia_prompt, add_special_tokens=True)

    # ── Step 1: Normal behaviour ─────────────────────────────
    print()
    print("── Step 1: Ask the model normally ──")
    print()

    r = pipeline.generate(france_prompt, max_new_tokens=3, temperature=0.0)
    print(f"  Prompt:  \"{france_prompt}\"")
    print(f"  Answer:  {r.text.strip()}")
    print()

    r = pipeline.generate(australia_prompt, max_new_tokens=3, temperature=0.0)
    print(f"  Prompt:  \"{australia_prompt}\"")
    print(f"  Answer:  {r.text.strip()}")

    # ── Step 2: Extract the internal maps ────────────────────
    print()
    print("── Step 2: Extract the internal map for each country ──")
    print()

    france_residual = extract_residual_at_layer(kv_gen, france_ids, PATCH_LAYER)
    print(f"  France map:    extracted  (2560 dimensions)")

    australia_residual = extract_residual_at_layer(kv_gen, australia_ids, PATCH_LAYER)
    print(f"  Australia map: extracted  (2560 dimensions)")

    # ── Step 3: The swap ─────────────────────────────────────
    print()
    print("── Step 3: Swap France's map into the Australia prompt ──")
    print()
    print(f"  Prompt tokens: \"The capital city of Australia is\"")
    print(f"  Internal map:  replaced with France's")
    print()
    print("  Running...", end=" ", flush=True)

    backbone = kv_gen.backbone
    B, S = 1, len(australia_ids)
    input_ids = mx.array(australia_ids)[None]

    h = backbone.embed(input_ids)

    for i, layer in enumerate(backbone.adapted_layers):
        if i == PATCH_LAYER:
            france_last = france_residual[0, -1:]
            h = mx.concatenate([h[:, :-1, :], france_last[None, :, :]], axis=1)

        mask = backbone.prefill_mask(i, h)
        x = layer.pre_attn_norm(h)
        q, k, v = layer.project_qkv(x, B, S, offset=0)

        k_rpt = mx.repeat(k, layer.n_rep, axis=1) if layer.n_rep > 1 else k
        v_rpt = mx.repeat(v, layer.n_rep, axis=1) if layer.n_rep > 1 else v

        attn_out = mx.fast.scaled_dot_product_attention(
            q, k_rpt, v_rpt, scale=layer.head_dim ** -0.5, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        attn_out = layer.output_project(attn_out)
        h = layer.residual_add_attn(h, attn_out)

        x2 = layer.pre_ffn_norm(h)
        ff_out = layer.ffn(x2)
        h = layer.residual_add_ffn(h, ff_out)

    h = backbone.final_norm(h)
    logits = backbone.unembed(h)
    mx.eval(logits)

    last_logits = logits[0, -1]
    top_idx = int(mx.argmax(last_logits).item())
    top_token = tokenizer.decode([top_idx]).strip()
    prob = mx.softmax(last_logits)[top_idx].item()

    print("done.")

    # ── Result ───────────────────────────────────────────────
    print()
    print("── Result ──")
    print()
    print(f"  Prompt:     \"The capital city of Australia is\"")
    print(f"  Map:        France")
    print(f"  Answer:     {top_token}  ({prob:.0%} confidence)")
    print()

    if "paris" in top_token.lower() or "Par" in top_token:
        print("  Every token says Australia. The model says Paris.")
        print("  The model reads the map, not the tokens.")
    print()


if __name__ == "__main__":
    main()
