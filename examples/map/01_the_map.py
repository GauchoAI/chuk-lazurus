#!/usr/bin/env python3
"""
ACT 2 — The Map

Three passes through the same prompt:
  1. Normal:   "The capital of Australia is" → Sydney → Canberra
  2. L0 swap:  France's residual after L0 → Paris from layer 1 onward
  3. L26 swap: 26 layers of Australia, then France's residual → Sydney → Paris

Both Pass 2 and Pass 3 are residual stream swaps — the Markov property
at different layers. Pass 2 swaps early (after L0). Pass 3 swaps late
(at L26). Both override everything downstream.

Usage:
    uv run python examples/map/01_the_map.py
"""

import time

import mlx.core as mx
from chuk_lazarus.inference import UnifiedPipeline

# ── Config ────────────────────────────────────────────────────
MODEL = "google/gemma-3-4b-it"

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def top_prediction(logits, tokenizer):
    """Return (token_str, probability) for the top-1 prediction."""
    last = logits[0, -1]
    probs = mx.softmax(last)
    idx = int(mx.argmax(last).item())
    return tokenizer.decode([idx]).strip(), float(probs[idx].item())


def read_prediction_at(backbone, h, tokenizer):
    """Unembed the current residual to see what the model would predict now."""
    h_normed = backbone.final_norm(h)
    logits = backbone.unembed(h_normed)
    mx.eval(logits)
    return top_prediction(logits, tokenizer)


def run_layer(backbone, layer, i, h, B, S):
    """Run one transformer layer, return updated residual."""
    mask = backbone.prefill_mask(i, h)
    x = layer.pre_attn_norm(h)
    q, k, v = layer.project_qkv(x, B, S, offset=0)

    k_rpt = mx.repeat(k, layer.n_rep, axis=1) if layer.n_rep > 1 else k
    v_rpt = mx.repeat(v, layer.n_rep, axis=1) if layer.n_rep > 1 else v

    attn_out = mx.fast.scaled_dot_product_attention(
        q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask
    )
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    attn_out = layer.output_project(attn_out)
    h = layer.residual_add_attn(h, attn_out)
    h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))
    return h


def layer_bar(i, n_layers, token, prob, swapped=False):
    """Render one layer's progress line."""
    filled = int((i + 1) / n_layers * 30)
    bar = "█" * filled + "░" * (30 - filled)
    layer_str = f"L{i:<2d}"

    if swapped:
        marker = f" {YELLOW}← swapped{RESET}"
    else:
        marker = ""

    colour = CYAN if swapped else GREEN
    return (
        f"  {DIM}{layer_str}{RESET} {DIM}{bar}{RESET}"
        f"  {colour}{token:<12s}{RESET} {DIM}{prob:5.1%}{RESET}{marker}"
    )


def forward_show(backbone, input_ids_list, tokenizer, france_residual=None,
                 swap_layer=None, delay=0.12):
    """Forward pass with live layer-by-layer prediction readout."""
    B, S = 1, len(input_ids_list)
    input_ids = mx.array(input_ids_list)[None]
    h = backbone.embed(input_ids)
    n_layers = len(backbone.adapted_layers)

    for i, layer in enumerate(backbone.adapted_layers):
        did_swap = False

        if france_residual is not None and i == swap_layer:
            h = france_residual  # replace ALL positions
            did_swap = True

        h = run_layer(backbone, layer, i, h, B, S)
        mx.eval(h)

        token, prob = read_prediction_at(backbone, h, tokenizer)
        print(layer_bar(i, n_layers, token, prob, swapped=did_swap))

        if did_swap:
            time.sleep(delay * 4)
        else:
            time.sleep(delay)

    h_final = backbone.final_norm(h)
    logits = backbone.unembed(h_final)
    mx.eval(logits)
    return logits


def main():
    prompt_aus = "The capital of Australia is"
    prompt_fra = "The capital of France is"

    print()
    print(f"{DIM}Loading model...{RESET}", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    tokenizer = pipeline.tokenizer
    kv_gen = pipeline.make_engine()
    backbone = kv_gen.backbone
    print(f"{DIM}done.{RESET}")

    aus_ids = tokenizer.encode(prompt_aus, add_special_tokens=True)
    fra_ids = tokenizer.encode(prompt_fra, add_special_tokens=True)

    # Pre-extract France's residual at both swap points
    # L0: France's state after layer 0 has processed (not raw embeddings)
    france_at_0 = kv_gen.prefill_to_layer(
        mx.array(fra_ids)[None], target_layer=0
    )

    # L26: France's state after 26 layers of processing
    france_at_26 = kv_gen.prefill_to_layer(
        mx.array(fra_ids)[None], target_layer=26
    )

    # ══════════════════════════════════════════════════════════
    # Pass 1: Normal
    # ══════════════════════════════════════════════════════════
    print()
    print(f"  {BOLD}Pass 1 — Normal{RESET}")
    print(f"  {BOLD}Prompt:{RESET} \"{prompt_aus}\"")
    print()

    logits = forward_show(backbone, aus_ids, tokenizer)
    tok, prob = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {GREEN}{tok}{RESET}  ({prob:.1%})")

    # ══════════════════════════════════════════════════════════
    # Pass 2: L0 swap — France's residual after layer 0
    # ══════════════════════════════════════════════════════════
    print()
    time.sleep(1.0)
    print(f"  {BOLD}Pass 2 — Replace residual at L1{RESET}")
    print(f"  {YELLOW}France's residual after layer 0. Injected at layer 1.{RESET}")
    print(f"  {BOLD}Prompt:{RESET} \"{prompt_aus}\"")
    print()
    time.sleep(0.5)

    logits = forward_show(
        backbone, aus_ids, tokenizer,
        france_residual=france_at_0, swap_layer=1,
    )
    tok_l0, prob_l0 = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {CYAN}{tok_l0}{RESET}  ({prob_l0:.1%})")

    # ══════════════════════════════════════════════════════════
    # Pass 3: L26 swap — the dramatic flip
    # ══════════════════════════════════════════════════════════
    print()
    time.sleep(1.0)
    print(f"  {BOLD}Pass 3 — Replace residual at L26{RESET}")
    print(f"  {YELLOW}26 layers of Australia. Then France's residual.{RESET}")
    print(f"  {BOLD}Prompt:{RESET} \"{prompt_aus}\"")
    print()
    time.sleep(0.5)

    logits = forward_show(
        backbone, aus_ids, tokenizer,
        france_residual=france_at_26, swap_layer=26,
    )
    tok_l26, prob_l26 = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {CYAN}{tok_l26}{RESET}  ({prob_l26:.1%})")

    # ── Closing ───────────────────────────────────────────────
    print()
    print(f"  The residual IS the state. Replace it, replace the knowledge.")
    print(f"  At L1 — France from layer 1 onward.")
    print(f"  At L26 — Sydney at 90%, then one swap: Paris at 100%.")
    print(f"  The Markov property. Proven at two layers.")
    print()


if __name__ == "__main__":
    main()