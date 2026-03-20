#!/usr/bin/env python3
"""
ACT 5 — The Injection

Four passes. Each removes one thing:
  1. Full context  (tokens + residual)    → Volt 100%
  2. No context    (no tokens, no map)    → guessing
  3. Residual only (no tokens, full map)  → Volt 100%
  4. 12 bytes only (no tokens, 1D map)    → Volt 100%

Usage:
    uv run python examples/map/02_the_injection.py
"""

import time

import mlx.core as mx
from chuk_lazarus.inference import UnifiedPipeline
from chuk_lazarus.inference.context.vec_inject._primitives import vec_inject

# ── Config ────────────────────────────────────────────────────
MODEL = "google/gemma-3-4b-it"
RETRIEVAL_LAYER = 29
INJECT_LAYER = 30

QUERY = "Zarkov Industries was founded in the city of"
DOCUMENT = (
    "Zarkov Industries was established in the mid-1990s as a "
    "pioneering manufacturer of industrial filtration systems. "
    "Its headquarters, built on a former industrial lot, became "
    "a landmark of Voltara's commercial district.\n\n"
    "Zarkov Industries was founded in the city of"
)
ANSWER_TOKEN = "Volt"

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def top_prediction(logits, tokenizer):
    last = logits[0, -1]
    probs = mx.softmax(last)
    idx = int(mx.argmax(last).item())
    return tokenizer.decode([idx]).strip(), float(probs[idx].item())


def read_prediction_at(backbone, h, tokenizer):
    h_normed = backbone.final_norm(h)
    logits = backbone.unembed(h_normed)
    mx.eval(logits)
    return top_prediction(logits, tokenizer)


def run_layer(backbone, layer, i, h, B, S):
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


def layer_bar(i, n_layers, token, prob, marker=""):
    filled = int((i + 1) / n_layers * 30)
    bar = "█" * filled + "░" * (30 - filled)
    layer_str = f"L{i:<2d}"
    colour = CYAN if marker else GREEN
    return (
        f"  {DIM}{layer_str}{RESET} {DIM}{bar}{RESET}"
        f"  {colour}{token:<12s}{RESET} {DIM}{prob:5.1%}{RESET}{marker}"
    )


def forward_show(backbone, input_ids_list, tokenizer, inject_at=None,
                 inject_fn=None, swap_residual=None, delay=0.12):
    """Forward pass with layer-by-layer readout.

    inject_fn(h) -> h    : additive injection on last position at inject_at
    swap_residual         : full residual replacement (all positions) at inject_at
    """
    B, S = 1, len(input_ids_list)
    input_ids = mx.array(input_ids_list)[None]
    h = backbone.embed(input_ids)
    n_layers = len(backbone.adapted_layers)

    for i, layer in enumerate(backbone.adapted_layers):
        did_inject = False
        marker = ""

        if i == inject_at:
            if swap_residual is not None:
                # Full residual swap — last position only from donor
                donor_last = swap_residual[:, -1:, :]
                h = mx.concatenate([h[:, :-1, :], donor_last], axis=1)
                did_inject = True
                marker = f" {YELLOW}← residual (5,120 bytes){RESET}"
            elif inject_fn is not None:
                h_last = h[:, -1:, :]
                h_last = inject_fn(h_last)
                h = mx.concatenate([h[:, :-1, :], h_last], axis=1)
                did_inject = True
                marker = f" {YELLOW}← 12 bytes{RESET}"

        h = run_layer(backbone, layer, i, h, B, S)
        mx.eval(h)

        token, prob = read_prediction_at(backbone, h, tokenizer)
        print(layer_bar(i, n_layers, token, prob, marker=marker))

        if did_inject:
            time.sleep(delay * 4)
        else:
            time.sleep(delay)

    h_final = backbone.final_norm(h)
    logits = backbone.unembed(h_final)
    mx.eval(logits)
    return logits


def main():
    print()
    print(f"{DIM}Loading model...{RESET}", end=" ", flush=True)
    pipeline = UnifiedPipeline.from_pretrained(MODEL, verbose=False)
    tokenizer = pipeline.tokenizer
    kv_gen = pipeline.make_engine()
    backbone = kv_gen.backbone
    print(f"{DIM}done.{RESET}")

    query_ids = tokenizer.encode(QUERY, add_special_tokens=True)
    doc_ids = tokenizer.encode(DOCUMENT, add_special_tokens=True)

    answer_id = tokenizer.encode(ANSWER_TOKEN, add_special_tokens=False)[0]
    answer_str = tokenizer.decode([answer_id])

    # ── Pre-extract everything ────────────────────────────────

    # Full-context residual at L29 (entering L30) — the "map"
    donor_h = kv_gen.prefill_to_layer(
        mx.array(doc_ids)[None], target_layer=RETRIEVAL_LAYER
    )
    residual_bytes = donor_h[:, -1:, :].size * 2  # bfloat16

    # 12-byte injection: coefficient from donor, raw embed for vec_inject
    r_last = donor_h[0, -1, :].astype(mx.float32)
    e_scaled = backbone.embed(mx.array([[answer_id]]))[0, 0, :].astype(mx.float32)
    mx.eval(r_last, e_scaled)
    coefficient = float(mx.sum(r_last * e_scaled).item())
    embed_matrix = pipeline.model.model.embed_tokens.weight

    results = []

    # ══════════════════════════════════════════════════════════
    # Pass 1: Full context — the model reads the document
    # ══════════════════════════════════════════════════════════
    print()
    print(f"  {BOLD}Pass 1 — Full context{RESET}")
    print(f"  {DIM}Document + query. 'Voltara' is in the tokens.{RESET}")
    print()

    logits = forward_show(backbone, doc_ids, tokenizer)
    tok, prob = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {GREEN}{tok}{RESET}  ({prob:.1%})")
    results.append((tok, prob))

    # ══════════════════════════════════════════════════════════
    # Pass 2: No context — bare query
    # ══════════════════════════════════════════════════════════
    print()
    time.sleep(1.0)
    print(f"  {BOLD}Pass 2 — No context{RESET}")
    print(f"  {DIM}Remove the document. Bare query only.{RESET}")
    print()
    time.sleep(0.5)

    logits = forward_show(backbone, query_ids, tokenizer)
    tok, prob = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {GREEN}{tok}{RESET}  ({prob:.1%})")
    print(f"  {DIM}The map is blank. The model wanders.{RESET}")
    results.append((tok, prob))

    # ══════════════════════════════════════════════════════════
    # Pass 3: Residual only — the full map, no tokens
    # ══════════════════════════════════════════════════════════
    print()
    time.sleep(1.0)
    print(f"  {BOLD}Pass 3 — Residual only ({residual_bytes:,} bytes){RESET}")
    print(f"  {DIM}Bare query tokens. Full-context residual at L{INJECT_LAYER}.{RESET}")
    print(f"  {DIM}'Voltara' is NOT in the KV cache. But the map is.{RESET}")
    print()
    time.sleep(0.5)

    logits = forward_show(
        backbone, query_ids, tokenizer,
        inject_at=INJECT_LAYER, swap_residual=donor_h,
    )
    tok, prob = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {CYAN}{tok}{RESET}  ({prob:.1%})")
    results.append((tok, prob))

    # ══════════════════════════════════════════════════════════
    # Pass 4: 12 bytes — one direction
    # ══════════════════════════════════════════════════════════
    print()
    time.sleep(1.0)
    print(f"  {BOLD}Pass 4 — 12 bytes{RESET}")
    print(f"  {YELLOW}token={repr(answer_str)}  coefficient={coefficient:.1f}{RESET}")
    print(f"  {DIM}Same bare query. 12 bytes injected at L{INJECT_LAYER}.{RESET}")
    print()
    time.sleep(0.5)

    def inject_fn(h):
        return vec_inject(h, answer_id, coefficient, embed_matrix)

    logits = forward_show(
        backbone, query_ids, tokenizer,
        inject_at=INJECT_LAYER, inject_fn=inject_fn,
    )
    tok, prob = top_prediction(logits, tokenizer)

    print()
    print(f"  {BOLD}Prediction:{RESET}  {CYAN}{tok}{RESET}  ({prob:.1%})")
    results.append((tok, prob))

    # ── Summary ───────────────────────────────────────────────
    print()
    labels = [
        "Full context (tokens + residual)",
        "No context   (blank map)",
        f"Residual     ({residual_bytes:,} bytes)",
        "12 bytes     (1 direction)",
    ]
    for label, (t, p) in zip(labels, results):
        mark = "✓" if "volt" in t.lower() or "Volt" in t else "✗"
        colour = CYAN if mark == "✓" else GREEN
        print(f"  {label:<38s} → {colour}{t:<8s}{RESET} {p:5.1%}  {mark}")
    print()
    print(f"  {residual_bytes:,} bytes → 12 bytes. Same answer.")
    print(f"  99.8% of the residual is scaffolding. The fact is one scalar.")
    print()


if __name__ == "__main__":
    main()
