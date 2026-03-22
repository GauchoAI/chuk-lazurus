"""Extract L26 residuals from interval positions and test routing.

The Lazarus experiments showed query identity crystallizes at L26 in a 6-9D
subspace with 8.3° angular separation between different queries. All prior
routing attempts used the final pre-norm residual (post-L33) which is dominated
by output formatting, not query routing.

This experiment:
1. Re-extract interval residuals at L26 (not final layer)
2. Compute PCA from diverse query directions at L26
3. Project interval residuals into the query subspace
4. Test if routing becomes query-sensitive AND content-accurate
"""

import sys
import time

import mlx.core as mx
import numpy as np


def extract_l26_residual(kv_gen, input_ids, target_layer=26):
    """Run a forward pass and capture the residual at layer target_layer.

    Returns the hidden state AFTER layer target_layer's residual additions,
    BEFORE layer target_layer+1.  Shape: (1, S, hidden_size).
    """
    backbone = kv_gen.backbone
    B, S = input_ids.shape
    h = backbone.embed(input_ids)

    for i, layer in enumerate(backbone.adapted_layers):
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

        if i == target_layer:
            # Capture h after this layer, before continuing
            mx.eval(h)
            return h

    return h  # fallback: return final


def main():
    from chuk_lazarus.inference import UnifiedPipeline
    from chuk_lazarus.inference.context import CheckpointLibrary
    from chuk_lazarus.inference.context.kv_generator import make_kv_generator

    lib_path = "/Users/christopherhay/chris-source/apollo-demo/apollo11_ctx_4k"
    model_id = "google/gemma-3-4b-it"
    TARGET_LAYER = 26
    N_SAMPLES = 8  # interval samples per window

    print(f"Loading library...", file=sys.stderr)
    lib = CheckpointLibrary(lib_path)

    print(f"Loading model...", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(model_id, verbose=False)
    kv_gen = make_kv_generator(pipeline.model)
    tokenizer = pipeline.tokenizer

    # ── Step 1: Extract L26 interval residuals for all windows ──
    print(f"\nExtracting L{TARGET_LAYER} interval residuals ({N_SAMPLES}/window)...", file=sys.stderr)

    all_vecs = []  # (wid, sample_idx, np_vector)
    t0 = time.time()

    for wid in range(lib.num_windows):
        w_tokens = lib.get_window_tokens(wid)
        w_len = len(w_tokens)
        w_ids = mx.array(w_tokens)[None]

        # Get L26 residual for full window
        h_l26 = extract_l26_residual(kv_gen, w_ids, target_layer=TARGET_LAYER)
        # h_l26 shape: (1, w_len, 2560)

        # Sample at evenly-spaced positions
        positions = [int(i * (w_len - 1) / max(N_SAMPLES - 1, 1)) for i in range(N_SAMPLES)]
        for si, pos in enumerate(positions):
            vec = h_l26[0, pos, :]
            mx.eval(vec)
            all_vecs.append((wid, si, np.array(vec.tolist(), dtype=np.float32)))

        elapsed = time.time() - t0
        rate = (wid + 1) / elapsed if elapsed > 0 else 0
        eta = (lib.num_windows - wid - 1) / rate if rate > 0 else 0
        print(f"\r  Window {wid+1}/{lib.num_windows}  {rate:.1f} w/s  ETA {eta:.0f}s",
              end="", file=sys.stderr, flush=True)

    print(f"\n  Done: {len(all_vecs)} L{TARGET_LAYER} residuals in {time.time()-t0:.0f}s", file=sys.stderr)

    # ── Step 2: Extract L26 query residuals ──────────────────
    queries = {
        "sports": "What sport and teams were discussed during the mission?",
        "aldrin": "What did Buzz Aldrin say when he first stepped on the moon?",
        "docking": "What were the docking procedures used?",
        "cincinnati": "What were the baseball scores from Cincinnati?",
    }

    query_vecs = {}
    for name, text in queries.items():
        if hasattr(tokenizer, "apply_chat_template"):
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer.encode(chat, add_special_tokens=False)
        else:
            ids = tokenizer.encode(text, add_special_tokens=False)

        q_arr = mx.array(ids)[None]
        h_l26 = extract_l26_residual(kv_gen, q_arr, target_layer=TARGET_LAYER)
        # Take last position as the query vector
        q_vec = h_l26[0, -1, :]
        mx.eval(q_vec)
        query_vecs[name] = np.array(q_vec.tolist(), dtype=np.float32)

    print(f"\nExtracted {len(query_vecs)} query L{TARGET_LAYER} residuals", file=sys.stderr)

    # ── Step 3: PCA on interval residuals ────────────────────
    X = np.stack([v for _, _, v in all_vecs], axis=0)  # (728, 2560)
    mean = X.mean(axis=0)
    X_centered = X - mean

    print("Computing PCA...", file=sys.stderr)
    U, S_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained = (S_vals ** 2) / np.sum(S_vals ** 2)

    print(f"  PC0: {explained[0]*100:.1f}%, PC1: {explained[1]*100:.1f}%, PC2: {explained[2]*100:.1f}%", file=sys.stderr)

    # ── Step 4: Sweep subspace ranges ────────────────────────
    wid_map = [(wid, si) for wid, si, _ in all_vecs]

    def route_in_subspace(q_vec, basis):
        q_proj = (q_vec - mean) @ basis.T
        all_proj = X_centered @ basis.T

        q_norm = np.linalg.norm(q_proj)
        if q_norm < 1e-10:
            return [(wid, 0.0) for wid, _ in wid_map]

        all_norms = np.linalg.norm(all_proj, axis=1)
        cosines = (all_proj @ q_proj) / (all_norms * q_norm + 1e-10)

        per_window = {}
        for idx, (wid, si) in enumerate(wid_map):
            c = cosines[idx]
            if wid not in per_window or c > per_window[wid]:
                per_window[wid] = c

        return sorted(per_window.items(), key=lambda x: -x[1])

    ranges = [
        (0, 8, "PC 0-7"),
        (0, 2560, "Full 2560D"),
        (4, 12, "PC 4-11"),
        (8, 16, "PC 8-15"),
        (1, 9, "PC 1-8 (skip structural)"),
        (2, 10, "PC 2-9"),
        (8, 24, "PC 8-23"),
        (16, 32, "PC 16-31"),
    ]

    print(f"\n{'='*80}")
    print(f"L{TARGET_LAYER} SUBSPACE ROUTING SWEEP")
    print(f"{'='*80}")

    for start, end, label in ranges:
        basis = Vt[start:end]
        var = sum(explained[start:min(end, len(explained))]) * 100
        print(f"\n--- {label} ({end-start} dims, var={var:.1f}%) ---")

        for qname, qvec in query_vecs.items():
            scores = route_in_subspace(qvec, basis)
            top5 = [(wid, f"{s:.4f}") for wid, s in scores[:5]]
            print(f"  {qname:>12}: {top5}")

        # Query sensitivity check
        sets = {}
        for qname, qvec in query_vecs.items():
            scores = route_in_subspace(qvec, basis)
            sets[qname] = set(wid for wid, _ in scores[:3])

        sports_aldrin = sets["sports"] & sets["aldrin"]
        sports_cincy = sets["sports"] & sets["cincinnati"]
        print(f"  sports∩aldrin: {len(sports_aldrin)}/3  sports∩cincy: {len(sports_cincy)}/3")

    # ── Step 5: Where does window 76 rank? ───────────────────
    print(f"\n{'='*80}")
    print("WINDOW 76 (sports) RANK BY QUERY AND SUBSPACE")
    print(f"{'='*80}")
    for start, end, label in ranges:
        basis = Vt[start:end]
        for qname in ["sports", "cincinnati"]:
            scores = route_in_subspace(query_vecs[qname], basis)
            rank_76 = next((i for i, (wid, _) in enumerate(scores) if wid == 76), -1) + 1
            score_76 = next((s for wid, s in scores if wid == 76), 0.0)
            print(f"  {label:>25} | {qname:>12}: rank={rank_76:>3}, score={score_76:.4f}")


if __name__ == "__main__":
    main()
