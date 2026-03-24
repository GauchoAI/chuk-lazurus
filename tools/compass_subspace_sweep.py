"""Sweep PCA component ranges to find the compass subspace.

Loads interval residuals, computes PCA, projects query + window residuals
into various component ranges, measures if routing becomes query-sensitive.

The hypothesis: structural dominance lives in the top PCs. Content signal
lives in components ~5-15. Projecting into that range and computing cosine
similarity should produce query-sensitive routing.
"""

import sys
import time

import mlx.core as mx
import numpy as np


def main():
    from chuk_lazarus.inference import UnifiedPipeline
    from chuk_lazarus.inference.context import CheckpointLibrary
    from chuk_lazarus.inference.context.kv_generator import make_kv_generator

    lib_path = "/Users/christopherhay/chris-source/apollo-demo/apollo11_ctx_4k"
    model_id = "google/gemma-3-4b-it"

    print("Loading library...", file=sys.stderr)
    lib = CheckpointLibrary(lib_path)

    print("Loading model...", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(model_id, verbose=False)
    kv_gen = make_kv_generator(pipeline.model)
    tokenizer = pipeline.tokenizer

    # ── Load all interval residuals ──────────────────────────
    assert lib.has_interval_residuals, "No interval residuals"
    n_samples = lib.interval_samples_per_window
    n_windows = lib.num_windows

    all_vecs = []  # (n_windows * n_samples, hidden_dim)
    wid_map = []   # (wid, sample_idx) for each row
    for wid in range(n_windows):
        for si, res in enumerate(lib.get_interval_residuals(wid)):
            vec = res.reshape(-1)
            all_vecs.append(np.array(vec.tolist(), dtype=np.float32))
            wid_map.append((wid, si))

    X = np.stack(all_vecs, axis=0)  # (728, hidden_dim)
    print(f"Loaded {X.shape[0]} interval residuals, dim={X.shape[1]}", file=sys.stderr)

    # ── Compute PCA ──────────────────────────────────────────
    mean = X.mean(axis=0)
    X_centered = X - mean

    print("Computing PCA (SVD)...", file=sys.stderr)
    t0 = time.time()
    # Economy SVD — we only need top ~50 components
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    print(f"SVD done in {time.time() - t0:.1f}s", file=sys.stderr)

    # Explained variance
    explained = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(explained)
    print("\nExplained variance by component:", file=sys.stderr)
    for i in range(min(30, len(explained))):
        print(f"  PC{i:>2}: {explained[i]*100:5.2f}%  (cumulative: {cum_var[i]*100:5.1f}%)", file=sys.stderr)

    # ── Encode two queries ───────────────────────────────────
    queries = {
        "sports": "What sport and teams were discussed during the mission?",
        "aldrin": "What did Buzz Aldrin say when he first stepped on the moon?",
    }

    query_residuals = {}
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
        _logits, _kv, q_res = kv_gen.prefill_with_residual(q_arr)
        mx.eval(q_res)
        query_residuals[name] = np.array(q_res.reshape(-1).tolist(), dtype=np.float32)

    # ── Sweep component ranges ───────────────────────────────
    # Project into PC[start:end], compute cosine, report top-3 windows

    def cosine_in_subspace(q_vec, all_vecs, basis, mean):
        """Project into subspace and compute cosine similarity."""
        q_proj = (q_vec - mean) @ basis.T  # (n_components,)
        all_proj = (all_vecs - mean) @ basis.T  # (728, n_components)

        q_norm = np.linalg.norm(q_proj)
        if q_norm < 1e-10:
            return [(wid, 0.0) for wid, _ in wid_map]

        # Per-sample cosine
        all_norms = np.linalg.norm(all_proj, axis=1)
        dots = all_proj @ q_proj
        cosines = dots / (all_norms * q_norm + 1e-10)

        # Aggregate: max cosine per window
        per_window = {}
        for idx, (wid, si) in enumerate(wid_map):
            c = cosines[idx]
            if wid not in per_window or c > per_window[wid]:
                per_window[wid] = c

        scores = sorted(per_window.items(), key=lambda x: -x[1])
        return scores

    # Test ranges
    ranges = [
        (0, 8, "PC 0-7 (top structural)"),
        (0, 16, "PC 0-15"),
        (4, 12, "PC 4-11"),
        (8, 16, "PC 8-15"),
        (8, 24, "PC 8-23"),
        (16, 32, "PC 16-31"),
        (0, 32, "PC 0-31"),
        (4, 20, "PC 4-19"),
        (2, 10, "PC 2-9"),
        (1, 8, "PC 1-7 (skip top structural)"),
    ]

    print("\n" + "=" * 80)
    print("SUBSPACE ROUTING SWEEP")
    print("=" * 80)

    for start, end, label in ranges:
        basis = Vt[start:end]  # (n_components, hidden_dim)
        print(f"\n--- {label} ({end-start} dims, var={sum(explained[start:end])*100:.1f}%) ---")

        for qname, qvec in query_residuals.items():
            scores = cosine_in_subspace(qvec, X, basis, mean)
            top5 = [(wid, f"{s:.4f}") for wid, s in scores[:5]]
            print(f"  {qname:>8}: top-5 = {top5}")

        # Check if top-3 differ between queries
        sports_top3 = set(wid for wid, _ in cosine_in_subspace(query_residuals["sports"], X, basis, mean)[:3])
        aldrin_top3 = set(wid for wid, _ in cosine_in_subspace(query_residuals["aldrin"], X, basis, mean)[:3])
        overlap = sports_top3 & aldrin_top3
        diff = "SAME" if sports_top3 == aldrin_top3 else f"DIFFERENT ({len(overlap)} overlap)"
        print(f"  Top-3 overlap: {diff}")

    # ── Also check: where does window 76 (sports) rank? ──────
    print("\n" + "=" * 80)
    print("WHERE DOES WINDOW 76 (sports content) RANK?")
    print("=" * 80)
    for start, end, label in ranges:
        basis = Vt[start:end]
        scores = cosine_in_subspace(query_residuals["sports"], X, basis, mean)
        rank_76 = next((i for i, (wid, _) in enumerate(scores) if wid == 76), -1)
        score_76 = next((s for wid, s in scores if wid == 76), 0.0)
        print(f"  {label:>35}: window 76 rank = {rank_76 + 1:>3}, score = {score_76:.4f}")


if __name__ == "__main__":
    main()
