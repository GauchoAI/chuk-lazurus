"""Two-pass speculative routing — the model's hallucination IS the signal."""

from __future__ import annotations

import sys
import time


def two_pass_generate(
    lib,
    kv_gen,
    prompt_ids: list[int],
    prompt_text: str,
    tokenizer,
    engine,
    max_new_tokens: int = 200,
    speculative_tokens: int = 10,
    top_k: int = 3,
    temperature: float = 0.0,
) -> dict:
    """Two-pass: generate N tokens, take residual, compare vs checkpoints, replay.

    1. Generate speculative_tokens without context
    2. Extract residual at each generated token
    3. Compare each step-residual against 91 checkpoint residuals
    4. Print the routing table at every step so we can see where the
       compass points as the model transitions from format to content
    5. Route using the residual at the final step
    6. Replay top-k windows, regenerate with context
    """
    import mlx.core as mx

    last_wid = lib.num_windows - 1

    use_interval = lib.has_interval_residuals
    if not use_interval and not lib.has_residuals:
        print("  Error: library has no residuals for twopass routing", file=sys.stderr)
        return {
            "tokens": [],
            "speculative_text": "",
            "selected_windows": [],
            "residual_scores": [],
            "source": "error",
            "context_tokens": 0,
        }

    # ── Load residuals (from npz, instant) ───────────────────
    if use_interval:
        # Interval residuals: multiple interior samples per window
        n_samples = lib.interval_samples_per_window
        # Flatten all interval residuals into a single list with window mapping
        all_vecs_raw = []  # list of (wid, sample_idx, vec)
        for wid in range(lib.num_windows):
            for si, res in enumerate(lib.get_interval_residuals(wid)):
                all_vecs_raw.append((wid, si, res.reshape(-1).astype(mx.float32)))

        all_flat = [v for _, _, v in all_vecs_raw]
        stacked = mx.stack(all_flat, axis=0)
        mean_vec = mx.mean(stacked, axis=0)
        all_centered = [v - mean_vec for v in all_flat]
        all_norms = [mx.sqrt(mx.sum(v * v)) for v in all_centered]

        print(
            f"  Loaded {len(all_flat)} interval residuals "
            f"({n_samples} per window × {lib.num_windows} windows)",
            file=sys.stderr,
        )

        def _rank_residual(res_vec):
            """Cosine similarity against all interval residuals, aggregated per window."""
            v = res_vec.reshape(-1).astype(mx.float32) - mean_vec
            vn = mx.sqrt(mx.sum(v * v))
            # Score each interval sample
            per_window_max: dict[int, float] = {}
            for idx, (wid, si, _) in enumerate(all_vecs_raw):
                cos = (mx.sum(v * all_centered[idx]) / (vn * all_norms[idx] + 1e-8)).item()
                # Take MAX across samples within a window — the best-matching interior point
                if wid not in per_window_max or cos > per_window_max[wid]:
                    per_window_max[wid] = cos
            scores = list(per_window_max.items())
            scores.sort(key=lambda x: -x[1])
            return scores
    else:
        # Boundary residuals only (fallback)
        window_vecs_raw = []
        for wid in range(lib.num_windows):
            window_vecs_raw.append(lib.get_residual(wid).reshape(-1).astype(mx.float32))

        stacked = mx.stack(window_vecs_raw, axis=0)
        mean_vec = mx.mean(stacked, axis=0)
        window_vecs = [v - mean_vec for v in window_vecs_raw]
        window_norms = [mx.sqrt(mx.sum(v * v)) for v in window_vecs]

        def _rank_residual(res_vec):
            v = res_vec.reshape(-1).astype(mx.float32) - mean_vec
            vn = mx.sqrt(mx.sum(v * v))
            scores = []
            for wid in range(lib.num_windows):
                cos = (mx.sum(v * window_vecs[wid]) / (vn * window_norms[wid] + 1e-8)).item()
                scores.append((wid, cos))
            scores.sort(key=lambda x: -x[1])
            return scores

    # ── Pass 1: Generate N tokens, capture residual at each step ─
    t0 = time.time()
    q_ids = mx.array(prompt_ids)[None]

    # Prefill query — get residual at the last query token
    logits, spec_kv, query_residual = kv_gen.prefill_with_residual(q_ids)
    mx.eval(logits, query_residual)

    # Show query residual ranking
    q_scores = _rank_residual(query_residual)
    print(
        f"  Step 0 (query residual): top-3 = [{q_scores[0][0]}, {q_scores[1][0]}, {q_scores[2][0]}]",
        file=sys.stderr,
    )

    spec_tokens = []
    seq_len = len(prompt_ids)
    eos_id = tokenizer.eos_token_id
    step_scores = None  # will hold scores at final step

    for step in range(speculative_tokens):
        if temperature == 0.0:
            next_tok = int(mx.argmax(logits[0, -1, :]).item())
        else:
            scaled = logits[0, -1, :] / temperature
            next_tok = int(mx.random.categorical(scaled[None]).item())

        if eos_id is not None and next_tok == eos_id:
            break
        spec_tokens.append(next_tok)

        # Step and capture residual
        logits, spec_kv, step_residual = kv_gen.extend_with_residual(
            mx.array([[next_tok]]), spec_kv, abs_start=seq_len
        )
        mx.eval(logits, step_residual)
        seq_len += 1

        # Rank this step's residual
        step_scores = _rank_residual(step_residual)
        tok_text = tokenizer.decode([next_tok], skip_special_tokens=True)
        top3 = [(wid, f"{s:+.4f}") for wid, s in step_scores[:3]]
        print(
            f"  Step {step + 1:>2} tok={next_tok:>6} '{tok_text}': top-3 = {top3}",
            file=sys.stderr,
        )

    spec_text = tokenizer.decode(spec_tokens, skip_special_tokens=True)
    pass1_ms = (time.time() - t0) * 1000
    print(f"  Pass 1 ({len(spec_tokens)} tokens, {pass1_ms:.0f}ms): {spec_text}", file=sys.stderr)

    # ── Route using final step residual ──────────────────────
    if step_scores is None:
        step_scores = q_scores  # fallback if no tokens generated

    selected = [wid for wid, _ in step_scores[:top_k]]
    if last_wid not in selected:
        selected.append(last_wid)

    # Print full routing table
    print(f"  Routing (step-{len(spec_tokens)} residual):", file=sys.stderr)
    show_n = max(top_k + 2, 5)
    for i, (wid, score) in enumerate(step_scores[:show_n]):
        marker = " *" if wid in selected else ""
        w = lib.windows[wid]
        print(
            f"    window {wid:>2} (score={score:+.4f}){marker}  {w.preview[:50]}",
            file=sys.stderr,
        )
        if i == top_k - 1 and top_k < len(step_scores):
            print(f"    {'─' * 60}", file=sys.stderr)

    all_scores = [s for _, s in step_scores]
    if all_scores:
        print(
            f"  Score range: min={min(all_scores):+.4f} max={max(all_scores):+.4f} "
            f"spread={max(all_scores) - min(all_scores):.2e}",
            file=sys.stderr,
        )

    # ── Pass 2: Grounded generation ──────────────────────────
    context_kv = engine._make_empty_kv()
    seq_len = 0

    for wid in sorted(selected):
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]
        t0_w = time.time()
        if seq_len == 0:
            _logits, context_kv = kv_gen.prefill(w_ids)
        else:
            _logits, context_kv = kv_gen.extend(w_ids, context_kv, abs_start=seq_len)
        mx.eval(*[t for pair in context_kv for t in pair])
        elapsed_ms = (time.time() - t0_w) * 1000
        print(
            f"  Replayed window {wid} @ pos {seq_len}–{seq_len + len(w_tokens) - 1} ({elapsed_ms:.0f}ms)",
            file=sys.stderr,
        )
        seq_len += len(w_tokens)

    logits, gen_kv = kv_gen.extend(q_ids, context_kv, abs_start=seq_len)
    mx.eval(logits)
    seq_len += len(prompt_ids)
    context_tokens = seq_len

    generated = []
    for _ in range(max_new_tokens):
        if temperature == 0.0:
            next_tok = int(mx.argmax(logits[0, -1, :]).item())
        else:
            scaled = logits[0, -1, :] / temperature
            next_tok = int(mx.random.categorical(scaled[None]).item())

        if eos_id is not None and next_tok == eos_id:
            break
        generated.append(next_tok)
        sys.stdout.write(tokenizer.decode([next_tok], skip_special_tokens=True))
        sys.stdout.flush()
        logits, gen_kv = kv_gen.step_uncompiled(mx.array([[next_tok]]), gen_kv, seq_len)
        mx.eval(logits)
        seq_len += 1

    print()

    return {
        "tokens": generated,
        "speculative_text": spec_text,
        "selected_windows": sorted(selected),
        "residual_scores": step_scores[:10],
        "source": "grounded",
        "context_tokens": context_tokens,
    }
