"""
Grounding detector test — does L26 PC1 detect when the model needs more context?

Test: replay windows W419-W430 (EVA sequence) one at a time with the moonwalk
query. At the first generated token, extract L26 and project onto the
grounding direction. Does the signal detect partial grounding at W419 and
grounding when the model reaches the "one small step" window?

Calibration: compute grounding PC1 on-the-fly from 4 contrastive examples.
  - Grounding: replay W170 (sports), ask about sports → model has context
  - Reaching: replay W170 (sports), ask about weather → model lacks context
  - Grounding: replay W76 (news), ask about news → model has context
  - Reaching: replay W76 (news), ask about fuel readings → model lacks context

PC1 of these 4 vectors = grounding direction (universal, 47% variance).
"""

import sys
import math
import time
import numpy as np
import mlx.core as mx


def angle_degrees(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def extract_first_token_l26(kv_gen, preamble_ids, window_tokens, postamble_ids,
                             tokenizer, compass_layer, temperature=0.0):
    """Replay window, generate first token, extract L26 at that position."""
    # Prefill preamble
    seq_len = 0
    p = mx.array(preamble_ids)[None]
    _l, kv = kv_gen.prefill(p)
    mx.eval(*[t for pair in kv for t in pair])
    seq_len += len(preamble_ids)

    # Replay window
    w = mx.array(window_tokens)[None]
    _l, kv = kv_gen.extend(w, kv, abs_start=seq_len)
    mx.eval(*[t for pair in kv for t in pair])
    seq_len += len(window_tokens)

    # Postamble
    q = mx.array(postamble_ids)[None]
    logits, kv = kv_gen.extend(q, kv, abs_start=seq_len)
    seq_len += len(postamble_ids)

    # Generate first token
    if temperature == 0.0:
        first_tok = int(mx.argmax(logits[0, -1]).item())
    else:
        first_tok = int(mx.random.categorical(logits[0, -1:] / temperature).item())

    first_text = tokenizer.decode([first_tok], skip_special_tokens=True)

    # Generate a few more tokens to see what the model says
    gen_tokens = [first_tok]
    logits, kv = kv_gen.step_uncompiled(mx.array([[first_tok]]), kv, seq_len=seq_len)
    seq_len += 1
    for _ in range(20):
        nt = int(mx.argmax(logits[0, -1]).item())
        if tokenizer.eos_token_id is not None and nt == tokenizer.eos_token_id:
            break
        gen_tokens.append(nt)
        logits, kv = kv_gen.step_uncompiled(mx.array([[nt]]), kv, seq_len=seq_len)
        seq_len += 1

    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Extract L26 at the first generated token position
    full_seq = list(preamble_ids) + list(window_tokens) + list(postamble_ids) + [first_tok]
    h = kv_gen.prefill_to_layer(mx.array(full_seq)[None], target_layer=compass_layer)
    mx.eval(h)
    vec = np.array(h[0, -1, :].tolist(), dtype=np.float32)

    return vec, first_text, gen_text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default="./apollo11_ctx_512")
    parser.add_argument("--model", "-m", default="google/gemma-3-4b-it")
    args = parser.parse_args()

    from chuk_lazarus.inference import UnifiedPipeline
    from chuk_lazarus.inference.context import CheckpointLibrary
    from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine

    print(f"Loading library: {args.checkpoint}")
    lib = CheckpointLibrary(args.checkpoint)
    compass_layer = lib.compass_layer

    print(f"Loading model: {args.model}")
    pipeline = UnifiedPipeline.from_pretrained(args.model, verbose=False)
    tokenizer = pipeline.tokenizer
    engine = UnlimitedContextEngine(pipeline.model, pipeline.config, window_size=lib.window_size)
    engine.load_library(lib)
    kv_gen = engine.kv_gen

    sys_content = (
        "You are answering questions based on a document transcript. "
        "Answer using only information from the transcript. Quote exact text when possible."
    )

    def make_prompt_ids(question):
        pre = f"<start_of_turn>user\n{sys_content}\n\nHere is the relevant transcript:\n\n"
        pre_ids = tokenizer.encode(pre, add_special_tokens=True)
        post = f"\n\n---\nBased on the transcript above, {question}<end_of_turn>\n<start_of_turn>model\n"
        post_ids = tokenizer.encode(post, add_special_tokens=False)
        return pre_ids, post_ids

    # ── Step 1: Calibrate grounding direction ──
    print("\n" + "=" * 60)
    print("Step 1: Calibrating grounding direction")
    print("=" * 60)

    calibration = [
        # (window_id, question, expected_state)
        (170, "What sports scores were mentioned?", "grounding"),
        (170, "What were the fuel pressure readings?", "reaching"),
        (76, "What was in the morning news?", "grounding"),
        (76, "What did the astronauts eat for breakfast?", "reaching"),
    ]

    cal_vecs = []
    cal_labels = []
    for wid, question, label in calibration:
        w_tokens = lib.get_window_tokens(wid)
        pre_ids, post_ids = make_prompt_ids(question)
        vec, first_tok, gen_preview = extract_first_token_l26(
            kv_gen, pre_ids, w_tokens, post_ids, tokenizer, compass_layer,
        )
        cal_vecs.append(vec)
        cal_labels.append(label)
        print(f"  W{wid} [{label}] Q: {question[:50]}...")
        print(f"    First token: '{first_tok}' | Preview: {gen_preview[:60]}...")

    # PCA on calibration vectors
    cal_arr = np.stack(cal_vecs, axis=0)  # (4, 2560)
    cal_mean = cal_arr.mean(axis=0)
    cal_centered = cal_arr - cal_mean
    _U, _S, Vt = np.linalg.svd(cal_centered, full_matrices=False)

    grounding_pc1 = Vt[0]  # (2560,) — the grounding direction
    projections = cal_centered @ grounding_pc1

    # Determine polarity: grounding should be positive
    ground_proj = np.mean([p for p, l in zip(projections, cal_labels) if l == "grounding"])
    reach_proj = np.mean([p for p, l in zip(projections, cal_labels) if l == "reaching"])
    if reach_proj > ground_proj:
        grounding_pc1 = -grounding_pc1
        projections = -projections
        ground_proj, reach_proj = -reach_proj, -ground_proj

    print(f"\n  Grounding PC1 calibrated:")
    print(f"    Variance explained: {(_S[0]**2 / np.sum(_S**2)) * 100:.1f}%")
    print(f"    Grounding mean projection: {ground_proj:+.1f}")
    print(f"    Reaching mean projection:  {reach_proj:+.1f}")
    print(f"    Separation: {abs(ground_proj - reach_proj):.1f}")

    # Thresholds: midpoint ± margin
    midpoint = (ground_proj + reach_proj) / 2
    margin = abs(ground_proj - reach_proj) * 0.2
    grounding_threshold = midpoint + margin  # above = grounded
    reaching_threshold = midpoint - margin   # below = reaching
    print(f"    Thresholds: grounding > {grounding_threshold:.1f}, reaching < {reaching_threshold:.1f}")

    # ── Step 2: Test on moonwalk query ──
    print("\n" + "=" * 60)
    print("Step 2: Moonwalk query — grounding signal per window")
    print("=" * 60)

    question = "What was the first thing said when someone stepped onto the lunar surface?"
    pre_ids, post_ids = make_prompt_ids(question)

    # Test W415-W435 (EVA sequence spanning the moonwalk)
    test_range = range(415, min(436, lib.num_windows))

    print(f"\n  {'Window':<10} {'Proj':>8} {'State':<12} {'First tok':<12} {'Preview'}")
    print(f"  {'─' * 80}")

    for wid in test_range:
        w_tokens = lib.get_window_tokens(wid)
        vec, first_tok, gen_preview = extract_first_token_l26(
            kv_gen, pre_ids, w_tokens, post_ids, tokenizer, compass_layer,
        )

        # Project onto grounding PC1
        proj = float((vec - cal_mean) @ grounding_pc1)

        if proj > grounding_threshold:
            state = "GROUNDED"
        elif proj < reaching_threshold:
            state = "REACHING"
        else:
            state = "PARTIAL"

        preview = gen_preview[:40].replace('\n', ' ')
        print(f"  W{wid:<7} {proj:>+8.1f} {state:<12} '{first_tok}'  {preview}")

    # ── Step 3: Compare with known-grounded window ──
    print(f"\n" + "=" * 60)
    print("Step 3: Control — known grounded vs reaching")
    print("=" * 60)

    # W170 with sports question = definitely grounded
    w170 = lib.get_window_tokens(170)
    pre_g, post_g = make_prompt_ids("What sports scores were mentioned?")
    vec_g, _, preview_g = extract_first_token_l26(
        kv_gen, pre_g, w170, post_g, tokenizer, compass_layer,
    )
    proj_g = float((vec_g - cal_mean) @ grounding_pc1)

    # W170 with unrelated question = definitely reaching
    pre_r, post_r = make_prompt_ids("What was the first thing said on the lunar surface?")
    vec_r, _, preview_r = extract_first_token_l26(
        kv_gen, pre_r, w170, post_r, tokenizer, compass_layer,
    )
    proj_r = float((vec_r - cal_mean) @ grounding_pc1)

    print(f"  W170 + sports question (grounded):  proj={proj_g:+.1f}  {preview_g[:50]}")
    print(f"  W170 + moonwalk question (reaching): proj={proj_r:+.1f}  {preview_r[:50]}")
    print(f"  Separation: {abs(proj_g - proj_r):.1f}")


if __name__ == "__main__":
    main()
