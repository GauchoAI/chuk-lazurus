"""
Experiment: Does the model's generation residual carry tonal signal?

Hypothesis: The L26 residual at the LAST GENERATED TOKEN is closer to the
tonal "amusing" direction than the post-read residual. Reading doesn't shift
tone (0.25° invariance). But the model's judgment — expressed in generation —
DOES encode tone.

Test:
  1. Construct tonal "amusing" direction via contrastive prompts
  2. Bare query: "Find 3 amusing moments" → extract L26
  3. Post-read: [W170 content + query] → extract L26 at query's last token
  4. Post-generation: model reads W170, generates response → extract L26
     at the LAST GENERATED TOKEN
  5. Compare all three against the tonal direction

Prediction: post-generation residual is significantly closer to the amusing
direction than post-read. The model's own judgment lives in its output state.

Experiment ID: tonal-generation-residual
"""

import sys
import time
import math
import numpy as np
import mlx.core as mx


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def extract_l26_last(kv_gen, token_ids: list[int], compass_layer: int) -> np.ndarray:
    """Extract L26 residual at the last token position."""
    ids = mx.array(token_ids)[None]
    h = kv_gen.prefill_to_layer(ids, target_layer=compass_layer)
    mx.eval(h)
    vec = np.array(h[0, -1, :].tolist(), dtype=np.float32)
    return vec


def generate_and_extract_l26(
    kv_gen, preamble_ids, window_tokens, postamble_ids, tokenizer,
    compass_layer: int, max_gen_tokens: int = 60, temperature: float = 0.0,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Generate after reading a window. Return (generation_L26, generated_text, post_read_L26).

    Also captures the post-read L26 (at the last postamble token, before generation).
    """
    # Prefill preamble
    seq_len = 0
    p_ids = mx.array(preamble_ids)[None]
    _logits, kv = kv_gen.prefill(p_ids)
    mx.eval(*[t for pair in kv for t in pair])
    seq_len += len(preamble_ids)

    # Extend with window content
    w_ids = mx.array(window_tokens)[None]
    _logits, kv = kv_gen.extend(w_ids, kv, abs_start=seq_len)
    mx.eval(*[t for pair in kv for t in pair])
    seq_len += len(window_tokens)

    # Extend with postamble — capture residual at last postamble token
    q_ids = mx.array(postamble_ids)[None]
    logits, kv = kv_gen.extend(q_ids, kv, abs_start=seq_len)
    mx.eval(logits, *[t for pair in kv for t in pair])
    seq_len += len(postamble_ids)

    # Extract post-read L26: re-run just the postamble through to L26
    # to get the residual at the "about to generate" position.
    # We need a full forward to L26 with the context — use the full sequence.
    all_ids = preamble_ids + list(window_tokens) + postamble_ids
    post_read_vec = extract_l26_last(kv_gen, all_ids, compass_layer)

    # Generate tokens
    eos_id = tokenizer.eos_token_id
    gen_tokens = []
    for _ in range(max_gen_tokens):
        last_logits = logits[0, -1]
        if temperature == 0.0:
            next_tok = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / temperature
            next_tok = int(mx.random.categorical(scaled[None]).item())
        if eos_id is not None and next_tok == eos_id:
            break
        gen_tokens.append(next_tok)
        logits, kv = kv_gen.step_uncompiled(mx.array([[next_tok]]), kv, seq_len=seq_len)
        seq_len += 1

    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Extract L26 at the last GENERATED token position.
    # Full sequence: preamble + window + postamble + generated tokens
    full_ids = all_ids + gen_tokens
    gen_vec = extract_l26_last(kv_gen, full_ids, compass_layer)

    return gen_vec, gen_text, post_read_vec


def main():
    from chuk_lazarus.inference import UnifiedPipeline
    from chuk_lazarus.inference.context import CheckpointLibrary

    import argparse
    parser = argparse.ArgumentParser(description="Test tonal signal in generation residuals")
    parser.add_argument("--checkpoint", "-c", default="./apollo11_ctx_512",
                        help="Path to checkpoint library")
    parser.add_argument("--model", "-m", default="google/gemma-3-4b-it",
                        help="Model ID")
    cli_args = parser.parse_args()

    model_id = cli_args.model
    lib_path = cli_args.checkpoint

    print(f"Loading library: {lib_path}")
    lib = CheckpointLibrary(lib_path)
    print(f"  {lib.num_windows} windows, {lib.total_tokens:,} tokens")
    compass_layer = lib.compass_layer
    print(f"  Compass layer: L{compass_layer}")

    print(f"Loading model: {model_id}")
    pipeline = UnifiedPipeline.from_pretrained(model_id, verbose=False)
    tokenizer = pipeline.tokenizer

    from chuk_lazarus.inference.context.unlimited_engine import UnlimitedContextEngine
    engine = UnlimitedContextEngine(pipeline.model, pipeline.config, window_size=lib.window_size)
    engine.load_library(lib)
    kv_gen = engine.kv_gen

    # ── Step 1: Construct tonal "amusing" direction ──
    # Contrastive pairs: amusing vs neutral descriptions of the same event
    amusing_prompts = [
        "That was hilarious, the porridge eating contest was the funniest thing I've read",
        "What an amusing and entertaining anecdote about the oatmeal competition",
        "The astronauts joking around was genuinely funny and lighthearted",
        "That's a delightful and comical moment from the mission transcript",
    ]
    neutral_prompts = [
        "The crew consumed their scheduled meal as part of the flight plan",
        "Mission control relayed standard communication updates to the crew",
        "The transcript records routine operational procedures during the mission",
        "Technical telemetry data was transmitted at the scheduled interval",
    ]

    print("\n" + "=" * 60)
    print("Step 1: Constructing tonal 'amusing' direction")
    print("=" * 60)

    amusing_vecs = []
    for p in amusing_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        vec = extract_l26_last(kv_gen, ids, compass_layer)
        amusing_vecs.append(vec)

    neutral_vecs = []
    for p in neutral_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        vec = extract_l26_last(kv_gen, ids, compass_layer)
        neutral_vecs.append(vec)

    # Tonal direction = mean(amusing) - mean(neutral)
    amusing_mean = np.mean(amusing_vecs, axis=0)
    neutral_mean = np.mean(neutral_vecs, axis=0)
    tonal_dir = amusing_mean - neutral_mean
    tonal_norm = np.linalg.norm(tonal_dir)
    tonal_unit = tonal_dir / (tonal_norm + 1e-10)
    print(f"  Tonal direction norm: {tonal_norm:.4f}")
    print(f"  Amusing centroid ↔ Neutral centroid angle: {angle_degrees(amusing_mean, neutral_mean):.2f}°")

    # ── Step 2: Bare query residual ──
    print("\n" + "=" * 60)
    print("Step 2: Bare query residual")
    print("=" * 60)

    query = "Find 3 amusing moments from the transcript"
    messages = [{"role": "user", "content": query}]
    query_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)
    bare_vec = extract_l26_last(kv_gen, query_ids, compass_layer)

    bare_tonal_angle = angle_degrees(bare_vec, tonal_unit)
    bare_tonal_proj = np.dot(bare_vec, tonal_unit)
    print(f"  Bare query → tonal direction: angle={bare_tonal_angle:.2f}°, projection={bare_tonal_proj:.4f}")

    # ── Step 3: Test windows ──
    # W170 = porridge/baseball (amusing content)
    # W118 = Earth observation (neutral/awe content)
    test_windows = [
        (170, "porridge/baseball — amusing"),
        (118, "Earth observation — neutral/awe"),
    ]

    print("\n" + "=" * 60)
    print("Step 3: Post-read vs post-generation residuals")
    print("=" * 60)

    for wid, desc in test_windows:
        print(f"\n{'─' * 60}")
        print(f"Window {wid}: {desc}")
        print(f"{'─' * 60}")

        w_tokens = lib.get_window_tokens(wid)
        w_text = tokenizer.decode(w_tokens, skip_special_tokens=True)
        print(f"  Content preview: {w_text[:100]}...")

        # Build framed prompt
        preamble_text = (
            "<start_of_turn>user\n"
            "You are reading a transcript from the Apollo 11 mission.\n\n"
            "Here is the relevant transcript:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)

        postamble_text = (
            "\n\n---\n"
            "Based on the transcript above, is there anything amusing or funny here? "
            "Describe what you found and why it's amusing.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)

        t0 = time.time()
        gen_vec, gen_text, post_read_vec = generate_and_extract_l26(
            kv_gen, preamble_ids, w_tokens, postamble_ids, tokenizer,
            compass_layer=compass_layer,
            max_gen_tokens=80,
            temperature=0.0,
        )
        elapsed = time.time() - t0

        print(f"  Generated ({elapsed:.1f}s): {gen_text[:150]}...")

        # Measurements
        post_read_tonal_angle = angle_degrees(post_read_vec, tonal_unit)
        post_read_tonal_proj = np.dot(post_read_vec, tonal_unit)

        gen_tonal_angle = angle_degrees(gen_vec, tonal_unit)
        gen_tonal_proj = np.dot(gen_vec, tonal_unit)

        # Angles between the three residuals
        bare_to_read = angle_degrees(bare_vec, post_read_vec)
        bare_to_gen = angle_degrees(bare_vec, gen_vec)
        read_to_gen = angle_degrees(post_read_vec, gen_vec)

        print(f"\n  Tonal projection (higher = more 'amusing'):")
        print(f"    Bare query:      proj={bare_tonal_proj:+.4f}  angle={bare_tonal_angle:.2f}°")
        print(f"    Post-read:       proj={post_read_tonal_proj:+.4f}  angle={post_read_tonal_angle:.2f}°")
        print(f"    Post-generation: proj={gen_tonal_proj:+.4f}  angle={gen_tonal_angle:.2f}°")

        # The key comparison
        tonal_shift_read = post_read_tonal_proj - bare_tonal_proj
        tonal_shift_gen = gen_tonal_proj - bare_tonal_proj
        print(f"\n  Tonal shift from bare query:")
        print(f"    Reading shifts tonal by:    {tonal_shift_read:+.4f}")
        print(f"    Generation shifts tonal by: {tonal_shift_gen:+.4f}")
        print(f"    Generation / Reading ratio: {abs(tonal_shift_gen) / (abs(tonal_shift_read) + 1e-10):.1f}×")

        print(f"\n  Pairwise angles:")
        print(f"    Bare → Post-read:       {bare_to_read:.2f}°")
        print(f"    Bare → Post-generation: {bare_to_gen:.2f}°")
        print(f"    Post-read → Post-gen:   {read_to_gen:.2f}°")

    # ── Step 4: Generation-mode tonal direction ──
    # The expression-mode direction was built from prompts EXPRESSING amusement.
    # But the model's generation is ANALYZING content. Build a generation-mode
    # direction: model reads amusing vs neutral content, generates response,
    # extract L26 at the last generated token.
    print("\n" + "=" * 60)
    print("Step 4: Generation-mode tonal direction")
    print("=" * 60)
    print("  Building tonal direction from generation residuals...")
    print("  (model reads content → generates analysis → extract L26 at last token)")

    # Use a few windows: some amusing, some neutral/technical
    amusing_wids = [170, 64]   # porridge/baseball, morning news with sports
    neutral_wids = [118, 605]  # Earth observation, technical comms

    def _generate_and_get_l26(wid: int) -> np.ndarray:
        """Read a window, generate short response, extract L26 at last token."""
        wt = lib.get_window_tokens(wid)
        pre_text = (
            "<start_of_turn>user\n"
            "Read this transcript excerpt and describe anything amusing or funny.\n\n"
        )
        pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
        post_text = (
            "\n\n---\nDescribe what you found. Is it amusing?<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        post_ids = tokenizer.encode(post_text, add_special_tokens=False)

        # Prefill + extend + generate
        p = mx.array(pre_ids)[None]
        _l, kv = kv_gen.prefill(p)
        mx.eval(*[t for pair in kv for t in pair])
        sl = len(pre_ids)

        w = mx.array(wt)[None]
        _l, kv = kv_gen.extend(w, kv, abs_start=sl)
        mx.eval(*[t for pair in kv for t in pair])
        sl += len(wt)

        q = mx.array(post_ids)[None]
        logits, kv = kv_gen.extend(q, kv, abs_start=sl)
        sl += len(post_ids)

        eos = tokenizer.eos_token_id
        gen_toks = []
        for _ in range(60):
            nt = int(mx.argmax(logits[0, -1]).item())
            if eos is not None and nt == eos:
                break
            gen_toks.append(nt)
            logits, kv = kv_gen.step_uncompiled(mx.array([[nt]]), kv, seq_len=sl)
            sl += 1

        # Extract L26 at last generated token
        full = list(pre_ids) + list(wt) + list(post_ids) + gen_toks
        h = kv_gen.prefill_to_layer(mx.array(full)[None], target_layer=compass_layer)
        mx.eval(h)
        vec = np.array(h[0, -1, :].tolist(), dtype=np.float32)

        gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)
        return vec, gen_text

    gen_amusing_vecs = []
    gen_neutral_vecs = []

    for wid in amusing_wids:
        vec, text = _generate_and_get_l26(wid)
        gen_amusing_vecs.append(vec)
        print(f"  W{wid} (amusing): {text[:80]}...")

    for wid in neutral_wids:
        vec, text = _generate_and_get_l26(wid)
        gen_neutral_vecs.append(vec)
        print(f"  W{wid} (neutral): {text[:80]}...")

    # Generation-mode tonal direction
    gen_amusing_mean = np.mean(gen_amusing_vecs, axis=0)
    gen_neutral_mean = np.mean(gen_neutral_vecs, axis=0)
    gen_tonal_dir = gen_amusing_mean - gen_neutral_mean
    gen_tonal_norm = np.linalg.norm(gen_tonal_dir)
    gen_tonal_unit = gen_tonal_dir / (gen_tonal_norm + 1e-10)

    print(f"\n  Generation-mode tonal direction norm: {gen_tonal_norm:.4f}")
    print(f"  Gen amusing ↔ Gen neutral angle: {angle_degrees(gen_amusing_mean, gen_neutral_mean):.2f}°")

    # Compare the two tonal directions
    expr_gen_angle = angle_degrees(tonal_unit, gen_tonal_unit)
    print(f"  Expression-mode ↔ Generation-mode tonal angle: {expr_gen_angle:.2f}°")
    print(f"  (90° = orthogonal, 0° = identical)")

    # ── Step 5: Discrimination comparison ──
    print("\n" + "=" * 60)
    print("Step 5: Which tonal direction discriminates better?")
    print("=" * 60)

    # Re-score the generation residuals from Step 3 against BOTH tonal directions
    # We stored the results in the loop above — need to regenerate
    print("\n  Scoring generation residuals against both tonal directions:")
    print(f"  {'Window':<12} {'Expr proj':>12} {'Gen proj':>12} {'Expr angle':>12} {'Gen angle':>12}")
    print(f"  {'─'*60}")

    for wid, desc in test_windows:
        w_tokens = lib.get_window_tokens(wid)
        preamble_text = (
            "<start_of_turn>user\n"
            "You are reading a transcript from the Apollo 11 mission.\n\n"
            "Here is the relevant transcript:\n\n"
        )
        preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)
        postamble_text = (
            "\n\n---\n"
            "Based on the transcript above, is there anything amusing or funny here? "
            "Describe what you found and why it's amusing.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)

        gen_vec, gen_text, _ = generate_and_extract_l26(
            kv_gen, preamble_ids, w_tokens, postamble_ids, tokenizer,
            compass_layer=compass_layer, max_gen_tokens=80, temperature=0.0,
        )

        # Score against both directions
        expr_proj = np.dot(gen_vec, tonal_unit)
        gen_proj = np.dot(gen_vec, gen_tonal_unit)
        expr_angle = angle_degrees(gen_vec, tonal_unit)
        gen_angle = angle_degrees(gen_vec, gen_tonal_unit)

        label = f"W{wid}"
        print(f"  {label:<12} {expr_proj:>+12.1f} {gen_proj:>+12.1f} {expr_angle:>11.2f}° {gen_angle:>11.2f}°")

    print(f"\n  Expression-mode discrimination (W170 - W118 gen projection):")
    print(f"    Larger gap = better discrimination for routing")
    print(f"    If generation-mode >> expression-mode, use generation-mode tonal")
    print(f"    direction for iterative routing")


if __name__ == "__main__":
    main()
