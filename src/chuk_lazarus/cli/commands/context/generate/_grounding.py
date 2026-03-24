"""Grounding calibration for iterative navigation."""

from __future__ import annotations

import sys


def _calibrate_grounding(kv_gen, lib, tokenizer, compass_layer, sys_content):
    """Calibrate the grounding/reaching direction at L26.

    Uses 4 contrastive examples: 2 grounding (content in context) and
    2 reaching (content not in context). PCA PC1 = grounding direction.

    Returns (grounding_pc1, cal_mean, ground_threshold, partial_threshold).
    """
    import mlx.core as mx
    import numpy as np

    calibration = [
        (170, "What sports scores were mentioned?", True),  # grounding
        (170, "What were the fuel pressure readings?", False),  # reaching
        (76, "What was in the morning news?", True),  # grounding
        (76, "What did the astronauts eat for breakfast?", False),  # reaching
    ]

    cal_vecs = []
    cal_is_grounding = []

    for wid, question, is_grounding in calibration:
        w_tokens = lib.get_window_tokens(wid)
        pre_text = f"<start_of_turn>user\n{sys_content}\n\nHere is the relevant transcript:\n\n"
        pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
        post_text = (
            f"\n\n---\nBased on the transcript above, "
            f"{question}<end_of_turn>\n<start_of_turn>model\n"
        )
        post_ids = tokenizer.encode(post_text, add_special_tokens=False)

        # Prefill + replay + postamble
        sl = 0
        _l, kv = kv_gen.prefill(mx.array(pre_ids)[None])
        mx.eval(*[t for p in kv for t in p])
        sl += len(pre_ids)
        _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=sl)
        mx.eval(*[t for p in kv for t in p])
        sl += len(w_tokens)
        logits, kv = kv_gen.extend(mx.array(post_ids)[None], kv, abs_start=sl)
        sl += len(post_ids)

        # Generate first token
        first_tok = int(mx.argmax(logits[0, -1]).item())

        # Extract L26 at first generated token
        full = list(pre_ids) + list(w_tokens) + list(post_ids) + [first_tok]
        h = kv_gen.prefill_to_layer(mx.array(full)[None], target_layer=compass_layer)
        mx.eval(h)
        vec = np.array(h[0, -1, :].tolist(), dtype=np.float32)

        cal_vecs.append(vec)
        cal_is_grounding.append(is_grounding)

    # PCA -> PC1
    cal_arr = np.stack(cal_vecs, axis=0)
    cal_mean = cal_arr.mean(axis=0)
    cal_centered = cal_arr - cal_mean
    _U, _S, Vt = np.linalg.svd(cal_centered, full_matrices=False)
    pc1 = Vt[0]

    # Ensure grounding = positive
    projs = cal_centered @ pc1
    g_mean = np.mean([p for p, g in zip(projs, cal_is_grounding) if g])
    r_mean = np.mean([p for p, g in zip(projs, cal_is_grounding) if not g])
    if r_mean > g_mean:
        pc1 = -pc1
        g_mean, r_mean = -r_mean, -g_mean

    midpoint = (g_mean + r_mean) / 2
    margin = abs(g_mean - r_mean) * 0.2
    ground_thresh = midpoint + margin
    partial_thresh = midpoint - margin

    variance_pct = (_S[0] ** 2 / np.sum(_S**2)) * 100
    print(
        f"  Grounding calibrated: PC1={variance_pct:.0f}% variance, "
        f"G={g_mean:+.0f} R={r_mean:+.0f} sep={abs(g_mean - r_mean):.0f}",
        file=sys.stderr,
    )

    return pc1, cal_mean, ground_thresh, partial_thresh
