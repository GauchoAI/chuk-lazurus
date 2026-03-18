"""On-the-fly probe calibration with caching.

Two probes:

  query_type:  Routes queries as EXPLORATION or FACTUAL.
               Generic query examples, no library content needed. ~0.5s.

  tonal:       Scores content as engaging vs routine in generation-mode.
               Samples library windows, model generates assessments,
               trains from relative ratings (top-5 vs bottom-5). ~15s.

Both cached in checkpoint directory. First run calibrates, then instant.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import mlx.core as mx
import numpy as np


# ── Data structures ──────────────────────────────────────────────────

class QueryType(str, Enum):
    """Mode 7 query classifications."""
    FACTUAL = "factual"
    ENGAGEMENT = "engagement"
    TENSION = "tension"
    GLOBAL = "global"
    TONE = "tone"


@dataclass
class ProbeSet:
    """Calibrated probe directions."""
    # Query-type: exploration vs factual (legacy 2-class)
    qt_direction: mx.array    # (hidden_dim,) positive = exploration
    qt_mean: mx.array         # (hidden_dim,)
    qt_threshold: float

    # Tonal: engaging vs routine (generation-mode L26)
    tonal_direction: mx.array  # (hidden_dim,) positive = engaging
    tonal_mean: mx.array       # (hidden_dim,)
    tonal_threshold: float
    tonal_available: bool      # False if calibration had insufficient contrast

    # Mode 7: 5-class query type classifier
    m7_class_directions: dict[str, mx.array] | None = None  # class_name → direction
    m7_class_means: dict[str, mx.array] | None = None
    m7_available: bool = False

    # Mode 7: tension probe (generation-mode, separate from tonal)
    tension_direction: mx.array | None = None
    tension_mean: mx.array | None = None
    tension_available: bool = False

    # Structural basis for cleaning residuals before projection.
    # Removes BOS-sink and format variance that dominates PCA but carries no content.
    # Shape: struct_basis (K, hidden_dim), struct_mean (hidden_dim,)
    struct_basis: mx.array | None = None
    struct_mean: mx.array | None = None


# ── Constants ────────────────────────────────────────────────────────

_QT_EXAMPLES: list[tuple[str, bool]] = [
    ("Find the most interesting or notable moments", True),
    ("What were the most surprising things that happened?", True),
    ("What amusing or entertaining parts are there?", True),
    ("What were the most dramatic or tense moments?", True),
    ("What human-interest stories are in the text?", True),
    ("What specific names or people were mentioned?", False),
    ("What numerical measurements or readings were given?", False),
    ("What instructions or procedures were described?", False),
    ("What times or dates were referenced?", False),
    ("What equipment or systems were discussed?", False),
]

_TONAL_SAMPLE_COUNT = 20
_TONAL_ASSESS_TOKENS = 20
_CACHE_VERSION = 5  # bump to invalidate stale caches (added structural PC removal)

# Mode 7: 5-class query type examples (from spec §3.3)
_M7_EXAMPLES: list[tuple[str, str]] = [
    # factual
    ("What sport was discussed?", "factual"),
    ("What was said when Eagle landed?", "factual"),
    ("What specific names were mentioned?", "factual"),
    ("What numerical measurements were given?", "factual"),
    # engagement
    ("Find 3 amusing moments", "engagement"),
    ("What surprised the crew?", "engagement"),
    ("Describe crew relationships", "engagement"),
    ("What human-interest stories are in the text?", "engagement"),
    # tension
    ("What worried the crew most?", "tension"),
    ("What was the most tense moment?", "tension"),
    ("What went wrong during the mission?", "tension"),
    ("What were the most dramatic moments?", "tension"),
    # global
    ("Summarise the mission timeline", "global"),
    ("How did the mission progress from launch to splashdown?", "global"),
    ("Give a broad overview of the document", "global"),
    # tone
    ("What was the mood during landing?", "tone"),
    ("How did the crew's mood change?", "tone"),
    ("Describe the emotional atmosphere", "tone"),
]


# ── Helpers ──────────────────────────────────────────────────────────

def _probe_cache_path(checkpoint_dir: str | Path, model_name: str) -> Path:
    h = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return Path(checkpoint_dir) / f".probe_cache_v{_CACHE_VERSION}_{h}.npz"


def _pca_direction(vecs: list[mx.array], labels: list[bool], positive_label: bool = True):
    """PCA PC1, oriented so positive_label projects positive."""
    X = mx.stack(vecs, axis=0).astype(mx.float32)
    mean = mx.mean(X, axis=0)
    centered = X - mean

    _U, S, Vt = mx.linalg.svd(centered, stream=mx.cpu)
    mx.eval(S, Vt)

    pc1 = Vt[0]
    projs = centered @ pc1
    mx.eval(projs)
    projs_list = projs.tolist()

    pos_projs = [p for p, lab in zip(projs_list, labels) if lab == positive_label]
    neg_projs = [p for p, lab in zip(projs_list, labels) if lab != positive_label]
    pos_mean = sum(pos_projs) / len(pos_projs)
    neg_mean = sum(neg_projs) / len(neg_projs)

    if neg_mean > pos_mean:
        pc1 = -pc1
        pos_mean, neg_mean = -neg_mean, -pos_mean

    threshold = (pos_mean + neg_mean) / 2
    variance_pct = float((S[0] * S[0] / mx.sum(S * S)).item()) * 100
    mx.eval(pc1)
    return pc1, mean, threshold, variance_pct, pos_mean, neg_mean


# ── Structural PC removal ─────────────────────────────────────────────

def _get_lib_structural_np(lib) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract structural basis from a library as numpy arrays.

    Returns (struct_basis_np, corpus_mean_np) or (None, None) if unavailable.
    struct_basis_np: (K, hidden_dim) — structural PCs to project out.
    corpus_mean_np:  (hidden_dim,)   — corpus mean used in compass calibration.
    """
    if lib is None or not lib.has_structural_basis:
        return None, None
    mean_vec, _, _, _ = lib.get_compass_basis()
    struct_basis = lib.get_structural_basis()
    mean_np = np.array(mean_vec.reshape(-1).tolist(), dtype=np.float32)
    struct_np = np.array(struct_basis.tolist(), dtype=np.float32)
    return struct_np, mean_np


def _clean_vecs(
    vecs: list[mx.array],
    struct_np: np.ndarray | None,
    mean_np: np.ndarray | None,
) -> list[mx.array]:
    """Subtract corpus mean and remove structural PCs from each vector.

    Returns cleaned vectors as mx.arrays. Structural directions (BOS-sink,
    format variance) dominate PCA but carry no content — removing them
    exposes the dark space where routing signal lives.

    If struct_np/mean_np are None, returns vecs unchanged.
    """
    if struct_np is None or mean_np is None:
        return vecs
    cleaned = []
    for v in vecs:
        v_np = np.array(v.tolist(), dtype=np.float32)
        centered = v_np - mean_np
        proj = centered @ struct_np.T        # (K,)
        cleaned.append(mx.array(centered - proj @ struct_np))
    return cleaned


def _clean_single_np(
    vec: mx.array,
    struct_np: np.ndarray | None,
    mean_np: np.ndarray | None,
) -> mx.array:
    """Clean one vector — used at query-time."""
    if struct_np is None or mean_np is None:
        return vec
    v_np = np.array(vec.tolist(), dtype=np.float32)
    centered = v_np - mean_np
    proj = centered @ struct_np.T
    return mx.array(centered - proj @ struct_np)


# ── Query-type calibration ───────────────────────────────────────────

def _calibrate_query_type(kv_gen, tokenizer, compass_layer: int, lib=None):
    """PCA direction from generic exploration/factual examples.

    Structural PCs from the library are removed before PCA so PC1
    captures query-type variance, not BOS-sink or format structure.
    """
    vecs: list[mx.array] = []
    labels: list[bool] = []

    for query_text, is_exploration in _QT_EXAMPLES:
        prompt = (
            f"<start_of_turn>user\n{query_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        h = kv_gen.prefill_to_layer(
            mx.array(ids)[None], target_layer=compass_layer,
            sample_positions=[len(ids) - 1],
        )
        mx.eval(h)
        vecs.append(h[0, 0, :])
        labels.append(is_exploration)

    struct_np, mean_np = _get_lib_structural_np(lib)
    vecs = _clean_vecs(vecs, struct_np, mean_np)

    pc1, mean, thresh, var_pct, pos_m, neg_m = _pca_direction(
        vecs, labels, positive_label=True,
    )
    cleaned_label = " (struct-cleaned)" if struct_np is not None else ""
    print(
        f"    Query-type{cleaned_label}: PC1={var_pct:.0f}% variance, "
        f"E={pos_m:+.0f} F={neg_m:+.0f} sep={abs(pos_m - neg_m):.0f}",
        file=sys.stderr,
    )
    return pc1, mean, thresh


# ── Tonal calibration (generation-mode) ──────────────────────────────

def _assess_window(kv_gen, tokenizer, w_tokens, compass_layer: int):
    """Model reads window + assessment prompt, generates 20 tokens.

    Returns (rating, L26_residual) where rating is parsed from text
    and L26_residual is extracted at the last generated token.

    Uses extend_to_layer on the final generated token to capture L26
    in the same forward pass — avoids the previous double forward pass
    (prefill_to_layer on the full sequence after generation).
    """
    pre_text = "<start_of_turn>user\nHere is a text excerpt:\n\n"
    pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
    post_text = (
        "\n\nIs there anything amusing, surprising, or human-interest "
        "in this excerpt? Rate from 1-5.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    post_ids = tokenizer.encode(post_text, add_special_tokens=False)

    sl = 0
    _l, kv = kv_gen.prefill(mx.array(pre_ids)[None])
    mx.eval(*[t for p in kv for t in p])
    sl += len(pre_ids)
    _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=sl)
    mx.eval(*[t for p in kv for t in p])
    sl += len(w_tokens)
    logits, kv = kv_gen.extend(mx.array(post_ids)[None], kv, abs_start=sl)
    sl += len(post_ids)

    gen_tokens: list[int] = []
    eos = tokenizer.eos_token_id
    # Track KV+position before each step so we can capture L26 at the last token
    last_tok_kv = kv
    last_tok_sl = sl
    for _ in range(_TONAL_ASSESS_TOKENS):
        tok = int(mx.argmax(logits[0, -1]).item())
        if eos is not None and tok == eos:
            break
        gen_tokens.append(tok)
        last_tok_kv = kv
        last_tok_sl = sl
        logits, kv = kv_gen.step_uncompiled(mx.array([[tok]]), kv, seq_len=sl)
        sl += 1

    # Parse rating
    judge_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    m = re.search(r'[1-5]', judge_text)
    rating = int(m.group()) if m else 3

    # Extract L26 at last generated token via extend_to_layer — one forward
    # pass of one token with the full KV context, not a full prefill replay.
    if gen_tokens:
        _, _, layer_h = kv_gen.extend_to_layer(
            mx.array([[gen_tokens[-1]]]), last_tok_kv,
            abs_start=last_tok_sl, target_layer=compass_layer,
        )
        vec = layer_h[0, 0, :].astype(mx.float32)
        mx.eval(vec)
    else:
        # Fallback: EOS on first token — extract from postamble end position
        full = list(pre_ids) + list(w_tokens) + list(post_ids)
        h = kv_gen.prefill_to_layer(
            mx.array(full)[None], target_layer=compass_layer,
            sample_positions=[len(full) - 1],
        )
        mx.eval(h)
        vec = h[0, 0, :].astype(mx.float32)

    return rating, vec


def _calibrate_tonal(kv_gen, tokenizer, compass_layer: int, lib):
    """Train tonal probe from sampled library windows — dark space, not tokens.

    The model rates all messy OCR windows as "1" in token space. But the
    dark space at L26 DOES discriminate — the generation-mode residual
    encodes the model's judgment even when the text output is identical.

    Method:
      1. Sample 20 windows → generate assessments → extract L26 residuals
      2. Unsupervised PCA on L26 residuals → PC1 (dominant variation)
      3. Orient PC1 with 2 synthetic references (amusing vs routine)

    No parsed ratings. No labels. Pure dark-space geometry.
    """
    n_windows = lib.num_windows
    sample_count = min(_TONAL_SAMPLE_COUNT, n_windows)
    step = max(1, n_windows // sample_count)
    sample_wids = list(range(0, n_windows, step))[:sample_count]

    residuals: list[mx.array] = []

    for i, wid in enumerate(sample_wids):
        w_tokens = lib.get_window_tokens(wid)
        _, vec = _assess_window(kv_gen, tokenizer, w_tokens, compass_layer)
        residuals.append(vec)
        if (i + 1) % 5 == 0:
            print(
                f"      ... assessed {i + 1}/{sample_count} windows",
                file=sys.stderr,
            )

    if len(residuals) < 4:
        print(f"    Tonal: too few samples ({len(residuals)}), skipping", file=sys.stderr)
        return None, None, 0.0

    # Remove structural PCs before PCA so PC1 captures tonal judgment,
    # not BOS-sink or format dominance.
    struct_np, mean_np = _get_lib_structural_np(lib)
    residuals = _clean_vecs(residuals, struct_np, mean_np)

    # Unsupervised PCA — PC1 captures the dominant variation in the
    # generation-mode judgment space. The experiment showed this is
    # 84.6% tonal judgment (amusing ↔ routine).
    X = mx.stack(residuals, axis=0).astype(mx.float32)
    mean = mx.mean(X, axis=0)
    centered = X - mean

    _U, S, Vt = mx.linalg.svd(centered, stream=mx.cpu)
    mx.eval(S, Vt)

    pc1 = Vt[0]
    variance_pct = float((S[0] * S[0] / mx.sum(S * S)).item()) * 100
    mx.eval(pc1)

    # Orient PC1: run 2 synthetic references through the same assessment
    # pipeline. "Amusing" should project positive, "routine" negative.
    amusing_tokens = tokenizer.encode(
        "Someone told a hilarious joke and the whole crew burst out laughing. "
        "Then they had a porridge eating contest while floating in zero gravity.",
        add_special_tokens=False,
    )
    routine_tokens = tokenizer.encode(
        "The pressure gauge reading was twelve point five nominal. "
        "Proceed with standard fuel cell purge procedure.",
        add_special_tokens=False,
    )

    _, amusing_vec = _assess_window(kv_gen, tokenizer, amusing_tokens, compass_layer)
    _, routine_vec = _assess_window(kv_gen, tokenizer, routine_tokens, compass_layer)

    # Clean synthetic references with the same structural basis
    amusing_vec = _clean_single_np(amusing_vec, struct_np, mean_np)
    routine_vec = _clean_single_np(routine_vec, struct_np, mean_np)

    amusing_proj = float(mx.sum((amusing_vec - mean) * pc1).item())
    routine_proj = float(mx.sum((routine_vec - mean) * pc1).item())

    if routine_proj > amusing_proj:
        pc1 = -pc1
        amusing_proj, routine_proj = -routine_proj, -amusing_proj

    threshold = (amusing_proj + routine_proj) / 2

    cleaned_label = " struct-cleaned" if struct_np is not None else ""
    print(
        f"    Tonal{cleaned_label}: PC1={variance_pct:.0f}% variance, "
        f"amusing={amusing_proj:+.0f} routine={routine_proj:+.0f} "
        f"sep={abs(amusing_proj - routine_proj):.0f} "
        f"(from {sample_count} dark-space samples)",
        file=sys.stderr,
    )
    return pc1, mean, threshold


# ── Tonal scoring (used at query time) ───────────────────────────────

def tonal_score_window(kv_gen, tokenizer, w_tokens, compass_layer, probes):
    """Score a single window with the tonal generation probe.

    Model reads window + assessment → generates 20 tokens →
    L26 at last token → project onto tonal direction.
    Returns float score (higher = more engaging/amusing).
    """
    _, vec = _assess_window(kv_gen, tokenizer, w_tokens, compass_layer)
    # Apply structural cleaning if available
    struct_np = (
        np.array(probes.struct_basis.tolist(), dtype=np.float32)
        if probes.struct_basis is not None else None
    )
    mean_np = (
        np.array(probes.struct_mean.reshape(-1).tolist(), dtype=np.float32)
        if probes.struct_mean is not None else None
    )
    vec = _clean_single_np(vec, struct_np, mean_np)
    proj = mx.sum((vec - probes.tonal_mean) * probes.tonal_direction)
    mx.eval(proj)
    return float(proj.item())


# ── Mode 7: 5-class query classifier ──────────────────────────────────

def _calibrate_m7_classifier(kv_gen, tokenizer, compass_layer: int, lib=None):
    """Train a 1-vs-rest linear classifier for 5 query types.

    For each class, compute the mean direction. At query time, project
    query onto each class direction, pick highest confidence.

    Structural PCs are removed before computing class centroids so the
    directions capture content-type differences, not format structure.

    Returns (class_directions, class_means) dicts keyed by class name.
    """
    classes = sorted(set(label for _, label in _M7_EXAMPLES))
    vecs_by_class: dict[str, list[mx.array]] = {c: [] for c in classes}
    all_vecs: list[mx.array] = []

    for query_text, label in _M7_EXAMPLES:
        prompt = (
            f"<start_of_turn>user\n{query_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        h = kv_gen.prefill_to_layer(
            mx.array(ids)[None], target_layer=compass_layer,
            sample_positions=[len(ids) - 1],
        )
        mx.eval(h)
        vec = h[0, 0, :].astype(mx.float32)
        vecs_by_class[label].append(vec)
        all_vecs.append(vec)

    # Remove structural PCs before computing class centroids
    struct_np, mean_np = _get_lib_structural_np(lib)
    all_vecs = _clean_vecs(all_vecs, struct_np, mean_np)
    # Rebuild vecs_by_class from cleaned all_vecs (same order as _M7_EXAMPLES)
    vecs_by_class = {c: [] for c in classes}
    for (_, label), vec in zip(_M7_EXAMPLES, all_vecs):
        vecs_by_class[label].append(vec)

    # Global mean
    global_mean = mx.mean(mx.stack(all_vecs, axis=0), axis=0)
    mx.eval(global_mean)

    # Per-class mean direction (centroid - global mean, normalized)
    class_directions: dict[str, mx.array] = {}
    class_means: dict[str, mx.array] = {}
    for cls_name, vecs in vecs_by_class.items():
        if not vecs:
            continue
        cls_mean = mx.mean(mx.stack(vecs, axis=0), axis=0)
        direction = cls_mean - global_mean
        norm = mx.sqrt(mx.sum(direction * direction))
        mx.eval(norm)
        if float(norm.item()) > 1e-6:
            direction = direction / norm
        mx.eval(direction)
        class_directions[cls_name] = direction
        class_means[cls_name] = global_mean  # all share global mean

    # Validate: classify training set using already-cleaned all_vecs
    correct = 0
    for (_, true_label), vec in zip(_M7_EXAMPLES, all_vecs):
        centered = vec - global_mean
        best_cls = max(
            class_directions.keys(),
            key=lambda c: float(mx.sum(centered * class_directions[c]).item()),
        )
        if best_cls == true_label:
            correct += 1

    accuracy = correct / len(_M7_EXAMPLES) * 100
    cleaned_label = " struct-cleaned" if struct_np is not None else ""
    print(
        f"    Mode 7 classifier{cleaned_label}: {len(classes)} classes, "
        f"train accuracy {accuracy:.0f}% ({correct}/{len(_M7_EXAMPLES)})",
        file=sys.stderr,
    )

    return class_directions, class_means


def _tension_assess_window(kv_gen, tokenizer, w_tokens, compass_layer: int):
    """Tension-specific assessment: model rates danger/criticality.

    Same structure as _assess_window but with tension-oriented prompt.
    Uses extend_to_layer on the final generated token — no double forward pass.
    Returns (rating, L26_residual).
    """
    pre_text = "<start_of_turn>user\nHere is a text excerpt:\n\n"
    pre_ids = tokenizer.encode(pre_text, add_special_tokens=True)
    post_text = (
        "\n\nHow tense, dangerous, or critical is the situation "
        "described? Rate from 1-5.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    post_ids = tokenizer.encode(post_text, add_special_tokens=False)

    sl = 0
    _l, kv = kv_gen.prefill(mx.array(pre_ids)[None])
    mx.eval(*[t for p in kv for t in p])
    sl += len(pre_ids)
    _l, kv = kv_gen.extend(mx.array(w_tokens)[None], kv, abs_start=sl)
    mx.eval(*[t for p in kv for t in p])
    sl += len(w_tokens)
    logits, kv = kv_gen.extend(mx.array(post_ids)[None], kv, abs_start=sl)
    sl += len(post_ids)

    gen_tokens: list[int] = []
    eos = tokenizer.eos_token_id
    last_tok_kv = kv
    last_tok_sl = sl
    for _ in range(_TONAL_ASSESS_TOKENS):
        tok = int(mx.argmax(logits[0, -1]).item())
        if eos is not None and tok == eos:
            break
        gen_tokens.append(tok)
        last_tok_kv = kv
        last_tok_sl = sl
        logits, kv = kv_gen.step_uncompiled(mx.array([[tok]]), kv, seq_len=sl)
        sl += 1

    # Parse rating
    judge_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    m = re.search(r'[1-5]', judge_text)
    rating = int(m.group()) if m else 3

    if gen_tokens:
        _, _, layer_h = kv_gen.extend_to_layer(
            mx.array([[gen_tokens[-1]]]), last_tok_kv,
            abs_start=last_tok_sl, target_layer=compass_layer,
        )
        vec = layer_h[0, 0, :].astype(mx.float32)
        mx.eval(vec)
    else:
        full = list(pre_ids) + list(w_tokens) + list(post_ids)
        h = kv_gen.prefill_to_layer(
            mx.array(full)[None], target_layer=compass_layer,
            sample_positions=[len(full) - 1],
        )
        mx.eval(h)
        vec = h[0, 0, :].astype(mx.float32)

    return rating, vec


def _calibrate_tension(kv_gen, tokenizer, compass_layer: int, lib):
    """Train tension probe using BM25-guided supervision.

    The old approach (unsupervised PCA on uniform samples) fails because
    most windows are routine — PC1 captures structural variance, not
    tension. sep=104 is essentially random.

    New approach: use BM25 tension indicators to find ~10 keyword-rich
    windows (likely tense) and ~10 keyword-free windows (likely calm).
    Compute difference-of-means as the probe direction. The supervision
    comes from keywords which cover ~55% of tense content — enough to
    orient the probe correctly. The dark space captures the rest.
    """
    # Use BM25 tension indicators to find tense vs calm windows
    try:
        from ..compass_routing._indicator_bm25 import (
            TENSION_INDICATORS,
            _indicator_bm25_score_windows,
        )
        bm25_scores = _indicator_bm25_score_windows(
            lib, tokenizer, TENSION_INDICATORS,
        )
    except Exception:
        bm25_scores = []

    # Split into tense (top BM25) and calm (zero BM25 score)
    tense_wids = [wid for wid, s in bm25_scores[:10] if s > 0.0]
    calm_wids = [wid for wid, s in bm25_scores if s == 0.0]

    if len(tense_wids) < 3:
        print(
            f"    Tension: only {len(tense_wids)} keyword windows, "
            "falling back to synthetic-only",
            file=sys.stderr,
        )
        tense_wids = []

    # Sample calm windows evenly across the document
    if len(calm_wids) > 10:
        step = max(1, len(calm_wids) // 10)
        calm_wids = calm_wids[::step][:10]

    # Assess tense windows
    tense_vecs: list[mx.array] = []
    for i, wid in enumerate(tense_wids):
        w_tokens = lib.get_window_tokens(wid)
        _, vec = _tension_assess_window(kv_gen, tokenizer, w_tokens, compass_layer)
        tense_vecs.append(vec)
        if (i + 1) % 5 == 0:
            print(
                f"      ... tension assessed {i + 1}/{len(tense_wids)} tense windows",
                file=sys.stderr,
            )

    # Assess calm windows
    calm_vecs: list[mx.array] = []
    for i, wid in enumerate(calm_wids):
        w_tokens = lib.get_window_tokens(wid)
        _, vec = _tension_assess_window(kv_gen, tokenizer, w_tokens, compass_layer)
        calm_vecs.append(vec)
        if (i + 1) % 5 == 0:
            print(
                f"      ... tension assessed {i + 1}/{len(calm_wids)} calm windows",
                file=sys.stderr,
            )

    # Add synthetic anchors to strengthen the direction
    tense_tokens = tokenizer.encode(
        "ALARM! The fuel cell pressure is dropping rapidly. Abort abort abort! "
        "We have a critical malfunction. All hands to emergency stations.",
        add_special_tokens=False,
    )
    calm_tokens = tokenizer.encode(
        "The crew is sleeping peacefully during the translunar coast. "
        "All systems nominal. Quiet night shift.",
        add_special_tokens=False,
    )
    _, tense_synth = _tension_assess_window(
        kv_gen, tokenizer, tense_tokens, compass_layer,
    )
    _, calm_synth = _tension_assess_window(
        kv_gen, tokenizer, calm_tokens, compass_layer,
    )
    tense_vecs.append(tense_synth)
    calm_vecs.append(calm_synth)

    if len(tense_vecs) < 2 or len(calm_vecs) < 2:
        print("    Tension: insufficient samples, skipping", file=sys.stderr)
        return None, None, 0.0

    # Remove structural PCs before diff-of-means so the probe direction
    # captures semantic tension, not format/structural variance.
    struct_np, mean_np = _get_lib_structural_np(lib)
    tense_vecs = _clean_vecs(tense_vecs, struct_np, mean_np)
    calm_vecs = _clean_vecs(calm_vecs, struct_np, mean_np)

    # Difference of means → probe direction
    tense_mean = mx.mean(mx.stack(tense_vecs, axis=0), axis=0)
    calm_mean = mx.mean(mx.stack(calm_vecs, axis=0), axis=0)
    all_vecs = tense_vecs + calm_vecs
    global_mean = mx.mean(mx.stack(all_vecs, axis=0), axis=0)

    direction = tense_mean - calm_mean
    norm = mx.sqrt(mx.sum(direction * direction))
    mx.eval(norm)
    if float(norm.item()) > 1e-6:
        direction = direction / norm
    mx.eval(direction, global_mean)

    # Measure separation
    tense_proj = float(mx.sum((tense_mean - global_mean) * direction).item())
    calm_proj = float(mx.sum((calm_mean - global_mean) * direction).item())
    threshold = (tense_proj + calm_proj) / 2

    cleaned_label = " struct-cleaned" if struct_np is not None else ""
    print(
        f"    Tension{cleaned_label}: supervised, {len(tense_vecs)} tense + {len(calm_vecs)} calm, "
        f"tense={tense_proj:+.0f} calm={calm_proj:+.0f} "
        f"sep={abs(tense_proj - calm_proj):.0f}",
        file=sys.stderr,
    )
    return direction, global_mean, threshold


def classify_query_m7(
    kv_gen,
    tokenizer,
    prompt_text: str,
    compass_layer: int,
    probes: ProbeSet,
) -> tuple[QueryType, float]:
    """Classify a query into one of 5 types using Mode 7 probes.

    Returns (query_type, confidence) where confidence is the margin
    between the best and second-best class projections.
    """
    prompt = (
        f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    h = kv_gen.prefill_to_layer(
        mx.array(ids)[None], target_layer=compass_layer,
        sample_positions=[len(ids) - 1],
    )
    mx.eval(h)
    vec = h[0, 0, :].astype(mx.float32)

    # Apply structural cleaning — same basis used during calibration
    struct_np = (
        np.array(probes.struct_basis.tolist(), dtype=np.float32)
        if probes.struct_basis is not None else None
    )
    mean_np = (
        np.array(probes.struct_mean.reshape(-1).tolist(), dtype=np.float32)
        if probes.struct_mean is not None else None
    )
    vec = _clean_single_np(vec, struct_np, mean_np)

    if not probes.m7_available or probes.m7_class_directions is None:
        # Fall back to legacy 2-class
        qt_proj = mx.sum((vec - probes.qt_mean) * probes.qt_direction)
        mx.eval(qt_proj)
        is_exploration = float(qt_proj.item()) > probes.qt_threshold
        return (QueryType.ENGAGEMENT if is_exploration else QueryType.FACTUAL), 0.0

    # Project onto each class direction
    global_mean = list(probes.m7_class_means.values())[0]
    centered = vec - global_mean

    projections: dict[str, float] = {}
    for cls_name, direction in probes.m7_class_directions.items():
        proj = float(mx.sum(centered * direction).item())
        projections[cls_name] = proj

    # Pick best
    sorted_classes = sorted(projections.items(), key=lambda x: -x[1])
    best_cls, best_score = sorted_classes[0]
    second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
    confidence = best_score - second_score

    return QueryType(best_cls), confidence


# ── Public API ───────────────────────────────────────────────────────

def load_or_calibrate(
    kv_gen, tokenizer, compass_layer: int, lib,
    checkpoint_dir: str | Path, model_name: str,
) -> ProbeSet:
    """Load cached probes or calibrate from scratch."""
    cache_path = _probe_cache_path(checkpoint_dir, model_name)

    if cache_path.exists():
        try:
            data = mx.load(str(cache_path))
            # Load Mode 7 classifier directions if present
            m7_classes = ["engagement", "factual", "global", "tension", "tone"]
            m7_dirs: dict[str, mx.array] | None = None
            m7_means: dict[str, mx.array] | None = None
            m7_avail = False
            if f"m7_dir_{m7_classes[0]}" in data:
                m7_dirs = {c: data[f"m7_dir_{c}"] for c in m7_classes}
                m7_means = {c: data["m7_global_mean"] for c in m7_classes}
                m7_avail = True

            # Load tension probe if present
            tension_dir = data.get("tension_direction")
            tension_mean = data.get("tension_mean")
            tension_avail = tension_dir is not None

            # Load structural basis if present
            struct_basis = data.get("struct_basis")
            struct_mean = data.get("struct_mean")

            probes = ProbeSet(
                qt_direction=data["qt_direction"],
                qt_mean=data["qt_mean"],
                qt_threshold=float(data["qt_threshold"].item()),
                tonal_direction=data["tonal_direction"],
                tonal_mean=data["tonal_mean"],
                tonal_threshold=float(data["tonal_threshold"].item()),
                tonal_available=bool(int(data["tonal_available"].item())),
                m7_class_directions=m7_dirs,
                m7_class_means=m7_means,
                m7_available=m7_avail,
                tension_direction=tension_dir,
                tension_mean=tension_mean,
                tension_available=tension_avail,
                struct_basis=struct_basis,
                struct_mean=struct_mean,
            )
            m7_label = " + Mode 7" if m7_avail else ""
            tension_label = " + tension" if tension_avail else ""
            struct_label = " + struct-clean" if struct_basis is not None else ""
            print(
                f"  Loaded cached probes: {cache_path.name}"
                f"{m7_label}{tension_label}{struct_label}",
                file=sys.stderr,
            )
            return probes
        except Exception:
            pass

    print("  Calibrating probes (first run, will cache)...", file=sys.stderr)
    t0 = time.time()

    # Extract structural basis from library once — shared across all probes
    struct_np, struct_mean_np = _get_lib_structural_np(lib)
    struct_basis_mx = mx.array(struct_np) if struct_np is not None else None
    struct_mean_mx = mx.array(struct_mean_np) if struct_mean_np is not None else None
    if struct_np is not None:
        print(
            f"  Structural basis: {struct_np.shape[0]} PCs from library compass",
            file=sys.stderr,
        )

    # Query-type (legacy 2-class)
    qt_pc1, qt_mean, qt_thresh = _calibrate_query_type(kv_gen, tokenizer, compass_layer, lib)

    # Tonal (engagement probe)
    tonal_pc1, tonal_mean, tonal_thresh = _calibrate_tonal(
        kv_gen, tokenizer, compass_layer, lib,
    )
    tonal_available = tonal_pc1 is not None
    if not tonal_available:
        dim = qt_pc1.shape[0]
        tonal_pc1 = mx.zeros((dim,))
        tonal_mean = mx.zeros((dim,))
        tonal_thresh = 0.0

    # Mode 7: 5-class query classifier
    m7_class_directions, m7_class_means = _calibrate_m7_classifier(
        kv_gen, tokenizer, compass_layer, lib,
    )

    # Mode 7: tension probe
    tension_pc1, tension_mean, tension_thresh = _calibrate_tension(
        kv_gen, tokenizer, compass_layer, lib,
    )
    tension_available = tension_pc1 is not None
    if not tension_available:
        dim = qt_pc1.shape[0]
        tension_pc1 = mx.zeros((dim,))
        tension_mean = mx.zeros((dim,))

    probes = ProbeSet(
        qt_direction=qt_pc1, qt_mean=qt_mean, qt_threshold=qt_thresh,
        tonal_direction=tonal_pc1, tonal_mean=tonal_mean,
        tonal_threshold=tonal_thresh, tonal_available=tonal_available,
        m7_class_directions=m7_class_directions,
        m7_class_means=m7_class_means,
        m7_available=bool(m7_class_directions),
        tension_direction=tension_pc1,
        tension_mean=tension_mean,
        tension_available=tension_available,
        struct_basis=struct_basis_mx,
        struct_mean=struct_mean_mx,
    )

    # Save everything to cache
    save_kwargs = dict(
        qt_direction=qt_pc1, qt_mean=qt_mean,
        qt_threshold=mx.array(qt_thresh),
        tonal_direction=tonal_pc1, tonal_mean=tonal_mean,
        tonal_threshold=mx.array(tonal_thresh),
        tonal_available=mx.array(1 if tonal_available else 0),
        tension_direction=tension_pc1, tension_mean=tension_mean,
    )

    # Save structural basis for runtime cleaning
    if struct_basis_mx is not None:
        save_kwargs["struct_basis"] = struct_basis_mx
        save_kwargs["struct_mean"] = struct_mean_mx

    # Save Mode 7 classifier directions
    if m7_class_directions:
        for cls_name, direction in m7_class_directions.items():
            save_kwargs[f"m7_dir_{cls_name}"] = direction
        # Save global mean (shared across classes)
        global_mean = list(m7_class_means.values())[0]
        save_kwargs["m7_global_mean"] = global_mean

    mx.savez(str(cache_path), **save_kwargs)

    elapsed = time.time() - t0
    print(f"  Probes calibrated in {elapsed:.1f}s → {cache_path.name}", file=sys.stderr)
    return probes
