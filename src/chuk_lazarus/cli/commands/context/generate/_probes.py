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
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class ProbeSet:
    """Calibrated probe directions."""
    # Query-type: exploration vs factual
    qt_direction: mx.array    # (hidden_dim,) positive = exploration
    qt_mean: mx.array         # (hidden_dim,)
    qt_threshold: float

    # Tonal: engaging vs routine (generation-mode L26)
    tonal_direction: mx.array  # (hidden_dim,) positive = engaging
    tonal_mean: mx.array       # (hidden_dim,)
    tonal_threshold: float
    tonal_available: bool      # False if calibration had insufficient contrast


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
_CACHE_VERSION = 3  # bump to invalidate stale caches


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


# ── Query-type calibration ───────────────────────────────────────────

def _calibrate_query_type(kv_gen, tokenizer, compass_layer: int):
    """PCA direction from generic exploration/factual examples."""
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

    pc1, mean, thresh, var_pct, pos_m, neg_m = _pca_direction(
        vecs, labels, positive_label=True,
    )
    print(
        f"    Query-type: PC1={var_pct:.0f}% variance, "
        f"E={pos_m:+.0f} F={neg_m:+.0f} sep={abs(pos_m - neg_m):.0f}",
        file=sys.stderr,
    )
    return pc1, mean, thresh


# ── Tonal calibration (generation-mode) ──────────────────────────────

def _assess_window(kv_gen, tokenizer, w_tokens, compass_layer: int):
    """Model reads window + assessment prompt, generates 20 tokens.

    Returns (rating, L26_residual) where rating is parsed from text
    and L26_residual is extracted at the last generated token.
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
    for _ in range(_TONAL_ASSESS_TOKENS):
        tok = int(mx.argmax(logits[0, -1]).item())
        if eos is not None and tok == eos:
            break
        gen_tokens.append(tok)
        logits, kv = kv_gen.step_uncompiled(mx.array([[tok]]), kv, seq_len=sl)
        sl += 1

    # Parse rating
    judge_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    m = re.search(r'[1-5]', judge_text)
    rating = int(m.group()) if m else 3

    # Extract L26 at last generated token
    full = list(pre_ids) + list(w_tokens) + list(post_ids) + gen_tokens
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

    amusing_proj = float(mx.sum((amusing_vec - mean) * pc1).item())
    routine_proj = float(mx.sum((routine_vec - mean) * pc1).item())

    if routine_proj > amusing_proj:
        pc1 = -pc1
        amusing_proj, routine_proj = -routine_proj, -amusing_proj

    threshold = (amusing_proj + routine_proj) / 2

    print(
        f"    Tonal: PC1={variance_pct:.0f}% variance, "
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
    proj = mx.sum((vec - probes.tonal_mean) * probes.tonal_direction)
    mx.eval(proj)
    return float(proj.item())


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
            probes = ProbeSet(
                qt_direction=data["qt_direction"],
                qt_mean=data["qt_mean"],
                qt_threshold=float(data["qt_threshold"].item()),
                tonal_direction=data["tonal_direction"],
                tonal_mean=data["tonal_mean"],
                tonal_threshold=float(data["tonal_threshold"].item()),
                tonal_available=bool(int(data["tonal_available"].item())),
            )
            print(f"  Loaded cached probes: {cache_path.name}", file=sys.stderr)
            return probes
        except Exception:
            pass

    print("  Calibrating probes (first run, will cache)...", file=sys.stderr)
    t0 = time.time()

    # Query-type
    qt_pc1, qt_mean, qt_thresh = _calibrate_query_type(kv_gen, tokenizer, compass_layer)

    # Tonal
    tonal_pc1, tonal_mean, tonal_thresh = _calibrate_tonal(
        kv_gen, tokenizer, compass_layer, lib,
    )
    tonal_available = tonal_pc1 is not None
    if not tonal_available:
        dim = qt_pc1.shape[0]
        tonal_pc1 = mx.zeros((dim,))
        tonal_mean = mx.zeros((dim,))
        tonal_thresh = 0.0

    probes = ProbeSet(
        qt_direction=qt_pc1, qt_mean=qt_mean, qt_threshold=qt_thresh,
        tonal_direction=tonal_pc1, tonal_mean=tonal_mean,
        tonal_threshold=tonal_thresh, tonal_available=tonal_available,
    )

    mx.savez(
        str(cache_path),
        qt_direction=qt_pc1, qt_mean=qt_mean,
        qt_threshold=mx.array(qt_thresh),
        tonal_direction=tonal_pc1, tonal_mean=tonal_mean,
        tonal_threshold=mx.array(tonal_thresh),
        tonal_available=mx.array(1 if tonal_available else 0),
    )

    elapsed = time.time() - t0
    print(f"  Probes calibrated in {elapsed:.1f}s → {cache_path.name}", file=sys.stderr)
    return probes
