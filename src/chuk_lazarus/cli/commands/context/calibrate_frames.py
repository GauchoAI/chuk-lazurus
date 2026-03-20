"""Calibrate dark space coordinate frames for a model.

Two methods:

  whitening (default):
    No predefined categories. Let the model's own geometry define the
    navigation space. Run diverse probes through L26, PCA, skip structural
    PCs (top K), keep the next N PCs, scale each by 1/sqrt(eigenvalue) to
    equalise variance. The whitening transform amplifies the dark space
    signal that structural dominance was hiding.

  category:
    Human-defined content categories (event, sports, technical, etc.).
    Cross-domain PCA with Fisher criterion scoring per category.
    Deduplicated across categories.

Usage:
    lazarus context calibrate-frames \
        --model google/gemma-3-4b-it \
        --output ./frame_bank.npz

    lazarus context calibrate-frames \
        --model google/gemma-3-4b-it \
        --method category \
        --output ./frame_bank.npz
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path

# ---------------------------------------------------------------------------
# Diverse probe prompts — used by both methods
# ---------------------------------------------------------------------------

PROBE_PROMPTS: list[str] = [
    # Events / happenings
    "The historic moment when the spacecraft landed on the surface was",
    "At the exact moment of touchdown everyone in the room erupted",
    "The announcement was made that the mission had successfully completed",
    "When they finally arrived at the destination the crew reported",
    "The celebration began immediately after the successful landing on",
    "Breaking news the team has just completed the first ever attempt",
    "The crowd erupted when they heard the words confirming that",
    "At precisely that moment history was made as the vehicle",
    "The mission achieved its primary objective when the module",
    "After years of preparation the event finally occurred and the",
    # Sports
    "The final score was seven to three in favor of the visiting",
    "In the bottom of the ninth inning the batter hit a deep",
    "The championship game ended with a dramatic overtime victory",
    "The teams winning streak continued with a victory over the",
    "The quarterback threw a touchdown pass to the wide receiver",
    "The league standings showed the team in first place with a",
    "The pitcher struck out twelve batters in the complete game",
    "Final results New York five Boston three with the winning run",
    "The all star game featured players from both the American and",
    "The season record now stands at fourteen wins and six losses",
    # Technical / measurements
    "The pressure reading on gauge four showed a value of approximately",
    "Temperature at the sensor was measured at exactly thirty two",
    "The voltage across the circuit was recorded as twelve point five",
    "Telemetry data indicated that the altitude was holding steady at",
    "The fuel remaining in tank two was approximately forty seven",
    "Instrument readings confirmed that the velocity was within nominal",
    "The signal strength measured at the antenna was negative twenty",
    "Calibration of the gyroscope showed a drift rate of zero point",
    "The oxygen flow rate was steady at point two liters per",
    "Power consumption readings indicated a total draw of eight hundred",
    # Dialogue / communication
    "Houston we are reading you loud and clear standing by for your",
    "Roger that we copy your transmission proceeding with the planned",
    "Go ahead with your report we are listening on this frequency",
    "Copy we have confirmation of your status please verify the",
    "Understood will comply with your instructions regarding the next",
    "This is mission control we need you to check the status of",
    "Affirmative we see the same readings on our displays down here",
    "Please repeat your last transmission we had some interference on",
    "All stations this is a priority call regarding the upcoming",
    "We read you five by five go ahead with your status report",
    # Weather / conditions
    "Current conditions wind from the northwest at fifteen knots gusting",
    "Visibility reduced to approximately two miles due to fog and low",
    "Scattered clouds at five thousand feet with a ceiling of eight",
    "The weather forecast calls for clear skies with temperatures reaching",
    "Barometric pressure has been falling steadily since early this morning",
    "Surface winds are gusting up to thirty knots from the southwest",
    "The storm system is expected to move through the area by late",
    "Temperature at the surface is seventy two degrees with humidity at",
    "Cloud cover has increased to broken at eight thousand with occasional",
    "Weather advisory expect moderate turbulence at flight level three five",
    # Narrative / human
    "Looking out the window he could see the Earth rising above the",
    "She remembered the first time they had discussed the possibility",
    "The feeling of weightlessness was unlike anything he had ever",
    "As they gathered around the table the conversation turned to memories",
    "In that quiet moment he reflected on everything that had led to",
    "The children watched in amazement as the rocket lifted off from",
    "Years later she would recall that day as the most significant",
    "His voice trembled slightly as he spoke the words that would",
    "They knew this would be their last chance to see the surface",
    "The photograph captured a moment that would be remembered for",
    # Procedural / checklists
    "Step one verify that all switches are in the nominal position",
    "Procedure requires confirmation before initiating the sequence for",
    "Checklist item seven ensure the backup system is fully operational",
    "Before proceeding verify that the following conditions are satisfied",
    "The sequence begins with activation of the primary control system",
    "Confirm that pressure in the manifold is within the specified range",
    "Following the abort procedure immediately set the master switch to",
    "The preflight checklist requires visual inspection of all external",
    "Execute the following steps in order first arm the pyrotechnic",
    "Standard operating procedure calls for a complete check of all",
    # Scientific / analysis
    "The geological samples collected from the surface revealed traces of",
    "Analysis of the spectral data indicated the presence of iron oxide",
    "The experiment confirmed the theoretical prediction that gravity",
    "Observations through the telescope showed that the surface features",
    "The chemical composition of the sample was determined to contain",
    "Research findings suggest that the formation was created by volcanic",
    "The data collected strongly supports the hypothesis that water once",
    "Microscopic examination of the material revealed a crystalline structure",
    "The radiation measurements were consistent with expected solar wind",
    "Preliminary analysis of the core sample indicates organic compounds",
]

# Category labels (only used by --method category)
PROBE_CATEGORIES: dict[str, list[int]] = {
    "event": list(range(0, 10)),
    "sports": list(range(10, 20)),
    "technical": list(range(20, 30)),
    "dialogue": list(range(30, 40)),
    "weather": list(range(40, 50)),
    "narrative": list(range(50, 60)),
    "procedural": list(range(60, 70)),
    "scientific": list(range(70, 80)),
}


async def context_calibrate_frames_cmd(args: Namespace) -> None:
    """CLI entry point: discover dark space coordinate frames for a model."""
    import mlx.core as mx
    import numpy as np

    from ....inference import UnifiedPipeline
    from ....inference.context.kv_generator import make_kv_generator

    output_path = Path(args.output)
    n_dims = getattr(args, "dims_per_frame", 64)
    target_layer_frac = getattr(args, "layer_frac", 0.77)
    method = getattr(args, "method", "whitening")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}", file=sys.stderr)
    pipeline = UnifiedPipeline.from_pretrained(args.model, verbose=False)
    tokenizer = pipeline.tokenizer
    kv_gen = make_kv_generator(pipeline.model, pipeline.config)

    num_layers = pipeline.config.num_hidden_layers
    compass_layer = round(num_layers * target_layer_frac)

    # Warm up
    _warm = mx.array([[1, 2, 3]])
    _, _kv = kv_gen.prefill(_warm)
    mx.eval()

    print(
        f"Model: {args.model}  |  {num_layers} layers  |  "
        f"commitment layer: L{compass_layer}  |  method: {method}",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 2. Extract L26 residuals for all probes
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    all_residuals: list[np.ndarray] = []

    for prompt in PROBE_PROMPTS:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array(token_ids)[None]
        h = kv_gen.prefill_to_layer(input_ids, target_layer=compass_layer)
        vec = h[0, -1, :]
        all_residuals.append(np.array(vec.tolist(), dtype=np.float32))

    elapsed = time.monotonic() - t0
    total_probes = len(PROBE_PROMPTS)
    print(
        f"  Extracted {total_probes} probe residuals in {elapsed:.1f}s "
        f"({total_probes / elapsed:.0f} probes/s)",
        file=sys.stderr,
    )

    X = np.stack(all_residuals, axis=0)  # (N, hidden_dim)

    # ------------------------------------------------------------------
    # 3. Build frame bank
    # ------------------------------------------------------------------
    if method == "whitening":
        all_frames, total_dims = _build_whitening_bank(X, n_dims)
    else:
        all_frames, total_dims = _build_category_bank(X, n_dims)

    # ------------------------------------------------------------------
    # 4. Save frame bank
    # ------------------------------------------------------------------
    save_dict = {
        "frame_bank": mx.array(all_frames),
        "compass_layer": mx.array(compass_layer),
        "n_categories": mx.array(0),  # 0 = model-driven, no categories
        "dims_per_frame": mx.array(total_dims),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output_path), **save_dict)

    file_size = output_path.stat().st_size
    print(
        f"\n  Saved: {output_path} ({file_size / 1024:.0f} KB)",
        file=sys.stderr,
    )
    print(
        f"\n  Use with: lazarus context prefill ... "
        f"--residual-mode darkspace --frame-bank {output_path}",
        file=sys.stderr,
    )


def _build_whitening_bank(X, n_dims: int):
    """Whitening method — no categories, pure model geometry.

    1. PCA on diverse probe residuals
    2. Auto-detect structural boundary (where eigenvalue ratios flatten)
    3. Keep next n_dims PCs after the structural cutoff
    4. Scale each by 1/sqrt(eigenvalue) → whitening transform

    The whitening equalises variance across content dimensions.
    Structural PCs (high variance, shared across all content) are removed.
    The remaining dimensions carry the dark space navigation signal.
    """
    import numpy as np

    mean = X.mean(axis=0)
    centered = X - mean
    _U, S_vals, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S_vals**2) / (X.shape[0] - 1)
    explained = eigenvalues / eigenvalues.sum()

    # Auto-detect structural boundary (same algorithm as compass calibration)
    structural_end = 0
    for i in range(min(len(explained) - 3, 50)):
        ratios = [explained[i + j] / max(explained[i + j + 1], 1e-10) for j in range(3)]
        if all(r < 1.5 for r in ratios):
            structural_end = i
            break
    else:
        structural_end = 4  # safe default

    # Keep n_dims PCs after structural boundary
    pc_start = structural_end
    pc_end = min(pc_start + n_dims, len(S_vals))
    actual_dims = pc_end - pc_start

    # Whitening transform: scale each PC by 1/sqrt(eigenvalue)
    # This equalises variance — no PC dominates in cosine matching
    whitening_scales = 1.0 / np.sqrt(eigenvalues[pc_start:pc_end] + 1e-10)
    frame_bank = Vt[pc_start:pc_end] * whitening_scales[:, None]
    # frame_bank shape: (actual_dims, hidden_dim)

    # Report
    structural_var = explained[:pc_start].sum() * 100
    content_var = explained[pc_start:pc_end].sum() * 100
    tail_var = explained[pc_end:].sum() * 100

    print("\n  Whitening frame bank:", file=sys.stderr)
    print(
        f"    Structural PCs 0-{pc_start - 1}: {structural_var:.1f}% variance (removed)",
        file=sys.stderr,
    )
    print(
        f"    Content PCs {pc_start}-{pc_end - 1}: {content_var:.1f}% variance (whitened)",
        file=sys.stderr,
    )
    print(
        f"    Tail PCs {pc_end}+: {tail_var:.1f}% variance (truncated)",
        file=sys.stderr,
    )
    print(
        f"    Frame bank: {actual_dims}D  |  eigenvalue range: "
        f"{eigenvalues[pc_start]:.4f} → {eigenvalues[pc_end - 1]:.4f}",
        file=sys.stderr,
    )

    return frame_bank, actual_dims


def _build_category_bank(X, n_dims: int):
    """Category method — Fisher criterion per predefined category.

    Cross-domain PCA + Fisher scoring. Deduplicated across categories.
    """
    import numpy as np

    mean = X.mean(axis=0)
    centered = X - mean
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)

    category_names = list(PROBE_CATEGORIES.keys())
    unique_pc_indices: list[int] = []
    seen_pcs: set[int] = set()

    print("\n  Discovering coordinate frames:", file=sys.stderr)

    for cat_name in category_names:
        target_idx = PROBE_CATEGORIES[cat_name]
        other_idx = [i for i in range(len(X)) if i not in target_idx]

        n_check = min(50, Vt.shape[0])
        target_proj = (X[target_idx] - mean) @ Vt[:n_check].T
        other_proj = (X[other_idx] - mean) @ Vt[:n_check].T

        fisher_scores = []
        for i in range(n_check):
            t_mean = target_proj[:, i].mean()
            o_mean = other_proj[:, i].mean()
            t_var = target_proj[:, i].var() + 1e-10
            o_var = other_proj[:, i].var() + 1e-10
            fisher = (t_mean - o_mean) ** 2 / (t_var + o_var)
            fisher_scores.append((i, float(fisher)))

        fisher_scores.sort(key=lambda x: -x[1])
        top_pcs = fisher_scores[:n_dims]

        for pc_idx, _ in top_pcs:
            if pc_idx not in seen_pcs:
                unique_pc_indices.append(pc_idx)
                seen_pcs.add(pc_idx)

        pcs_used = [idx for idx, _ in top_pcs]
        print(
            f"    {cat_name:12s}: PCs {pcs_used}  Fisher {top_pcs[0][1]:.2f}→{top_pcs[-1][1]:.2f}",
            file=sys.stderr,
        )

    unique_pc_indices.sort()
    frame_bank = Vt[unique_pc_indices]
    total_dims = len(unique_pc_indices)

    print(
        f"\n  Frame bank: {total_dims}D (deduplicated from "
        f"{len(category_names)} categories × {n_dims}D)",
        file=sys.stderr,
    )

    return frame_bank, total_dims


__all__ = ["context_calibrate_frames_cmd"]
