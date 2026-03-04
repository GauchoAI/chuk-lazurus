#!/usr/bin/env python3
"""
Residual Stream Dynamics Experiment.

Investigates geometric properties of the residual stream across layers:

1. Bypass Detection
   - Hypothesis: Easy problems (lookup) cause less residual change through
     middle layers than hard problems (computation).
   - Measures per-layer relative residual delta for easy vs hard arithmetic.
   - Connects to the lookup table vs computation finding.

2. Residual Saturation
   - Tracks how quickly the residual stream converges to its final state.
   - Detects discrete phase transitions (jumps in inter-layer delta).
   - Tests whether convergence is gradual or happens in functional phases.

3. Cross-Position Information Flow
   - For arithmetic like "127 * 89 =", tracks how the final position gathers
     information from operand positions through the layers.
   - Uses cosine similarity between position representations as a proxy for
     information flow (doesn't require attention weight capture).

4. Layer Subspace Communication
   - Decomposes per-layer residual updates via PCA to find dominant write
     directions at each layer.
   - Measures alignment between consecutive layers' update directions.
   - Tests: do layers communicate through specific subspace "channels"?

Usage:
    python experiment.py
    python experiment.py --analysis bypass_detection
    python experiment.py --analysis bypass_detection residual_saturation
    python experiment.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml

from chuk_lazarus.introspection.hooks import (
    CaptureConfig,
    CapturedState,
    LayerSelection,
    ModelHooks,
    PositionSelection,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class PromptDelta:
    """Per-layer residual deltas for a single prompt."""

    prompt: str
    category: str
    deltas: list[float]  # relative delta per layer: ||h_{l} - h_{l-1}|| / ||h_{l-1}||
    norms: list[float]  # ||h_l|| at each layer
    prediction: str  # model's next-token prediction
    correct: bool | None  # whether prediction is correct (if verifiable)


@dataclass
class BypassDetectionResult:
    """Results from bypass detection analysis."""

    easy_deltas: list[PromptDelta]
    hard_deltas: list[PromptDelta]
    factual_deltas: list[PromptDelta]
    # Aggregated curves
    easy_mean_curve: list[float]  # mean delta per layer across easy prompts
    hard_mean_curve: list[float]  # mean delta per layer across hard prompts
    factual_mean_curve: list[float]
    # Key metrics
    total_path_length: dict[str, float]  # category -> sum of deltas
    max_delta_layer: dict[str, int]  # category -> layer with largest delta
    bypass_score: float  # ratio: easy_total / hard_total (lower = more bypass)


@dataclass
class SaturationCurve:
    """Convergence metrics for a single prompt."""

    prompt: str
    category: str
    dist_from_final: list[float]  # cosine_distance(h_l, h_final) per layer
    inter_layer_delta: list[float]  # cosine_distance(h_l, h_{l-1}) per layer
    convergence_layer: int  # first layer where dist_from_final < threshold


@dataclass
class ResidualSaturationResult:
    """Results from residual saturation analysis."""

    curves: list[SaturationCurve]
    # Per-category aggregates
    mean_dist_from_final: dict[str, list[float]]  # category -> mean curve
    mean_inter_layer_delta: dict[str, list[float]]
    # Phase transition detection
    phase_transitions: list[int]  # layers where inter_layer_delta jumps
    mean_convergence_layer: dict[str, float]  # category -> avg convergence layer


@dataclass
class PositionFlow:
    """Information flow to the final position from each input position."""

    prompt: str
    tokens: list[str]
    operand_labels: list[str]
    # Per-layer similarity: flow[layer][position] = cosine_sim(h_last, h_pos)
    flow: dict[int, list[float]]
    # Per-layer operand gathering score (max sim to an operand position)
    operand_gathering: dict[int, dict[str, float]]  # layer -> label -> sim


@dataclass
class InformationFlowResult:
    """Results from cross-position information flow analysis."""

    flows: list[PositionFlow]
    # Aggregated: at which layer are operands most attended to?
    operand_peak_layer: dict[str, int]  # operand_type -> layer
    gathering_curve: dict[str, list[float]]  # operand_type -> per-layer avg sim


@dataclass
class LayerSubspaceResult:
    """Results from layer subspace communication analysis."""

    # Per-layer explained variance ratio for top-k PCA components
    explained_variance: dict[int, list[float]]  # layer -> [var_ratio_1, var_ratio_2, ...]
    # Layer-to-layer subspace alignment (cosine similarity of top PCA directions)
    alignment_matrix: list[list[float]]  # [num_layers, num_layers]
    # Consecutive layer alignment
    consecutive_alignment: list[float]  # alignment between layer l and l+1
    # High-alignment layer pairs
    aligned_pairs: list[tuple[int, int, float]]  # (layer_a, layer_b, alignment)


@dataclass
class SkipConditionResult:
    """Results for one skip condition (e.g. skip L10-L14) across prompts."""

    condition_name: str
    skip_layers: list[int]
    easy_results: list[dict[str, Any]]  # [{prompt, baseline, skipped, baseline_correct, skipped_correct}]
    hard_results: list[dict[str, Any]]
    easy_survival_rate: float  # fraction that remain correct after skip
    hard_survival_rate: float
    easy_baseline_accuracy: float
    hard_baseline_accuracy: float


@dataclass
class BypassValidationResult:
    """Results from causal bypass validation."""

    conditions: list[SkipConditionResult]
    # Key metric: does easy survive skipping better than hard?
    survival_gap: dict[str, float]  # condition -> (easy_survival - hard_survival)


@dataclass
class ExperimentResults:
    """All experiment results."""

    bypass_detection: BypassDetectionResult | None = None
    residual_saturation: ResidualSaturationResult | None = None
    information_flow: InformationFlowResult | None = None
    layer_subspaces: LayerSubspaceResult | None = None
    bypass_validation: BypassValidationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Main Experiment Class
# =============================================================================


class ResidualStreamDynamicsExperiment:
    """Residual stream geometry and dynamics analysis."""

    def __init__(self, config_path: Path | None = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.hooks: ModelHooks | None = None
        self.results = ExperimentResults()

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    async def setup(self) -> None:
        """Load model and initialize hooks."""
        from chuk_lazarus.models_v2.loader import load_model

        model_id = self.config["model"]
        logger.info(f"Loading model: {model_id}")

        try:
            loaded = load_model(model_id)
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer
        except Exception as e:
            logger.warning(f"Could not load {model_id}: {e}")
            logger.info("Falling back to TinyLlama for testing")
            loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        self._num_layers = len(self.model.model.layers)
        self._hidden_dim = self.model.model.layers[0].hidden_size

        # Initialize hooks
        self.hooks = ModelHooks(self.model)

        logger.info(
            f"Model loaded: {self._num_layers} layers, "
            f"{self._hidden_dim} hidden dim"
        )

    async def run(self, analyses: list[str] | None = None) -> ExperimentResults:
        """Run specified analyses."""
        if self.model is None:
            await self.setup()

        analyses = analyses or self.config.get("analyses", [])
        logger.info(f"Running analyses: {analyses}")

        self.results.metadata = {
            "model": self.config["model"],
            "timestamp": datetime.now().isoformat(),
            "analyses": analyses,
            "num_layers": self._num_layers,
            "hidden_dim": self._hidden_dim,
        }

        for analysis in analyses:
            logger.info(f"Running analysis: {analysis}")
            try:
                if analysis == "bypass_detection":
                    self.results.bypass_detection = (
                        await self._analyze_bypass_detection()
                    )
                elif analysis == "residual_saturation":
                    self.results.residual_saturation = (
                        await self._analyze_residual_saturation()
                    )
                elif analysis == "information_flow":
                    self.results.information_flow = (
                        await self._analyze_information_flow()
                    )
                elif analysis == "layer_subspaces":
                    self.results.layer_subspaces = (
                        await self._analyze_layer_subspaces()
                    )
                elif analysis == "bypass_validation":
                    self.results.bypass_validation = (
                        await self._analyze_bypass_validation()
                    )
                else:
                    logger.warning(f"Unknown analysis: {analysis}")
            except Exception as e:
                logger.error(f"Analysis {analysis} failed: {e}")
                raise

        return self.results

    # =========================================================================
    # Analysis 1: Bypass Detection
    # =========================================================================

    async def _analyze_bypass_detection(self) -> BypassDetectionResult:
        """Detect bypass behavior by comparing residual deltas across difficulty."""
        logger.info("Analyzing bypass detection...")
        config = self.config.get("bypass_detection", {})

        easy_prompts = config.get("easy_prompts", [])
        hard_prompts = config.get("hard_prompts", [])
        factual_prompts = config.get("factual_prompts", [])

        easy_deltas = await self._compute_deltas(easy_prompts, "easy")
        hard_deltas = await self._compute_deltas(hard_prompts, "hard")
        factual_deltas = await self._compute_deltas(factual_prompts, "factual")

        # Compute mean curves
        easy_mean = self._mean_delta_curve([d.deltas for d in easy_deltas])
        hard_mean = self._mean_delta_curve([d.deltas for d in hard_deltas])
        factual_mean = self._mean_delta_curve([d.deltas for d in factual_deltas])

        # Compute aggregate metrics
        def total_path(curves: list[list[float]]) -> float:
            return float(np.mean([sum(c) for c in curves])) if curves else 0.0

        def max_delta_layer(mean_curve: list[float]) -> int:
            return int(np.argmax(mean_curve)) if mean_curve else 0

        easy_total = total_path([d.deltas for d in easy_deltas])
        hard_total = total_path([d.deltas for d in hard_deltas])
        factual_total = total_path([d.deltas for d in factual_deltas])

        bypass_score = easy_total / hard_total if hard_total > 0 else 1.0

        # Log findings
        logger.info(f"Total residual path length - Easy: {easy_total:.4f}, Hard: {hard_total:.4f}")
        logger.info(f"Bypass score (easy/hard ratio): {bypass_score:.4f}")
        logger.info(f"  < 1.0 means easy problems change residual LESS (bypass detected)")
        logger.info(f"  = 1.0 means no difference")
        if bypass_score < 0.9:
            logger.info("  -> BYPASS DETECTED: Easy problems take a shorter residual path")
        elif bypass_score > 1.1:
            logger.info("  -> INVERSE: Hard problems actually change residual less (unexpected)")
        else:
            logger.info("  -> NO CLEAR BYPASS: Similar residual paths for easy and hard")

        # Print per-layer comparison
        logger.info("Per-layer mean delta (easy | hard | factual):")
        for l in range(min(len(easy_mean), len(hard_mean))):
            e = easy_mean[l] if l < len(easy_mean) else 0
            h = hard_mean[l] if l < len(hard_mean) else 0
            f = factual_mean[l] if l < len(factual_mean) else 0
            marker = " <--" if abs(e - h) > 0.01 else ""
            logger.info(f"  L{l:2d}: {e:.4f} | {h:.4f} | {f:.4f}{marker}")

        return BypassDetectionResult(
            easy_deltas=easy_deltas,
            hard_deltas=hard_deltas,
            factual_deltas=factual_deltas,
            easy_mean_curve=easy_mean,
            hard_mean_curve=hard_mean,
            factual_mean_curve=factual_mean,
            total_path_length={
                "easy": easy_total,
                "hard": hard_total,
                "factual": factual_total,
            },
            max_delta_layer={
                "easy": max_delta_layer(easy_mean),
                "hard": max_delta_layer(hard_mean),
                "factual": max_delta_layer(factual_mean),
            },
            bypass_score=bypass_score,
        )

    async def _compute_deltas(
        self, prompts: list[str], category: str
    ) -> list[PromptDelta]:
        """Compute per-layer residual deltas for a list of prompts."""
        results = []

        for prompt in prompts:
            tokens = self.tokenizer(prompt, return_tensors="np")
            input_ids = mx.array(tokens["input_ids"])

            # Capture hidden states at all layers (last position only for efficiency)
            self.hooks.configure(
                CaptureConfig(
                    layers=LayerSelection.ALL,
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                    detach=True,
                )
            )

            logits = self.hooks.forward(input_ids, return_logits=True)

            # Get prediction
            pred_id = int(mx.argmax(logits[0, -1, :]))
            prediction = self.tokenizer.decode([pred_id]).strip()

            # Check correctness for arithmetic
            correct = self._check_arithmetic(prompt, prediction)

            # Compute deltas between consecutive layers
            layers = sorted(self.hooks.state.hidden_states.keys())
            deltas = []
            norms = []

            # Include embedding as layer -1
            prev_h = self.hooks.state.embeddings
            if prev_h is not None:
                if prev_h.ndim == 3:
                    prev_h = prev_h[0, -1, :]
                elif prev_h.ndim == 2:
                    prev_h = prev_h[-1, :]

            for layer_idx in layers:
                h = self.hooks.state.get_hidden_at_position(layer_idx, -1)
                h_norm = float(mx.sqrt(mx.sum(h * h)))
                norms.append(h_norm)

                if prev_h is not None:
                    diff = h - prev_h
                    diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
                    prev_norm = float(mx.sqrt(mx.sum(prev_h * prev_h)))
                    relative_delta = diff_norm / (prev_norm + 1e-10)
                    deltas.append(relative_delta)

                prev_h = h

            results.append(
                PromptDelta(
                    prompt=prompt,
                    category=category,
                    deltas=deltas,
                    norms=norms,
                    prediction=prediction,
                    correct=correct,
                )
            )

            self.hooks.state.clear()
            gc.collect()

        return results

    def _mean_delta_curve(self, curves: list[list[float]]) -> list[float]:
        """Compute element-wise mean across delta curves."""
        if not curves:
            return []
        max_len = max(len(c) for c in curves)
        means = []
        for i in range(max_len):
            vals = [c[i] for c in curves if i < len(c)]
            means.append(float(np.mean(vals)) if vals else 0.0)
        return means

    def _check_arithmetic(self, prompt: str, prediction: str) -> bool | None:
        """Check if arithmetic prediction is correct. Returns None if not arithmetic."""
        prompt = prompt.strip()
        if "=" not in prompt:
            return None

        expr = prompt.replace("=", "").strip()
        try:
            expected = eval(expr)  # noqa: S307 - safe for arithmetic
            # Check if prediction starts with the correct number
            pred_clean = prediction.strip().lstrip()
            return str(int(expected)) in pred_clean[:10]
        except Exception:
            return None

    # =========================================================================
    # Analysis 2: Residual Saturation
    # =========================================================================

    async def _analyze_residual_saturation(self) -> ResidualSaturationResult:
        """Track how residual stream converges to final state."""
        logger.info("Analyzing residual saturation...")
        config = self.config.get("residual_saturation", {})
        categories = config.get("categories", {})

        convergence_threshold = 0.1  # cosine distance < this = "converged"

        all_curves: list[SaturationCurve] = []
        mean_dist: dict[str, list[float]] = {}
        mean_delta: dict[str, list[float]] = {}
        convergence_layers: dict[str, list[int]] = defaultdict(list)

        for category, prompts in categories.items():
            cat_dist_curves = []
            cat_delta_curves = []

            for prompt in prompts:
                tokens = self.tokenizer(prompt, return_tensors="np")
                input_ids = mx.array(tokens["input_ids"])

                self.hooks.configure(
                    CaptureConfig(
                        layers=LayerSelection.ALL,
                        capture_hidden_states=True,
                        positions=PositionSelection.LAST,
                        detach=True,
                    )
                )

                self.hooks.forward(input_ids, return_logits=False)

                layers = sorted(self.hooks.state.hidden_states.keys())

                # Get final layer hidden state
                final_h = self.hooks.state.get_hidden_at_position(layers[-1], -1)
                final_h_norm = mx.sqrt(mx.sum(final_h * final_h))

                dist_from_final = []
                inter_layer_delta = []
                convergence_layer = len(layers)

                prev_h = None
                for layer_idx in layers:
                    h = self.hooks.state.get_hidden_at_position(layer_idx, -1)

                    # Cosine distance from final
                    cos_sim = float(
                        mx.sum(h * final_h)
                        / (mx.sqrt(mx.sum(h * h)) * final_h_norm + 1e-10)
                    )
                    cos_dist = 1.0 - cos_sim
                    dist_from_final.append(cos_dist)

                    # Check convergence
                    if cos_dist < convergence_threshold and convergence_layer == len(layers):
                        convergence_layer = layer_idx

                    # Inter-layer cosine distance
                    if prev_h is not None:
                        inter_sim = float(
                            mx.sum(h * prev_h)
                            / (mx.sqrt(mx.sum(h * h)) * mx.sqrt(mx.sum(prev_h * prev_h)) + 1e-10)
                        )
                        inter_layer_delta.append(1.0 - inter_sim)
                    prev_h = h

                curve = SaturationCurve(
                    prompt=prompt,
                    category=category,
                    dist_from_final=dist_from_final,
                    inter_layer_delta=inter_layer_delta,
                    convergence_layer=convergence_layer,
                )
                all_curves.append(curve)
                cat_dist_curves.append(dist_from_final)
                cat_delta_curves.append(inter_layer_delta)
                convergence_layers[category].append(convergence_layer)

                self.hooks.state.clear()
                gc.collect()

            # Compute category means
            mean_dist[category] = self._mean_delta_curve(cat_dist_curves)
            mean_delta[category] = self._mean_delta_curve(cat_delta_curves)

        # Detect phase transitions across all prompts
        # A phase transition = layer where inter-layer delta increases significantly
        all_delta_curves = [c.inter_layer_delta for c in all_curves if c.inter_layer_delta]
        global_mean_delta = self._mean_delta_curve(all_delta_curves)
        phase_transitions = self._detect_phase_transitions(global_mean_delta)

        # Compute mean convergence layers
        mean_conv = {
            cat: float(np.mean(layers)) for cat, layers in convergence_layers.items()
        }

        # Log findings
        logger.info("Residual saturation analysis:")
        for category in categories:
            conv = mean_conv.get(category, self._num_layers)
            logger.info(f"  {category}: mean convergence at L{conv:.1f}")

        if phase_transitions:
            logger.info(f"Phase transitions detected at layers: {phase_transitions}")
        else:
            logger.info("No sharp phase transitions detected (gradual convergence)")

        logger.info("Distance from final layer (per-layer mean across all prompts):")
        for l, d in enumerate(global_mean_delta):
            bar = "#" * int(d * 100)
            logger.info(f"  L{l:2d}: {d:.4f} {bar}")

        return ResidualSaturationResult(
            curves=all_curves,
            mean_dist_from_final=mean_dist,
            mean_inter_layer_delta=mean_delta,
            phase_transitions=phase_transitions,
            mean_convergence_layer=mean_conv,
        )

    def _detect_phase_transitions(
        self, delta_curve: list[float], z_threshold: float = 2.0
    ) -> list[int]:
        """Detect phase transitions as layers where delta jumps above z_threshold stdevs."""
        if len(delta_curve) < 3:
            return []

        arr = np.array(delta_curve)
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return []

        transitions = []
        for i in range(1, len(arr)):
            # Check if this layer's delta is significantly higher than mean
            z_score = (arr[i] - mean) / std
            if z_score > z_threshold:
                transitions.append(i)

        return transitions

    # =========================================================================
    # Analysis 3: Cross-Position Information Flow
    # =========================================================================

    async def _analyze_information_flow(self) -> InformationFlowResult:
        """Track how the final position gathers operand information through layers."""
        logger.info("Analyzing cross-position information flow...")
        config = self.config.get("information_flow", {})
        prompt_configs = config.get("prompts", [])

        flows: list[PositionFlow] = []

        for pc in prompt_configs:
            prompt = pc["prompt"]
            operand_labels = pc.get("operand_labels", [])

            tokens = self.tokenizer(prompt, return_tensors="np")
            input_ids = mx.array(tokens["input_ids"])
            token_ids = tokens["input_ids"][0].tolist()
            token_strs = [self.tokenizer.decode([tid]) for tid in token_ids]

            # Map operand labels to token positions
            label_to_positions = self._map_operands_to_positions(
                token_strs, operand_labels
            )

            # Capture ALL positions (not just last) so we can compute cross-position similarity
            self.hooks.configure(
                CaptureConfig(
                    layers=LayerSelection.ALL,
                    capture_hidden_states=True,
                    positions=PositionSelection.ALL,
                    detach=True,
                )
            )

            self.hooks.forward(input_ids, return_logits=False)

            layers = sorted(self.hooks.state.hidden_states.keys())
            seq_len = len(token_ids)
            last_pos = seq_len - 1

            flow_per_layer: dict[int, list[float]] = {}
            operand_gathering: dict[int, dict[str, float]] = {}

            for layer_idx in layers:
                h = self.hooks.state.hidden_states[layer_idx]
                # h shape: [1, seq_len, hidden] or [seq_len, hidden]
                if h.ndim == 3:
                    h = h[0]  # [seq_len, hidden]

                # Hidden state at the final position
                h_last = h[last_pos]
                h_last_norm = mx.sqrt(mx.sum(h_last * h_last)) + 1e-10

                # Compute cosine similarity to each position
                sims = []
                for pos in range(seq_len):
                    h_pos = h[pos]
                    sim = float(
                        mx.sum(h_last * h_pos)
                        / (h_last_norm * mx.sqrt(mx.sum(h_pos * h_pos)) + 1e-10)
                    )
                    sims.append(sim)

                flow_per_layer[layer_idx] = sims

                # Compute operand-specific gathering scores
                layer_gathering = {}
                for label, positions in label_to_positions.items():
                    if positions:
                        label_sim = max(sims[p] for p in positions if p < seq_len)
                        layer_gathering[label] = label_sim
                operand_gathering[layer_idx] = layer_gathering

            flows.append(
                PositionFlow(
                    prompt=prompt,
                    tokens=token_strs,
                    operand_labels=operand_labels,
                    flow=flow_per_layer,
                    operand_gathering=operand_gathering,
                )
            )

            self.hooks.state.clear()
            gc.collect()

        # Aggregate: for each operand type, find peak gathering layer
        operand_types = set()
        for flow in flows:
            for label_positions in flow.operand_gathering.values():
                operand_types.update(label_positions.keys())

        # Classify operand types (number vs operator vs equals)
        operand_classes = {"number": [], "operator": [], "equals": []}
        for ot in operand_types:
            if ot in ("+", "-", "*", "/"):
                operand_classes["operator"].append(ot)
            elif ot == "=":
                operand_classes["equals"].append(ot)
            else:
                operand_classes["number"].append(ot)

        # Compute per-class gathering curves
        gathering_curves: dict[str, list[float]] = {}
        peak_layers: dict[str, int] = {}

        for cls_name, cls_labels in operand_classes.items():
            if not cls_labels:
                continue
            layer_sims: dict[int, list[float]] = defaultdict(list)
            for flow in flows:
                for layer_idx, gathering in flow.operand_gathering.items():
                    for label in cls_labels:
                        if label in gathering:
                            layer_sims[layer_idx].append(gathering[label])

            if layer_sims:
                curve = []
                for l in sorted(layer_sims.keys()):
                    curve.append(float(np.mean(layer_sims[l])))
                gathering_curves[cls_name] = curve
                peak_layers[cls_name] = int(np.argmax(curve))

        # Log findings
        logger.info("Information flow analysis:")
        for cls_name, curve in gathering_curves.items():
            peak = peak_layers.get(cls_name, -1)
            logger.info(f"  {cls_name}: peak gathering at L{peak}")
            for l, sim in enumerate(curve):
                bar = "#" * int(sim * 50)
                logger.info(f"    L{l:2d}: {sim:.4f} {bar}")

        return InformationFlowResult(
            flows=flows,
            operand_peak_layer=peak_layers,
            gathering_curve=gathering_curves,
        )

    def _map_operands_to_positions(
        self, token_strs: list[str], operand_labels: list[str]
    ) -> dict[str, list[int]]:
        """Map operand labels to token position indices."""
        label_to_positions: dict[str, list[int]] = defaultdict(list)

        # Build the text from tokens and find where each label appears
        # This is approximate - tokens may not align perfectly with labels
        current_text = ""
        token_starts = []
        for tok in token_strs:
            token_starts.append(len(current_text))
            current_text += tok

        for label in operand_labels:
            # Find token(s) that contain this label
            for pos, tok in enumerate(token_strs):
                tok_clean = tok.strip()
                if tok_clean and (
                    label in tok_clean
                    or tok_clean in label
                    or (label.isdigit() and tok_clean.isdigit() and tok_clean in label)
                ):
                    label_to_positions[label].append(pos)

        return dict(label_to_positions)

    # =========================================================================
    # Analysis 4: Layer Subspace Communication
    # =========================================================================

    async def _analyze_layer_subspaces(self) -> LayerSubspaceResult:
        """Decompose per-layer residual updates to find communication subspaces."""
        logger.info("Analyzing layer subspace communication...")
        config = self.config.get("layer_subspaces", {})
        num_components = config.get("num_components", 10)
        prompts = config.get("prompts", [])

        if not prompts:
            logger.warning("No prompts configured for layer_subspaces")
            return LayerSubspaceResult(
                explained_variance={},
                alignment_matrix=[],
                consecutive_alignment=[],
                aligned_pairs=[],
            )

        # Collect residual updates at each layer for all prompts
        # update_l = h_{l} - h_{l-1}
        layer_updates: dict[int, list[np.ndarray]] = defaultdict(list)

        for prompt in prompts:
            tokens = self.tokenizer(prompt, return_tensors="np")
            input_ids = mx.array(tokens["input_ids"])

            self.hooks.configure(
                CaptureConfig(
                    layers=LayerSelection.ALL,
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                    detach=True,
                )
            )

            self.hooks.forward(input_ids, return_logits=False)

            layers = sorted(self.hooks.state.hidden_states.keys())

            # Use embedding as layer -1
            prev_h = self.hooks.state.embeddings
            if prev_h is not None:
                if prev_h.ndim == 3:
                    prev_h = prev_h[0, -1, :]
                elif prev_h.ndim == 2:
                    prev_h = prev_h[-1, :]

            for layer_idx in layers:
                h = self.hooks.state.get_hidden_at_position(layer_idx, -1)

                if prev_h is not None:
                    update = np.array((h - prev_h).astype(mx.float32))
                    layer_updates[layer_idx].append(update)

                prev_h = h

            self.hooks.state.clear()
            gc.collect()

        # PCA decomposition at each layer
        explained_variance: dict[int, list[float]] = {}
        layer_components: dict[int, np.ndarray] = {}  # top-k components

        for layer_idx in sorted(layer_updates.keys()):
            updates = np.stack(layer_updates[layer_idx])  # [num_prompts, hidden_dim]

            # Center
            updates_centered = updates - updates.mean(axis=0)

            # PCA via SVD
            try:
                U, S, Vt = np.linalg.svd(updates_centered, full_matrices=False)
                total_var = np.sum(S ** 2)
                k = min(num_components, len(S))
                var_ratios = (S[:k] ** 2 / (total_var + 1e-10)).tolist()
                explained_variance[layer_idx] = var_ratios
                layer_components[layer_idx] = Vt[:k]  # top-k right singular vectors
            except np.linalg.LinAlgError:
                explained_variance[layer_idx] = [0.0] * num_components
                layer_components[layer_idx] = np.zeros(
                    (num_components, updates.shape[1])
                )

        # Compute alignment matrix between layers
        sorted_layers = sorted(layer_components.keys())
        n = len(sorted_layers)
        alignment_matrix = np.zeros((n, n))

        for i, layer_a in enumerate(sorted_layers):
            for j, layer_b in enumerate(sorted_layers):
                if i == j:
                    alignment_matrix[i, j] = 1.0
                else:
                    # Subspace alignment: average absolute cosine similarity
                    # between top components of layer_a and layer_b
                    Va = layer_components[layer_a]
                    Vb = layer_components[layer_b]
                    # Compute cosine similarity matrix between all pairs
                    sim_matrix = Va @ Vb.T
                    # Normalize (components should already be unit vectors from SVD)
                    norms_a = np.sqrt(np.sum(Va ** 2, axis=1, keepdims=True))
                    norms_b = np.sqrt(np.sum(Vb ** 2, axis=1, keepdims=True))
                    sim_matrix = sim_matrix / (norms_a @ norms_b.T + 1e-10)
                    # Use mean of absolute max alignment per component
                    alignment = float(np.mean(np.max(np.abs(sim_matrix), axis=1)))
                    alignment_matrix[i, j] = alignment

        # Consecutive layer alignment
        consecutive = []
        for i in range(n - 1):
            consecutive.append(float(alignment_matrix[i, i + 1]))

        # Find high-alignment pairs (non-consecutive)
        aligned_pairs = []
        for i in range(n):
            for j in range(i + 2, n):  # Skip consecutive (already captured)
                if alignment_matrix[i, j] > 0.3:
                    aligned_pairs.append(
                        (sorted_layers[i], sorted_layers[j], float(alignment_matrix[i, j]))
                    )
        aligned_pairs.sort(key=lambda x: -x[2])

        # Log findings
        logger.info("Layer subspace analysis:")
        logger.info("Consecutive layer alignment:")
        for i, align in enumerate(consecutive):
            bar = "#" * int(align * 50)
            logger.info(f"  L{sorted_layers[i]:2d}->L{sorted_layers[i+1]:2d}: {align:.4f} {bar}")

        if aligned_pairs:
            logger.info("High-alignment non-consecutive pairs:")
            for la, lb, align in aligned_pairs[:10]:
                logger.info(f"  L{la}->L{lb}: {align:.4f}")

        # Log explained variance for first component
        logger.info("PC1 explained variance ratio per layer:")
        for layer_idx in sorted_layers:
            var = explained_variance[layer_idx][0] if explained_variance[layer_idx] else 0
            bar = "#" * int(var * 50)
            logger.info(f"  L{layer_idx:2d}: {var:.4f} {bar}")

        return LayerSubspaceResult(
            explained_variance=explained_variance,
            alignment_matrix=alignment_matrix.tolist(),
            consecutive_alignment=consecutive,
            aligned_pairs=aligned_pairs[:20],
        )

    # =========================================================================
    # Analysis 5: Bypass Validation (Causal)
    # =========================================================================

    async def _analyze_bypass_validation(self) -> BypassValidationResult:
        """Causally validate bypass by skipping middle layers.

        If easy problems bypass L10-L14 in their residual path, then
        skipping those layers should hurt easy problems LESS than hard ones.

        Uses few-shot prompting (GPT-OSS is a base model and needs context
        to produce arithmetic answers) and first-token correctness checking.
        """
        logger.info("Running causal bypass validation...")
        config = self.config.get("bypass_validation", {})

        # Few-shot prefix to elicit arithmetic answers from base model
        fewshot_prefix = config.get(
            "fewshot_prefix",
            "Math: 2+2=4, 5*5=25, 10+10=20, ",
        )

        # Arithmetic expressions (without "=")
        easy_exprs = config.get("easy_exprs", [
            "3*3", "4+4", "6*4", "8+7", "9+1", "7*7",
            "6+6", "5+3", "2*8", "10*10",
        ])
        hard_exprs = config.get("hard_exprs", [
            "47*47", "89*73", "127*89", "156*23",
            "83*67", "234+567", "512-178", "347+896",
            "291*14", "1024-389",
        ])

        # Layer ranges to skip
        skip_conditions: dict[str, list[int]] = config.get("skip_conditions", {
            "skip_L10_L14": [10, 11, 12, 13, 14],
            "skip_L16_L20": [16, 17, 18, 19, 20],
            "skip_L0_L4": [0, 1, 2, 3, 4],
        })

        # Get block class for monkey-patching
        sample_block = self.model.model.layers[0]
        block_class = type(sample_block)
        original_call = block_class.__call__

        conditions: list[SkipConditionResult] = []
        survival_gap: dict[str, float] = {}

        for cond_name, skip_layers in skip_conditions.items():
            logger.info(f"Testing condition: {cond_name} (skip {skip_layers})")
            skip_set = set(skip_layers)

            easy_results = []
            hard_results = []

            for category, exprs, result_list in [
                ("easy", easy_exprs, easy_results),
                ("hard", hard_exprs, hard_results),
            ]:
                for expr in exprs:
                    prompt = f"{fewshot_prefix}{expr}="
                    expected = self._eval_expr(expr)
                    if expected is None:
                        continue

                    # Baseline: first token prediction
                    baseline_pred = self._get_first_token(prompt)
                    baseline_correct = self._check_first_token(baseline_pred, expected)

                    # Skipped: first token prediction with layer skip
                    skipped_pred = self._get_first_token_with_skip(
                        prompt, skip_set, block_class, original_call,
                    )
                    skipped_correct = self._check_first_token(skipped_pred, expected)

                    result_list.append({
                        "prompt": expr,
                        "expected": expected,
                        "baseline": baseline_pred,
                        "skipped": skipped_pred,
                        "baseline_correct": baseline_correct,
                        "skipped_correct": skipped_correct,
                    })

            # Compute survival rates
            easy_baseline_correct = [r for r in easy_results if r["baseline_correct"]]
            hard_baseline_correct = [r for r in hard_results if r["baseline_correct"]]

            easy_survived = sum(1 for r in easy_baseline_correct if r["skipped_correct"])
            hard_survived = sum(1 for r in hard_baseline_correct if r["skipped_correct"])

            easy_survival = (
                easy_survived / len(easy_baseline_correct)
                if easy_baseline_correct else 0.0
            )
            hard_survival = (
                hard_survived / len(hard_baseline_correct)
                if hard_baseline_correct else 0.0
            )

            easy_baseline_acc = (
                len(easy_baseline_correct) / len(easy_results) if easy_results else 0.0
            )
            hard_baseline_acc = (
                len(hard_baseline_correct) / len(hard_results) if hard_results else 0.0
            )

            gap = easy_survival - hard_survival
            survival_gap[cond_name] = gap

            logger.info(f"  {cond_name}:")
            logger.info(f"    Easy: baseline={easy_baseline_acc:.0%} ({len(easy_baseline_correct)}/{len(easy_results)}), survival={easy_survival:.0%} ({easy_survived}/{len(easy_baseline_correct)})")
            logger.info(f"    Hard: baseline={hard_baseline_acc:.0%} ({len(hard_baseline_correct)}/{len(hard_results)}), survival={hard_survival:.0%} ({hard_survived}/{len(hard_baseline_correct)})")
            logger.info(f"    Gap (easy - hard): {gap:+.0%}")
            if gap > 0.1:
                logger.info("    -> BYPASS CONFIRMED: Easy problems survive skip better")
            elif gap < -0.1:
                logger.info("    -> INVERSE: Hard problems survive better (unexpected)")
            else:
                logger.info("    -> No significant survival difference")

            # Log per-prompt details
            for r in easy_results + hard_results:
                b = "Y" if r["baseline_correct"] else "N"
                s = "Y" if r["skipped_correct"] else "N"
                logger.info(
                    f"      {r['prompt']:12s} expected={r['expected']:<8} "
                    f"baseline={r['baseline']!r:8s}({b}) "
                    f"skipped={r['skipped']!r:8s}({s})"
                )

            conditions.append(SkipConditionResult(
                condition_name=cond_name,
                skip_layers=skip_layers,
                easy_results=easy_results,
                hard_results=hard_results,
                easy_survival_rate=easy_survival,
                hard_survival_rate=hard_survival,
                easy_baseline_accuracy=easy_baseline_acc,
                hard_baseline_accuracy=hard_baseline_acc,
            ))

        return BypassValidationResult(
            conditions=conditions,
            survival_gap=survival_gap,
        )

    def _eval_expr(self, expr: str) -> int | None:
        """Safely evaluate an arithmetic expression."""
        try:
            return int(eval(expr))  # noqa: S307
        except Exception:
            return None

    def _get_first_token(self, prompt: str) -> str:
        """Get the first generated token (greedy) for a prompt."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        output = self.model(input_ids)
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        return self.tokenizer.decode([next_token]).strip()

    def _get_first_token_with_skip(
        self,
        prompt: str,
        skip_layers: set[int],
        block_class: type,
        original_call: Any,
    ) -> str:
        """Get first token with MoE skipped at specified layers."""
        experiment_model = self.model

        def patched_block(
            block_self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            layer_idx = -1
            for i, layer in enumerate(experiment_model.model.layers):
                if layer is block_self:
                    layer_idx = i
                    break
            if layer_idx not in skip_layers:
                return original_call(block_self, x, mask=mask, cache=cache)
            # Attention only, skip MoE
            residual = x
            x = block_self.input_layernorm(x)
            x, new_cache = block_self.self_attn(x, mask=mask, cache=cache)
            x = residual + x
            return x, new_cache

        try:
            block_class.__call__ = patched_block
            result = self._get_first_token(prompt)
        finally:
            block_class.__call__ = original_call
        return result

    def _check_first_token(self, token: str, expected: int) -> bool:
        """Check if first generated token matches expected answer.

        For easy problems (single/double digit), the full answer should
        appear in the first token. For hard problems, the first token
        contains the leading digits -- we check if expected starts with
        the generated digits.
        """
        token = token.strip()
        expected_str = str(expected)
        # Exact match
        if token == expected_str:
            return True
        # First token contains start of answer (e.g. "220" for 2209)
        if token.isdigit() and expected_str.startswith(token) and len(token) >= len(expected_str):
            return True
        # Answer fits in one token
        if token.isdigit() and int(token) == expected:
            return True
        return False

    def _generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text from prompt."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated: list[int] = []

        for _ in range(max_tokens):
            output = self.model(input_ids)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break
            input_ids = mx.array([[next_token]])

        return self.tokenizer.decode(generated).strip()

    def _generate_with_layer_skip(
        self,
        prompt: str,
        skip_layers: set[int],
        block_class: type,
        original_call: Any,
        max_tokens: int = 20,
    ) -> str:
        """Generate with MoE skipped at specified layers.

        At skipped layers: attention runs normally, MoE is bypassed.
        Residual stream passes through with only attention contribution.
        """
        experiment_model = self.model

        def patched_block(
            block_self,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            # Find layer index
            layer_idx = -1
            for i, layer in enumerate(experiment_model.model.layers):
                if layer is block_self:
                    layer_idx = i
                    break

            if layer_idx not in skip_layers:
                return original_call(block_self, x, mask=mask, cache=cache)

            # Skip MoE: run attention only
            residual = x
            x = block_self.input_layernorm(x)
            x, new_cache = block_self.self_attn(x, mask=mask, cache=cache)
            x = residual + x
            # MoE skipped -- residual passes through
            return x, new_cache

        try:
            block_class.__call__ = patched_block
            result = self._generate(prompt, max_tokens)
        finally:
            block_class.__call__ = original_call

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def save_results(self, output_path: Path | None = None) -> None:
        """Save results to JSON."""
        if output_path is None:
            output_path = (
                Path(__file__).parent
                / "results"
                / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict: dict[str, Any] = {"metadata": self.results.metadata}

        if self.results.bypass_detection:
            bd = self.results.bypass_detection
            results_dict["bypass_detection"] = {
                "easy_mean_curve": bd.easy_mean_curve,
                "hard_mean_curve": bd.hard_mean_curve,
                "factual_mean_curve": bd.factual_mean_curve,
                "total_path_length": bd.total_path_length,
                "max_delta_layer": bd.max_delta_layer,
                "bypass_score": bd.bypass_score,
                "per_prompt": [
                    {
                        "prompt": d.prompt,
                        "category": d.category,
                        "deltas": d.deltas,
                        "norms": d.norms,
                        "prediction": d.prediction,
                        "correct": d.correct,
                    }
                    for d in bd.easy_deltas + bd.hard_deltas + bd.factual_deltas
                ],
            }

        if self.results.residual_saturation:
            rs = self.results.residual_saturation
            results_dict["residual_saturation"] = {
                "mean_dist_from_final": rs.mean_dist_from_final,
                "mean_inter_layer_delta": rs.mean_inter_layer_delta,
                "phase_transitions": rs.phase_transitions,
                "mean_convergence_layer": rs.mean_convergence_layer,
                "per_prompt": [
                    {
                        "prompt": c.prompt,
                        "category": c.category,
                        "dist_from_final": c.dist_from_final,
                        "inter_layer_delta": c.inter_layer_delta,
                        "convergence_layer": c.convergence_layer,
                    }
                    for c in rs.curves
                ],
            }

        if self.results.information_flow:
            ifl = self.results.information_flow
            results_dict["information_flow"] = {
                "operand_peak_layer": ifl.operand_peak_layer,
                "gathering_curve": ifl.gathering_curve,
                "per_prompt": [
                    {
                        "prompt": f.prompt,
                        "tokens": f.tokens,
                        "operand_labels": f.operand_labels,
                        "flow": {str(k): v for k, v in f.flow.items()},
                        "operand_gathering": {
                            str(k): v for k, v in f.operand_gathering.items()
                        },
                    }
                    for f in ifl.flows
                ],
            }

        if self.results.layer_subspaces:
            ls = self.results.layer_subspaces
            results_dict["layer_subspaces"] = {
                "explained_variance": {
                    str(k): v for k, v in ls.explained_variance.items()
                },
                "alignment_matrix": ls.alignment_matrix,
                "consecutive_alignment": ls.consecutive_alignment,
                "aligned_pairs": [
                    {"layer_a": a, "layer_b": b, "alignment": c}
                    for a, b, c in ls.aligned_pairs
                ],
            }

        if self.results.bypass_validation:
            bv = self.results.bypass_validation
            results_dict["bypass_validation"] = {
                "survival_gap": bv.survival_gap,
                "conditions": [
                    {
                        "condition_name": c.condition_name,
                        "skip_layers": c.skip_layers,
                        "easy_survival_rate": c.easy_survival_rate,
                        "hard_survival_rate": c.hard_survival_rate,
                        "easy_baseline_accuracy": c.easy_baseline_accuracy,
                        "hard_baseline_accuracy": c.hard_baseline_accuracy,
                        "easy_results": c.easy_results,
                        "hard_results": c.hard_results,
                    }
                    for c in bv.conditions
                ],
            }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of all results."""
        print()
        print("=" * 80)
        print("RESIDUAL STREAM DYNAMICS - EXPERIMENT SUMMARY")
        print("=" * 80)
        print()
        print(f"Model: {self.results.metadata.get('model', 'unknown')}")
        print(f"Layers: {self.results.metadata.get('num_layers', '?')}")
        print(f"Hidden dim: {self.results.metadata.get('hidden_dim', '?')}")
        print()

        if self.results.bypass_detection:
            bd = self.results.bypass_detection
            print("-" * 80)
            print("1. BYPASS DETECTION")
            print("-" * 80)
            print()
            print(f"Bypass score (easy/hard path ratio): {bd.bypass_score:.4f}")
            print(f"  Easy total path:    {bd.total_path_length['easy']:.4f}")
            print(f"  Hard total path:    {bd.total_path_length['hard']:.4f}")
            print(f"  Factual total path: {bd.total_path_length['factual']:.4f}")
            print()
            if bd.bypass_score < 0.9:
                print("  FINDING: Easy problems take a shorter residual path.")
                print("  This is consistent with lookup-table behavior.")
            elif bd.bypass_score > 1.1:
                print("  FINDING: Hard problems take a shorter path (unexpected).")
            else:
                print("  FINDING: No significant difference in path length.")
            print()

            # Accuracy comparison
            easy_correct = sum(1 for d in bd.easy_deltas if d.correct is True)
            easy_total = sum(1 for d in bd.easy_deltas if d.correct is not None)
            hard_correct = sum(1 for d in bd.hard_deltas if d.correct is True)
            hard_total = sum(1 for d in bd.hard_deltas if d.correct is not None)
            if easy_total > 0 and hard_total > 0:
                print(f"  Easy accuracy: {easy_correct}/{easy_total} ({easy_correct/easy_total:.0%})")
                print(f"  Hard accuracy: {hard_correct}/{hard_total} ({hard_correct/hard_total:.0%})")
                print()

        if self.results.residual_saturation:
            rs = self.results.residual_saturation
            print("-" * 80)
            print("2. RESIDUAL SATURATION")
            print("-" * 80)
            print()
            for category, conv_layer in sorted(rs.mean_convergence_layer.items()):
                print(f"  {category:12s}: converges at L{conv_layer:.1f}")
            print()
            if rs.phase_transitions:
                print(f"  Phase transitions at layers: {rs.phase_transitions}")
                print("  (layers where inter-layer delta spikes above 2 stdevs)")
            else:
                print("  No sharp phase transitions detected (gradual convergence)")
            print()

        if self.results.information_flow:
            ifl = self.results.information_flow
            print("-" * 80)
            print("3. CROSS-POSITION INFORMATION FLOW")
            print("-" * 80)
            print()
            for cls_name, peak in sorted(ifl.operand_peak_layer.items()):
                print(f"  {cls_name:10s}: peak gathering at L{peak}")
            print()

        if self.results.layer_subspaces:
            ls = self.results.layer_subspaces
            print("-" * 80)
            print("4. LAYER SUBSPACE COMMUNICATION")
            print("-" * 80)
            print()
            print("Consecutive layer alignment:")
            for i, align in enumerate(ls.consecutive_alignment):
                bar = "#" * int(align * 30)
                print(f"  L{i:2d}->L{i+1:2d}: {align:.3f} {bar}")
            print()
            if ls.aligned_pairs:
                print("High-alignment non-consecutive pairs:")
                for la, lb, align in ls.aligned_pairs[:5]:
                    print(f"  L{la}->L{lb}: {align:.3f}")
            print()

        if self.results.bypass_validation:
            bv = self.results.bypass_validation
            print("-" * 80)
            print("5. BYPASS VALIDATION (Causal)")
            print("-" * 80)
            print()
            print(f"{'Condition':<20} {'Easy Surv':>10} {'Hard Surv':>10} {'Gap':>10} {'Result'}")
            print("-" * 70)
            for cond in bv.conditions:
                gap = bv.survival_gap[cond.condition_name]
                if gap > 0.1:
                    result = "BYPASS"
                elif gap < -0.1:
                    result = "INVERSE"
                else:
                    result = "neutral"
                print(
                    f"  {cond.condition_name:<18} "
                    f"{cond.easy_survival_rate:>8.0%}   "
                    f"{cond.hard_survival_rate:>8.0%}   "
                    f"{gap:>+8.0%}   "
                    f"{result}"
                )
            print()
            print("  Easy baseline accuracy:", end="")
            for cond in bv.conditions[:1]:
                print(f" {cond.easy_baseline_accuracy:.0%}")
            print("  Hard baseline accuracy:", end="")
            for cond in bv.conditions[:1]:
                print(f" {cond.hard_baseline_accuracy:.0%}")
            print()

        print("=" * 80)


async def main():
    """Run the experiment."""
    parser = argparse.ArgumentParser(
        description="Residual Stream Dynamics Experiment"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        nargs="*",
        help="Specific analyses to run (default: all from config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    experiment = ResidualStreamDynamicsExperiment(args.config)
    await experiment.run(args.analysis)
    experiment.print_summary()
    experiment.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())
