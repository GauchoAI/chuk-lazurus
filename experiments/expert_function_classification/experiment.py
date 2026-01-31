#!/usr/bin/env python3
"""Expert Function Classification Experiment.

Classifies each MoE expert as:
- STORAGE: Externalizable via memory bank (ablation causes fact errors, recoverable)
- COMPUTATION: Irreducible (ablation causes structure errors, not recoverable)
- ROUTING: Structural (ablation disrupts downstream expert selection)
- REDUNDANT: Removable (ablation causes no measurable change)

4 Phases:
1. Baseline generation (one-time, all prompts)
2. Systematic ablation with error classification
3. Recovery testing for storage experts via memory bank injection
4. Capacity estimation

Run: python experiments/expert_function_classification/experiment.py
     python experiments/expert_function_classification/experiment.py --layer 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptResult:
    """Baseline result for a single prompt."""
    prompt: str
    category: str
    expected: str
    text: str


@dataclass
class AblationSignature:
    """Result of ablating one expert on one prompt."""
    expert_id: str
    prompt: str
    category: str
    baseline_text: str
    ablated_text: str
    output_changed: bool
    fact_error: bool
    structure_error: bool
    repetition_ratio: float
    length_ratio: float


@dataclass
class ExpertClassification:
    """Classification for one expert."""
    expert_id: str
    layer_idx: int
    expert_idx: int
    classification: str  # storage, computation, routing, redundant
    confidence: float
    fact_error_rate: float
    structure_error_rate: float
    no_change_rate: float
    routing_disruption: float
    recovery_rate: float | None
    num_prompts_tested: int
    signatures: list[AblationSignature] = field(default_factory=list)


@dataclass
class CapacityEstimate:
    """Overall capacity estimation."""
    total_experts_tested: int
    by_category: dict[str, int]
    by_category_pct: dict[str, float]
    storage_experts: list[str]
    recovery_success_rate: float
    estimated_externalizable_fraction: float


# =============================================================================
# Experiment
# =============================================================================


class ExpertFunctionClassification:
    """Classify MoE experts by function via systematic ablation."""

    def __init__(self, config_path: Path | None = None):
        self.config = self._load_config(config_path)
        self.router = None
        self.model = None
        self.tokenizer = None
        self.baselines: dict[str, PromptResult] = {}
        self.classifications: dict[str, ExpertClassification] = {}

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    # ─────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Load model via ExpertRouter."""
        from chuk_lazarus.introspection.moe import ExpertRouter

        model_id = self.config["model"]
        logger.info(f"Loading model: {model_id}")

        self.router = await ExpertRouter.from_pretrained(model_id)
        self.model = self.router._model
        self.tokenizer = self.router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        info = self.router._info
        logger.info(
            f"Model loaded: {info.num_experts} experts, "
            f"{len(info.moe_layers)} MoE layers, "
            f"top-{info.num_experts_per_tok}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Generation helpers
    # ─────────────────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_tokens: int = 30) -> str:
        """Generate text without ablation."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            next_token = mx.argmax(output.logits[0, -1, :])
            token_id = next_token.item()
            if token_id == self.tokenizer.eos_token_id:
                break
            generated.append(token_id)
            input_ids = mx.concatenate(
                [input_ids, next_token.reshape(1, 1)], axis=1
            )

        return self.tokenizer.decode(generated).strip()

    def _get_all_prompts(self) -> list[dict[str, str]]:
        """Get all test prompts from config, flattened with category."""
        prompts = []
        for category, items in self.config["test_prompts"].items():
            for item in items:
                prompts.append({
                    "prompt": item["prompt"],
                    "expected": item.get("expected", ""),
                    "category": category,
                    "memory_bank": item.get("memory_bank", ""),
                })
        return prompts

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Baselines
    # ─────────────────────────────────────────────────────────────────

    async def _generate_baselines(self) -> dict[str, PromptResult]:
        """Generate baseline outputs for all test prompts (no ablation)."""
        logger.info("Phase 1: Generating baselines")
        prompts = self._get_all_prompts()
        baselines = {}

        for i, p in enumerate(prompts):
            text = self._generate(p["prompt"], self.config["ablation"]["max_tokens"])
            baselines[p["prompt"]] = PromptResult(
                prompt=p["prompt"],
                category=p["category"],
                expected=p["expected"],
                text=text,
            )
            logger.info(f"  [{i+1}/{len(prompts)}] {p['prompt'][:50]}... -> {text[:60]}")

        logger.info(f"  Baselines generated: {len(baselines)}")
        return baselines

    # ─────────────────────────────────────────────────────────────────
    # Error Detection
    # ─────────────────────────────────────────────────────────────────

    def _detect_fact_error(
        self,
        baseline: str,
        ablated: str,
        expected: str,
        category: str,
    ) -> bool:
        """Detect if ablation caused a factual error.

        A fact error means the expected answer is present in the baseline
        but missing from the ablated output.
        """
        if not expected:
            return False

        expected_lower = expected.lower()
        baseline_has = expected_lower in baseline.lower()
        ablated_has = expected_lower in ablated.lower()

        # Fact error = baseline had it, ablation lost it
        return baseline_has and not ablated_has

    def _detect_structure_error(self, baseline: str, ablated: str) -> bool:
        """Detect if ablation CAUSED a structural/coherence error.

        Only flags structure errors that are worse than baseline.
        If baseline is already repetitive, the ablated output being equally
        repetitive is not an ablation-caused error.

        Heuristics:
        1. Output became empty when baseline wasn't
        2. Repetition increased significantly vs baseline
        3. Extreme length change vs baseline
        """
        thresholds = self.config["classification"]

        # If output didn't change, it's not an ablation-caused error
        if baseline == ablated:
            return False

        # Empty or whitespace when baseline wasn't
        if not ablated.strip() and baseline.strip():
            return True

        # Repetition: only flag if ablated is MORE repetitive than baseline
        baseline_rep = self._compute_repetition_ratio(baseline)
        ablated_rep = self._compute_repetition_ratio(ablated)
        rep_increase = ablated_rep - baseline_rep
        if rep_increase > 0.2:  # Significant increase in repetition
            return True

        # Length ratio check
        baseline_len = max(len(baseline.split()), 1)
        ablated_len = max(len(ablated.split()), 1)
        length_ratio = ablated_len / baseline_len

        if length_ratio < thresholds["length_ratio_min"]:
            return True
        if length_ratio > thresholds["length_ratio_max"]:
            return True

        return False

    @staticmethod
    def _compute_repetition_ratio(text: str, n: int = 3) -> float:
        """Fraction of n-grams that are repeated."""
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        unique = set(ngrams)
        return 1.0 - len(unique) / len(ngrams)

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: Ablation
    # ─────────────────────────────────────────────────────────────────

    async def _quick_scan(self, layer_idx: int) -> list[int]:
        """Quick scan: test all 32 experts on a few prompts to find causal ones.

        Returns indices of experts whose ablation changed output.
        """
        num_experts = self.config["moe"]["num_experts"]
        scan_prompts_count = self.config["sampling"]["quick_scan_prompts"]

        # Use the first N factual prompts for scanning
        factual = self.config["test_prompts"]["factual"][:scan_prompts_count]
        scan_prompts = [p["prompt"] for p in factual]

        causal_experts = []

        for expert_idx in range(num_experts):
            changed = False
            for prompt in scan_prompts:
                baseline_text = self.baselines[prompt].text

                ablated_text, _ = await self.router.generate_with_ablation(
                    prompt,
                    [expert_idx],
                    max_tokens=self.config["ablation"]["max_tokens"],
                    layers=[layer_idx],
                )
                ablated_text = ablated_text.strip()

                if ablated_text != baseline_text:
                    changed = True
                    break

            if changed:
                causal_experts.append(expert_idx)

        logger.info(
            f"  Layer {layer_idx}: {len(causal_experts)}/{num_experts} causal experts"
        )
        return causal_experts

    async def _deep_ablation(
        self, layer_idx: int, expert_idx: int
    ) -> list[AblationSignature]:
        """Run full prompt suite with one expert ablated."""
        expert_id = f"L{layer_idx}E{expert_idx}"
        prompts = self._get_all_prompts()
        signatures = []
        max_tokens = self.config["ablation"]["max_tokens"]

        for p in prompts:
            baseline_text = self.baselines[p["prompt"]].text

            ablated_text, _ = await self.router.generate_with_ablation(
                p["prompt"],
                [expert_idx],
                max_tokens=max_tokens,
                layers=[layer_idx],
            )
            ablated_text = ablated_text.strip()

            output_changed = ablated_text != baseline_text
            fact_error = self._detect_fact_error(
                baseline_text, ablated_text, p["expected"], p["category"]
            )
            structure_error = self._detect_structure_error(baseline_text, ablated_text)

            baseline_len = max(len(baseline_text.split()), 1)
            ablated_len = max(len(ablated_text.split()), 1)

            signatures.append(AblationSignature(
                expert_id=expert_id,
                prompt=p["prompt"],
                category=p["category"],
                baseline_text=baseline_text,
                ablated_text=ablated_text,
                output_changed=output_changed,
                fact_error=fact_error,
                structure_error=structure_error,
                repetition_ratio=self._compute_repetition_ratio(ablated_text),
                length_ratio=ablated_len / baseline_len,
            ))

        return signatures

    async def _measure_routing_disruption(
        self, layer_idx: int, expert_idx: int
    ) -> float:
        """Measure how ablation changes downstream expert selection.

        Uses JS divergence between baseline and ablated routing histograms.
        """
        # Use first 2 factual prompts
        prompts = [p["prompt"] for p in self.config["test_prompts"]["factual"][:2]]

        # Downstream layers to check
        all_layers = self.config["ablation"]["target_layers"]
        downstream = [l for l in all_layers if l > layer_idx][:3]
        if not downstream:
            return 0.0

        num_experts = self.config["moe"]["num_experts"]
        max_divergence = 0.0

        for prompt in prompts:
            # Baseline routing
            baseline_weights = await self.router.capture_router_weights(
                prompt, layers=downstream
            )

            # Build baseline histograms per layer
            baseline_hists: dict[int, list[float]] = {}
            for lw in baseline_weights:
                hist = [0.0] * num_experts
                for pos in lw.positions:
                    for exp_idx in pos.expert_indices:
                        hist[exp_idx] += 1.0
                total = sum(hist) or 1.0
                baseline_hists[lw.layer_idx] = [h / total for h in hist]

            # Ablated routing: patch router to mask expert, then capture weights
            # We capture routing by running a forward pass with both patches:
            # 1. Router class patch to mask the ablated expert
            # 2. MLP class patch to capture downstream routing
            ablated_hists = await self._capture_routing_with_ablation(
                prompt, layer_idx, expert_idx, downstream
            )

            # Compute JS divergence per downstream layer
            for dl in downstream:
                if dl in baseline_hists and dl in ablated_hists:
                    jsd = self._js_divergence(baseline_hists[dl], ablated_hists[dl])
                    max_divergence = max(max_divergence, jsd)

        return max_divergence

    async def _capture_routing_with_ablation(
        self,
        prompt: str,
        ablate_layer: int,
        ablate_expert: int,
        capture_layers: list[int],
    ) -> dict[int, list[float]]:
        """Capture downstream routing with an expert ablated.

        Combines two class-level patches:
        1. Router class: mask ablated expert (from generate_with_ablation pattern)
        2. MLP class: capture routing decisions (from capture_router_weights pattern)
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        num_experts = self.config["moe"]["num_experts"]
        top_k = self.config["moe"]["top_k"]

        captured: dict[int, list[int]] = {l: [] for l in capture_layers}

        # Get classes for patching
        sample_layer = self.model.model.layers[ablate_layer]
        router_class = type(sample_layer.mlp.router)
        mlp_class = type(sample_layer.mlp)
        original_router_call = router_class.__call__
        original_mlp_call = mlp_class.__call__

        model_ref = self.model
        ablate_set = {ablate_expert}
        target_ablate = {ablate_layer}

        def patched_router(router_self, x):
            """Mask ablated expert at target layer."""
            layer_idx = -1
            for i, layer in enumerate(model_ref.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            if layer_idx not in target_ablate:
                return original_router_call(router_self, x)

            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])

            logits = x @ router_self.weight.T
            if hasattr(router_self, "bias") and router_self.bias is not None:
                logits = logits + router_self.bias

            mask_values = [
                -1e9 if i in ablate_set else 0.0
                for i in range(logits.shape[-1])
            ]
            logits = logits + mx.array(mask_values, dtype=logits.dtype)

            k = router_self.num_experts_per_tok
            partitioned = mx.argpartition(logits, kth=-k, axis=-1)
            top_k_indices = partitioned[..., -k:]
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            top_k_weights = mx.softmax(top_k_logits, axis=-1)

            return top_k_weights, top_k_indices

        def patched_mlp(mlp_self, x):
            """Capture routing at downstream layers."""
            layer_idx = -1
            for i, layer in enumerate(model_ref.model.layers):
                if layer.mlp is mlp_self:
                    layer_idx = i
                    break

            if layer_idx in capture_layers:
                router = mlp_self.router
                if x.ndim == 3:
                    x_flat = x.reshape(-1, x.shape[-1])
                else:
                    x_flat = x

                router_result = router(x_flat)
                if isinstance(router_result, tuple):
                    _, indices = router_result
                    for pos in range(indices.shape[0]):
                        captured[layer_idx].extend(indices[pos].tolist())

            return original_mlp_call(mlp_self, x)

        try:
            router_class.__call__ = patched_router
            mlp_class.__call__ = patched_mlp
            self.model(input_ids)
        finally:
            router_class.__call__ = original_router_call
            mlp_class.__call__ = original_mlp_call

        # Convert to histograms
        hists: dict[int, list[float]] = {}
        for layer_idx, expert_indices in captured.items():
            hist = [0.0] * num_experts
            for exp_idx in expert_indices:
                if 0 <= exp_idx < num_experts:
                    hist[exp_idx] += 1.0
            total = sum(hist) or 1.0
            hists[layer_idx] = [h / total for h in hist]

        return hists

    @staticmethod
    def _js_divergence(p: list[float], q: list[float]) -> float:
        """Jensen-Shannon divergence between two distributions."""
        n = len(p)
        m = [(p[i] + q[i]) / 2.0 for i in range(n)]

        def kl(a, b):
            total = 0.0
            for i in range(n):
                if a[i] > 0 and b[i] > 0:
                    total += a[i] * math.log(a[i] / b[i])
            return total

        return (kl(p, m) + kl(q, m)) / 2.0

    # ─────────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────────

    def _classify_expert(
        self,
        signatures: list[AblationSignature],
        routing_disruption: float,
    ) -> tuple[str, float]:
        """Classify expert based on ablation signatures."""
        thresholds = self.config["classification"]
        n = len(signatures)

        no_change_rate = sum(1 for s in signatures if not s.output_changed) / n
        fact_rate = sum(1 for s in signatures if s.fact_error) / n
        struct_rate = sum(1 for s in signatures if s.structure_error) / n

        # Redundant: almost no output changes
        if no_change_rate > thresholds["redundant_threshold"]:
            return "redundant", no_change_rate

        # Routing: high downstream disruption, low direct errors
        if (routing_disruption > thresholds["routing_threshold"]
                and fact_rate < 0.2
                and struct_rate < 0.2):
            return "routing", routing_disruption

        # Storage: primarily fact errors, structure preserved
        if (fact_rate > struct_rate
                and fact_rate >= thresholds["fact_error_threshold"]):
            return "storage", fact_rate

        # Computation: primarily structure errors
        if (struct_rate > fact_rate
                and struct_rate >= thresholds["structure_error_threshold"]):
            return "computation", struct_rate

        # Mixed: classify by dominant signal
        if fact_rate >= struct_rate and fact_rate > 0:
            return "storage", fact_rate
        if struct_rate > 0:
            return "computation", struct_rate

        return "redundant", no_change_rate

    # ─────────────────────────────────────────────────────────────────
    # Phase 3: Recovery
    # ─────────────────────────────────────────────────────────────────

    async def _test_recovery(self, expert: ExpertClassification) -> float:
        """Test if memory bank injection recovers ablation damage.

        For each prompt that showed a fact error during ablation,
        inject the correct fact via [Memory Bank] and re-generate
        with the expert still ablated.

        Returns recovery rate (fraction of fact errors recovered).
        """
        max_tokens = self.config.get("recovery", {}).get("max_tokens", 30)
        prompts = self._get_all_prompts()
        prompt_lookup = {p["prompt"]: p for p in prompts}

        fact_error_sigs = [s for s in expert.signatures if s.fact_error]
        if not fact_error_sigs:
            return 0.0

        recovered = 0
        total = len(fact_error_sigs)

        for sig in fact_error_sigs:
            p = prompt_lookup.get(sig.prompt)
            if not p or not p.get("memory_bank"):
                continue

            # Build memory bank context
            memory_context = self._build_memory_bank(p["memory_bank"])
            augmented = (
                f"{memory_context}\n\n"
                f"Using the memory bank above, answer: {p['prompt']}\n"
                f"Answer:"
            )

            # Generate with expert ablated + memory bank
            text, _ = await self.router.generate_with_ablation(
                augmented,
                [expert.expert_idx],
                max_tokens=max_tokens,
                layers=[expert.layer_idx],
            )
            text = text.strip()

            # Check if expected answer now present
            if p["expected"] and p["expected"].lower() in text.lower():
                recovered += 1
                logger.info(
                    f"    Recovery SUCCESS: {expert.expert_id} | "
                    f"{p['prompt'][:40]}... -> {text[:40]}"
                )
            else:
                logger.info(
                    f"    Recovery FAILED:  {expert.expert_id} | "
                    f"{p['prompt'][:40]}... -> {text[:40]}"
                )

        return recovered / total

    @staticmethod
    def _build_memory_bank(entry: str) -> str:
        """Build [Memory Bank] context string."""
        return f"[Memory Bank]\n- {entry}\n[End Memory Bank]"

    # ─────────────────────────────────────────────────────────────────
    # Phase 4: Capacity
    # ─────────────────────────────────────────────────────────────────

    def _compute_capacity(self) -> CapacityEstimate:
        """Compute overall capacity estimates."""
        if not self.classifications:
            return CapacityEstimate(
                total_experts_tested=0,
                by_category={},
                by_category_pct={},
                storage_experts=[],
                recovery_success_rate=0.0,
                estimated_externalizable_fraction=0.0,
            )

        total = len(self.classifications)
        counts = Counter(c.classification for c in self.classifications.values())
        pcts = {k: v / total for k, v in counts.items()}

        storage = [
            c.expert_id for c in self.classifications.values()
            if c.classification == "storage"
        ]

        # Recovery rate across all storage experts
        recovery_rates = [
            c.recovery_rate for c in self.classifications.values()
            if c.classification == "storage" and c.recovery_rate is not None
        ]
        avg_recovery = sum(recovery_rates) / len(recovery_rates) if recovery_rates else 0.0

        # Extrapolate: fraction of all experts that are storage
        # We tested a subset, so this is an estimate
        num_total_experts = (
            self.config["moe"]["num_experts"]
            * len(self.config["ablation"]["target_layers"])
        )
        storage_fraction = len(storage) / num_total_experts if num_total_experts > 0 else 0.0

        return CapacityEstimate(
            total_experts_tested=total,
            by_category=dict(counts),
            by_category_pct=pcts,
            storage_experts=storage,
            recovery_success_rate=avg_recovery,
            estimated_externalizable_fraction=storage_fraction,
        )

    # ─────────────────────────────────────────────────────────────────
    # Orchestration
    # ─────────────────────────────────────────────────────────────────

    async def run(self, target_layer: int | None = None) -> dict[str, Any]:
        """Run all 4 phases."""
        await self.setup()

        # Phase 1: Baselines
        self.baselines = await self._generate_baselines()

        # Phase 2: Systematic ablation
        target_layers = self.config["ablation"]["target_layers"]
        if target_layer is not None:
            target_layers = [target_layer]

        use_quick_scan = self.config["sampling"]["use_quick_scan"]

        logger.info(f"\nPhase 2: Systematic ablation across {len(target_layers)} layers")

        for layer_idx in target_layers:
            logger.info(f"\n  Layer {layer_idx}:")

            # Quick scan to find causal experts
            if use_quick_scan:
                causal_experts = await self._quick_scan(layer_idx)
            else:
                causal_experts = list(range(self.config["moe"]["num_experts"]))

            if not causal_experts:
                logger.info(f"    No causal experts found, skipping layer")
                continue

            # Deep ablation on causal experts
            for expert_idx in causal_experts:
                expert_id = f"L{layer_idx}E{expert_idx}"
                logger.info(f"    Deep testing {expert_id}...")

                signatures = await self._deep_ablation(layer_idx, expert_idx)

                # Measure routing disruption
                disruption = await self._measure_routing_disruption(
                    layer_idx, expert_idx
                )

                # Classify
                classification, confidence = self._classify_expert(
                    signatures, disruption
                )

                n = len(signatures)
                self.classifications[expert_id] = ExpertClassification(
                    expert_id=expert_id,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    classification=classification,
                    confidence=confidence,
                    fact_error_rate=sum(1 for s in signatures if s.fact_error) / n,
                    structure_error_rate=sum(1 for s in signatures if s.structure_error) / n,
                    no_change_rate=sum(1 for s in signatures if not s.output_changed) / n,
                    routing_disruption=disruption,
                    recovery_rate=None,
                    num_prompts_tested=n,
                    signatures=signatures,
                )

                logger.info(
                    f"      -> {classification.upper()} "
                    f"(conf={confidence:.2f}, fact_err={self.classifications[expert_id].fact_error_rate:.2f}, "
                    f"struct_err={self.classifications[expert_id].structure_error_rate:.2f}, "
                    f"routing={disruption:.3f})"
                )

        # Phase 3: Recovery testing for storage experts
        storage_experts = [
            c for c in self.classifications.values()
            if c.classification == "storage"
        ]

        if storage_experts:
            logger.info(f"\nPhase 3: Recovery testing for {len(storage_experts)} storage experts")
            for expert in storage_experts:
                logger.info(f"  Testing recovery for {expert.expert_id}...")
                expert.recovery_rate = await self._test_recovery(expert)
                logger.info(f"    Recovery rate: {expert.recovery_rate:.1%}")
        else:
            logger.info("\nPhase 3: No storage experts found, skipping recovery")

        # Phase 4: Capacity estimation
        logger.info("\nPhase 4: Capacity estimation")
        capacity = self._compute_capacity()

        # Save and print
        self._save_results(capacity)
        self._print_summary(capacity)

        return self._build_results_dict(capacity)

    # ─────────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────────

    def _build_results_dict(self, capacity: CapacityEstimate) -> dict[str, Any]:
        """Build results dictionary for JSON serialization."""
        return {
            "metadata": {
                "model": self.config["model"],
                "timestamp": datetime.now().isoformat(),
                "target_layers": self.config["ablation"]["target_layers"],
                "num_prompts": len(self._get_all_prompts()),
            },
            "baselines": {
                prompt: {
                    "category": result.category,
                    "expected": result.expected,
                    "text": result.text,
                }
                for prompt, result in self.baselines.items()
            },
            "classifications": [
                {
                    "expert_id": c.expert_id,
                    "layer_idx": c.layer_idx,
                    "expert_idx": c.expert_idx,
                    "classification": c.classification,
                    "confidence": c.confidence,
                    "fact_error_rate": c.fact_error_rate,
                    "structure_error_rate": c.structure_error_rate,
                    "no_change_rate": c.no_change_rate,
                    "routing_disruption": c.routing_disruption,
                    "recovery_rate": c.recovery_rate,
                    "num_prompts_tested": c.num_prompts_tested,
                    "signatures": [
                        {
                            "prompt": s.prompt[:60],
                            "category": s.category,
                            "output_changed": s.output_changed,
                            "fact_error": s.fact_error,
                            "structure_error": s.structure_error,
                            "ablated_text": s.ablated_text[:100],
                        }
                        for s in c.signatures
                    ],
                }
                for c in self.classifications.values()
            ],
            "layer_summaries": self._build_layer_summaries(),
            "capacity_estimate": asdict(capacity),
        }

    def _build_layer_summaries(self) -> list[dict[str, Any]]:
        """Build per-layer summary."""
        layers: dict[int, list[ExpertClassification]] = {}
        for c in self.classifications.values():
            layers.setdefault(c.layer_idx, []).append(c)

        summaries = []
        for layer_idx in sorted(layers):
            experts = layers[layer_idx]
            counts = Counter(e.classification for e in experts)
            summaries.append({
                "layer_idx": layer_idx,
                "experts_tested": len(experts),
                "storage": counts.get("storage", 0),
                "computation": counts.get("computation", 0),
                "routing": counts.get("routing", 0),
                "redundant": counts.get("redundant", 0),
            })
        return summaries

    def _save_results(self, capacity: CapacityEstimate) -> None:
        """Save results to JSON."""
        output_path = (
            Path(__file__).parent / "results"
            / f"expert_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = self._build_results_dict(capacity)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_path}")

    def _print_summary(self, capacity: CapacityEstimate) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("EXPERT FUNCTION CLASSIFICATION - SUMMARY")
        print("=" * 70)

        # Layer map
        print("\nLayer  | Storage | Computation | Routing | Redundant | Tested")
        print("-------|---------|-------------|---------|-----------|-------")

        layers: dict[int, list[ExpertClassification]] = {}
        for c in self.classifications.values():
            layers.setdefault(c.layer_idx, []).append(c)

        for layer_idx in sorted(layers):
            experts = layers[layer_idx]
            counts = Counter(e.classification for e in experts)
            print(
                f"L{layer_idx:<5}| "
                f"{counts.get('storage', 0):>7} | "
                f"{counts.get('computation', 0):>11} | "
                f"{counts.get('routing', 0):>7} | "
                f"{counts.get('redundant', 0):>9} | "
                f"{len(experts):>5}"
            )

        # Totals
        print("-------|---------|-------------|---------|-----------|-------")
        print(
            f"Total  | "
            f"{capacity.by_category.get('storage', 0):>7} | "
            f"{capacity.by_category.get('computation', 0):>11} | "
            f"{capacity.by_category.get('routing', 0):>7} | "
            f"{capacity.by_category.get('redundant', 0):>9} | "
            f"{capacity.total_experts_tested:>5}"
        )

        # Percentages
        if capacity.total_experts_tested > 0:
            print(
                f"       | "
                f"{capacity.by_category_pct.get('storage', 0):>6.0%} | "
                f"{capacity.by_category_pct.get('computation', 0):>10.0%} | "
                f"{capacity.by_category_pct.get('routing', 0):>6.0%} | "
                f"{capacity.by_category_pct.get('redundant', 0):>8.0%} |"
            )

        # Recovery
        if capacity.storage_experts:
            print(f"\nStorage experts: {', '.join(capacity.storage_experts)}")
            print(f"Recovery success rate: {capacity.recovery_success_rate:.1%}")

        # Compression potential
        print(f"\nEstimated externalizable fraction: {capacity.estimated_externalizable_fraction:.1%}")
        total_params_b = 20.0  # GPT-OSS 20B
        ext_params = total_params_b * capacity.estimated_externalizable_fraction
        print(f"Estimated externalizable params: {ext_params:.1f}B of {total_params_b:.0f}B")

        # Validation criteria
        print("\n" + "-" * 70)
        print("VALIDATION CRITERIA")
        print("-" * 70)

        l16e4 = self.classifications.get("L16E4")
        if l16e4:
            v1 = l16e4.classification == "storage"
            print(f"  1. L16E4 is storage: {'PASS' if v1 else 'FAIL'} ({l16e4.classification})")
        else:
            print("  1. L16E4 is storage: NOT TESTED")

        storage_experts_with_recovery = [
            c for c in self.classifications.values()
            if c.classification == "storage" and c.recovery_rate is not None
        ]
        if storage_experts_with_recovery:
            avg_rec = sum(c.recovery_rate for c in storage_experts_with_recovery) / len(storage_experts_with_recovery)
            v2 = avg_rec > 0.7
            print(f"  2. Storage recovery >70%: {'PASS' if v2 else 'FAIL'} ({avg_rec:.1%})")
        else:
            print("  2. Storage recovery >70%: NO STORAGE EXPERTS")

        comp_experts = [
            c for c in self.classifications.values()
            if c.classification == "computation" and c.recovery_rate is not None
        ]
        if comp_experts:
            avg_comp_rec = sum(c.recovery_rate for c in comp_experts) / len(comp_experts)
            v3 = avg_comp_rec < 0.3
            print(f"  3. Computation recovery <30%: {'PASS' if v3 else 'FAIL'} ({avg_comp_rec:.1%})")
        else:
            print("  3. Computation recovery <30%: NO COMPUTATION EXPERTS TESTED")

        storage_pct = capacity.by_category_pct.get("storage", 0)
        v4 = storage_pct >= 0.10
        print(f"  4. >=10% storage experts: {'PASS' if v4 else 'FAIL'} ({storage_pct:.1%})")

        print("=" * 70)


# =============================================================================
# CLI
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Expert Function Classification")
    parser.add_argument(
        "--config", type=Path, help="Path to config file"
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Test a single layer only (e.g. --layer 16)"
    )
    args = parser.parse_args()

    experiment = ExpertFunctionClassification(args.config)
    await experiment.run(target_layer=args.layer)


if __name__ == "__main__":
    asyncio.run(main())
