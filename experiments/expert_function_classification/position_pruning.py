#!/usr/bin/env python3
"""Position-Class Pruning Experiment.

Position analysis (Part 11) showed expert routing is 93% position-coded.
Cross-layer ablation (Part 10) showed 4 arbitrary experts at all layers
causes model collapse. But was that because of insufficient QUANTITY or
insufficient POSITION COVERAGE?

This experiment tests: can we keep just 4 experts per layer (87.5% pruning)
if we ensure one expert per position class?

Conditions:
A. 4 position-diverse experts at all layers (1 per position class)
B. 4 same-position experts at all layers (all "end" specialists)
C. 4 arbitrary experts at all layers (Part 10 baseline replication)
D. 4 fact-specific experts at all layers (Part 10 baseline replication)

If A >> B ≈ C, position coverage is the critical variable for model coherence.

Phase 4 also tests within-class redundancy at a single layer: group L16's
32 experts by position class, prune to 1 per class, and test quality.

Run: python experiments/expert_function_classification/position_pruning.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


FACTS = [
    {"prompt": "The capital of France is", "expected": "Paris"},
    {"prompt": "The chemical symbol for gold is", "expected": "Au"},
    {"prompt": "The author of Romeo and Juliet is", "expected": "Shakespeare"},
    {"prompt": "The speed of light is approximately", "expected": "299"},
    {"prompt": "The CEO of Microsoft is", "expected": "Nadella"},
    {"prompt": "The capital of Japan is", "expected": "Tokyo"},
    {"prompt": "The chemical symbol for silver is", "expected": "Ag"},
    {"prompt": "The capital of Australia is", "expected": "Canberra"},
]

# Extra prompts for position profiling (diverse structures)
PROFILE_PROMPTS = [
    "The capital of Germany is",
    "The chemical symbol for iron is",
    "France's capital city is",
    "What is the capital of France?",
    "Once upon a time there was a",
    "The opposite of hot is",
    "If all cats are mammals then Fluffy is a",
    "To convert Celsius to Fahrenheit multiply by",
    "The president of the United States is",
]

TARGET_LAYERS = [8, 12, 16, 20]
POSITION_CLASSES = ["start", "early_mid", "late_mid", "end"]
N_BINS = 4


@dataclass
class ExpertProfile:
    """Position profile for one expert at one layer."""
    expert_idx: int
    layer_idx: int
    bin_counts: list[int]  # [start, early_mid, late_mid, end]
    total_activations: int
    dominant_class: str
    dominance_score: float  # Fraction of activations in dominant class
    selectivity: float


@dataclass
class PruningResult:
    """Result of testing one pruning condition."""
    condition_name: str
    kept_experts: list[int]
    layers: list[int]
    n_kept: int
    n_layers: int
    fact_results: list[dict]  # [{prompt, expected, text, correct}]
    facts_correct: int
    facts_total: int
    accuracy: float


class PositionPruning:
    """Test whether position-diverse expert selection enables 87.5% pruning."""

    def __init__(self):
        self.router = None
        self.model = None
        self.tokenizer = None
        self.all_moe_layers: list[int] = []

    async def setup(self):
        from chuk_lazarus.introspection.moe import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        self.router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = self.router._model
        self.tokenizer = self.router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        info = self.router._info
        self.all_moe_layers = sorted(info.moe_layers)
        logger.info(
            f"Model loaded: {info.num_experts} experts, "
            f"{len(info.moe_layers)} MoE layers, "
            f"top-{info.num_experts_per_tok}"
        )

    def _generate(self, prompt: str, max_tokens: int = 30) -> str:
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

    def _has_answer(self, text: str, expected: str) -> bool:
        return expected.lower() in text.lower()

    def _compute_repetition_ratio(self, text: str, n: int = 3) -> float:
        """Fraction of n-grams that are repeated."""
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        return 1.0 - len(set(ngrams)) / len(ngrams)

    # -----------------------------------------------------------------
    # Phase 1: Profile experts by position
    # -----------------------------------------------------------------

    async def _profile_experts(
        self, prompts: list[str]
    ) -> dict[int, list[ExpertProfile]]:
        """Profile each expert's position distribution at each layer."""
        # Capture routing for all prompts
        all_routing = {}
        for prompt in prompts:
            weights_list = await self.router.capture_router_weights(
                prompt, layers=TARGET_LAYERS
            )
            all_routing[prompt] = weights_list

        # Build position profiles
        profiles: dict[int, list[ExpertProfile]] = {}

        for layer_idx in TARGET_LAYERS:
            # expert_idx -> bin counts
            expert_bins: dict[int, list[int]] = {
                e: [0] * N_BINS for e in range(32)
            }

            for prompt, weights_list in all_routing.items():
                for lw in weights_list:
                    if lw.layer_idx != layer_idx:
                        continue
                    n_pos = len(lw.positions)
                    if n_pos == 0:
                        continue

                    for pos in lw.positions:
                        rel = pos.position_idx / max(n_pos - 1, 1)
                        bin_idx = min(int(rel * N_BINS), N_BINS - 1)
                        for expert_idx in pos.expert_indices:
                            expert_bins[expert_idx][bin_idx] += 1

            layer_profiles = []
            for expert_idx in range(32):
                bins = expert_bins[expert_idx]
                total = sum(bins)

                if total == 0:
                    layer_profiles.append(ExpertProfile(
                        expert_idx=expert_idx,
                        layer_idx=layer_idx,
                        bin_counts=bins,
                        total_activations=0,
                        dominant_class="none",
                        dominance_score=0.0,
                        selectivity=0.0,
                    ))
                    continue

                # Dominant class
                max_bin = max(bins)
                dominant_idx = bins.index(max_bin)
                dominance = max_bin / total

                # Entropy-based selectivity
                max_entropy = math.log2(N_BINS)
                entropy = 0.0
                for count in bins:
                    if count > 0:
                        p = count / total
                        entropy -= p * math.log2(p)
                selectivity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

                layer_profiles.append(ExpertProfile(
                    expert_idx=expert_idx,
                    layer_idx=layer_idx,
                    bin_counts=bins,
                    total_activations=total,
                    dominant_class=POSITION_CLASSES[dominant_idx],
                    dominance_score=dominance,
                    selectivity=selectivity,
                ))

            profiles[layer_idx] = layer_profiles

        return profiles

    # -----------------------------------------------------------------
    # Phase 2: Select expert sets
    # -----------------------------------------------------------------

    def _select_diverse(
        self, profiles: dict[int, list[ExpertProfile]]
    ) -> list[int]:
        """Select 4 experts that cover all position classes across layers.

        For each position class, find the expert with the highest average
        selectivity for that class across all target layers.
        """
        # For each expert, compute average bin score across layers
        expert_class_scores: dict[int, dict[str, float]] = {
            e: {c: 0.0 for c in POSITION_CLASSES} for e in range(32)
        }

        for layer_idx, layer_profiles in profiles.items():
            for prof in layer_profiles:
                if prof.total_activations == 0:
                    continue
                for i, cls_name in enumerate(POSITION_CLASSES):
                    score = prof.bin_counts[i] / prof.total_activations
                    expert_class_scores[prof.expert_idx][cls_name] += score

        # Normalize by number of layers
        n_layers = len(profiles)
        for expert_idx in expert_class_scores:
            for cls_name in POSITION_CLASSES:
                expert_class_scores[expert_idx][cls_name] /= n_layers

        # Greedy selection: for each position class, pick the best expert
        selected = []
        used = set()
        for cls_name in POSITION_CLASSES:
            best_expert = -1
            best_score = -1
            for expert_idx in range(32):
                if expert_idx in used:
                    continue
                score = expert_class_scores[expert_idx][cls_name]
                if score > best_score:
                    best_score = score
                    best_expert = expert_idx
            if best_expert >= 0:
                selected.append(best_expert)
                used.add(best_expert)

        return selected

    def _select_same_position(
        self, profiles: dict[int, list[ExpertProfile]]
    ) -> list[int]:
        """Select 4 experts that all prefer the same position class ("end")."""
        # Find experts with highest "end" score across layers
        expert_end_scores: dict[int, float] = {}
        for layer_idx, layer_profiles in profiles.items():
            for prof in layer_profiles:
                if prof.total_activations == 0:
                    continue
                end_score = prof.bin_counts[3] / prof.total_activations
                expert_end_scores[prof.expert_idx] = (
                    expert_end_scores.get(prof.expert_idx, 0) + end_score
                )

        ranked = sorted(
            expert_end_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [e for e, _ in ranked[:4]]

    # -----------------------------------------------------------------
    # Phase 3: All-layer pruning test
    # -----------------------------------------------------------------

    async def _test_pruning(
        self,
        condition_name: str,
        kept_experts: list[int],
        layers: list[int],
    ) -> PruningResult:
        """Ablate all experts except the kept set and test facts."""
        ablated = [e for e in range(32) if e not in kept_experts]

        fact_results = []
        correct_count = 0

        for fact in FACTS:
            text, _ = await self.router.generate_with_ablation(
                fact["prompt"],
                ablated,
                max_tokens=30,
                layers=layers,
            )
            text = text.strip()
            correct = self._has_answer(text, fact["expected"])
            if correct:
                correct_count += 1

            rep = self._compute_repetition_ratio(text)
            fact_results.append({
                "prompt": fact["prompt"],
                "expected": fact["expected"],
                "text": text[:120],
                "correct": correct,
                "repetition": round(rep, 3),
            })

            status = "OK" if correct else "LOST"
            logger.info(
                f"    {fact['prompt'][:35]:35} | {status:4} | "
                f"rep={rep:.2f} | {text[:45]}"
            )

        total = len(FACTS)
        return PruningResult(
            condition_name=condition_name,
            kept_experts=kept_experts,
            layers=layers,
            n_kept=len(kept_experts),
            n_layers=len(layers),
            fact_results=fact_results,
            facts_correct=correct_count,
            facts_total=total,
            accuracy=correct_count / total if total else 0,
        )

    # -----------------------------------------------------------------
    # Phase 4: Single-layer within-class pruning
    # -----------------------------------------------------------------

    async def _test_single_layer_pruning(
        self,
        profiles: dict[int, list[ExpertProfile]],
        layer_idx: int = 16,
    ) -> list[PruningResult]:
        """At one layer, prune within each position class."""
        results = []
        layer_profiles = profiles.get(layer_idx, [])

        # Group experts by dominant class
        class_experts: dict[str, list[ExpertProfile]] = defaultdict(list)
        for prof in layer_profiles:
            if prof.total_activations > 0:
                class_experts[prof.dominant_class].append(prof)

        # Sort each class by selectivity (best first)
        for cls_name in class_experts:
            class_experts[cls_name].sort(
                key=lambda p: p.selectivity, reverse=True
            )

        logger.info(f"\n  Expert position classes at L{layer_idx}:")
        for cls_name in POSITION_CLASSES:
            experts = class_experts.get(cls_name, [])
            expert_ids = [f"E{p.expert_idx}" for p in experts]
            logger.info(
                f"    {cls_name:>9}: {len(experts)} experts - "
                f"{', '.join(expert_ids[:8])}"
                + (f" (+{len(experts)-8} more)" if len(experts) > 8 else "")
            )

        # Test: keep only top-1 per class at this layer
        keep_1_per_class = []
        for cls_name in POSITION_CLASSES:
            experts = class_experts.get(cls_name, [])
            if experts:
                keep_1_per_class.append(experts[0].expert_idx)

        logger.info(
            f"\n  Pruning to 1 per class at L{layer_idx}: "
            f"keep {keep_1_per_class}"
        )
        result_1 = await self._test_pruning(
            f"L{layer_idx}_1_per_class",
            keep_1_per_class,
            [layer_idx],
        )
        results.append(result_1)

        # Test: keep top-2 per class at this layer
        keep_2_per_class = []
        for cls_name in POSITION_CLASSES:
            experts = class_experts.get(cls_name, [])
            for p in experts[:2]:
                keep_2_per_class.append(p.expert_idx)

        logger.info(
            f"\n  Pruning to 2 per class at L{layer_idx}: "
            f"keep {keep_2_per_class} ({len(keep_2_per_class)} total)"
        )
        result_2 = await self._test_pruning(
            f"L{layer_idx}_2_per_class",
            keep_2_per_class,
            [layer_idx],
        )
        results.append(result_2)

        return results

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        await self.setup()

        # Phase 0: Baselines
        logger.info("\nPhase 0: Generating baselines")
        baselines = {}
        for fact in FACTS:
            text = self._generate(fact["prompt"])
            correct = self._has_answer(text, fact["expected"])
            rep = self._compute_repetition_ratio(text)
            baselines[fact["prompt"]] = {
                "text": text[:120],
                "correct": correct,
                "repetition": round(rep, 3),
            }
            logger.info(
                f"  {fact['prompt'][:40]:40} -> "
                f"{'OK' if correct else 'WRONG'}: {text[:50]}"
            )

        # Phase 1: Profile experts
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1: PROFILING EXPERT POSITION CLASSES")
        logger.info("=" * 70)

        all_prompts = (
            [f["prompt"] for f in FACTS] + PROFILE_PROMPTS
        )
        profiles = await self._profile_experts(all_prompts)

        for layer_idx in TARGET_LAYERS:
            class_counts = Counter()
            for prof in profiles[layer_idx]:
                if prof.total_activations > 0:
                    class_counts[prof.dominant_class] += 1
            logger.info(
                f"  L{layer_idx}: "
                + " | ".join(
                    f"{cls}={class_counts.get(cls, 0)}"
                    for cls in POSITION_CLASSES
                )
            )

        # Phase 2: Select expert sets
        logger.info("\n" + "=" * 70)
        logger.info("Phase 2: SELECTING EXPERT SETS")
        logger.info("=" * 70)

        diverse_set = self._select_diverse(profiles)
        same_pos_set = self._select_same_position(profiles)
        arbitrary_set = [28, 29, 30, 31]  # Same as Part 10 baseline

        # Get fact-specific experts at L16 for comparison
        routing_data = await self.router.capture_router_weights(
            FACTS[0]["prompt"], layers=[16]
        )
        fact_set = []
        for lw in routing_data:
            if lw.layer_idx == 16 and lw.positions:
                fact_set = list(lw.positions[-1].expert_indices)

        logger.info(f"  Diverse (1/class):   {diverse_set}")
        for i, expert_idx in enumerate(diverse_set):
            for layer_idx in TARGET_LAYERS:
                prof = profiles[layer_idx][expert_idx]
                logger.info(
                    f"    E{expert_idx} at L{layer_idx}: "
                    f"class={prof.dominant_class:>9} "
                    f"dom={prof.dominance_score:.2f} "
                    f"sel={prof.selectivity:.2f}"
                )

        logger.info(f"  Same-position (end): {same_pos_set}")
        logger.info(f"  Arbitrary [28-31]:   {arbitrary_set}")
        logger.info(f"  Fact-specific (L16): {fact_set}")

        # Phase 3: All-layer pruning test
        logger.info("\n" + "=" * 70)
        logger.info("Phase 3: ALL-LAYER PRUNING (keep 4, ablate 28 at all layers)")
        logger.info("=" * 70)

        all_layer_results: list[PruningResult] = []

        for name, kept in [
            ("diverse_4", diverse_set),
            ("same_pos_4", same_pos_set),
            ("arbitrary_4", arbitrary_set),
            ("fact_specific_4", fact_set),
        ]:
            logger.info(f"\n  Condition: {name} — keep {kept}")
            result = await self._test_pruning(
                name, kept, self.all_moe_layers
            )
            all_layer_results.append(result)

        # Phase 4: Single-layer within-class pruning
        logger.info("\n" + "=" * 70)
        logger.info("Phase 4: SINGLE-LAYER POSITION-CLASS PRUNING (L16)")
        logger.info("=" * 70)

        single_layer_results = await self._test_single_layer_pruning(
            profiles, layer_idx=16
        )

        # Save and print
        results = self._build_results(
            baselines, profiles, diverse_set, same_pos_set,
            arbitrary_set, fact_set, all_layer_results,
            single_layer_results,
        )
        self._save_results(results)
        self._print_summary(
            profiles, all_layer_results, single_layer_results
        )
        return results

    def _build_results(
        self, baselines, profiles, diverse_set, same_pos_set,
        arbitrary_set, fact_set, all_layer_results, single_layer_results,
    ) -> dict[str, Any]:
        def serialize_pruning(pr: PruningResult) -> dict:
            return {
                "condition": pr.condition_name,
                "kept_experts": pr.kept_experts,
                "n_kept": pr.n_kept,
                "n_layers": pr.n_layers,
                "facts_correct": pr.facts_correct,
                "facts_total": pr.facts_total,
                "accuracy": pr.accuracy,
                "fact_results": pr.fact_results,
            }

        profiles_data = {}
        for layer_idx, layer_profiles in profiles.items():
            profiles_data[str(layer_idx)] = [
                {
                    "expert_idx": p.expert_idx,
                    "bin_counts": p.bin_counts,
                    "dominant_class": p.dominant_class,
                    "dominance_score": p.dominance_score,
                    "selectivity": p.selectivity,
                    "total_activations": p.total_activations,
                }
                for p in layer_profiles
            ]

        return {
            "metadata": {
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "experiment": "position_pruning",
            },
            "baselines": baselines,
            "expert_profiles": profiles_data,
            "selected_sets": {
                "diverse": diverse_set,
                "same_position": same_pos_set,
                "arbitrary": arbitrary_set,
                "fact_specific": fact_set,
            },
            "all_layer_pruning": [
                serialize_pruning(r) for r in all_layer_results
            ],
            "single_layer_pruning": [
                serialize_pruning(r) for r in single_layer_results
            ],
        }

    def _save_results(self, results: dict) -> None:
        output_path = (
            Path(__file__).parent
            / "results"
            / f"position_pruning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")

    def _print_summary(
        self,
        profiles: dict[int, list[ExpertProfile]],
        all_layer_results: list[PruningResult],
        single_layer_results: list[PruningResult],
    ) -> None:
        print("\n" + "=" * 70)
        print("POSITION-CLASS PRUNING - SUMMARY")
        print("=" * 70)

        # Expert class distribution
        print("\n--- Expert Position Classes ---")
        for layer_idx in TARGET_LAYERS:
            class_counts = Counter()
            for prof in profiles[layer_idx]:
                if prof.total_activations > 0:
                    class_counts[prof.dominant_class] += 1
            print(
                f"  L{layer_idx}: "
                + " | ".join(
                    f"{cls}={class_counts.get(cls, 0)}"
                    for cls in POSITION_CLASSES
                )
            )

        # All-layer pruning comparison
        print("\n--- All-Layer Pruning: Keep 4 Experts at All 24 Layers ---")
        print(
            f"{'Condition':>20} | {'Kept':>12} | "
            f"Correct | Total | Accuracy"
        )
        print("-" * 70)

        for r in all_layer_results:
            experts_str = str(r.kept_experts)
            print(
                f"{r.condition_name:>20} | {experts_str:>12} | "
                f"{r.facts_correct:>7} | {r.facts_total:>5} | "
                f"{r.accuracy:.0%}"
            )

        # Quality detail for diverse vs others
        print("\n--- Output Quality (All-Layer Pruning) ---")
        for r in all_layer_results:
            avg_rep = (
                sum(f["repetition"] for f in r.fact_results)
                / len(r.fact_results)
                if r.fact_results else 0
            )
            print(
                f"  {r.condition_name:>20}: "
                f"accuracy={r.accuracy:.0%} | "
                f"avg_repetition={avg_rep:.2f}"
            )

        # Single-layer pruning
        print("\n--- Single-Layer Pruning (L16) ---")
        for r in single_layer_results:
            print(
                f"  {r.condition_name:>25}: "
                f"keep {r.n_kept} experts → "
                f"{r.facts_correct}/{r.facts_total} facts "
                f"({r.accuracy:.0%})"
            )

        # Key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        diverse_result = next(
            (r for r in all_layer_results if r.condition_name == "diverse_4"),
            None,
        )
        same_result = next(
            (r for r in all_layer_results if r.condition_name == "same_pos_4"),
            None,
        )
        arbitrary_result = next(
            (r for r in all_layer_results if r.condition_name == "arbitrary_4"),
            None,
        )

        if diverse_result and same_result and arbitrary_result:
            print(
                f"  Position-diverse (1/class):  "
                f"{diverse_result.accuracy:.0%}"
            )
            print(
                f"  Same-position (all end):     "
                f"{same_result.accuracy:.0%}"
            )
            print(
                f"  Arbitrary [28-31]:           "
                f"{arbitrary_result.accuracy:.0%}"
            )

            if diverse_result.accuracy > same_result.accuracy + 0.1:
                print(
                    "\n  FINDING: Position-diverse selection OUTPERFORMS "
                    "same-position and arbitrary selection."
                )
                print(
                    "  Position COVERAGE, not expert COUNT, is the critical "
                    "variable for model coherence."
                )
                print(
                    f"  87.5% expert pruning is achievable with "
                    f"position-aware selection "
                    f"({diverse_result.accuracy:.0%} fact preservation)."
                )
            elif diverse_result.accuracy == same_result.accuracy == 0:
                print(
                    "\n  FINDING: 4 experts at all layers is below the "
                    "minimum viable threshold"
                )
                print(
                    "  regardless of position selection strategy."
                )
            else:
                print(
                    "\n  FINDING: Position diversity provides "
                    f"{diverse_result.accuracy - same_result.accuracy:+.0%} "
                    "improvement"
                )
                print("  over same-position selection.")

        # Single-layer finding
        if single_layer_results:
            best_single = max(single_layer_results, key=lambda r: r.accuracy)
            print(
                f"\n  Single-layer (L16): {best_single.condition_name} → "
                f"{best_single.accuracy:.0%} with {best_single.n_kept} experts"
            )

        print("=" * 70)


async def main():
    experiment = PositionPruning()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
