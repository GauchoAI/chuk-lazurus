#!/usr/bin/env python3
"""Cross-Layer Ablation Experiment.

Progressive ablation proved facts survive removal of all 4 top-k experts
at any single layer. This experiment tests the next questions:

1. Do facts survive when the same experts are disabled across MULTIPLE
   layers simultaneously? (Layer escalation)
2. How many of the 32 experts can be disabled at a single layer before
   a fact breaks? (Expert count escalation)
3. Do facts survive when 28-30 of 32 experts are disabled at ALL 24
   layers simultaneously? (Maximum stress test)

If facts survive condition 3, the residual stream + attention alone
carry all factual knowledge. MoE expert routing is entirely dispensable
for fact retrieval.

Run: python experiments/expert_function_classification/cross_layer_ablation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
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


# =============================================================================
# Data
# =============================================================================

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


@dataclass
class AblationCondition:
    """One ablation test condition."""

    name: str
    expert_indices: list[int]
    layers: list[int]
    n_experts_ablated: int
    n_layers_ablated: int


@dataclass
class FactTestResult:
    """Result of testing one fact under one condition."""

    prompt: str
    expected: str
    condition_name: str
    text: str
    correct: bool
    n_experts_ablated: int
    n_layers_ablated: int


# =============================================================================
# Experiment
# =============================================================================


class CrossLayerAblation:
    """Cross-layer expert ablation to test residual stream knowledge."""

    def __init__(self):
        self.router = None
        self.model = None
        self.tokenizer = None
        self.all_moe_layers: list[int] = []

    async def setup(self):
        """Load model via ExpertRouter."""
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
        """Generate text (no ablation)."""
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
        """Check if expected answer appears in text."""
        return expected.lower() in text.lower()

    async def _get_routing(self, prompt: str) -> dict[int, list[int]]:
        """Get top-4 expert indices at each MoE layer for last token."""
        weights_list = await self.router.capture_router_weights(
            prompt, layers=self.all_moe_layers
        )

        routing = {}
        for lw in weights_list:
            if lw.positions:
                last_pos = lw.positions[-1]
                routing[lw.layer_idx] = list(last_pos.expert_indices)
        return routing

    async def _test_condition(
        self, fact: dict, condition: AblationCondition
    ) -> FactTestResult:
        """Test one fact under one ablation condition."""
        text, _ = await self.router.generate_with_ablation(
            fact["prompt"],
            condition.expert_indices,
            max_tokens=30,
            layers=condition.layers,
        )
        text = text.strip()
        correct = self._has_answer(text, fact["expected"])
        return FactTestResult(
            prompt=fact["prompt"],
            expected=fact["expected"],
            condition_name=condition.name,
            text=text[:100],
            correct=correct,
            n_experts_ablated=condition.n_experts_ablated,
            n_layers_ablated=condition.n_layers_ablated,
        )

    def _build_layer_sets(self) -> list[tuple[str, list[int]]]:
        """Build layer sets for escalation, centered on L16."""
        all_layers = self.all_moe_layers
        return [
            ("L16_only", [16]),
            ("L14-L18", [l for l in all_layers if 14 <= l <= 18]),
            ("L8-L20", [l for l in all_layers if 8 <= l <= 20]),
            ("all_layers", all_layers),
        ]

    def _build_expert_sets(
        self, fact_experts: list[int]
    ) -> list[tuple[str, list[int]]]:
        """Build expert sets for escalation at a single layer.

        Starts with the fact's top-4, then adds more by index.
        """
        all_experts = list(range(32))
        remaining = [e for e in all_experts if e not in fact_experts]

        return [
            ("ablate_4", list(fact_experts)),
            ("ablate_8", list(fact_experts) + remaining[:4]),
            ("ablate_16", list(fact_experts) + remaining[:12]),
            ("ablate_24", list(fact_experts) + remaining[:20]),
            ("ablate_28", list(fact_experts) + remaining[:24]),
            ("ablate_30", list(fact_experts) + remaining[:26]),
        ]

    async def run(self) -> dict[str, Any]:
        """Run the full experiment."""
        await self.setup()

        # =====================================================================
        # Phase 0: Baselines
        # =====================================================================
        logger.info("\nPhase 0: Generating baselines")
        baselines: dict[str, dict] = {}
        for fact in FACTS:
            text = self._generate(fact["prompt"])
            correct = self._has_answer(text, fact["expected"])
            baselines[fact["prompt"]] = {"text": text[:100], "correct": correct}
            logger.info(
                f"  {fact['prompt'][:40]:40} -> "
                f"{'OK' if correct else 'WRONG'}: {text[:50]}"
            )

        # =====================================================================
        # Phase 1: Capture routing at ALL layers for all facts
        # =====================================================================
        logger.info("\nPhase 1: Capturing expert routing at all layers")
        fact_routing: dict[str, dict[int, list[int]]] = {}
        for fact in FACTS:
            routing = await self._get_routing(fact["prompt"])
            fact_routing[fact["prompt"]] = routing
            l16 = routing.get(16, [])
            logger.info(f"  {fact['prompt'][:40]:40} L16={l16}")

        # =====================================================================
        # Phase 2: Layer escalation
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 2: LAYER ESCALATION")
        logger.info(
            "Ablate fact-specific L16 top-4 across increasing numbers of layers"
        )
        logger.info("=" * 70)

        layer_sets = self._build_layer_sets()
        layer_results: list[FactTestResult] = []

        for fact in FACTS:
            routing = fact_routing[fact["prompt"]]
            l16_experts = routing.get(16, [])

            for set_name, layers in layer_sets:
                condition = AblationCondition(
                    name=f"layer_esc_{set_name}",
                    expert_indices=l16_experts,
                    layers=layers,
                    n_experts_ablated=len(l16_experts),
                    n_layers_ablated=len(layers),
                )
                result = await self._test_condition(fact, condition)
                layer_results.append(result)
                status = "OK" if result.correct else "LOST"
                logger.info(
                    f"  {fact['prompt'][:30]:30} | {set_name:10} "
                    f"({len(layers):2} layers) | "
                    f"{status:4} | {result.text[:40]}"
                )

        # =====================================================================
        # Phase 3: Expert count escalation at L16
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 3: EXPERT COUNT ESCALATION")
        logger.info("Ablate increasing numbers of experts at Layer 16 only")
        logger.info("=" * 70)

        expert_results: list[FactTestResult] = []

        for fact in FACTS:
            routing = fact_routing[fact["prompt"]]
            l16_experts = routing.get(16, [])
            expert_sets = self._build_expert_sets(l16_experts)

            for set_name, experts in expert_sets:
                condition = AblationCondition(
                    name=f"expert_esc_{set_name}",
                    expert_indices=experts,
                    layers=[16],
                    n_experts_ablated=len(experts),
                    n_layers_ablated=1,
                )
                result = await self._test_condition(fact, condition)
                expert_results.append(result)
                status = "OK" if result.correct else "LOST"
                logger.info(
                    f"  {fact['prompt'][:30]:30} | {set_name:10} "
                    f"({len(experts):2} experts) | "
                    f"{status:4} | {result.text[:40]}"
                )

        # =====================================================================
        # Phase 4: Maximum stress test
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 4: MAXIMUM STRESS TEST")
        logger.info("Ablate 28-30 experts at ALL MoE layers simultaneously")
        logger.info("=" * 70)

        stress_results: list[FactTestResult] = []

        for fact in FACTS:
            routing = fact_routing[fact["prompt"]]
            l16_experts = routing.get(16, [])

            # Condition A: Ablate 28, keep [28-31] (arbitrary survivors)
            condition_a = AblationCondition(
                name="stress_28_keep_high",
                expert_indices=list(range(28)),
                layers=self.all_moe_layers,
                n_experts_ablated=28,
                n_layers_ablated=len(self.all_moe_layers),
            )
            result_a = await self._test_condition(fact, condition_a)
            stress_results.append(result_a)
            logger.info(
                f"  {fact['prompt'][:30]:30} | 28 ablated, keep [28-31] | "
                f"{'OK' if result_a.correct else 'LOST':4} | {result_a.text[:40]}"
            )

            # Condition B: Ablate 28, keep fact's L16 top-4 experts
            others = [e for e in range(32) if e not in l16_experts]
            condition_b = AblationCondition(
                name="stress_28_keep_fact",
                expert_indices=others[:28],  # ablate 28 non-fact experts
                layers=self.all_moe_layers,
                n_experts_ablated=len(others[:28]),
                n_layers_ablated=len(self.all_moe_layers),
            )
            result_b = await self._test_condition(fact, condition_b)
            stress_results.append(result_b)
            logger.info(
                f"  {fact['prompt'][:30]:30} | 28 ablated, keep fact-4  | "
                f"{'OK' if result_b.correct else 'LOST':4} | {result_b.text[:40]}"
            )

            # Condition C: Ablate 30, keep only [30-31]
            condition_c = AblationCondition(
                name="stress_30_keep_high",
                expert_indices=list(range(30)),
                layers=self.all_moe_layers,
                n_experts_ablated=30,
                n_layers_ablated=len(self.all_moe_layers),
            )
            result_c = await self._test_condition(fact, condition_c)
            stress_results.append(result_c)
            logger.info(
                f"  {fact['prompt'][:30]:30} | 30 ablated, keep [30-31] | "
                f"{'OK' if result_c.correct else 'LOST':4} | {result_c.text[:40]}"
            )

        # =====================================================================
        # Save and print
        # =====================================================================
        results = self._build_results(
            baselines, fact_routing, layer_results, expert_results, stress_results
        )
        self._save_results(results)
        self._print_summary(layer_results, expert_results, stress_results)
        return results

    def _build_results(
        self,
        baselines: dict,
        routing: dict,
        layer_results: list[FactTestResult],
        expert_results: list[FactTestResult],
        stress_results: list[FactTestResult],
    ) -> dict[str, Any]:
        def serialize(results: list[FactTestResult]) -> list[dict]:
            return [
                {
                    "prompt": r.prompt,
                    "expected": r.expected,
                    "condition": r.condition_name,
                    "text": r.text,
                    "correct": r.correct,
                    "n_experts_ablated": r.n_experts_ablated,
                    "n_layers_ablated": r.n_layers_ablated,
                }
                for r in results
            ]

        return {
            "metadata": {
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "num_facts": len(FACTS),
                "experiment": "cross_layer_ablation",
            },
            "baselines": baselines,
            "routing": {
                prompt: {str(k): v for k, v in layers.items()}
                for prompt, layers in routing.items()
            },
            "layer_escalation": serialize(layer_results),
            "expert_escalation": serialize(expert_results),
            "stress_test": serialize(stress_results),
        }

    def _save_results(self, results: dict) -> None:
        output_path = (
            Path(__file__).parent
            / "results"
            / f"cross_layer_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")

    def _print_summary(
        self,
        layer_results: list[FactTestResult],
        expert_results: list[FactTestResult],
        stress_results: list[FactTestResult],
    ) -> None:
        print("\n" + "=" * 70)
        print("CROSS-LAYER ABLATION - SUMMARY")
        print("=" * 70)

        # --- Layer escalation ---
        print("\n--- Layer Escalation (fact-specific top-4 across N layers) ---")
        print(f"{'Layers':>15} | {'N':>3} | Correct | Total | Rate")
        print("-" * 55)

        layer_groups: dict[str, dict] = {}
        for r in layer_results:
            key = r.condition_name
            if key not in layer_groups:
                layer_groups[key] = {
                    "correct": 0,
                    "total": 0,
                    "n_layers": r.n_layers_ablated,
                }
            layer_groups[key]["total"] += 1
            if r.correct:
                layer_groups[key]["correct"] += 1

        for name, data in sorted(
            layer_groups.items(), key=lambda x: x[1]["n_layers"]
        ):
            rate = data["correct"] / data["total"] if data["total"] else 0
            label = name.replace("layer_esc_", "")
            print(
                f"{label:>15} | {data['n_layers']:>3} | "
                f"{data['correct']:>7} | {data['total']:>5} | {rate:.0%}"
            )

        # --- Expert escalation ---
        print("\n--- Expert Escalation (N experts at Layer 16 only) ---")
        print(f"{'Experts':>15} | {'N':>3} | Correct | Total | Rate")
        print("-" * 55)

        expert_groups: dict[str, dict] = {}
        for r in expert_results:
            key = r.condition_name
            if key not in expert_groups:
                expert_groups[key] = {
                    "correct": 0,
                    "total": 0,
                    "n_experts": r.n_experts_ablated,
                }
            expert_groups[key]["total"] += 1
            if r.correct:
                expert_groups[key]["correct"] += 1

        for name, data in sorted(
            expert_groups.items(), key=lambda x: x[1]["n_experts"]
        ):
            rate = data["correct"] / data["total"] if data["total"] else 0
            label = name.replace("expert_esc_", "")
            print(
                f"{label:>15} | {data['n_experts']:>3} | "
                f"{data['correct']:>7} | {data['total']:>5} | {rate:.0%}"
            )

        # --- Stress test ---
        print("\n--- Maximum Stress Test (N experts at ALL MoE layers) ---")
        print(f"{'Condition':>25} | Correct | Total | Rate")
        print("-" * 55)

        stress_groups: dict[str, dict] = {}
        for r in stress_results:
            key = r.condition_name
            if key not in stress_groups:
                stress_groups[key] = {
                    "correct": 0,
                    "total": 0,
                    "n_experts": r.n_experts_ablated,
                }
            stress_groups[key]["total"] += 1
            if r.correct:
                stress_groups[key]["correct"] += 1

        for name, data in sorted(
            stress_groups.items(), key=lambda x: x[1]["n_experts"]
        ):
            rate = data["correct"] / data["total"] if data["total"] else 0
            print(
                f"{name:>25} | {data['correct']:>7} | "
                f"{data['total']:>5} | {rate:.0%}"
            )

        # --- Overall ---
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        all_results = layer_results + expert_results + stress_results
        total = len(all_results)
        correct = sum(1 for r in all_results if r.correct)

        max_experts_survived = 0
        max_layers_survived = 0
        for r in all_results:
            if r.correct:
                max_experts_survived = max(
                    max_experts_survived, r.n_experts_ablated
                )
                max_layers_survived = max(
                    max_layers_survived, r.n_layers_ablated
                )

        print(f"  Total tests: {total}")
        print(f"  Facts preserved: {correct}/{total} ({correct / total:.0%})")
        print(
            f"  Max experts ablated (fact survived): "
            f"{max_experts_survived}/32"
        )
        print(
            f"  Max layers ablated (fact survived): "
            f"{max_layers_survived}/{len(self.all_moe_layers)}"
        )

        broken = [r for r in all_results if not r.correct]
        if broken:
            print(f"\n  Facts LOST ({len(broken)} cases):")
            for r in broken:
                print(
                    f"    {r.prompt[:35]:35} | {r.condition_name} | "
                    f"got: {r.text[:40]}"
                )
        else:
            print(
                "\n  NO FACTS WERE LOST UNDER ANY CONDITION."
            )
            print("  The residual stream carries all factual knowledge.")
            print(
                "  MoE expert routing is entirely dispensable "
                "for fact retrieval."
            )

        print("=" * 70)


async def main():
    experiment = CrossLayerAblation()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
