#!/usr/bin/env python3
"""Knowledge Ablation Experiment.

The single-expert ablation experiment showed that removing any 1 of 4
top-k experts doesn't cause fact loss - the other 3 compensate. This
script tests progressive ablation: for each fact, find which experts
handle it, then remove 1, 2, 3, and all 4. At what point does the
model lose the fact? And can a memory bank recover it?

This directly answers: "Where does knowledge live in a top-k MoE?"

Run: python experiments/expert_function_classification/knowledge_ablation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
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
    {
        "prompt": "The capital of France is",
        "expected": "Paris",
        "memory_bank": "France | capital | Paris",
    },
    {
        "prompt": "The chemical symbol for gold is",
        "expected": "Au",
        "memory_bank": "gold | chemical_symbol | Au",
    },
    {
        "prompt": "The author of Romeo and Juliet is",
        "expected": "Shakespeare",
        "memory_bank": "Romeo and Juliet | author | William Shakespeare",
    },
    {
        "prompt": "The speed of light is approximately",
        "expected": "299",
        "memory_bank": "speed of light | value | 299,792,458 m/s",
    },
    {
        "prompt": "The CEO of Microsoft is",
        "expected": "Nadella",
        "memory_bank": "Microsoft | CEO | Satya Nadella",
    },
    {
        "prompt": "The capital of Japan is",
        "expected": "Tokyo",
        "memory_bank": "Japan | capital | Tokyo",
    },
    {
        "prompt": "The chemical symbol for silver is",
        "expected": "Ag",
        "memory_bank": "silver | chemical_symbol | Ag",
    },
    {
        "prompt": "The capital of Australia is",
        "expected": "Canberra",
        "memory_bank": "Australia | capital | Canberra",
    },
]


@dataclass
class FactAblationResult:
    """Result of progressively ablating experts for one fact."""
    prompt: str
    expected: str
    baseline_text: str
    baseline_correct: bool

    # Which experts handle this fact at each layer
    expert_routing: dict[int, list[int]]  # layer -> [expert indices]

    # Progressive ablation: remove 1, 2, 3, 4 experts
    ablation_results: list[dict[str, Any]]  # [{n_ablated, text, correct, experts_ablated}]

    # At which point does the fact break?
    break_point: int | None  # Number of experts removed when fact is lost (None = never)

    # Recovery: can memory bank fix it after full ablation?
    recovery_text: str
    recovery_correct: bool


@dataclass
class LayerKnowledgeProfile:
    """How knowledge is distributed across experts at one layer."""
    layer_idx: int
    facts_tested: int
    avg_break_point: float  # Average N experts to remove before fact loss
    never_broke: int  # Facts that survived even full top-4 ablation
    broke_at_1: int  # Facts lost after removing just 1 expert
    broke_at_2: int
    broke_at_3: int
    broke_at_4: int
    recovery_rate: float  # Fraction of broken facts recovered by memory bank


# =============================================================================
# Experiment
# =============================================================================


class KnowledgeAblation:
    """Progressive expert ablation to find where knowledge lives."""

    def __init__(self):
        self.router = None
        self.model = None
        self.tokenizer = None

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

    async def _get_fact_experts(
        self, prompt: str, layers: list[int]
    ) -> dict[int, list[int]]:
        """Find which experts are selected for this prompt at each layer.

        Returns the top-4 expert indices per layer for the LAST token
        (the prediction position).
        """
        weights_list = await self.router.capture_router_weights(
            prompt, layers=layers
        )

        routing = {}
        for lw in weights_list:
            # Take the last token position (where the fact is predicted)
            if lw.positions:
                last_pos = lw.positions[-1]
                routing[lw.layer_idx] = list(last_pos.expert_indices)

        return routing

    async def _test_fact_at_layer(
        self, fact: dict, layer_idx: int
    ) -> FactAblationResult:
        """Test progressive ablation for one fact at one layer."""
        prompt = fact["prompt"]
        expected = fact["expected"]

        # Baseline
        baseline = self._generate(prompt)
        baseline_correct = self._has_answer(baseline, expected)

        # Get expert routing for this fact
        routing = await self._get_fact_experts(prompt, [layer_idx])
        experts = routing.get(layer_idx, [])

        logger.info(
            f"    {prompt[:45]:45} | baseline={'correct' if baseline_correct else 'WRONG':7} "
            f"| experts={experts}"
        )

        # Progressive ablation: remove 1, 2, 3, 4 experts
        ablation_results = []
        break_point = None

        for n in range(1, len(experts) + 1):
            ablate_set = experts[:n]

            text, _ = await self.router.generate_with_ablation(
                prompt,
                ablate_set,
                max_tokens=30,
                layers=[layer_idx],
            )
            text = text.strip()
            correct = self._has_answer(text, expected)

            ablation_results.append({
                "n_ablated": n,
                "experts_ablated": ablate_set,
                "text": text[:100],
                "correct": correct,
            })

            if baseline_correct and not correct and break_point is None:
                break_point = n

            status = "correct" if correct else "LOST"
            logger.info(
                f"      ablate {n}/{len(experts)} {ablate_set}: "
                f"{status:7} | {text[:50]}"
            )

        # Recovery test: ablate ALL top-4 experts + inject memory bank
        memory_bank = (
            f"[Memory Bank]\n"
            f"- {fact['memory_bank']}\n"
            f"[End Memory Bank]\n\n"
            f"Using the memory bank above, answer: {prompt}\n"
            f"Answer:"
        )

        recovery_text, _ = await self.router.generate_with_ablation(
            memory_bank,
            experts,  # All top-4 ablated
            max_tokens=30,
            layers=[layer_idx],
        )
        recovery_text = recovery_text.strip()
        recovery_correct = self._has_answer(recovery_text, expected)

        logger.info(
            f"      recovery (all {len(experts)} ablated + bank): "
            f"{'RECOVERED' if recovery_correct else 'FAILED':9} | {recovery_text[:50]}"
        )

        return FactAblationResult(
            prompt=prompt,
            expected=expected,
            baseline_text=baseline[:100],
            baseline_correct=baseline_correct,
            expert_routing={layer_idx: experts},
            ablation_results=ablation_results,
            break_point=break_point,
            recovery_text=recovery_text[:100],
            recovery_correct=recovery_correct,
        )

    async def run(self) -> dict[str, Any]:
        """Run the experiment."""
        await self.setup()

        # Test at multiple layers where specialization occurs
        target_layers = [8, 12, 16, 20]

        all_results: dict[int, list[FactAblationResult]] = {}
        layer_profiles: list[LayerKnowledgeProfile] = []

        for layer_idx in target_layers:
            logger.info(f"\n{'='*70}")
            logger.info(f"LAYER {layer_idx}")
            logger.info(f"{'='*70}")

            layer_results = []
            for fact in FACTS:
                result = await self._test_fact_at_layer(fact, layer_idx)
                layer_results.append(result)

            all_results[layer_idx] = layer_results

            # Compute layer profile
            break_points = [r.break_point for r in layer_results if r.baseline_correct]
            valid = [bp for bp in break_points if bp is not None]
            never_broke = sum(1 for bp in break_points if bp is None)

            broke_at = {1: 0, 2: 0, 3: 0, 4: 0}
            for bp in valid:
                if bp in broke_at:
                    broke_at[bp] += 1

            recoveries = [
                r for r in layer_results
                if r.break_point is not None  # fact did break
            ]
            recovery_rate = (
                sum(1 for r in recoveries if r.recovery_correct) / len(recoveries)
                if recoveries else 0.0
            )

            profile = LayerKnowledgeProfile(
                layer_idx=layer_idx,
                facts_tested=len(layer_results),
                avg_break_point=(
                    sum(valid) / len(valid) if valid else float("inf")
                ),
                never_broke=never_broke,
                broke_at_1=broke_at[1],
                broke_at_2=broke_at[2],
                broke_at_3=broke_at[3],
                broke_at_4=broke_at[4],
                recovery_rate=recovery_rate,
            )
            layer_profiles.append(profile)

        # Save and print
        results = self._build_results(all_results, layer_profiles)
        self._save_results(results)
        self._print_summary(layer_profiles, all_results)

        return results

    def _build_results(
        self,
        all_results: dict[int, list[FactAblationResult]],
        profiles: list[LayerKnowledgeProfile],
    ) -> dict[str, Any]:
        return {
            "metadata": {
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "num_facts": len(FACTS),
                "target_layers": list(all_results.keys()),
            },
            "layer_profiles": [
                {
                    "layer_idx": p.layer_idx,
                    "facts_tested": p.facts_tested,
                    "avg_break_point": p.avg_break_point,
                    "never_broke": p.never_broke,
                    "broke_at_1": p.broke_at_1,
                    "broke_at_2": p.broke_at_2,
                    "broke_at_3": p.broke_at_3,
                    "broke_at_4": p.broke_at_4,
                    "recovery_rate": p.recovery_rate,
                }
                for p in profiles
            ],
            "fact_results": {
                str(layer_idx): [
                    {
                        "prompt": r.prompt,
                        "expected": r.expected,
                        "baseline_text": r.baseline_text,
                        "baseline_correct": r.baseline_correct,
                        "expert_routing": {
                            str(k): v for k, v in r.expert_routing.items()
                        },
                        "ablation_results": r.ablation_results,
                        "break_point": r.break_point,
                        "recovery_text": r.recovery_text,
                        "recovery_correct": r.recovery_correct,
                    }
                    for r in results
                ]
                for layer_idx, results in all_results.items()
            },
        }

    def _save_results(self, results: dict) -> None:
        output_path = (
            Path(__file__).parent / "results"
            / f"knowledge_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")

    def _print_summary(
        self,
        profiles: list[LayerKnowledgeProfile],
        all_results: dict[int, list[FactAblationResult]],
    ) -> None:
        print("\n" + "=" * 70)
        print("KNOWLEDGE ABLATION - SUMMARY")
        print("=" * 70)

        print("\nHow many experts must be removed before a fact is lost?")
        print()
        print("Layer | Broke@1 | Broke@2 | Broke@3 | Broke@4 | Never | Avg | Recovery")
        print("------|---------|---------|---------|---------|-------|-----|----------")
        for p in profiles:
            avg = f"{p.avg_break_point:.1f}" if p.avg_break_point != float("inf") else "inf"
            print(
                f"L{p.layer_idx:<4}| "
                f"{p.broke_at_1:>7} | "
                f"{p.broke_at_2:>7} | "
                f"{p.broke_at_3:>7} | "
                f"{p.broke_at_4:>7} | "
                f"{p.never_broke:>5} | "
                f"{avg:>3} | "
                f"{p.recovery_rate:.0%}"
            )

        # Per-fact breakdown at the most interesting layer
        print()
        print("-" * 70)
        print("PER-FACT DETAIL (Layer 16)")
        print("-" * 70)

        if 16 in all_results:
            for r in all_results[16]:
                bp = r.break_point if r.break_point is not None else "never"
                rec = "yes" if r.recovery_correct else "no"
                experts = r.expert_routing.get(16, [])
                print(
                    f"  {r.prompt[:40]:40} | "
                    f"experts={experts} | "
                    f"breaks@{bp} | "
                    f"recovery={rec}"
                )

        # Key findings
        print()
        print("=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        total_facts = sum(p.facts_tested for p in profiles)
        total_broke = sum(
            p.broke_at_1 + p.broke_at_2 + p.broke_at_3 + p.broke_at_4
            for p in profiles
        )
        total_never = sum(p.never_broke for p in profiles)
        all_recovery = [
            p.recovery_rate for p in profiles
            if p.broke_at_1 + p.broke_at_2 + p.broke_at_3 + p.broke_at_4 > 0
        ]

        print(f"  Facts tested: {total_facts} ({len(FACTS)} facts x {len(profiles)} layers)")
        print(f"  Facts that broke: {total_broke}")
        print(f"  Facts that never broke (full top-4 ablation): {total_never}")
        if all_recovery:
            avg_rec = sum(all_recovery) / len(all_recovery)
            print(f"  Memory bank recovery rate: {avg_rec:.0%}")

        if total_never > total_broke:
            print("\n  FINDING: Knowledge survives even full top-4 ablation at most layers.")
            print("  Knowledge is NOT concentrated in the top-4 experts alone.")
            print("  The residual stream carries facts independently of expert routing.")
        elif total_broke > 0 and all_recovery and sum(all_recovery) / len(all_recovery) > 0.5:
            print("\n  FINDING: Knowledge IS in the experts, and memory bank CAN recover it.")
            print("  This validates the virtual expert architecture for externalization.")

        print("=" * 70)


async def main():
    experiment = KnowledgeAblation()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
