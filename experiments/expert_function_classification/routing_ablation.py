#!/usr/bin/env python3
"""Routing Ablation Experiment.

Tests whether learned routing is valuable by comparing:
  A. Normal (learned) routing - baseline
  B. Random routing - random 4 of 32 each token, each layer
  C. Fixed routing - always experts [0, 8, 16, 24] (1 per position class)
  D. Fixed routing - always experts [0, 1, 2, 3] (arbitrary first 4)
  E. Inverse frequency - always use the LEAST popular experts

If random ≈ learned: routing is overhead, experts are interchangeable.
If fixed ≈ learned: position coding doesn't matter.

Run: python experiments/expert_function_classification/routing_ablation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

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

# Open-ended prompts to test coherence
COHERENCE_PROMPTS = [
    "Once upon a time there was a",
    "The process of photosynthesis involves",
    "To make a peanut butter sandwich, first",
    "The main difference between Python and Java is",
]

MAX_TOKENS = 40


@dataclass
class RoutingResult:
    """Result of one generation under a routing condition."""

    condition: str
    prompt: str
    expected: str | None
    text: str
    fact_preserved: bool | None  # None for coherence prompts
    repetition_ratio: float
    is_degenerate: bool


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def is_degenerate(text: str) -> bool:
    """Check if output is degenerate (empty, single char repeated, etc.)."""
    stripped = text.strip()
    if len(stripped) < 3:
        return True
    # All same character
    if len(set(stripped.replace(" ", ""))) <= 2:
        return True
    return False


class RoutingAblation:
    """Test whether learned routing adds value."""

    def __init__(self):
        self.router = None
        self.model = None
        self.tokenizer = None
        self.results: list[RoutingResult] = []
        self._original_router_call = None
        self._router_class = None
        self._rng = np.random.RandomState(42)

    async def setup(self):
        """Load model via ExpertRouter."""
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model via ExpertRouter...")
        self.router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = self.router._model
        self.tokenizer = self.router._tokenizer

        # Get router class for monkey-patching
        sample_layer = self.model.model.layers[0]
        self._router_class = type(sample_layer.mlp.router)
        self._original_router_call = self._router_class.__call__

        logger.info("  Model loaded. Router class identified.")

    def _generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        """Generate text token-by-token (greedy)."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated: list[int] = []
        cache = None

        for _ in range(max_tokens):
            output = self.model(input_ids, cache=cache)
            if hasattr(output, "logits"):
                logits = output.logits
                cache = getattr(output, "cache", None)
            elif isinstance(output, tuple):
                logits, cache = output
            else:
                logits = output
                cache = None

            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

            input_ids = mx.array([[next_token]])

        return self.tokenizer.decode(generated).strip()

    def _generate_with_routing(
        self, prompt: str, condition: str, max_tokens: int = MAX_TOKENS
    ) -> str:
        """Generate with a specific routing condition."""
        if condition == "normal":
            return self._generate(prompt, max_tokens)

        experiment = self
        original_call = self._original_router_call

        def patched_router(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            """Patched router that replaces learned routing."""
            # Handle both 2D and 3D inputs
            if x.ndim == 3:
                batch_size, seq_len, hidden_size = x.shape
                x_flat = x.reshape(-1, hidden_size)
            else:
                x_flat = x

            num_tokens = x_flat.shape[0]
            k = router_self.num_experts_per_tok  # 4
            num_experts = 32

            if condition == "random":
                # Random 4 experts per token
                indices = np.array([
                    experiment._rng.choice(num_experts, size=k, replace=False)
                    for _ in range(num_tokens)
                ])
                indices_mx = mx.array(indices.astype(np.int32))
                # Equal weights
                weights_mx = mx.ones((num_tokens, k)) / k
                return weights_mx, indices_mx

            elif condition == "fixed_diverse":
                # Fixed: 1 per position class [0(end), 8(start), 13(start), 5(early_mid)]
                # Actually use position-diverse set from Part 12
                fixed = [0, 5, 8, 22]  # end, early_mid, start, late_mid
                indices = np.tile(np.array(fixed, dtype=np.int32), (num_tokens, 1))
                indices_mx = mx.array(indices)
                weights_mx = mx.ones((num_tokens, k)) / k
                return weights_mx, indices_mx

            elif condition == "fixed_arbitrary":
                # Fixed: arbitrary [0, 1, 2, 3]
                fixed = [0, 1, 2, 3]
                indices = np.tile(np.array(fixed, dtype=np.int32), (num_tokens, 1))
                indices_mx = mx.array(indices)
                weights_mx = mx.ones((num_tokens, k)) / k
                return weights_mx, indices_mx

            elif condition == "fixed_popular":
                # Fixed: most popular experts (E4, E27, E11, E31 at L16)
                # These are the highest-activation experts from Part 12
                fixed = [4, 27, 11, 31]
                indices = np.tile(np.array(fixed, dtype=np.int32), (num_tokens, 1))
                indices_mx = mx.array(indices)
                weights_mx = mx.ones((num_tokens, k)) / k
                return weights_mx, indices_mx

            elif condition == "fixed_cold":
                # Fixed: least popular active experts
                # E7(1 activation), E10(1), E25(1), E26(1) from Part 12
                fixed = [7, 10, 25, 26]
                indices = np.tile(np.array(fixed, dtype=np.int32), (num_tokens, 1))
                indices_mx = mx.array(indices)
                weights_mx = mx.ones((num_tokens, k)) / k
                return weights_mx, indices_mx

            elif condition == "random_weighted":
                # Random experts but with learned weights
                # Use learned logits for weight magnitudes, random selection
                logits = x_flat @ router_self.weight.T
                if hasattr(router_self, "bias") and router_self.bias is not None:
                    logits = logits + router_self.bias

                # Random indices
                indices = np.array([
                    experiment._rng.choice(num_experts, size=k, replace=False)
                    for _ in range(num_tokens)
                ])
                indices_mx = mx.array(indices.astype(np.int32))

                # Use softmax of random subset logits as weights
                selected_logits = mx.take_along_axis(logits, indices_mx, axis=-1)
                weights_mx = mx.softmax(selected_logits, axis=-1)
                return weights_mx, indices_mx

            else:
                return original_call(router_self, x)

        try:
            self._router_class.__call__ = patched_router
            result = self._generate(prompt, max_tokens)
        finally:
            self._router_class.__call__ = self._original_router_call

        return result

    @staticmethod
    def _check_fact(text: str, expected: str) -> bool:
        return expected.lower() in text.lower()

    async def run_condition(self, condition: str):
        """Run all prompts under one routing condition."""
        logger.info(f"\n  Condition: {condition}")
        loop = asyncio.get_event_loop()

        # Reset RNG for reproducibility
        self._rng = np.random.RandomState(42)

        # Facts
        facts_preserved = 0
        total_rep = 0.0
        degenerate_count = 0

        for fact in FACTS:
            text = await loop.run_in_executor(
                None, self._generate_with_routing, fact["prompt"], condition
            )
            mx.eval(mx.zeros(1))

            preserved = self._check_fact(text, fact["expected"])
            rep = compute_repetition_ratio(text)
            degen = is_degenerate(text)

            if preserved:
                facts_preserved += 1
            total_rep += rep
            if degen:
                degenerate_count += 1

            self.results.append(
                RoutingResult(
                    condition=condition,
                    prompt=fact["prompt"],
                    expected=fact["expected"],
                    text=text,
                    fact_preserved=preserved,
                    repetition_ratio=rep,
                    is_degenerate=degen,
                )
            )

        fact_pct = facts_preserved / len(FACTS) * 100
        avg_rep = total_rep / len(FACTS)
        logger.info(
            f"    Facts: {facts_preserved}/{len(FACTS)} ({fact_pct:.0f}%), "
            f"avg repetition: {avg_rep:.3f}, degenerate: {degenerate_count}"
        )

        # Coherence prompts
        for prompt in COHERENCE_PROMPTS:
            text = await loop.run_in_executor(
                None, self._generate_with_routing, prompt, condition
            )
            mx.eval(mx.zeros(1))

            rep = compute_repetition_ratio(text)
            degen = is_degenerate(text)

            self.results.append(
                RoutingResult(
                    condition=condition,
                    prompt=prompt,
                    expected=None,
                    text=text,
                    fact_preserved=None,
                    repetition_ratio=rep,
                    is_degenerate=degen,
                )
            )

            logger.info(f"    Coherence: '{prompt[:40]}...' -> rep={rep:.3f}, degen={degen}")

    def _print_summary(self):
        """Print summary table."""
        print("\n" + "=" * 70)
        print("ROUTING ABLATION RESULTS")
        print("=" * 70)

        conditions = []
        seen = set()
        for r in self.results:
            if r.condition not in seen:
                conditions.append(r.condition)
                seen.add(r.condition)

        print(
            f"\n{'Condition':>20} | {'Facts':>6} | {'Fact%':>5} | "
            f"{'AvgRep':>6} | {'Degen':>5} | {'Coherent':>8}"
        )
        print("-" * 70)

        for cond in conditions:
            cond_results = [r for r in self.results if r.condition == cond]
            fact_results = [r for r in cond_results if r.expected is not None]
            coherence_results = [r for r in cond_results if r.expected is None]

            facts_ok = sum(1 for r in fact_results if r.fact_preserved)
            fact_pct = facts_ok / len(fact_results) * 100 if fact_results else 0
            avg_rep = (
                sum(r.repetition_ratio for r in cond_results) / len(cond_results)
                if cond_results
                else 0
            )
            n_degen = sum(1 for r in cond_results if r.is_degenerate)
            n_coherent = sum(
                1 for r in coherence_results if not r.is_degenerate
            )

            print(
                f"{cond:>20} | {facts_ok:>2}/{len(fact_results):<3} | "
                f"{fact_pct:>4.0f}% | {avg_rep:>6.3f} | {n_degen:>5} | "
                f"{n_coherent}/{len(coherence_results)}"
            )

        # Sample outputs for comparison
        print("\n--- Sample Outputs ---")
        sample_prompt = "The capital of France is"
        for cond in conditions:
            for r in self.results:
                if r.condition == cond and r.prompt == sample_prompt:
                    text_short = r.text[:80].replace("\n", " ")
                    print(f"  {cond:>20}: {text_short}")

        sample_prompt2 = "Once upon a time there was a"
        print()
        for cond in conditions:
            for r in self.results:
                if r.condition == cond and r.prompt == sample_prompt2:
                    text_short = r.text[:80].replace("\n", " ")
                    print(f"  {cond:>20}: {text_short}")

        # Interpretation
        print("\n--- INTERPRETATION ---")
        normal_facts = sum(
            1
            for r in self.results
            if r.condition == "normal" and r.expected and r.fact_preserved
        )
        random_facts = sum(
            1
            for r in self.results
            if r.condition == "random" and r.expected and r.fact_preserved
        )

        if random_facts >= normal_facts - 1:
            print(
                f"Random routing ({random_facts}/8) ≈ learned ({normal_facts}/8)"
            )
            print("-> Learned routing is NOT adding significant value for facts")
            print("-> Experts are functionally interchangeable for fact recall")
        else:
            print(
                f"Random routing ({random_facts}/8) < learned ({normal_facts}/8)"
            )
            print("-> Learned routing IS valuable for fact recall")
            print("-> Expert selection matters")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"routing_ablation_{timestamp}.json"

        output = {
            "metadata": {
                "experiment": "routing_ablation",
                "model": "openai/gpt-oss-20b",
                "timestamp": timestamp,
                "max_tokens": MAX_TOKENS,
                "conditions": [
                    "normal",
                    "random",
                    "fixed_diverse",
                    "fixed_arbitrary",
                    "fixed_popular",
                    "fixed_cold",
                ],
            },
            "results": [
                {
                    "condition": r.condition,
                    "prompt": r.prompt,
                    "expected": r.expected,
                    "text": r.text,
                    "fact_preserved": r.fact_preserved,
                    "repetition_ratio": r.repetition_ratio,
                    "is_degenerate": r.is_degenerate,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")
        return output_path

    async def run(self):
        """Run the full experiment."""
        await self.setup()

        conditions = [
            "normal",
            "random",
            "fixed_diverse",
            "fixed_arbitrary",
            "fixed_popular",
            "fixed_cold",
        ]

        logger.info("=" * 60)
        logger.info("ROUTING ABLATION EXPERIMENT")
        logger.info(f"  Conditions: {conditions}")
        logger.info(f"  Facts: {len(FACTS)}, Coherence: {len(COHERENCE_PROMPTS)}")
        logger.info(
            f"  Total passes: {len(conditions) * (len(FACTS) + len(COHERENCE_PROMPTS))}"
        )
        logger.info("=" * 60)

        for condition in conditions:
            await self.run_condition(condition)

        self._print_summary()
        self._save_results()


async def main():
    experiment = RoutingAblation()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
