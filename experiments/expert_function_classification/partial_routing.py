#!/usr/bin/env python3
"""Partial Routing Ablation Experiment.

Part 15 showed global routing replacement breaks everything (0/8).
Parts 8-12 showed single-layer ablation is harmless (8/8).

This experiment maps the boundary: how many layers need learned routing?

Conditions:
  A. normal - baseline (learned routing at all 24 layers)
  B. first_half - learned at L0-11, fixed at L12-23
  C. second_half - fixed at L0-11, learned at L12-23
  D. alternating - learned at even layers, fixed at odd
  E. bookends - learned at L0-3 + L20-23, fixed at L4-19
  F. every_6th - learned at [0,6,12,18,23], fixed at rest
  G. middle_only - learned at L8-15, fixed at rest
  H. first_quarter - learned at L0-5, fixed at L6-23
  I. last_quarter - learned at L18-23, fixed at L0-17

"Fixed" routing = always select experts [0, 8, 16, 24] with equal weights.

Run: python experiments/expert_function_classification/partial_routing.py
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

COHERENCE_PROMPTS = [
    "Once upon a time there was a",
    "The process of photosynthesis involves",
]

MAX_TOKENS = 40

# Fixed expert set for "broken" layers
FIXED_EXPERTS = [0, 8, 16, 24]

# Routing conditions: name -> list of layer indices that keep learned routing
CONDITIONS: dict[str, list[int]] = {
    "normal": list(range(24)),
    "first_half": list(range(12)),
    "second_half": list(range(12, 24)),
    "alternating": list(range(0, 24, 2)),
    "bookends": list(range(4)) + list(range(20, 24)),
    "every_6th": [0, 6, 12, 18, 23],
    "middle_only": list(range(8, 16)),
    "first_quarter": list(range(6)),
    "last_quarter": list(range(18, 24)),
    # Finer-grained: how many evenly-spaced layers are enough?
    "every_4th": [0, 4, 8, 12, 16, 20, 23],
    "every_3rd": list(range(0, 24, 3)) + [23],
    "every_2nd": list(range(0, 24, 2)),
}


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


class PartialRouting:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results: list[dict] = []
        self._original_router_call = None
        self._router_class = None

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model...")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        sample_layer = self.model.model.layers[0]
        self._router_class = type(sample_layer.mlp.router)
        self._original_router_call = self._router_class.__call__
        logger.info("  Ready.")

    def _generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
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

    def _generate_with_partial_routing(
        self, prompt: str, learned_layers: set[int]
    ) -> str:
        """Generate with learned routing at specified layers, fixed at rest."""
        experiment = self
        original_call = self._original_router_call

        def patched_router(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            # Find which layer this router belongs to
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            # Learned routing for designated layers
            if layer_idx in learned_layers:
                return original_call(router_self, x)

            # Fixed routing for other layers
            if x.ndim == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x

            num_tokens = x_flat.shape[0]
            k = router_self.num_experts_per_tok

            fixed = FIXED_EXPERTS[:k]
            indices = np.tile(np.array(fixed, dtype=np.int32), (num_tokens, 1))
            indices_mx = mx.array(indices)
            weights_mx = mx.ones((num_tokens, k)) / k
            return weights_mx, indices_mx

        try:
            self._router_class.__call__ = patched_router
            result = self._generate(prompt)
        finally:
            self._router_class.__call__ = self._original_router_call

        return result

    async def run_condition(self, name: str, learned_layers: list[int]):
        n_learned = len(learned_layers)
        n_fixed = 24 - n_learned
        logger.info(
            f"\n  {name}: {n_learned} learned, {n_fixed} fixed"
        )

        loop = asyncio.get_event_loop()
        learned_set = set(learned_layers)

        facts_ok = 0
        total_rep = 0.0

        for fact in FACTS:
            if name == "normal":
                text = await loop.run_in_executor(
                    None, self._generate, fact["prompt"]
                )
            else:
                text = await loop.run_in_executor(
                    None,
                    self._generate_with_partial_routing,
                    fact["prompt"],
                    learned_set,
                )
            mx.eval(mx.zeros(1))

            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)
            if preserved:
                facts_ok += 1
            total_rep += rep

            self.results.append({
                "condition": name,
                "n_learned": n_learned,
                "learned_layers": learned_layers,
                "prompt": fact["prompt"],
                "expected": fact["expected"],
                "text": text,
                "fact_preserved": preserved,
                "repetition_ratio": rep,
            })

        avg_rep = total_rep / len(FACTS)

        # Coherence
        coherence_rep = 0.0
        for prompt in COHERENCE_PROMPTS:
            if name == "normal":
                text = await loop.run_in_executor(None, self._generate, prompt)
            else:
                text = await loop.run_in_executor(
                    None,
                    self._generate_with_partial_routing,
                    prompt,
                    learned_set,
                )
            mx.eval(mx.zeros(1))
            rep = compute_repetition_ratio(text)
            coherence_rep += rep

            self.results.append({
                "condition": name,
                "n_learned": n_learned,
                "learned_layers": learned_layers,
                "prompt": prompt,
                "expected": None,
                "text": text,
                "fact_preserved": None,
                "repetition_ratio": rep,
            })

        avg_coh_rep = coherence_rep / len(COHERENCE_PROMPTS)
        logger.info(
            f"    Facts: {facts_ok}/{len(FACTS)}, "
            f"fact_rep: {avg_rep:.3f}, coherence_rep: {avg_coh_rep:.3f}"
        )

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("PARTIAL ROUTING ABLATION RESULTS")
        print("=" * 70)

        # Group by condition
        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        print(
            f"\n{'Condition':>15} | {'Learned':>7} | {'Fixed':>5} | "
            f"{'Facts':>6} | {'AvgRep':>6}"
        )
        print("-" * 60)

        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue
            fact_results = [r for r in results if r["expected"] is not None]
            facts_ok = sum(1 for r in fact_results if r["fact_preserved"])
            n_learned = results[0]["n_learned"]
            avg_rep = (
                sum(r["repetition_ratio"] for r in results) / len(results)
                if results
                else 0
            )
            pct = facts_ok / len(fact_results) * 100 if fact_results else 0

            print(
                f"{name:>15} | {n_learned:>4}/24 | "
                f"{24 - n_learned:>2}/24 | "
                f"{facts_ok}/{len(fact_results)} {pct:>3.0f}% | "
                f"{avg_rep:>6.3f}"
            )

        # Coverage curve
        print("\n--- COVERAGE CURVE ---")
        coverage_data = []
        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue
            fact_results = [r for r in results if r["expected"] is not None]
            facts_ok = sum(1 for r in fact_results if r["fact_preserved"])
            n_learned = results[0]["n_learned"]
            coverage_data.append((n_learned, name, facts_ok))

        coverage_data.sort()
        for n_learned, name, facts_ok in coverage_data:
            bar = "#" * facts_ok + "." * (8 - facts_ok)
            print(f"  {n_learned:>2} layers | {bar} {facts_ok}/8 | {name}")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"partial_routing_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "partial_routing",
                        "model": "openai/gpt-oss-20b",
                        "timestamp": timestamp,
                        "fixed_experts": FIXED_EXPERTS,
                        "conditions": {
                            k: {"learned_layers": v, "n_learned": len(v)}
                            for k, v in CONDITIONS.items()
                        },
                    },
                    "results": self.results,
                },
                f,
                indent=2,
            )

        logger.info(f"\nResults saved to {output_path}")
        return output_path

    async def run(self):
        await self.setup()

        n_passes = len(CONDITIONS) * (len(FACTS) + len(COHERENCE_PROMPTS))
        logger.info("=" * 60)
        logger.info("PARTIAL ROUTING ABLATION")
        logger.info(f"  Conditions: {len(CONDITIONS)}")
        logger.info(f"  Total passes: {n_passes}")
        logger.info("=" * 60)

        for name, learned_layers in CONDITIONS.items():
            await self.run_condition(name, learned_layers)

        self._print_summary()
        self._save_results()


async def main():
    experiment = PartialRouting()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
