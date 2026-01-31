#!/usr/bin/env python3
"""Layer Spacing Experiment.

Part 20 showed L0 is critical and spacing matters. Best 12-layer config
was L0 + spaced odds = 6/8 facts.

This experiment tests: how sparse can we go while keeping L0?

Conditions:
  A. normal          - all 24 learned (baseline)
  B. every_2nd_L0    - L0 + every 2nd: [0,2,4,...,22] (12 layers)
  C. every_3rd_L0    - L0 + every 3rd: [0,3,6,9,12,15,18,21,23] (9 layers)
  D. every_4th_L0    - L0 + every 4th: [0,4,8,12,16,20,23] (7 layers)
  E. every_6th_L0    - L0 + every 6th: [0,6,12,18,23] (5 layers)
  F. every_8th_L0    - L0 + every 8th: [0,8,16,23] (4 layers)
  G. L0_L23_only     - just bookends: [0,23] (2 layers)
  H. L0_only         - just L0: [0] (1 layer)

All tested bare and with memory bank.

Run: python experiments/expert_function_classification/layer_spacing.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
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
    {
        "prompt": "What is the capital of France?",
        "bare": "The capital of France is",
        "expected": "Paris",
        "mb_entry": "France | capital | Paris",
    },
    {
        "prompt": "What is the chemical symbol for gold?",
        "bare": "The chemical symbol for gold is",
        "expected": "Au",
        "mb_entry": "Gold | chemical symbol | Au",
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "bare": "The author of Romeo and Juliet is",
        "expected": "Shakespeare",
        "mb_entry": "Romeo and Juliet | author | William Shakespeare",
    },
    {
        "prompt": "What is the speed of light in m/s?",
        "bare": "The speed of light is approximately",
        "expected": "299",
        "mb_entry": "Speed of light | value | 299,792,458 meters per second",
    },
    {
        "prompt": "Who is the CEO of Microsoft?",
        "bare": "The CEO of Microsoft is",
        "expected": "Nadella",
        "mb_entry": "Microsoft | CEO | Satya Nadella",
    },
    {
        "prompt": "What is the capital of Japan?",
        "bare": "The capital of Japan is",
        "expected": "Tokyo",
        "mb_entry": "Japan | capital | Tokyo",
    },
    {
        "prompt": "What is the chemical symbol for silver?",
        "bare": "The chemical symbol for silver is",
        "expected": "Ag",
        "mb_entry": "Silver | chemical symbol | Ag",
    },
    {
        "prompt": "What is the capital of Australia?",
        "bare": "The capital of Australia is",
        "expected": "Canberra",
        "mb_entry": "Australia | capital | Canberra",
    },
]

COHERENCE_PROMPTS = [
    "Once upon a time there was a",
    "The process of photosynthesis involves",
]

MAX_TOKENS = 40

FIXED_EXPERTS = [0, 8, 16, 24]

CONDITIONS: dict[str, list[int]] = {
    "normal": list(range(24)),
    "every_2nd_L0": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # 12 layers
    "every_3rd_L0": [0, 3, 6, 9, 12, 15, 18, 21, 23],               # 9 layers
    "every_4th_L0": [0, 4, 8, 12, 16, 20, 23],                       # 7 layers
    "every_6th_L0": [0, 6, 12, 18, 23],                               # 5 layers
    "every_8th_L0": [0, 8, 16, 23],                                    # 4 layers
    "L0_L23_only": [0, 23],                                            # 2 layers
    "L0_only": [0],                                                     # 1 layer
}


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def build_memory_bank_prompt(question: str, all_entries: list[str]) -> str:
    mb_block = "\n".join(f"- {entry}" for entry in all_entries)
    return (
        f"[Memory Bank]\n{mb_block}\n[End Memory Bank]\n\n"
        f"Using the memory bank above, answer: {question}\nAnswer:"
    )


class LayerSpacing:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results: list[dict] = []
        self._router_class = None
        self._original_router_call = None

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
        experiment = self
        original_call = self._original_router_call

        def patched_router(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            if layer_idx in learned_layers:
                return original_call(router_self, x)

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
        gap = "N/A"
        if n_learned >= 2:
            gaps = [learned_layers[i+1] - learned_layers[i] for i in range(len(learned_layers)-1)]
            gap = f"avg {sum(gaps)/len(gaps):.1f}"
        logger.info(f"\n  {name}: {n_learned} learned, {n_fixed} fixed (gap: {gap})")

        loop = asyncio.get_event_loop()
        learned_set = set(learned_layers)
        all_mb_entries = [f["mb_entry"] for f in FACTS]

        # Bare prompts
        facts_ok = 0
        total_rep = 0.0

        for fact in FACTS:
            if name == "normal":
                text = await loop.run_in_executor(None, self._generate, fact["bare"])
            else:
                text = await loop.run_in_executor(
                    None, self._generate_with_partial_routing, fact["bare"], learned_set,
                )
            mx.eval(mx.zeros(1))

            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)
            if preserved:
                facts_ok += 1
            total_rep += rep

            self.results.append({
                "condition": name, "use_mb": False, "n_learned": n_learned,
                "learned_layers": learned_layers, "prompt": fact["bare"],
                "expected": fact["expected"], "text": text,
                "fact_preserved": preserved, "repetition_ratio": rep,
            })

        avg_rep = total_rep / len(FACTS)
        logger.info(f"    Bare: {facts_ok}/8 facts, avg_rep={avg_rep:.3f}")

        # Memory bank prompts
        mb_facts_ok = 0
        mb_total_rep = 0.0

        for fact in FACTS:
            mb_prompt = build_memory_bank_prompt(fact["prompt"], all_mb_entries)
            if name == "normal":
                text = await loop.run_in_executor(None, self._generate, mb_prompt)
            else:
                text = await loop.run_in_executor(
                    None, self._generate_with_partial_routing, mb_prompt, learned_set,
                )
            mx.eval(mx.zeros(1))

            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)
            if preserved:
                mb_facts_ok += 1
            mb_total_rep += rep

            self.results.append({
                "condition": name, "use_mb": True, "n_learned": n_learned,
                "learned_layers": learned_layers, "prompt": fact["prompt"],
                "expected": fact["expected"], "text": text,
                "fact_preserved": preserved, "repetition_ratio": rep,
            })

        mb_avg_rep = mb_total_rep / len(FACTS)
        logger.info(f"    MB:   {mb_facts_ok}/8 facts, avg_rep={mb_avg_rep:.3f}")

        # Coherence (bare only)
        for prompt in COHERENCE_PROMPTS:
            if name == "normal":
                text = await loop.run_in_executor(None, self._generate, prompt)
            else:
                text = await loop.run_in_executor(
                    None, self._generate_with_partial_routing, prompt, learned_set,
                )
            mx.eval(mx.zeros(1))
            rep = compute_repetition_ratio(text)

            self.results.append({
                "condition": name, "use_mb": False, "n_learned": n_learned,
                "learned_layers": learned_layers, "prompt": prompt,
                "expected": None, "text": text,
                "fact_preserved": None, "repetition_ratio": rep,
            })

    def _print_summary(self):
        print("\n" + "=" * 100)
        print("LAYER SPACING RESULTS")
        print("=" * 100)

        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        print(f"\n{'Condition':>16} | {'Layers':>6} | {'Gap':>5} | {'Bare':>5} | {'MB':>5} | {'BareRep':>7} | {'MBRep':>7}")
        print("-" * 80)

        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue

            bare_facts = [r for r in results if not r["use_mb"] and r["expected"] is not None]
            mb_facts = [r for r in results if r["use_mb"] and r["expected"] is not None]

            bare_ok = sum(1 for r in bare_facts if r["fact_preserved"])
            mb_ok = sum(1 for r in mb_facts if r["fact_preserved"])

            bare_rep = sum(r["repetition_ratio"] for r in bare_facts) / len(bare_facts) if bare_facts else 0
            mb_rep = sum(r["repetition_ratio"] for r in mb_facts) / len(mb_facts) if mb_facts else 0

            n = len(CONDITIONS[name])
            layers = CONDITIONS[name]
            if n >= 2:
                gaps = [layers[i+1] - layers[i] for i in range(n-1)]
                avg_gap = sum(gaps) / len(gaps)
                gap_str = f"{avg_gap:.1f}"
            else:
                gap_str = "N/A"

            print(
                f"{name:>16} | {n:>2}/24 | {gap_str:>5} | "
                f"{bare_ok}/8 | {mb_ok}/8 | "
                f"{bare_rep:>7.3f} | {mb_rep:>7.3f}"
            )

        # Coverage curve
        print("\n--- COVERAGE CURVE (bare facts, all configs include L0) ---")
        coverage = []
        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue
            bare_facts = [r for r in results if not r["use_mb"] and r["expected"] is not None]
            bare_ok = sum(1 for r in bare_facts if r["fact_preserved"])
            n = len(CONDITIONS[name])
            coverage.append((n, name, bare_ok))

        coverage.sort()
        for n, name, ok in coverage:
            bar = "#" * ok + "." * (8 - ok)
            print(f"  {n:>2} layers | {bar} {ok}/8 | {name}")

        # MB coverage curve
        print("\n--- COVERAGE CURVE (with memory bank) ---")
        mb_coverage = []
        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue
            mb_facts = [r for r in results if r["use_mb"] and r["expected"] is not None]
            mb_ok = sum(1 for r in mb_facts if r["fact_preserved"])
            n = len(CONDITIONS[name])
            mb_coverage.append((n, name, mb_ok))

        mb_coverage.sort()
        for n, name, ok in mb_coverage:
            bar = "#" * ok + "." * (8 - ok)
            print(f"  {n:>2} layers | {bar} {ok}/8 | {name}")

        # Efficiency: facts per learned layer
        print("\n--- EFFICIENCY (facts per learned layer) ---")
        for n, name, ok in sorted(coverage):
            if n > 0:
                eff = ok / n
                print(f"  {name:>16}: {ok}/8 facts / {n} layers = {eff:.2f} facts/layer")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"layer_spacing_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "layer_spacing",
                        "model": "openai/gpt-oss-20b",
                        "timestamp": timestamp,
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

        n_passes = len(CONDITIONS) * (len(FACTS) * 2 + len(COHERENCE_PROMPTS))
        logger.info("=" * 60)
        logger.info("LAYER SPACING EXPERIMENT")
        logger.info(f"  Conditions: {len(CONDITIONS)}")
        logger.info(f"  Total passes: ~{n_passes}")
        logger.info("=" * 60)

        for name, learned_layers in CONDITIONS.items():
            await self.run_condition(name, learned_layers)

        self._print_summary()
        self._save_results()


async def main():
    experiment = LayerSpacing()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
