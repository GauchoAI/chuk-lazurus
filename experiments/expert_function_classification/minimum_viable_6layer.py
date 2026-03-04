#!/usr/bin/env python3
"""Quick 6-layer test to pin the cliff between 5 and 7 layers.

7 layers [0,3,7,11,15,19,23]: 8/8 MB
5 layers [0,5,11,17,23]:      4/8 MB

Test 6 layers to find the exact boundary.

Run: python experiments/expert_function_classification/minimum_viable_6layer.py
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

# 6-layer configs to test: two spacing strategies
CONDITIONS: dict[str, list[int]] = {
    "L0_plus_6_even": [0, 5, 9, 14, 18, 23],     # 6 layers, ~gap 4.6 (evenly spaced)
    "L0_plus_6_tight": [0, 3, 7, 11, 15, 19],     # 6 layers, gap 3-4 (tighter, front-loaded)
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


class SixLayerTest:
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
        logger.info(f"\n  {name}: {n_learned} learned, {n_fixed} fixed")

        loop = asyncio.get_event_loop()
        learned_set = set(learned_layers)
        all_mb_entries = [f["mb_entry"] for f in FACTS]

        # Bare
        facts_ok = 0
        total_rep = 0.0
        for fact in FACTS:
            text = await loop.run_in_executor(
                None, self._generate_with_partial_routing,
                fact["bare"], learned_set,
            )
            mx.eval(mx.zeros(1))
            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)
            if preserved:
                facts_ok += 1
            total_rep += rep
            self.results.append({
                "condition": name, "use_mb": False,
                "n_learned": n_learned, "learned_layers": learned_layers,
                "prompt": fact["bare"], "expected": fact["expected"],
                "text": text, "fact_preserved": preserved,
                "repetition_ratio": rep,
            })

        avg_rep = total_rep / len(FACTS)
        logger.info(f"    Bare: {facts_ok}/8 facts, avg_rep={avg_rep:.3f}")

        # MB
        mb_facts_ok = 0
        mb_total_rep = 0.0
        for fact in FACTS:
            mb_prompt = build_memory_bank_prompt(fact["prompt"], all_mb_entries)
            text = await loop.run_in_executor(
                None, self._generate_with_partial_routing,
                mb_prompt, learned_set,
            )
            mx.eval(mx.zeros(1))
            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)
            if preserved:
                mb_facts_ok += 1
            mb_total_rep += rep
            self.results.append({
                "condition": name, "use_mb": True,
                "n_learned": n_learned, "learned_layers": learned_layers,
                "prompt": fact["prompt"], "expected": fact["expected"],
                "text": text, "fact_preserved": preserved,
                "repetition_ratio": rep,
            })

        mb_avg_rep = mb_total_rep / len(FACTS)
        logger.info(f"    MB:   {mb_facts_ok}/8 facts, avg_rep={mb_avg_rep:.3f}")

        # Coherence
        for prompt in COHERENCE_PROMPTS:
            text = await loop.run_in_executor(
                None, self._generate_with_partial_routing,
                prompt, learned_set,
            )
            mx.eval(mx.zeros(1))
            rep = compute_repetition_ratio(text)
            self.results.append({
                "condition": name, "use_mb": False,
                "n_learned": n_learned, "learned_layers": learned_layers,
                "prompt": prompt, "expected": None,
                "text": text, "fact_preserved": None,
                "repetition_ratio": rep,
            })

    def _print_summary(self):
        print("\n" + "=" * 90)
        print("6-LAYER CLIFF TEST RESULTS")
        print("=" * 90)

        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        print(f"\n{'Condition':>18} | {'Learned':>7} | {'Bare':>6} | {'MB':>6} | {'BareRep':>7} | {'MBRep':>7} | Layers")
        print("-" * 105)

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
            n_learned = results[0]["n_learned"]
            layers_str = str(CONDITIONS[name])

            print(
                f"{name:>18} | {n_learned:>3}/24 | "
                f"{bare_ok}/8 | {mb_ok}/8 | "
                f"{bare_rep:>7.3f} | {mb_rep:>7.3f} | {layers_str}"
            )

        # Context from prior experiments
        print("\n--- FULL SCALING CURVE (with prior results) ---")
        print()
        prior = [
            (0,  "none",           "0/8", "0/8"),
            (1,  "L0_only",        "0/8", "0/8"),
            (2,  "L0_endpoints",   "0/8", "0/8"),
            (3,  "L0_plus_mid",    "0/8", "1/8"),
            (5,  "L0_plus_5",      "0/8", "4/8"),
        ]
        print("  Prior results:")
        for n, name, bare, mb in prior:
            print(f"    {n:>2} layers ({name:>18}): bare={bare}, MB={mb}")

        print("  New results:")
        for name in CONDITIONS:
            results = by_cond.get(name, [])
            bare_facts = [r for r in results if not r["use_mb"] and r["expected"] is not None]
            mb_facts = [r for r in results if r["use_mb"] and r["expected"] is not None]
            bare_ok = sum(1 for r in bare_facts if r["fact_preserved"])
            mb_ok = sum(1 for r in mb_facts if r["fact_preserved"])
            n = len(CONDITIONS[name])
            print(f"    {n:>2} layers ({name:>18}): bare={bare_ok}/8, MB={mb_ok}/8  *** NEW ***")

        print("  Prior results:")
        print(f"    {'7':>2} layers ({'L0_plus_7':>18}): bare=1/8, MB=8/8")
        print(f"    {'12':>2} layers ({'L0_only_extra':>18}): bare=6/8, MB=8/8  (Part 20)")
        print(f"    {'24':>2} layers ({'normal':>18}): bare=8/8, MB=8/8")

        # Sample outputs
        print("\n--- Sample: 'The capital of France is' ---")
        for name in CONDITIONS:
            for r in self.results:
                if r["condition"] == name and "France" in r["prompt"]:
                    text_short = r["text"][:60].replace("\n", " ")
                    status = "ok" if r["fact_preserved"] else "FAIL"
                    mode = "MB" if r["use_mb"] else "bare"
                    print(f"  {name:>18} ({mode:>4}): [{status:>4}] {text_short}")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"minimum_viable_6layer_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "minimum_viable_6layer",
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
        logger.info("6-LAYER CLIFF TEST")
        logger.info(f"  Conditions: {len(CONDITIONS)}")
        logger.info(f"  Total passes: ~{n_passes}")
        logger.info("=" * 60)

        for name, learned_layers in CONDITIONS.items():
            await self.run_condition(name, learned_layers)

        self._save_results()
        self._print_summary()


async def main():
    experiment = SixLayerTest()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
