#!/usr/bin/env python3
"""Layer Parity Experiment.

Part 16 tested alternating routing with learned at EVEN layers [0,2,4,...,22].
But we never tested the inverse: learned at ODD layers [1,3,5,...,23].

Is it:
  A. Even layers are special (they do something critical)?
  B. Just needs spacing (any alternating pattern works)?
  C. L0 must be learned (first layer is special)?

Conditions:
  A. normal        - all 24 learned (baseline)
  B. even_learned  - learned at [0,2,4,...,22], fixed at odds (Part 16 original)
  C. odd_learned   - learned at [1,3,5,...,23], fixed at evens (inverse)
  D. L0_then_odd   - learned at [0,1,3,5,...,23] (13 layers: L0 + all odds)
  E. skip_L0       - learned at [2,4,6,...,22] (11 layers: evens minus L0)
  F. L0_only_extra - learned at [0,3,5,7,...,23] (L0 + odds, 13 layers)

Also run each with memory bank to test MB rescue across parity.

Run: python experiments/expert_function_classification/layer_parity.py
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

# Routing conditions: name -> list of layer indices that keep learned routing
CONDITIONS: dict[str, list[int]] = {
    "normal": list(range(24)),
    "even_learned": list(range(0, 24, 2)),               # [0,2,4,...,22] - 12 layers
    "odd_learned": list(range(1, 24, 2)),                 # [1,3,5,...,23] - 12 layers
    "L0_then_odd": [0] + list(range(1, 24, 2)),           # [0,1,3,5,...,23] - 13 layers
    "skip_L0": list(range(2, 24, 2)),                     # [2,4,6,...,22] - 11 layers (evens minus L0)
    "L0_only_extra": [0] + list(range(3, 24, 2)),         # [0,3,5,7,...,23] - L0 + odds starting at 3 - 12 layers
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


class LayerParity:
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

        # Test bare prompts (no memory bank)
        facts_ok = 0
        total_rep = 0.0

        for fact in FACTS:
            if name == "normal":
                text = await loop.run_in_executor(
                    None, self._generate, fact["bare"]
                )
            else:
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
                "condition": name,
                "use_mb": False,
                "n_learned": n_learned,
                "learned_layers": learned_layers,
                "prompt": fact["bare"],
                "expected": fact["expected"],
                "text": text,
                "fact_preserved": preserved,
                "repetition_ratio": rep,
            })

        avg_rep = total_rep / len(FACTS)
        logger.info(f"    Bare: {facts_ok}/8 facts, avg_rep={avg_rep:.3f}")

        # Test with memory bank
        mb_facts_ok = 0
        mb_total_rep = 0.0

        for fact in FACTS:
            mb_prompt = build_memory_bank_prompt(fact["prompt"], all_mb_entries)
            if name == "normal":
                text = await loop.run_in_executor(
                    None, self._generate, mb_prompt
                )
            else:
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
                "condition": name,
                "use_mb": True,
                "n_learned": n_learned,
                "learned_layers": learned_layers,
                "prompt": fact["prompt"],
                "expected": fact["expected"],
                "text": text,
                "fact_preserved": preserved,
                "repetition_ratio": rep,
            })

        mb_avg_rep = mb_total_rep / len(FACTS)
        logger.info(f"    MB:   {mb_facts_ok}/8 facts, avg_rep={mb_avg_rep:.3f}")

        # Coherence (bare only)
        for prompt in COHERENCE_PROMPTS:
            if name == "normal":
                text = await loop.run_in_executor(None, self._generate, prompt)
            else:
                text = await loop.run_in_executor(
                    None, self._generate_with_partial_routing,
                    prompt, learned_set,
                )
            mx.eval(mx.zeros(1))
            rep = compute_repetition_ratio(text)

            self.results.append({
                "condition": name,
                "use_mb": False,
                "n_learned": n_learned,
                "learned_layers": learned_layers,
                "prompt": prompt,
                "expected": None,
                "text": text,
                "fact_preserved": None,
                "repetition_ratio": rep,
            })

    def _print_summary(self):
        print("\n" + "=" * 90)
        print("LAYER PARITY RESULTS")
        print("=" * 90)

        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        print(f"\n{'Condition':>16} | {'Layers':>6} | {'Bare':>6} | {'MB':>6} | {'BareRep':>7} | {'MBRep':>7} | Learned layers")
        print("-" * 100)

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
            layers_str = str(CONDITIONS[name][:6]) + ("..." if len(CONDITIONS[name]) > 6 else "")

            print(
                f"{name:>16} | {n_learned:>2}/24 | "
                f"{bare_ok}/8 | {mb_ok}/8 | "
                f"{bare_rep:>7.3f} | {mb_rep:>7.3f} | {layers_str}"
            )

        # Key comparison
        print("\n--- KEY COMPARISON: Even vs Odd ---")
        for name in ["even_learned", "odd_learned"]:
            bare_facts = [r for r in by_cond.get(name, []) if not r["use_mb"] and r["expected"] is not None]
            mb_facts = [r for r in by_cond.get(name, []) if r["use_mb"] and r["expected"] is not None]
            bare_ok = sum(1 for r in bare_facts if r["fact_preserved"])
            mb_ok = sum(1 for r in mb_facts if r["fact_preserved"])
            bare_rep = sum(r["repetition_ratio"] for r in bare_facts) / len(bare_facts) if bare_facts else 0
            print(f"  {name:>16}: bare={bare_ok}/8 (rep={bare_rep:.3f}), MB={mb_ok}/8")

        even_bare = sum(1 for r in by_cond.get("even_learned", []) if not r["use_mb"] and r["expected"] is not None and r["fact_preserved"])
        odd_bare = sum(1 for r in by_cond.get("odd_learned", []) if not r["use_mb"] and r["expected"] is not None and r["fact_preserved"])

        if abs(even_bare - odd_bare) <= 1:
            print("\n  → Even ≈ Odd: It's about SPACING, not layer identity")
        elif even_bare > odd_bare + 1:
            print(f"\n  → Even ({even_bare}/8) > Odd ({odd_bare}/8): Even layers are special")
        else:
            print(f"\n  → Odd ({odd_bare}/8) > Even ({even_bare}/8): Odd layers are special")

        # L0 test
        print("\n--- L0 IMPORTANCE ---")
        for name in ["even_learned", "skip_L0", "L0_then_odd"]:
            bare_facts = [r for r in by_cond.get(name, []) if not r["use_mb"] and r["expected"] is not None]
            bare_ok = sum(1 for r in bare_facts if r["fact_preserved"])
            n = CONDITIONS[name]
            has_L0 = 0 in n
            print(f"  {name:>16}: {bare_ok}/8 facts, L0={'YES' if has_L0 else 'NO '}, {len(n)} layers")

        # Sample outputs
        print("\n--- Sample: 'The capital of France is' (bare) ---")
        for name in CONDITIONS:
            for r in self.results:
                if r["condition"] == name and "France" in r["prompt"] and not r["use_mb"]:
                    text_short = r["text"][:60].replace("\n", " ")
                    status = "✓" if r["fact_preserved"] else "✗"
                    print(f"  {name:>16} {status}: {text_short}")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"layer_parity_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "layer_parity",
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
        logger.info("LAYER PARITY EXPERIMENT")
        logger.info(f"  Conditions: {len(CONDITIONS)}")
        logger.info(f"  Total passes: ~{n_passes}")
        logger.info("=" * 60)

        for name, learned_layers in CONDITIONS.items():
            await self.run_condition(name, learned_layers)

        self._print_summary()
        self._save_results()


async def main():
    experiment = LayerParity()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
