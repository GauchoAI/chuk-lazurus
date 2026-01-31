#!/usr/bin/env python3
"""Routing Resilience Experiment.

Part 16 showed one prompt survived alternating routing almost perfectly:
  Photosynthesis: 0.04 repetition (vs 0.52 average, 0.06 normal)

This experiment investigates WHY. Hypotheses:
  H1: Technical/structured topics are more resilient than creative/open ones
  H2: Prompt length matters (longer = more context = more resilient)
  H3: Prompts that strongly constrain the output space survive better
  H4: Domain-specific vocabulary anchors generation against drift

Method: Test ~30 prompts across categories under alternating routing.
  Categories: technical, factual, creative, conversational, constrained, code

Run: python experiments/expert_function_classification/routing_resilience.py
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

# Prompts organized by hypothesized resilience category
PROMPTS: dict[str, list[str]] = {
    "technical": [
        "The process of photosynthesis involves",
        "In organic chemistry, a nucleophilic substitution reaction",
        "The TCP/IP protocol stack consists of",
        "Mitochondria are often called the powerhouse of the cell because",
        "The Fourier transform converts a signal from",
    ],
    "factual_constrained": [
        "The chemical symbol for gold is",
        "The boiling point of water at sea level is",
        "The speed of light in a vacuum is approximately",
        "DNA stands for",
        "The square root of 144 is",
    ],
    "factual_open": [
        "The capital of France is",
        "The CEO of Microsoft is",
        "The author of Romeo and Juliet is",
        "The capital of Australia is",
        "The largest planet in our solar system is",
    ],
    "creative": [
        "Once upon a time there was a",
        "The dragon soared above the mountains, its wings",
        "She opened the letter and gasped because",
        "In a world where gravity worked backwards,",
        "The old lighthouse keeper had one secret:",
    ],
    "conversational": [
        "I think the most important thing in life is",
        "If I could travel anywhere in the world,",
        "The best way to learn programming is",
        "Cats and dogs are different because",
        "My favorite thing about summer is",
    ],
    "code_like": [
        "def fibonacci(n):",
        "SELECT * FROM users WHERE",
        "import numpy as np; x = np.array([1, 2, 3]);",
        "class Node:\n    def __init__(self, value):",
        "for i in range(10):",
    ],
    "structured_list": [
        "The three branches of the US government are",
        "The steps to make a peanut butter sandwich are: 1.",
        "The primary colors are red,",
        "The planets in order from the sun are Mercury,",
        "The five senses are sight,",
    ],
}

MAX_TOKENS = 60  # Longer to measure sustained fluency


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def compute_vocab_diversity(text: str) -> float:
    """Unique words / total words. Higher = more diverse."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


class RoutingResilience:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results: list[dict] = []
        self._original_block_call = None
        self._block_class = None

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model...")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        sample_block = self.model.model.layers[0]
        self._block_class = type(sample_block)
        self._original_block_call = self._block_class.__call__

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

    def _generate_alternating(self, prompt: str) -> str:
        """Generate with alternating routing (MoE at even layers, skip at odd)."""
        experiment = self
        original_call = self._original_block_call
        even_layers = set(range(0, 24, 2))

        def patched_block(
            block_self: Any,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if layer is block_self:
                    layer_idx = i
                    break

            if layer_idx in even_layers:
                return original_call(block_self, x, mask=mask, cache=cache)

            # Skip MoE: run attention only
            residual = x
            x = block_self.input_layernorm(x)
            x, new_cache = block_self.self_attn(x, mask=mask, cache=cache)
            x = residual + x
            # MoE skipped
            return x, new_cache

        try:
            self._block_class.__call__ = patched_block
            result = self._generate(prompt)
        finally:
            self._block_class.__call__ = self._original_block_call

        return result

    def _generate_alternating_fixed(self, prompt: str) -> str:
        """Generate with alternating fixed routing (Part 16 style for comparison)."""
        experiment = self
        original_block_call = self._original_block_call

        # We need to patch at the router level for fixed routing
        sample_layer = self.model.model.layers[0]
        router_class = type(sample_layer.mlp.router)
        original_router_call = router_class.__call__

        even_layers = set(range(0, 24, 2))
        fixed_experts = [0, 8, 16, 24]

        def patched_router(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            if layer_idx in even_layers:
                return original_router_call(router_self, x)

            # Fixed routing
            if x.ndim == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x

            num_tokens = x_flat.shape[0]
            k = router_self.num_experts_per_tok
            fixed = fixed_experts[:k]
            indices = np.tile(np.array(fixed, dtype=np.int32), (num_tokens, 1))
            indices_mx = mx.array(indices)
            weights_mx = mx.ones((num_tokens, k)) / k
            return weights_mx, indices_mx

        try:
            router_class.__call__ = patched_router
            result = self._generate(prompt)
        finally:
            router_class.__call__ = original_router_call

        return result

    async def run_prompt(self, category: str, prompt: str):
        loop = asyncio.get_event_loop()

        # Normal baseline
        normal_text = await loop.run_in_executor(None, self._generate, prompt)
        mx.eval(mx.zeros(1))

        # Alternating skip (Part 17 style)
        skip_text = await loop.run_in_executor(None, self._generate_alternating, prompt)
        mx.eval(mx.zeros(1))

        # Alternating fixed routing (Part 16 style)
        fixed_text = await loop.run_in_executor(None, self._generate_alternating_fixed, prompt)
        mx.eval(mx.zeros(1))

        prompt_tokens = len(self.tokenizer.encode(prompt))

        for condition, text in [("normal", normal_text), ("skip", skip_text), ("fixed", fixed_text)]:
            rep = compute_repetition_ratio(text)
            vocab_div = compute_vocab_diversity(text)
            word_count = len(text.split())

            self.results.append({
                "category": category,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "condition": condition,
                "text": text,
                "repetition_ratio": rep,
                "vocab_diversity": vocab_div,
                "word_count": word_count,
            })

        # Log summary
        normal_rep = compute_repetition_ratio(normal_text)
        skip_rep = compute_repetition_ratio(skip_text)
        fixed_rep = compute_repetition_ratio(fixed_text)
        logger.info(
            f"    {category:>20} | N:{normal_rep:.2f} S:{skip_rep:.2f} F:{fixed_rep:.2f} | "
            f"{prompt[:50]}"
        )

    def _print_summary(self):
        print("\n" + "=" * 90)
        print("ROUTING RESILIENCE RESULTS")
        print("=" * 90)

        # Group by category
        by_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in self.results:
            by_cat[r["category"]][r["condition"]].append(r)

        print(f"\n{'Category':>22} | {'Normal':>8} | {'Skip':>8} | {'Fixed':>8} | {'Skip/Norm':>9} | {'Fixed/Norm':>10}")
        print("-" * 85)

        category_scores = []

        for category in PROMPTS:
            cat_data = by_cat.get(category, {})

            normal_reps = [r["repetition_ratio"] for r in cat_data.get("normal", [])]
            skip_reps = [r["repetition_ratio"] for r in cat_data.get("skip", [])]
            fixed_reps = [r["repetition_ratio"] for r in cat_data.get("fixed", [])]

            if not normal_reps:
                continue

            avg_normal = sum(normal_reps) / len(normal_reps)
            avg_skip = sum(skip_reps) / len(skip_reps)
            avg_fixed = sum(fixed_reps) / len(fixed_reps)

            skip_ratio = avg_skip / max(avg_normal, 0.001)
            fixed_ratio = avg_fixed / max(avg_normal, 0.001)

            print(
                f"{category:>22} | {avg_normal:>8.3f} | {avg_skip:>8.3f} | {avg_fixed:>8.3f} | "
                f"{skip_ratio:>9.1f}x | {fixed_ratio:>10.1f}x"
            )

            category_scores.append((category, avg_normal, avg_skip, avg_fixed))

        # Rank categories by resilience (lowest skip_rep = most resilient)
        print("\n--- RESILIENCE RANKING (by skip repetition, lower = more resilient) ---")
        category_scores.sort(key=lambda x: x[2])
        for rank, (cat, n_rep, s_rep, f_rep) in enumerate(category_scores, 1):
            bar_len = int(s_rep * 40)
            bar = "#" * bar_len + "." * (40 - bar_len)
            print(f"  {rank}. {cat:>22}: {bar} {s_rep:.3f}")

        # Per-prompt detail for best and worst
        print("\n--- MOST RESILIENT PROMPTS (skip rep < 0.2) ---")
        skip_results = [r for r in self.results if r["condition"] == "skip"]
        skip_results.sort(key=lambda r: r["repetition_ratio"])
        for r in skip_results[:10]:
            text_short = r["text"][:60].replace("\n", " ")
            print(f"  rep={r['repetition_ratio']:.3f} [{r['category']}] \"{r['prompt'][:40]}\"")
            print(f"         → {text_short}")

        print("\n--- LEAST RESILIENT PROMPTS (skip rep > 0.7) ---")
        for r in skip_results[-5:]:
            text_short = r["text"][:60].replace("\n", " ")
            print(f"  rep={r['repetition_ratio']:.3f} [{r['category']}] \"{r['prompt'][:40]}\"")
            print(f"         → {text_short}")

        # Vocab diversity comparison
        print("\n--- VOCAB DIVERSITY (normal vs skip) ---")
        for category in PROMPTS:
            cat_data = by_cat.get(category, {})
            normal_div = [r["vocab_diversity"] for r in cat_data.get("normal", [])]
            skip_div = [r["vocab_diversity"] for r in cat_data.get("skip", [])]
            if normal_div and skip_div:
                avg_n = sum(normal_div) / len(normal_div)
                avg_s = sum(skip_div) / len(skip_div)
                print(f"  {category:>22}: normal={avg_n:.3f}, skip={avg_s:.3f}, ratio={avg_s/max(avg_n,0.001):.2f}")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"routing_resilience_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "routing_resilience",
                        "model": "openai/gpt-oss-20b",
                        "timestamp": timestamp,
                        "max_tokens": MAX_TOKENS,
                        "conditions": ["normal", "skip (alternating)", "fixed (alternating)"],
                        "categories": list(PROMPTS.keys()),
                        "n_prompts": sum(len(v) for v in PROMPTS.values()),
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

        total = sum(len(v) for v in PROMPTS.values())
        logger.info("=" * 60)
        logger.info("ROUTING RESILIENCE EXPERIMENT")
        logger.info(f"  Categories: {len(PROMPTS)}")
        logger.info(f"  Prompts: {total}")
        logger.info(f"  Conditions: normal, skip, fixed (3 x {total} = {3 * total} passes)")
        logger.info("=" * 60)

        for category, prompts in PROMPTS.items():
            logger.info(f"\n  Category: {category}")
            for prompt in prompts:
                await self.run_prompt(category, prompt)

        self._print_summary()
        self._save_results()


async def main():
    experiment = RoutingResilience()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
