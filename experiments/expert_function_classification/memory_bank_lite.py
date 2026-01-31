#!/usr/bin/env python3
"""Memory Bank + Lite Model Experiment.

Tests whether external fact injection (memory banks) compensates for
routing degradation, enabling a "lite model + RAG" compression path.

Part 9:  Memory bank → 100% fact override on full model
Part 16: Alternating fixed routing → 5/8 facts (no memory bank)
Part 17: Alternating skip → 1/8 facts (no memory bank)

Key question: Does memory bank + degraded routing recover to ~8/8?
If yes → lite model + RAG is viable for factual workloads.

Conditions:
  A. normal              - Full model, no memory bank (baseline)
  B. normal_mb           - Full model + memory bank
  C. fixed_alt           - Alternating fixed routing, no memory bank
  D. fixed_alt_mb        - Alternating fixed routing + memory bank
  E. skip_alt            - Alternating MoE skip, no memory bank
  F. skip_alt_mb         - Alternating MoE skip + memory bank
  G. fixed_heavy         - Heavy degradation (every_3rd), no memory bank
  H. fixed_heavy_mb      - Heavy degradation (every_3rd) + memory bank

Run: python experiments/expert_function_classification/memory_bank_lite.py
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

# Facts with memory bank entries
FACTS = [
    {
        "prompt": "What is the capital of France?",
        "expected": "Paris",
        "mb_entry": "France | capital | Paris",
    },
    {
        "prompt": "What is the chemical symbol for gold?",
        "expected": "Au",
        "mb_entry": "Gold | chemical symbol | Au",
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "expected": "Shakespeare",
        "mb_entry": "Romeo and Juliet | author | William Shakespeare",
    },
    {
        "prompt": "What is the speed of light in m/s?",
        "expected": "299",
        "mb_entry": "Speed of light | value | 299,792,458 meters per second",
    },
    {
        "prompt": "Who is the CEO of Microsoft?",
        "expected": "Nadella",
        "mb_entry": "Microsoft | CEO | Satya Nadella",
    },
    {
        "prompt": "What is the capital of Japan?",
        "expected": "Tokyo",
        "mb_entry": "Japan | capital | Tokyo",
    },
    {
        "prompt": "What is the chemical symbol for silver?",
        "expected": "Ag",
        "mb_entry": "Silver | chemical symbol | Ag",
    },
    {
        "prompt": "What is the capital of Australia?",
        "expected": "Canberra",
        "mb_entry": "Australia | capital | Canberra",
    },
]

# Also test counterfactuals: does the lite model still obey the memory bank?
COUNTERFACTUALS = [
    {
        "prompt": "What is the capital of France?",
        "expected": "Lyon",
        "mb_entry": "France | capital | Lyon",
    },
    {
        "prompt": "What is the chemical symbol for gold?",
        "expected": "Gd",
        "mb_entry": "Gold | chemical symbol | Gd",
    },
    {
        "prompt": "What is the capital of Japan?",
        "expected": "Osaka",
        "mb_entry": "Japan | capital | Osaka",
    },
]

MAX_TOKENS = 40

# Fixed experts for "broken" routing layers
FIXED_EXPERTS = [0, 8, 16, 24]


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def build_memory_bank_prompt(question: str, mb_entries: list[str]) -> str:
    """Build a prompt with memory bank injection."""
    mb_block = "\n".join(f"- {entry}" for entry in mb_entries)
    return (
        f"[Memory Bank]\n"
        f"{mb_block}\n"
        f"[End Memory Bank]\n\n"
        f"Using the memory bank above, answer: {question}\n"
        f"Answer:"
    )


class MemoryBankLite:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results: list[dict] = []
        self._original_block_call = None
        self._block_class = None
        self._router_class = None
        self._original_router_call = None

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model...")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        sample_block = self.model.model.layers[0]
        self._block_class = type(sample_block)
        self._original_block_call = self._block_class.__call__

        self._router_class = type(sample_block.mlp.router)
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

    def _generate_fixed_alternating(self, prompt: str, learned_layers: set[int]) -> str:
        """Generate with fixed routing at non-learned layers."""
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

    def _generate_skip_alternating(self, prompt: str, moe_layers: set[int]) -> str:
        """Generate with MoE skipped at non-learned layers."""
        experiment = self
        original_call = self._original_block_call

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

            if layer_idx in moe_layers:
                return original_call(block_self, x, mask=mask, cache=cache)

            # Attention only, skip MoE
            residual = x
            x = block_self.input_layernorm(x)
            x, new_cache = block_self.self_attn(x, mask=mask, cache=cache)
            x = residual + x
            return x, new_cache

        try:
            self._block_class.__call__ = patched_block
            result = self._generate(prompt)
        finally:
            self._block_class.__call__ = self._original_block_call

        return result

    def _run_condition(self, prompt: str, condition: str) -> str:
        """Run a single prompt under a condition."""
        even_layers = set(range(0, 24, 2))
        every_3rd_layers = set(list(range(0, 24, 3)) + [23])  # 9 layers

        if condition in ("normal", "normal_mb"):
            return self._generate(prompt)
        elif condition in ("fixed_alt", "fixed_alt_mb"):
            return self._generate_fixed_alternating(prompt, even_layers)
        elif condition in ("skip_alt", "skip_alt_mb"):
            return self._generate_skip_alternating(prompt, even_layers)
        elif condition in ("fixed_heavy", "fixed_heavy_mb"):
            return self._generate_fixed_alternating(prompt, every_3rd_layers)
        elif condition in ("fixed_heavy_mb",):
            return self._generate_fixed_alternating(prompt, every_3rd_layers)
        else:
            return self._generate(prompt)

    async def run_fact(self, fact: dict, conditions: list[str], is_counterfactual: bool = False):
        """Run one fact across all conditions."""
        loop = asyncio.get_event_loop()
        all_mb_entries = [f["mb_entry"] for f in (COUNTERFACTUALS if is_counterfactual else FACTS)]

        for condition in conditions:
            use_mb = condition.endswith("_mb")

            if use_mb:
                prompt = build_memory_bank_prompt(fact["prompt"], all_mb_entries)
            else:
                # Simple prompt (matching earlier experiments)
                prompt = fact["prompt"].replace("What is ", "The ").replace("Who is ", "The ").replace("Who wrote ", "The author of ").rstrip("?") + " is"
                if "speed of light" in prompt:
                    prompt = "The speed of light is approximately"

            text = await loop.run_in_executor(
                None, self._run_condition, prompt, condition
            )
            mx.eval(mx.zeros(1))

            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)

            self.results.append({
                "condition": condition,
                "use_memory_bank": use_mb,
                "is_counterfactual": is_counterfactual,
                "question": fact["prompt"],
                "expected": fact["expected"],
                "prompt_used": prompt[:100] + ("..." if len(prompt) > 100 else ""),
                "text": text,
                "fact_preserved": preserved,
                "repetition_ratio": rep,
            })

            status = "✓" if preserved else "✗"
            logger.info(
                f"    {condition:>16} | {status} {fact['expected']:>12} | rep={rep:.3f} | "
                f"{text[:50].replace(chr(10), ' ')}"
            )

    def _print_summary(self):
        print("\n" + "=" * 90)
        print("MEMORY BANK + LITE MODEL RESULTS")
        print("=" * 90)

        # Group by condition
        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        # Standard facts
        print("\n### Standard Facts (correct answers in memory bank)")
        print(f"\n{'Condition':>18} | {'Facts':>6} | {'AvgRep':>6} | Description")
        print("-" * 80)

        conditions_order = [
            "normal", "normal_mb",
            "fixed_alt", "fixed_alt_mb",
            "skip_alt", "skip_alt_mb",
            "fixed_heavy", "fixed_heavy_mb",
        ]

        descriptions = {
            "normal": "Full model, no memory bank",
            "normal_mb": "Full model + memory bank",
            "fixed_alt": "Alternating fixed routing, no MB",
            "fixed_alt_mb": "Alternating fixed routing + MB",
            "skip_alt": "Alternating MoE skip, no MB",
            "skip_alt_mb": "Alternating MoE skip + MB",
            "fixed_heavy": "Every-3rd routing (9/24), no MB",
            "fixed_heavy_mb": "Every-3rd routing (9/24) + MB",
        }

        for cond in conditions_order:
            results = [r for r in by_cond.get(cond, []) if not r["is_counterfactual"]]
            if not results:
                continue
            facts_ok = sum(1 for r in results if r["fact_preserved"])
            avg_rep = sum(r["repetition_ratio"] for r in results) / len(results)
            pct = facts_ok / len(results) * 100

            print(
                f"{cond:>18} | {facts_ok}/{len(results)} {pct:>3.0f}% | "
                f"{avg_rep:>6.3f} | {descriptions.get(cond, '')}"
            )

        # Counterfactuals
        print("\n### Counterfactual Facts (wrong answers in memory bank)")
        print(f"\n{'Condition':>18} | {'Override':>8} | {'AvgRep':>6}")
        print("-" * 50)

        cf_conditions = [c for c in conditions_order if c.endswith("_mb")]
        for cond in cf_conditions:
            results = [r for r in by_cond.get(cond, []) if r["is_counterfactual"]]
            if not results:
                continue
            overrides = sum(1 for r in results if r["fact_preserved"])
            avg_rep = sum(r["repetition_ratio"] for r in results) / len(results)
            pct = overrides / len(results) * 100

            print(
                f"{cond:>18} | {overrides}/{len(results)} {pct:>3.0f}% | "
                f"{avg_rep:>6.3f}"
            )

        # Key comparison
        print("\n### Key Comparison: Does Memory Bank Rescue the Lite Model?")
        pairs = [
            ("fixed_alt", "fixed_alt_mb", "Alternating fixed"),
            ("skip_alt", "skip_alt_mb", "Alternating skip"),
            ("fixed_heavy", "fixed_heavy_mb", "Every-3rd fixed"),
        ]
        for no_mb, with_mb, label in pairs:
            no_mb_results = [r for r in by_cond.get(no_mb, []) if not r["is_counterfactual"]]
            mb_results = [r for r in by_cond.get(with_mb, []) if not r["is_counterfactual"]]
            if not no_mb_results or not mb_results:
                continue

            no_mb_ok = sum(1 for r in no_mb_results if r["fact_preserved"])
            mb_ok = sum(1 for r in mb_results if r["fact_preserved"])
            no_mb_rep = sum(r["repetition_ratio"] for r in no_mb_results) / len(no_mb_results)
            mb_rep = sum(r["repetition_ratio"] for r in mb_results) / len(mb_results)

            recovery = "YES" if mb_ok >= 6 else "PARTIAL" if mb_ok > no_mb_ok else "NO"
            print(
                f"  {label:>20}: {no_mb_ok}/8 → {mb_ok}/8 "
                f"(rep: {no_mb_rep:.3f} → {mb_rep:.3f}) [{recovery}]"
            )

        # Sample outputs
        print("\n### Sample Outputs: 'What is the capital of France?'")
        for cond in conditions_order:
            for r in self.results:
                if r["condition"] == cond and "France" in r["question"] and not r["is_counterfactual"]:
                    text_short = r["text"][:70].replace("\n", " ")
                    status = "✓" if r["fact_preserved"] else "✗"
                    print(f"  {cond:>18} {status}: {text_short}")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"memory_bank_lite_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "memory_bank_lite",
                        "model": "openai/gpt-oss-20b",
                        "timestamp": timestamp,
                        "max_tokens": MAX_TOKENS,
                        "conditions": [
                            "normal", "normal_mb",
                            "fixed_alt", "fixed_alt_mb",
                            "skip_alt", "skip_alt_mb",
                            "fixed_heavy", "fixed_heavy_mb",
                        ],
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

        conditions = [
            "normal", "normal_mb",
            "fixed_alt", "fixed_alt_mb",
            "skip_alt", "skip_alt_mb",
            "fixed_heavy", "fixed_heavy_mb",
        ]

        total_passes = len(FACTS) * len(conditions) + len(COUNTERFACTUALS) * 4  # CF only on _mb conditions
        logger.info("=" * 60)
        logger.info("MEMORY BANK + LITE MODEL EXPERIMENT")
        logger.info(f"  Conditions: {len(conditions)}")
        logger.info(f"  Facts: {len(FACTS)}, Counterfactuals: {len(COUNTERFACTUALS)}")
        logger.info(f"  Total passes: ~{total_passes}")
        logger.info("=" * 60)

        # Run standard facts across all conditions
        for fact in FACTS:
            logger.info(f"\n  Fact: {fact['expected']}")
            await self.run_fact(fact, conditions)

        # Run counterfactuals (only on memory bank conditions)
        mb_conditions = [c for c in conditions if c.endswith("_mb")]
        logger.info("\n--- COUNTERFACTUALS ---")
        for fact in COUNTERFACTUALS:
            logger.info(f"\n  Counterfactual: {fact['expected']}")
            await self.run_fact(fact, mb_conditions, is_counterfactual=True)

        self._print_summary()
        self._save_results()


async def main():
    experiment = MemoryBankLite()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
