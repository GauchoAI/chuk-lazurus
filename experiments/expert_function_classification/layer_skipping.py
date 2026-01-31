#!/usr/bin/env python3
"""Layer Skipping Experiment.

Part 16 showed fixed routing (wrong experts, wrong signal) degrades badly.
This experiment tests: what if we SKIP the MoE sublayer entirely?

Key insight:
  Fixed routing: residual + WRONG_MoE_signal → corrupts residual stream
  Layer skipping: residual + 0 → residual passes through unchanged

If skipping works better than fixed routing, the MoE signal at skipped layers
is actively harmful (wrong signal worse than no signal).

Conditions use the same layer sets as Part 16 for direct comparison:
  A. normal - baseline (MoE at all 24 layers)
  B. skip_first_half - skip MoE at L0-11, keep at L12-23
  C. skip_second_half - skip MoE at L12-23, keep at L0-11
  D. skip_alternating - keep MoE at even layers, skip at odd
  E. skip_bookends - keep MoE at L0-3 + L20-23, skip L4-19
  F. skip_every_6th - keep MoE at [0,6,12,18,23], skip rest
  G. skip_middle_only - keep MoE at L8-15, skip rest
  H. skip_first_quarter - keep MoE at L0-5, skip L6-23
  I. skip_last_quarter - keep MoE at L18-23, skip L0-17
  J. skip_all - skip MoE at ALL layers (attention-only model)

Run: python experiments/expert_function_classification/layer_skipping.py
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

# Conditions: name -> list of layer indices that KEEP their MoE sublayer
# (layers NOT in this list have MoE skipped entirely)
CONDITIONS: dict[str, list[int]] = {
    "normal": list(range(24)),
    "skip_first_half": list(range(12, 24)),       # keep L12-23
    "skip_second_half": list(range(12)),           # keep L0-11
    "skip_alternating": list(range(0, 24, 2)),     # keep even layers
    "skip_bookends": list(range(4)) + list(range(20, 24)),  # keep L0-3,L20-23
    "skip_every_6th": [0, 6, 12, 18, 23],         # keep 5 evenly-spaced
    "skip_middle_only": list(range(8, 16)),        # keep L8-15
    "skip_first_quarter": list(range(6)),          # keep L0-5
    "skip_last_quarter": list(range(18, 24)),      # keep L18-23
    "skip_every_4th": [0, 4, 8, 12, 16, 20, 23],  # keep 7 evenly-spaced
    "skip_every_3rd": list(range(0, 24, 3)) + [23],  # keep 9 evenly-spaced
    "skip_all": [],                                 # no MoE at all (attention-only)
}


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


class LayerSkipping:
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

        # Get block class for monkey-patching
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

    def _generate_with_skipping(
        self, prompt: str, moe_layers: set[int]
    ) -> str:
        """Generate with MoE active only at specified layers, skipped at rest.

        At skipped layers, the block does:
          x = residual + attention(layernorm(x))   # attention still runs
          x = x                                     # MoE skipped (identity)

        Instead of:
          x = residual + attention(layernorm(x))
          x = residual + MoE(layernorm(x))          # normal MoE
        """
        experiment = self
        original_call = self._original_block_call

        def patched_block(
            block_self: Any,
            x: mx.array,
            mask: mx.array | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            # Find which layer this block belongs to
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if layer is block_self:
                    layer_idx = i
                    break

            # If this layer keeps MoE, use original forward pass
            if layer_idx in moe_layers:
                return original_call(block_self, x, mask=mask, cache=cache)

            # Skip MoE: run attention normally, skip MoE sublayer
            # Self-attention with residual (unchanged from original)
            residual = x
            x = block_self.input_layernorm(x)
            x, new_cache = block_self.self_attn(x, mask=mask, cache=cache)
            x = residual + x

            # MoE SKIPPED: x passes through unchanged
            # (no post_attention_layernorm, no mlp, no residual addition)

            return x, new_cache

        try:
            self._block_class.__call__ = patched_block
            result = self._generate(prompt)
        finally:
            self._block_class.__call__ = self._original_block_call

        return result

    async def run_condition(self, name: str, moe_layers: list[int]):
        n_active = len(moe_layers)
        n_skipped = 24 - n_active
        logger.info(
            f"\n  {name}: {n_active} MoE active, {n_skipped} skipped"
        )

        loop = asyncio.get_event_loop()
        moe_set = set(moe_layers)

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
                    self._generate_with_skipping,
                    fact["prompt"],
                    moe_set,
                )
            mx.eval(mx.zeros(1))

            preserved = fact["expected"].lower() in text.lower()
            rep = compute_repetition_ratio(text)
            if preserved:
                facts_ok += 1
            total_rep += rep

            self.results.append({
                "condition": name,
                "n_moe_active": n_active,
                "moe_layers": moe_layers,
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
                    self._generate_with_skipping,
                    prompt,
                    moe_set,
                )
            mx.eval(mx.zeros(1))
            rep = compute_repetition_ratio(text)
            coherence_rep += rep

            self.results.append({
                "condition": name,
                "n_moe_active": n_active,
                "moe_layers": moe_layers,
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
        print("\n" + "=" * 80)
        print("LAYER SKIPPING RESULTS")
        print("=" * 80)

        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        print(
            f"\n{'Condition':>20} | {'MoE':>4} | {'Skip':>4} | "
            f"{'Facts':>6} | {'AvgRep':>6}"
        )
        print("-" * 65)

        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue
            fact_results = [r for r in results if r["expected"] is not None]
            facts_ok = sum(1 for r in fact_results if r["fact_preserved"])
            n_active = results[0]["n_moe_active"]
            avg_rep = (
                sum(r["repetition_ratio"] for r in results) / len(results)
                if results
                else 0
            )
            pct = facts_ok / len(fact_results) * 100 if fact_results else 0

            print(
                f"{name:>20} | {n_active:>2}/24 | "
                f"{24 - n_active:>2}/24 | "
                f"{facts_ok}/{len(fact_results)} {pct:>3.0f}% | "
                f"{avg_rep:>6.3f}"
            )

        # Coverage curve
        print("\n--- COVERAGE CURVE (SKIP vs FIXED ROUTING from Part 16) ---")
        print("  Layers | Skip       | Part16-Fixed")
        print("  -------+------------+-------------")

        coverage_data = []
        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue
            fact_results = [r for r in results if r["expected"] is not None]
            facts_ok = sum(1 for r in fact_results if r["fact_preserved"])
            n_active = results[0]["n_moe_active"]
            coverage_data.append((n_active, name, facts_ok))

        coverage_data.sort()
        for n_active, name, facts_ok in coverage_data:
            bar = "#" * facts_ok + "." * (8 - facts_ok)
            print(f"  {n_active:>2} MoE | {bar} {facts_ok}/8 | {name}")

        # Sample outputs
        print("\n--- Sample Outputs ---")
        sample_prompt = "The capital of France is"
        for name in CONDITIONS:
            for r in self.results:
                if r["condition"] == name and r["prompt"] == sample_prompt:
                    text_short = r["text"][:80].replace("\n", " ")
                    print(f"  {name:>20}: {text_short}")

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"layer_skipping_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "layer_skipping",
                        "model": "openai/gpt-oss-20b",
                        "timestamp": timestamp,
                        "conditions": {
                            k: {"moe_layers": v, "n_moe_active": len(v)}
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
        logger.info("LAYER SKIPPING EXPERIMENT")
        logger.info(f"  Conditions: {len(CONDITIONS)}")
        logger.info(f"  Total passes: {n_passes}")
        logger.info("=" * 60)

        for name, moe_layers in CONDITIONS.items():
            await self.run_condition(name, moe_layers)

        self._print_summary()
        self._save_results()


async def main():
    experiment = LayerSkipping()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
