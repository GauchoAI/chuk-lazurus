#!/usr/bin/env python3
"""Layer Skip at Emergence Point Experiment.

Residual fact emergence showed facts crystallize at L20-21 (top-1 avg L20.8).
Knowledge ablation showed 0/8 facts break under expert ablation at any layer.

This experiment tests: are the emergence layers (L20-21) *necessary* for
fact crystallization, or can the computation be deferred to later layers?

We skip the MoE FFN entirely at specific layers (attention still runs,
residual passes through, but expert computation is zeroed). Then we:
  1. Generate text and check if the fact is preserved
  2. Run logit lens to see if/where the fact emerges under each condition

Conditions:
  A. normal          - no skip (baseline)
  B. skip_L20        - skip MoE at L20 only
  C. skip_L21        - skip MoE at L21 only
  D. skip_L20_L21    - skip MoE at L20 and L21
  E. skip_L19_L20_L21 - skip the full emergence window
  F. skip_L15        - skip where first signal appears (control)
  G. skip_L22_L23    - skip post-emergence layers (control)

If skipping L20-21 kills facts: those layers are the bottleneck.
If facts still emerge at L22-23: the residual stream can crystallize later.

Run: python experiments/expert_function_classification/layer_skip_emergence.py
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
    {"prompt": "The capital of France is", "expected_keyword": "Paris"},
    {"prompt": "The chemical symbol for gold is", "expected_keyword": "Au"},
    {"prompt": "The author of Romeo and Juliet is", "expected_keyword": "Shakespeare"},
    {"prompt": "The CEO of Microsoft is", "expected_keyword": "Nadella"},
    {"prompt": "The capital of Japan is", "expected_keyword": "Tokyo"},
    {"prompt": "The chemical symbol for silver is", "expected_keyword": "Ag"},
    {"prompt": "The capital of Australia is", "expected_keyword": "Canberra"},
]

# Skip conditions: name -> list of layer indices where MoE FFN is zeroed
CONDITIONS: dict[str, list[int]] = {
    "normal": [],
    "skip_L20": [20],
    "skip_L21": [21],
    "skip_L20_L21": [20, 21],
    "skip_L19_L20_L21": [19, 20, 21],
    "skip_L15": [15],
    "skip_L22_L23": [22, 23],
}

MAX_TOKENS = 40


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


class LayerSkipEmergence:
    """Test whether emergence layers are necessary for fact crystallization."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._mlp_class = None
        self._original_mlp_call = None
        self.results: list[dict] = []

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        # Capture the MLP class for monkey-patching
        sample_layer = self.model.model.layers[0]
        self._mlp_class = type(sample_layer.mlp)
        self._original_mlp_call = self._mlp_class.__call__

        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded: {self.num_layers} layers. Ready.")

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

    def _generate_with_skip(self, prompt: str, skip_layers: set[int]) -> str:
        """Generate text with MoE FFN zeroed at specified layers.

        Attention still runs. The residual stream passes through unchanged
        at skipped layers (x = x + 0 = x).
        """
        experiment = self
        original_call = self._original_mlp_call

        def patched_mlp(mlp_self: Any, x: mx.array) -> mx.array:
            # Find which layer this MLP belongs to
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if hasattr(layer, "mlp") and layer.mlp is mlp_self:
                    layer_idx = i
                    break

            if layer_idx in skip_layers:
                # Return zeros: residual + 0 = residual (MoE skipped)
                return mx.zeros_like(x)

            return original_call(mlp_self, x)

        try:
            self._mlp_class.__call__ = patched_mlp
            result = self._generate(prompt)
        finally:
            self._mlp_class.__call__ = self._original_mlp_call

        return result

    def _discover_fact_token(self, prompt: str) -> tuple[int, str]:
        """Get the model's actual predicted next token for this prompt."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        output = self.model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        next_token_id = int(mx.argmax(logits[0, -1, :]).item())
        decoded = self.tokenizer.decode([next_token_id])
        return next_token_id, decoded

    def _run_logit_lens_with_skip(
        self, prompt: str, skip_layers: set[int], token_id: int
    ) -> dict[str, Any]:
        """Run logit lens with MoE skipped at specified layers.

        Returns probability and rank of token_id at each layer.
        """
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks
        from chuk_lazarus.introspection.logit_lens import LogitLens

        experiment = self
        original_call = self._original_mlp_call

        def patched_mlp(mlp_self: Any, x: mx.array) -> mx.array:
            layer_idx = -1
            for i, layer in enumerate(experiment.model.model.layers):
                if hasattr(layer, "mlp") and layer.mlp is mlp_self:
                    layer_idx = i
                    break
            if layer_idx in skip_layers:
                return mx.zeros_like(x)
            return original_call(mlp_self, x)

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                positions="last",
            )
        )

        try:
            self._mlp_class.__call__ = patched_mlp
            hooks.forward(input_ids)
        finally:
            self._mlp_class.__call__ = self._original_mlp_call

        lens = LogitLens(hooks, self.tokenizer)
        evolution = lens.track_token(token_id, position=-1, top_k_for_rank=200)

        # Find emergence points
        emergence_top1 = evolution.emergence_layer
        emergence_top5 = None
        threshold_50 = None

        for layer, rank, prob in zip(
            evolution.layers, evolution.ranks, evolution.probabilities
        ):
            if emergence_top5 is None and rank is not None and rank <= 5:
                emergence_top5 = layer
            if threshold_50 is None and prob >= 0.50:
                threshold_50 = layer

        return {
            "layers": evolution.layers,
            "probabilities": evolution.probabilities,
            "ranks": evolution.ranks,
            "emergence_top1": emergence_top1,
            "emergence_top5": emergence_top5,
            "threshold_50pct": threshold_50,
        }

    async def run_condition(self, name: str, skip_layers: list[int]):
        skip_set = set(skip_layers)
        skip_str = f"skip {skip_layers}" if skip_layers else "no skip"
        logger.info(f"\n  {name}: {skip_str}")

        loop = asyncio.get_event_loop()

        for fact in FACTS:
            prompt = fact["prompt"]
            keyword = fact["expected_keyword"]

            # Discover the correct token under normal conditions
            token_id, token_str = self._discover_fact_token(prompt)

            # Generate with skip
            if skip_layers:
                text = await loop.run_in_executor(
                    None, self._generate_with_skip, prompt, skip_set,
                )
            else:
                text = await loop.run_in_executor(
                    None, self._generate, prompt,
                )
            mx.eval(mx.zeros(1))

            preserved = keyword.lower() in text.lower()
            rep = compute_repetition_ratio(text)

            # Logit lens with skip
            lens_data = self._run_logit_lens_with_skip(prompt, skip_set, token_id)

            logger.info(
                f"    {prompt[:38]:38} | "
                f"{'ok' if preserved else 'LOST':>4} | "
                f"top1@L{lens_data['emergence_top1']} | "
                f">50%@L{lens_data['threshold_50pct']} | "
                f"{text[:40]}"
            )

            self.results.append({
                "condition": name,
                "skip_layers": skip_layers,
                "prompt": prompt,
                "expected_keyword": keyword,
                "discovered_token": token_str,
                "discovered_token_id": token_id,
                "text": text[:120],
                "fact_preserved": preserved,
                "repetition_ratio": rep,
                "emergence_top1": lens_data["emergence_top1"],
                "emergence_top5": lens_data["emergence_top5"],
                "threshold_50pct": lens_data["threshold_50pct"],
                "probability_curve": {
                    str(l): round(p, 6)
                    for l, p in zip(lens_data["layers"], lens_data["probabilities"])
                },
            })

    def _print_summary(self):
        print("\n" + "=" * 95)
        print("LAYER SKIP AT EMERGENCE POINT - RESULTS")
        print("=" * 95)

        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            by_cond[r["condition"]].append(r)

        # Overview table
        print(
            f"\n{'Condition':<22} | {'Skip':>18} | "
            f"{'Facts':>5} | {'AvgTop1':>7} | {'Avg>50%':>7} | {'AvgRep':>7}"
        )
        print("-" * 95)

        for name in CONDITIONS:
            results = by_cond.get(name, [])
            if not results:
                continue

            facts_ok = sum(1 for r in results if r["fact_preserved"])
            top1s = [r["emergence_top1"] for r in results if r["emergence_top1"] is not None]
            th50s = [r["threshold_50pct"] for r in results if r["threshold_50pct"] is not None]
            avg_rep = sum(r["repetition_ratio"] for r in results) / len(results)

            avg_top1 = f"L{sum(top1s)/len(top1s):.1f}" if top1s else "-"
            avg_th50 = f"L{sum(th50s)/len(th50s):.1f}" if th50s else "-"
            skip_str = str(CONDITIONS[name]) if CONDITIONS[name] else "none"

            print(
                f"{name:<22} | {skip_str:>18} | "
                f"{facts_ok}/{len(results):>3} | {avg_top1:>7} | {avg_th50:>7} | "
                f"{avg_rep:>7.3f}"
            )

        # Per-fact detail for key conditions
        print("\n" + "-" * 95)
        print("PER-FACT EMERGENCE SHIFT")
        print("-" * 95)

        for fact in FACTS:
            prompt_short = fact["prompt"][:35]
            print(f"\n  {prompt_short}:")

            for name in CONDITIONS:
                results = by_cond.get(name, [])
                match = [r for r in results if r["prompt"] == fact["prompt"]]
                if not match:
                    continue
                r = match[0]
                status = "ok" if r["fact_preserved"] else "LOST"
                top1 = f"L{r['emergence_top1']}" if r["emergence_top1"] is not None else "never"
                th50 = f"L{r['threshold_50pct']}" if r["threshold_50pct"] is not None else "never"
                print(
                    f"    {name:<22}: [{status:>4}] top1={top1:<6} >50%={th50:<6} "
                    f"| {r['text'][:45]}"
                )

        # Average probability curves for key conditions
        print("\n" + "-" * 95)
        print("AVERAGE PROBABILITY CURVE COMPARISON")
        print("-" * 95)

        key_conditions = ["normal", "skip_L20_L21", "skip_L19_L20_L21", "skip_L22_L23"]
        header = f"{'Layer':>5}"
        for name in key_conditions:
            if name in by_cond:
                header += f" | {name:>18}"
        print(header)
        print("-" * (6 + 21 * len(key_conditions)))

        for layer in range(24):
            row = f"L{layer:>3}:"
            for name in key_conditions:
                results = by_cond.get(name, [])
                if not results:
                    continue
                probs = []
                for r in results:
                    curve = r["probability_curve"]
                    if str(layer) in curve:
                        probs.append(curve[str(layer)])
                avg_p = sum(probs) / len(probs) if probs else 0.0
                bar_len = int(avg_p * 12)
                bar = "#" * bar_len
                row += f" | {avg_p:>6.4f} {bar:<11}"
            print(row)

        # Key findings
        print("\n" + "=" * 95)
        print("KEY FINDINGS")
        print("=" * 95)

        normal_facts = sum(1 for r in by_cond.get("normal", []) if r["fact_preserved"])
        skip20_facts = sum(1 for r in by_cond.get("skip_L20", []) if r["fact_preserved"])
        skip21_facts = sum(1 for r in by_cond.get("skip_L21", []) if r["fact_preserved"])
        skip2021_facts = sum(1 for r in by_cond.get("skip_L20_L21", []) if r["fact_preserved"])
        skip192021_facts = sum(1 for r in by_cond.get("skip_L19_L20_L21", []) if r["fact_preserved"])

        print(f"\n  Fact preservation:")
        print(f"    normal:             {normal_facts}/{len(FACTS)}")
        print(f"    skip L20:           {skip20_facts}/{len(FACTS)}")
        print(f"    skip L21:           {skip21_facts}/{len(FACTS)}")
        print(f"    skip L20+L21:       {skip2021_facts}/{len(FACTS)}")
        print(f"    skip L19+L20+L21:   {skip192021_facts}/{len(FACTS)}")

        if skip2021_facts == normal_facts:
            print("\n  FINDING: Skipping emergence layers does NOT kill facts.")
            print("  Facts crystallize at later layers when L20-L21 are removed.")
            print("  The residual stream is robust - fact computation is deferrable.")
        elif skip2021_facts == 0:
            print("\n  FINDING: Skipping emergence layers KILLS all facts.")
            print("  L20-L21 are the critical bottleneck for fact crystallization.")
            print("  The residual stream cannot compensate without these layers.")
        else:
            print(f"\n  FINDING: Partial survival ({skip2021_facts}/{len(FACTS)}).")
            print("  Some facts can defer to later layers, others cannot.")
            print("  Fact difficulty correlates with emergence robustness.")

        print("=" * 95)

    def _save_results(self) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"layer_skip_emergence_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "experiment": "layer_skip_emergence",
                        "model": "openai/gpt-oss-20b",
                        "timestamp": timestamp,
                        "num_facts": len(FACTS),
                        "conditions": {
                            k: {"skip_layers": v, "n_skipped": len(v)}
                            for k, v in CONDITIONS.items()
                        },
                        "description": (
                            "Tests whether fact emergence layers (L20-21) are "
                            "necessary by zeroing MoE FFN output at those layers. "
                            "Attention still runs; residual passes through unchanged."
                        ),
                        "prior_results": {
                            "fact_emergence_avg_top1": "L20.8",
                            "knowledge_ablation": "0/8 facts break at any layer",
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

        n_passes = len(CONDITIONS) * len(FACTS)
        logger.info("=" * 70)
        logger.info("LAYER SKIP AT EMERGENCE POINT")
        logger.info(f"  Conditions: {len(CONDITIONS)}")
        logger.info(f"  Facts: {len(FACTS)}")
        logger.info(f"  Total passes: ~{n_passes} generate + {n_passes} logit lens")
        logger.info("=" * 70)

        for name, skip_layers in CONDITIONS.items():
            await self.run_condition(name, skip_layers)

        self._save_results()
        self._print_summary()


async def main():
    experiment = LayerSkipEmergence()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
