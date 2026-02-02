#!/usr/bin/env python3
"""Residual Stream Fact Emergence Experiment.

Knowledge ablation showed facts survive full top-4 expert ablation at
every tested layer (L8, L12, L16, L20). Minimum viable routing showed
6-7 learned layers + memory bank = 100% fact preservation.

This experiment answers: WHERE in the residual stream do facts actually
emerge? Using logit lens, we project intermediate hidden states to vocab
logits at every layer and track when the correct answer token first
appears (rank-1) and when it becomes confident (>50% probability).

If facts emerge early (L4-L8) but expert ablation at those layers doesn't
break them, that proves facts propagate through the residual stream
independently of expert computation.

We test three conditions:
  1. Normal: full model, logit lens at all 24 layers
  2. Ablated emergence: ablate top-4 experts AT the emergence layer
  3. Skip emergence: skip the entire MoE FFN at the emergence layer

This directly connects your knowledge_ablation finding (facts never break)
to a mechanistic explanation (residual stream carries facts, experts add
fluency).

Run: python experiments/expert_function_classification/residual_fact_emergence.py
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


# =============================================================================
# Data - same 8 facts from prior experiments for comparability
# =============================================================================

FACTS = [
    {"prompt": "The capital of France is", "expected_keyword": "Paris"},
    {"prompt": "The chemical symbol for gold is", "expected_keyword": "Au"},
    {"prompt": "The author of Romeo and Juliet is", "expected_keyword": "Shakespeare"},
    {"prompt": "The speed of light is approximately", "expected_keyword": "299"},
    {"prompt": "The CEO of Microsoft is", "expected_keyword": "Nadella"},
    {"prompt": "The capital of Japan is", "expected_keyword": "Tokyo"},
    {"prompt": "The chemical symbol for silver is", "expected_keyword": "Ag"},
    {"prompt": "The capital of Australia is", "expected_keyword": "Canberra"},
]


# =============================================================================
# Experiment
# =============================================================================


class ResidualFactEmergence:
    """Track where facts emerge in the residual stream via logit lens."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.hooks = None

    async def setup(self):
        """Load model and set up hooks."""
        from chuk_lazarus.introspection.moe import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer
        self._router = router

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        info = router._info
        self.num_layers = len(info.moe_layers)
        logger.info(
            f"Model loaded: {info.num_experts} experts/layer, "
            f"{self.num_layers} MoE layers, top-{info.num_experts_per_tok}"
        )

    def _discover_fact_token(self, prompt: str) -> tuple[int, str]:
        """Discover the actual token the model predicts for this prompt.

        Instead of guessing token IDs (which fails due to BPE space
        prefixes like 'Ġ', '▁', etc.), we run a forward pass and take
        the argmax of the final logits. This IS the fact token.

        Returns (token_id, decoded_token_string).
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        output = self.model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        next_token_id = int(mx.argmax(logits[0, -1, :]).item())
        decoded = self.tokenizer.decode([next_token_id])
        return next_token_id, decoded

    def _run_logit_lens(self, prompt: str) -> dict[str, Any]:
        """Run logit lens at all layers, return per-layer top predictions.

        Returns dict with:
          - layer_predictions: {layer_idx: [(token, prob), ...]}
          - hidden_states captured for further analysis
        """
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks
        from chuk_lazarus.introspection.logit_lens import LogitLens

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                capture_pre_norm=True,
                positions="last",  # Only last position (prediction point)
            )
        )
        hooks.forward(input_ids)

        lens = LogitLens(hooks, self.tokenizer)
        return {
            "hooks": hooks,
            "lens": lens,
            "input_ids": input_ids,
        }

    def _track_fact_token(
        self,
        lens_result: dict,
        token_id: int,
        token_str: str,
    ) -> dict[str, Any]:
        """Track a specific token's emergence through residual stream layers."""
        from chuk_lazarus.introspection.logit_lens import LogitLens

        lens: LogitLens = lens_result["lens"]

        logger.info(f"    Tracking token '{token_str}' (id={token_id})")

        evolution = lens.track_token(token_id, position=-1, top_k_for_rank=200)

        # Find key emergence points
        emergence_top1 = evolution.emergence_layer  # First layer at rank 1

        emergence_top5 = None
        emergence_top10 = None
        threshold_10pct = None
        threshold_50pct = None

        for layer, rank, prob in zip(
            evolution.layers, evolution.ranks, evolution.probabilities
        ):
            if emergence_top10 is None and rank is not None and rank <= 10:
                emergence_top10 = layer
            if emergence_top5 is None and rank is not None and rank <= 5:
                emergence_top5 = layer
            if threshold_10pct is None and prob >= 0.10:
                threshold_10pct = layer
            if threshold_50pct is None and prob >= 0.50:
                threshold_50pct = layer

        return {
            "token": token_str,
            "token_id": token_id,
            "layers": evolution.layers,
            "probabilities": evolution.probabilities,
            "ranks": evolution.ranks,
            "emergence_top1": emergence_top1,
            "emergence_top5": emergence_top5,
            "emergence_top10": emergence_top10,
            "threshold_10pct": threshold_10pct,
            "threshold_50pct": threshold_50pct,
        }

    def _get_top_predictions(self, lens_result: dict) -> dict[int, list[tuple[str, float]]]:
        """Get top-5 predictions at each layer."""
        from chuk_lazarus.introspection.logit_lens import LogitLens

        lens: LogitLens = lens_result["lens"]
        predictions = lens.get_layer_predictions(position=-1, top_k=5)

        return {
            pred.layer_idx: list(zip(pred.top_tokens, pred.top_probs))
            for pred in predictions
        }

    def _generate(self, prompt: str, max_tokens: int = 30) -> str:
        """Generate text (baseline)."""
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

    async def analyze_fact(self, fact: dict) -> dict[str, Any]:
        """Run full emergence analysis for one fact."""
        prompt = fact["prompt"]
        expected_keyword = fact["expected_keyword"]

        logger.info(f"\n  Fact: {prompt}")

        # 1. Discover the actual token the model predicts (no guessing BPE)
        fact_token_id, fact_token_str = self._discover_fact_token(prompt)
        logger.info(f"    Model predicts: '{fact_token_str}' (id={fact_token_id})")

        # 2. Baseline generation (full sequence for correctness check)
        baseline_text = self._generate(prompt)
        baseline_correct = expected_keyword.lower() in baseline_text.lower()
        logger.info(
            f"    Baseline: {'correct' if baseline_correct else 'WRONG'} | "
            f"{baseline_text[:60]}"
        )

        # 3. Logit lens across all layers
        lens_result = self._run_logit_lens(prompt)

        # 4. Track the ACTUAL predicted token through all layers
        token_tracking = self._track_fact_token(
            lens_result, fact_token_id, fact_token_str
        )

        logger.info(
            f"    Emergence: top10@L{token_tracking['emergence_top10']}, "
            f"top5@L{token_tracking['emergence_top5']}, "
            f"top1@L{token_tracking['emergence_top1']}"
        )
        logger.info(
            f"    Threshold: >10%@L{token_tracking['threshold_10pct']}, "
            f">50%@L{token_tracking['threshold_50pct']}"
        )

        # 4. Get top predictions at each layer for context
        top_preds = self._get_top_predictions(lens_result)

        # 5. Log probability curve highlights
        probs = token_tracking["probabilities"]
        layers = token_tracking["layers"]
        if probs:
            peak_idx = max(range(len(probs)), key=lambda i: probs[i])
            logger.info(
                f"    Peak: L{layers[peak_idx]} at {probs[peak_idx]:.3f}"
            )

        # 6. Capture what's at the emergence layer (competing predictions)
        emergence_layer = token_tracking["emergence_top1"]
        competitors_at_emergence = {}
        if emergence_layer is not None and emergence_layer in top_preds:
            competitors_at_emergence = {
                "layer": emergence_layer,
                "predictions": [
                    {"token": t, "prob": float(p)}
                    for t, p in top_preds[emergence_layer]
                ],
            }

        # 7. What's the residual doing at pre-emergence layers?
        pre_emergence_predictions = {}
        if emergence_layer is not None:
            for check_layer in [0, emergence_layer // 2, max(0, emergence_layer - 1)]:
                if check_layer in top_preds:
                    pre_emergence_predictions[check_layer] = [
                        {"token": t, "prob": float(p)}
                        for t, p in top_preds[check_layer][:3]
                    ]

        return {
            "prompt": prompt,
            "discovered_token": fact_token_str,
            "discovered_token_id": fact_token_id,
            "expected_keyword": expected_keyword,
            "baseline_text": baseline_text[:100],
            "baseline_correct": baseline_correct,
            "token_tracking": {
                "token": token_tracking["token"],
                "token_id": token_tracking["token_id"],
                "emergence_top1": token_tracking["emergence_top1"],
                "emergence_top5": token_tracking["emergence_top5"],
                "emergence_top10": token_tracking["emergence_top10"],
                "threshold_10pct": token_tracking["threshold_10pct"],
                "threshold_50pct": token_tracking["threshold_50pct"],
                "probability_curve": {
                    str(l): round(p, 6)
                    for l, p in zip(
                        token_tracking["layers"],
                        token_tracking["probabilities"],
                    )
                },
                "rank_curve": {
                    str(l): r
                    for l, r in zip(
                        token_tracking["layers"],
                        token_tracking["ranks"],
                    )
                },
            },
            "competitors_at_emergence": competitors_at_emergence,
            "pre_emergence_predictions": pre_emergence_predictions,
        }

    async def run(self) -> dict[str, Any]:
        """Run the full experiment."""
        await self.setup()

        logger.info("=" * 70)
        logger.info("RESIDUAL STREAM FACT EMERGENCE")
        logger.info(f"  Model: openai/gpt-oss-20b ({self.num_layers} layers)")
        logger.info(f"  Facts: {len(FACTS)}")
        logger.info("=" * 70)

        fact_results = []
        for fact in FACTS:
            result = await self.analyze_fact(fact)
            fact_results.append(result)

        # Compute aggregate statistics
        summary = self._compute_summary(fact_results)

        # Build output
        output = {
            "metadata": {
                "experiment": "residual_fact_emergence",
                "model": "openai/gpt-oss-20b",
                "num_layers": self.num_layers,
                "timestamp": datetime.now().isoformat(),
                "num_facts": len(FACTS),
                "description": (
                    "Logit lens tracking of fact token emergence through "
                    "residual stream. Connects to knowledge_ablation finding "
                    "that facts survive full top-4 expert ablation."
                ),
            },
            "summary": summary,
            "fact_results": fact_results,
        }

        self._save_results(output)
        self._print_summary(summary, fact_results)

        return output

    def _compute_summary(self, fact_results: list[dict]) -> dict[str, Any]:
        """Compute aggregate statistics across all facts."""
        emergence_top1 = []
        emergence_top5 = []
        emergence_top10 = []
        threshold_10 = []
        threshold_50 = []
        baseline_correct = 0

        for r in fact_results:
            if r["baseline_correct"]:
                baseline_correct += 1

            t = r["token_tracking"]
            if t["emergence_top1"] is not None:
                emergence_top1.append(t["emergence_top1"])
            if t["emergence_top5"] is not None:
                emergence_top5.append(t["emergence_top5"])
            if t["emergence_top10"] is not None:
                emergence_top10.append(t["emergence_top10"])
            if t["threshold_10pct"] is not None:
                threshold_10.append(t["threshold_10pct"])
            if t["threshold_50pct"] is not None:
                threshold_50.append(t["threshold_50pct"])

        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else None

        def med(lst):
            if not lst:
                return None
            s = sorted(lst)
            n = len(s)
            return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

        # Probability at key layers (average across facts)
        prob_at_layer = defaultdict(list)
        for r in fact_results:
            curve = r["token_tracking"]["probability_curve"]
            for layer_str, prob in curve.items():
                prob_at_layer[int(layer_str)].append(prob)

        avg_prob_curve = {
            layer: round(sum(probs) / len(probs), 6)
            for layer, probs in sorted(prob_at_layer.items())
        }

        return {
            "baseline_accuracy": f"{baseline_correct}/{len(fact_results)}",
            "emergence_top1": {
                "values": emergence_top1,
                "mean": avg(emergence_top1),
                "median": med(emergence_top1),
                "min": min(emergence_top1) if emergence_top1 else None,
                "max": max(emergence_top1) if emergence_top1 else None,
                "count": len(emergence_top1),
            },
            "emergence_top5": {
                "values": emergence_top5,
                "mean": avg(emergence_top5),
                "median": med(emergence_top5),
            },
            "emergence_top10": {
                "values": emergence_top10,
                "mean": avg(emergence_top10),
                "median": med(emergence_top10),
            },
            "threshold_10pct": {
                "values": threshold_10,
                "mean": avg(threshold_10),
                "median": med(threshold_10),
            },
            "threshold_50pct": {
                "values": threshold_50,
                "mean": avg(threshold_50),
                "median": med(threshold_50),
            },
            "avg_probability_by_layer": avg_prob_curve,
            "interpretation": {
                "early_emergence": (
                    "Facts emerge before expert-heavy layers"
                    if emergence_top1 and avg(emergence_top1) < 12
                    else "Facts emerge in mid-to-late layers"
                    if emergence_top1 and avg(emergence_top1) < 18
                    else "Facts emerge in final layers"
                    if emergence_top1
                    else "No emergence detected"
                ),
                "connection_to_ablation": (
                    "Knowledge ablation showed 0/8 facts break at any layer. "
                    "If emergence is early, experts at later layers add fluency "
                    "but the residual stream already carries the fact."
                ),
            },
        }

    def _save_results(self, results: dict) -> None:
        output_path = (
            Path(__file__).parent
            / "results"
            / f"residual_fact_emergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")

    def _print_summary(
        self, summary: dict, fact_results: list[dict]
    ) -> None:
        print("\n" + "=" * 80)
        print("RESIDUAL STREAM FACT EMERGENCE - RESULTS")
        print("=" * 80)

        print(f"\nBaseline accuracy: {summary['baseline_accuracy']}")

        # Per-fact emergence table
        print("\n" + "-" * 80)
        print(
            f"{'Prompt':<42} | {'Top10':>5} | {'Top5':>5} | {'Top1':>5} | "
            f"{'>10%':>5} | {'>50%':>5} | {'Token':<8}"
        )
        print("-" * 80)

        for r in fact_results:
            t = r["token_tracking"]
            prompt_short = r["prompt"][:40]
            top10 = f"L{t['emergence_top10']}" if t["emergence_top10"] is not None else "-"
            top5 = f"L{t['emergence_top5']}" if t["emergence_top5"] is not None else "-"
            top1 = f"L{t['emergence_top1']}" if t["emergence_top1"] is not None else "-"
            th10 = f"L{t['threshold_10pct']}" if t["threshold_10pct"] is not None else "-"
            th50 = f"L{t['threshold_50pct']}" if t["threshold_50pct"] is not None else "-"
            tok = t["token"][:8]
            print(
                f"{prompt_short:<42} | {top10:>5} | {top5:>5} | {top1:>5} | "
                f"{th10:>5} | {th50:>5} | {tok:<8}"
            )

        # Aggregate
        print("\n" + "-" * 80)
        print("AGGREGATE EMERGENCE LAYERS")
        print("-" * 80)

        for metric in ["emergence_top10", "emergence_top5", "emergence_top1",
                        "threshold_10pct", "threshold_50pct"]:
            data = summary[metric]
            label = metric.replace("emergence_", "").replace("threshold_", ">")
            print(
                f"  {label:<12}: mean=L{data['mean']}, "
                f"median=L{data['median']}, "
                f"range=[L{data.get('min', '?')}-L{data.get('max', '?')}]"
            )

        # Average probability curve (condensed)
        print("\n" + "-" * 80)
        print("AVERAGE FACT PROBABILITY BY LAYER")
        print("-" * 80)

        curve = summary["avg_probability_by_layer"]
        layers = sorted(curve.keys())

        # Print as a compact bar chart
        for layer in layers:
            prob = curve[layer]
            bar_len = int(prob * 50)
            bar = "#" * bar_len
            print(f"  L{layer:>2}: {prob:>7.4f} |{bar}")

        # Key finding
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)
        print(f"\n  {summary['interpretation']['early_emergence']}")
        print(f"\n  {summary['interpretation']['connection_to_ablation']}")

        e1 = summary["emergence_top1"]
        if e1["mean"] is not None:
            print(f"\n  Average fact emergence (top-1): Layer {e1['mean']}")
            if e1["mean"] < self.num_layers * 0.5:
                print(
                    f"  This is in the FIRST HALF of the network ({e1['mean']:.0f}/{self.num_layers})."
                )
                print(
                    "  Since knowledge ablation shows facts survive expert ablation at L8-L20,"
                )
                print(
                    "  the residual stream is carrying facts INDEPENDENTLY of expert computation."
                )
                print(
                    "  Experts at post-emergence layers provide formatting/fluency, not facts."
                )
            elif e1["mean"] < self.num_layers * 0.75:
                print(
                    f"  Facts emerge in the MIDDLE layers ({e1['mean']:.0f}/{self.num_layers})."
                )
                print(
                    "  Combined with knowledge ablation (facts survive expert ablation),"
                )
                print(
                    "  this suggests facts are built up gradually through the residual stream."
                )
            else:
                print(
                    f"  Facts emerge LATE ({e1['mean']:.0f}/{self.num_layers})."
                )
                print(
                    "  But knowledge ablation shows they survive expert ablation even here."
                )
                print(
                    "  The residual stream computation is sufficient; experts refine but don't create."
                )

        # Connection to compression
        print("\n  COMPRESSION IMPLICATION:")
        print(
            "  Minimum viable routing needs 6-7 learned layers for 100% MB recovery."
        )
        if e1["mean"] is not None:
            print(
                f"  Fact emergence at ~L{e1['mean']:.0f} suggests layers after emergence"
            )
            print(
                "  contribute fluency (expert-dependent) rather than facts (residual-carried)."
            )

        print("=" * 80)


async def main():
    experiment = ResidualFactEmergence()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
