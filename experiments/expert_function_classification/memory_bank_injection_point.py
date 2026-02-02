#!/usr/bin/env python3
"""Memory Bank Injection Point Experiment.

Prior experiments established:
  - Facts crystallize at L20-21 in the residual stream (logit lens)
  - L20-21 are important but not irreplaceable (5/7 facts survive skip)
  - Attention focuses on entity at L19/L21 (1.3x increase)
  - Memory bank injection + 6-7 learned layers = 100% fact preservation

This experiment answers: WHERE does memory bank injection influence the
residual stream? When we prepend [Memory Bank] facts to a prompt, does
the fact appear EARLIER in the residual stream (bypassing the L20-21
computation), or does it simply reinforce what the model already computes?

Method:
  For each fact, we run two conditions:
  1. BARE: just the prompt ("The capital of France is")
  2. MB: memory bank + prompt ("[Memory Bank]\n- France | capital | Paris\n...")

  At each layer, we capture hidden states (last position) and compare:
  - Cosine distance between bare and MB hidden states
  - Fact token probability (logit lens) under each condition
  - Probability lift (MB - bare) at each layer
  - Emergence point shift (does MB cause earlier emergence?)

Predictions:
  - MB should cause earlier fact emergence (facts enter via attention at L0-L4)
  - Residual delta should peak at early-to-mid layers where MB context is encoded
  - By L20+, bare and MB residual streams should converge (both carry the fact)

Run: python experiments/expert_function_classification/memory_bank_injection_point.py
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
    {
        "prompt": "The capital of France is",
        "expected_keyword": "Paris",
        "mb_entry": "France | capital | Paris",
    },
    {
        "prompt": "The chemical symbol for gold is",
        "expected_keyword": "Au",
        "mb_entry": "Gold | chemical symbol | Au",
    },
    {
        "prompt": "The author of Romeo and Juliet is",
        "expected_keyword": "Shakespeare",
        "mb_entry": "Romeo and Juliet | author | William Shakespeare",
    },
    {
        "prompt": "The CEO of Microsoft is",
        "expected_keyword": "Nadella",
        "mb_entry": "Microsoft | CEO | Satya Nadella",
    },
    {
        "prompt": "The capital of Japan is",
        "expected_keyword": "Tokyo",
        "mb_entry": "Japan | capital | Tokyo",
    },
    {
        "prompt": "The chemical symbol for silver is",
        "expected_keyword": "Ag",
        "mb_entry": "Silver | chemical symbol | Ag",
    },
    {
        "prompt": "The capital of Australia is",
        "expected_keyword": "Canberra",
        "mb_entry": "Australia | capital | Canberra",
    },
]


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


class MemoryBankInjectionPoint:
    """Compare residual streams between bare and memory-bank-injected prompts."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded: {self.num_layers} layers. Ready.")

    def _discover_fact_token(self, prompt: str) -> tuple[int, str]:
        """Get the model's actual predicted next token for this prompt."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        output = self.model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        next_token_id = int(mx.argmax(logits[0, -1, :]).item())
        decoded = self.tokenizer.decode([next_token_id])
        return next_token_id, decoded

    def _run_logit_lens(self, prompt: str) -> dict[str, Any]:
        """Run logit lens at all layers. Returns hooks, lens, and hidden states."""
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks
        from chuk_lazarus.introspection.logit_lens import LogitLens

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                capture_pre_norm=True,
                positions="last",
            )
        )
        hooks.forward(input_ids)

        lens = LogitLens(hooks, self.tokenizer)
        return {
            "hooks": hooks,
            "lens": lens,
            "input_ids": input_ids,
        }

    def _track_token(
        self, lens_result: dict, token_id: int
    ) -> dict[str, Any]:
        """Track a token's probability and rank through all layers."""
        from chuk_lazarus.introspection.logit_lens import LogitLens

        lens: LogitLens = lens_result["lens"]
        evolution = lens.track_token(token_id, position=-1, top_k_for_rank=200)

        emergence_top1 = evolution.emergence_layer
        emergence_top5 = None
        threshold_10pct = None
        threshold_50pct = None

        for layer, rank, prob in zip(
            evolution.layers, evolution.ranks, evolution.probabilities
        ):
            if emergence_top5 is None and rank is not None and rank <= 5:
                emergence_top5 = layer
            if threshold_10pct is None and prob >= 0.10:
                threshold_10pct = layer
            if threshold_50pct is None and prob >= 0.50:
                threshold_50pct = layer

        return {
            "layers": evolution.layers,
            "probabilities": evolution.probabilities,
            "ranks": evolution.ranks,
            "emergence_top1": emergence_top1,
            "emergence_top5": emergence_top5,
            "threshold_10pct": threshold_10pct,
            "threshold_50pct": threshold_50pct,
        }

    def _extract_hidden_states(self, hooks) -> dict[int, mx.array]:
        """Extract hidden states at each layer from hooks.

        Returns {layer_idx: hidden_state} where hidden_state is 1D (hidden_dim).
        Uses hooks.state.get_hidden_at_position() for clean extraction.
        """
        states = {}
        for layer_idx in hooks.state.captured_layers:
            hs = hooks.state.get_hidden_at_position(layer_idx, position=-1)
            if hs is not None:
                states[layer_idx] = hs.reshape(-1)
        return states

    def _cosine_distance(self, a: mx.array, b: mx.array) -> float:
        """Compute cosine distance (1 - cosine_similarity) between two vectors."""
        dot = float(mx.sum(a * b).item())
        norm_a = float(mx.sqrt(mx.sum(a * a)).item())
        norm_b = float(mx.sqrt(mx.sum(b * b)).item())
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity

    def _l2_distance(self, a: mx.array, b: mx.array) -> float:
        """Compute L2 distance between two vectors."""
        diff = a - b
        return float(mx.sqrt(mx.sum(diff * diff)).item())

    async def analyze_fact(self, fact: dict) -> dict[str, Any]:
        """Compare bare vs MB residual streams for one fact."""
        prompt = fact["prompt"]
        keyword = fact["expected_keyword"]
        mb_entry = fact["mb_entry"]

        logger.info(f"\n  Fact: {prompt}")

        # Build prompts
        bare_prompt = prompt
        all_mb_entries = [f["mb_entry"] for f in FACTS]
        mb_prompt = build_memory_bank_prompt(prompt, all_mb_entries)

        logger.info(f"    Bare prompt length: {len(self.tokenizer.encode(bare_prompt))} tokens")
        logger.info(f"    MB prompt length: {len(self.tokenizer.encode(mb_prompt))} tokens")

        # Discover fact tokens for each condition
        bare_token_id, bare_token_str = self._discover_fact_token(bare_prompt)
        mb_token_id, mb_token_str = self._discover_fact_token(mb_prompt)

        logger.info(f"    Bare predicts: '{bare_token_str}' (id={bare_token_id})")
        logger.info(f"    MB predicts:   '{mb_token_str}' (id={mb_token_id})")

        # Use the bare token as the tracking target (consistent across conditions)
        track_token_id = bare_token_id
        track_token_str = bare_token_str

        # Run logit lens for both conditions
        bare_lens = self._run_logit_lens(bare_prompt)
        mb_lens = self._run_logit_lens(mb_prompt)

        # Track the fact token through both residual streams
        bare_tracking = self._track_token(bare_lens, track_token_id)
        mb_tracking = self._track_token(mb_lens, track_token_id)

        logger.info(
            f"    Bare emergence: top5@L{bare_tracking['emergence_top5']}, "
            f"top1@L{bare_tracking['emergence_top1']}, "
            f">50%@L{bare_tracking['threshold_50pct']}"
        )
        logger.info(
            f"    MB emergence:   top5@L{mb_tracking['emergence_top5']}, "
            f"top1@L{mb_tracking['emergence_top1']}, "
            f">50%@L{mb_tracking['threshold_50pct']}"
        )

        # Extract hidden states and compute distances
        bare_states = self._extract_hidden_states(bare_lens["hooks"])
        mb_states = self._extract_hidden_states(mb_lens["hooks"])

        cosine_distances = {}
        l2_distances = {}

        for layer_idx in sorted(bare_states.keys()):
            if layer_idx in mb_states:
                cos_d = self._cosine_distance(bare_states[layer_idx], mb_states[layer_idx])
                l2_d = self._l2_distance(bare_states[layer_idx], mb_states[layer_idx])
                cosine_distances[layer_idx] = cos_d
                l2_distances[layer_idx] = l2_d

        # Compute probability lift at each layer
        probability_lift = {}
        for i, layer in enumerate(bare_tracking["layers"]):
            bare_prob = bare_tracking["probabilities"][i]
            # Find corresponding MB probability
            mb_prob = 0.0
            if layer in mb_tracking["layers"]:
                mb_idx = mb_tracking["layers"].index(layer)
                mb_prob = mb_tracking["probabilities"][mb_idx]
            probability_lift[layer] = mb_prob - bare_prob

        # Log key layers
        for l in [0, 4, 8, 12, 15, 19, 20, 21, 22, 23]:
            if l in cosine_distances:
                bare_p = bare_tracking["probabilities"][bare_tracking["layers"].index(l)] if l in bare_tracking["layers"] else 0
                mb_p = mb_tracking["probabilities"][mb_tracking["layers"].index(l)] if l in mb_tracking["layers"] else 0
                lift = probability_lift.get(l, 0)
                logger.info(
                    f"    L{l:>2}: cos_dist={cosine_distances[l]:.4f}, "
                    f"bare={bare_p:.4f}, mb={mb_p:.4f}, lift={lift:+.4f}"
                )

        # Emergence shift
        bare_top1 = bare_tracking["emergence_top1"]
        mb_top1 = mb_tracking["emergence_top1"]
        emergence_shift = None
        if bare_top1 is not None and mb_top1 is not None:
            emergence_shift = bare_top1 - mb_top1  # positive = MB is earlier

        return {
            "prompt": prompt,
            "expected_keyword": keyword,
            "bare_token": bare_token_str,
            "mb_token": mb_token_str,
            "track_token": track_token_str,
            "track_token_id": track_token_id,
            "bare_emergence": {
                "top5": bare_tracking["emergence_top5"],
                "top1": bare_tracking["emergence_top1"],
                "threshold_10pct": bare_tracking["threshold_10pct"],
                "threshold_50pct": bare_tracking["threshold_50pct"],
            },
            "mb_emergence": {
                "top5": mb_tracking["emergence_top5"],
                "top1": mb_tracking["emergence_top1"],
                "threshold_10pct": mb_tracking["threshold_10pct"],
                "threshold_50pct": mb_tracking["threshold_50pct"],
            },
            "emergence_shift_top1": emergence_shift,
            "cosine_distance_by_layer": {
                str(k): round(v, 6) for k, v in cosine_distances.items()
            },
            "l2_distance_by_layer": {
                str(k): round(v, 4) for k, v in l2_distances.items()
            },
            "bare_probability_by_layer": {
                str(l): round(p, 6)
                for l, p in zip(bare_tracking["layers"], bare_tracking["probabilities"])
            },
            "mb_probability_by_layer": {
                str(l): round(p, 6)
                for l, p in zip(mb_tracking["layers"], mb_tracking["probabilities"])
            },
            "probability_lift_by_layer": {
                str(k): round(v, 6) for k, v in probability_lift.items()
            },
        }

    def _compute_summary(self, fact_results: list[dict]) -> dict[str, Any]:
        """Compute aggregate statistics across all facts."""
        valid = [r for r in fact_results if "error" not in r]

        # Average cosine distance by layer
        avg_cosine: dict[int, list[float]] = defaultdict(list)
        avg_lift: dict[int, list[float]] = defaultdict(list)
        avg_bare_prob: dict[int, list[float]] = defaultdict(list)
        avg_mb_prob: dict[int, list[float]] = defaultdict(list)

        for r in valid:
            for layer_str, val in r["cosine_distance_by_layer"].items():
                avg_cosine[int(layer_str)].append(val)
            for layer_str, val in r["probability_lift_by_layer"].items():
                avg_lift[int(layer_str)].append(val)
            for layer_str, val in r["bare_probability_by_layer"].items():
                avg_bare_prob[int(layer_str)].append(val)
            for layer_str, val in r["mb_probability_by_layer"].items():
                avg_mb_prob[int(layer_str)].append(val)

        avg_cosine_curve = {
            l: round(sum(v) / len(v), 6) for l, v in sorted(avg_cosine.items())
        }
        avg_lift_curve = {
            l: round(sum(v) / len(v), 6) for l, v in sorted(avg_lift.items())
        }
        avg_bare_curve = {
            l: round(sum(v) / len(v), 6) for l, v in sorted(avg_bare_prob.items())
        }
        avg_mb_curve = {
            l: round(sum(v) / len(v), 6) for l, v in sorted(avg_mb_prob.items())
        }

        # Peak cosine distance layer
        if avg_cosine_curve:
            peak_cos_layer = max(avg_cosine_curve, key=avg_cosine_curve.get)
            peak_cos_value = avg_cosine_curve[peak_cos_layer]
        else:
            peak_cos_layer = None
            peak_cos_value = None

        # Peak probability lift layer
        if avg_lift_curve:
            peak_lift_layer = max(avg_lift_curve, key=avg_lift_curve.get)
            peak_lift_value = avg_lift_curve[peak_lift_layer]
        else:
            peak_lift_layer = None
            peak_lift_value = None

        # Emergence shifts
        shifts_top1 = [r["emergence_shift_top1"] for r in valid if r["emergence_shift_top1"] is not None]
        bare_top1s = [r["bare_emergence"]["top1"] for r in valid if r["bare_emergence"]["top1"] is not None]
        mb_top1s = [r["mb_emergence"]["top1"] for r in valid if r["mb_emergence"]["top1"] is not None]

        avg_shift = round(sum(shifts_top1) / len(shifts_top1), 1) if shifts_top1 else None
        avg_bare_emergence = round(sum(bare_top1s) / len(bare_top1s), 1) if bare_top1s else None
        avg_mb_emergence = round(sum(mb_top1s) / len(mb_top1s), 1) if mb_top1s else None

        # Phase averages for cosine distance
        early_cos = [avg_cosine_curve.get(l, 0) for l in range(0, 10)]
        mid_cos = [avg_cosine_curve.get(l, 0) for l in range(10, 18)]
        late_cos = [avg_cosine_curve.get(l, 0) for l in range(18, 24)]

        avg_early_cos = sum(early_cos) / len(early_cos) if early_cos else 0
        avg_mid_cos = sum(mid_cos) / len(mid_cos) if mid_cos else 0
        avg_late_cos = sum(late_cos) / len(late_cos) if late_cos else 0

        return {
            "num_facts": len(valid),
            "emergence_comparison": {
                "avg_bare_top1": avg_bare_emergence,
                "avg_mb_top1": avg_mb_emergence,
                "avg_shift": avg_shift,
                "per_fact_shifts": shifts_top1,
                "interpretation": (
                    f"MB shifts emergence by {avg_shift:+.1f} layers on average "
                    f"(bare: L{avg_bare_emergence}, MB: L{avg_mb_emergence})"
                    if avg_shift is not None else "No shift data"
                ),
            },
            "residual_distance": {
                "avg_cosine_by_layer": avg_cosine_curve,
                "peak_cosine_layer": peak_cos_layer,
                "peak_cosine_value": peak_cos_value,
                "phase_averages": {
                    "early_L0_L9": round(avg_early_cos, 6),
                    "mid_L10_L17": round(avg_mid_cos, 6),
                    "late_L18_L23": round(avg_late_cos, 6),
                },
            },
            "probability_lift": {
                "avg_lift_by_layer": avg_lift_curve,
                "peak_lift_layer": peak_lift_layer,
                "peak_lift_value": peak_lift_value,
            },
            "avg_bare_probability": avg_bare_curve,
            "avg_mb_probability": avg_mb_curve,
        }

    def _print_summary(self, summary: dict, fact_results: list[dict]):
        valid = [r for r in fact_results if "error" not in r]

        print("\n" + "=" * 100)
        print("MEMORY BANK INJECTION POINT - RESULTS")
        print("=" * 100)

        # Per-fact emergence comparison
        print(
            f"\n{'Prompt':<36} | {'Bare':>4} | {'MB':>4} | "
            f"{'Token':<10} | {'MB Token':<10} | {'Shift':>5}"
        )
        print("-" * 100)

        for r in valid:
            prompt_short = r["prompt"][:34]
            bare_top1 = f"L{r['bare_emergence']['top1']}" if r["bare_emergence"]["top1"] is not None else "-"
            mb_top1 = f"L{r['mb_emergence']['top1']}" if r["mb_emergence"]["top1"] is not None else "-"
            shift = r["emergence_shift_top1"]
            shift_str = f"{shift:+d}" if shift is not None else "-"
            print(
                f"{prompt_short:<36} | {bare_top1:>4} | {mb_top1:>4} | "
                f"{r['bare_token']:<10} | {r['mb_token']:<10} | {shift_str:>5}"
            )

        # Emergence summary
        ec = summary["emergence_comparison"]
        print(f"\n  Average bare emergence (top-1): L{ec['avg_bare_top1']}")
        print(f"  Average MB emergence (top-1):   L{ec['avg_mb_top1']}")
        print(f"  Average shift: {ec['avg_shift']:+.1f} layers")

        # Cosine distance curve
        print("\n" + "-" * 100)
        print("RESIDUAL STREAM DISTANCE (bare vs MB) - Cosine Distance by Layer")
        print("-" * 100)

        cos_curve = summary["residual_distance"]["avg_cosine_by_layer"]
        max_cos = max(cos_curve.values()) if cos_curve else 1
        for layer in sorted(cos_curve.keys()):
            val = cos_curve[layer]
            bar_len = int((val / max(max_cos, 0.001)) * 40)
            bar = "#" * bar_len
            marker = ""
            if layer == summary["residual_distance"]["peak_cosine_layer"]:
                marker = "  <- peak divergence"
            print(f"  L{layer:>2}: {val:>8.5f} |{bar}{marker}")

        phases = summary["residual_distance"]["phase_averages"]
        print(f"\n  Phase averages:")
        print(f"    Early  (L0-L9):   {phases['early_L0_L9']:.5f}")
        print(f"    Mid    (L10-L17): {phases['mid_L10_L17']:.5f}")
        print(f"    Late   (L18-L23): {phases['late_L18_L23']:.5f}")

        # Probability comparison
        print("\n" + "-" * 100)
        print("FACT PROBABILITY: BARE vs MB")
        print("-" * 100)

        bare_curve = summary["avg_bare_probability"]
        mb_curve = summary["avg_mb_probability"]
        lift_curve = summary["probability_lift"]["avg_lift_by_layer"]

        print(f"  {'Layer':>5} | {'Bare':>8} | {'MB':>8} | {'Lift':>8} | {'Bare':15} | {'MB':15}")
        print("  " + "-" * 90)

        for layer in sorted(bare_curve.keys()):
            bare_p = bare_curve.get(layer, 0)
            mb_p = mb_curve.get(layer, 0)
            lift = lift_curve.get(layer, 0)
            bare_bar = "#" * int(bare_p * 15)
            mb_bar = "#" * int(mb_p * 15)
            print(
                f"  L{layer:>3}: {bare_p:>8.4f} | {mb_p:>8.4f} | {lift:>+8.4f} | "
                f"{bare_bar:<15} | {mb_bar:<15}"
            )

        # Peak lift
        pl = summary["probability_lift"]
        print(f"\n  Peak probability lift: L{pl['peak_lift_layer']} ({pl['peak_lift_value']:+.4f})")

        # Key findings
        print("\n" + "=" * 100)
        print("KEY FINDINGS")
        print("=" * 100)

        shift = ec["avg_shift"]
        if shift is not None and shift > 2:
            print(
                f"\n  MB shifts fact emergence {shift:+.1f} layers EARLIER."
            )
            print(
                "  Memory bank injection bypasses the normal L20-21 crystallization,"
            )
            print(
                "  providing facts via attention at earlier layers."
            )
        elif shift is not None and shift > 0:
            print(
                f"\n  MB shifts fact emergence modestly earlier ({shift:+.1f} layers)."
            )
            print(
                "  Memory bank provides some acceleration but doesn't fundamentally"
            )
            print(
                "  change the computation pathway."
            )
        elif shift is not None and abs(shift) <= 0.5:
            print(
                f"\n  MB does NOT shift emergence ({shift:+.1f} layers)."
            )
            print(
                "  Facts emerge at the same layers regardless of memory bank."
            )
            print(
                "  MB may work by reinforcing the residual stream rather than"
            )
            print(
                "  providing an alternative pathway."
            )
        elif shift is not None:
            print(
                f"\n  MB shifts emergence LATER ({shift:+.1f} layers) -- unexpected."
            )
            print(
                "  The memory bank may introduce interference or distraction"
            )
            print(
                "  that delays fact crystallization."
            )

        peak_cos = summary["residual_distance"]["peak_cosine_layer"]
        if peak_cos is not None:
            print(
                f"\n  Residual streams diverge most at L{peak_cos}."
            )
            if peak_cos < 10:
                print(
                    "  Early divergence: MB context is encoded in early layers,"
                )
                print(
                    "  then the representations converge as the model processes."
                )
            elif peak_cos < 18:
                print(
                    "  Mid-network divergence: MB influence peaks during the"
                )
                print(
                    "  representational buildup phase."
                )
            else:
                print(
                    "  Late divergence: MB influence peaks at the crystallization"
                )
                print(
                    "  layers, suggesting it modifies the final computation."
                )

        print("=" * 100)

    def _save_results(self, results: dict) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"memory_bank_injection_point_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
        return output_path

    async def run(self):
        await self.setup()

        logger.info("=" * 70)
        logger.info("MEMORY BANK INJECTION POINT")
        logger.info(f"  Facts: {len(FACTS)}")
        logger.info(f"  Conditions: bare vs memory bank")
        logger.info(f"  Measures: cosine distance, probability lift, emergence shift")
        logger.info("=" * 70)

        fact_results = []
        for fact in FACTS:
            result = await self.analyze_fact(fact)
            fact_results.append(result)

        summary = self._compute_summary(fact_results)

        output = {
            "metadata": {
                "experiment": "memory_bank_injection_point",
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "num_facts": len(FACTS),
                "description": (
                    "Compares residual streams between bare prompts and "
                    "memory-bank-injected prompts at each layer. Measures "
                    "cosine distance, fact probability lift, and emergence "
                    "point shift to determine WHERE memory bank injection "
                    "influences the computation."
                ),
                "method": (
                    "For each fact, runs logit lens under two conditions: "
                    "(1) bare prompt, (2) [Memory Bank]...[End Memory Bank] + prompt. "
                    "Captures hidden states at all 24 layers (last position) and "
                    "computes cosine distance between bare/MB representations, "
                    "plus tracks fact token probability under both conditions."
                ),
                "prior_results": {
                    "fact_emergence_avg_top1": "L20.8 (bare, logit lens)",
                    "layer_skip_L20_L21": "5/7 facts survive",
                    "entity_attention_increase": "1.3x at emergence",
                    "min_viable_routing_with_mb": "6-7 layers + MB = 100%",
                },
            },
            "summary": summary,
            "fact_results": fact_results,
        }

        self._save_results(output)
        self._print_summary(summary, fact_results)


async def main():
    experiment = MemoryBankInjectionPoint()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
