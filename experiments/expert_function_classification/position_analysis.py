#!/usr/bin/env python3
"""Sequence-Position Expert Analysis.

Prior experiments show experts don't store facts (they're in the residual
stream) but L16E4 has 3x the routing disruption of other experts. WHY?

Hypothesis: experts specialize by **token position** within the sequence,
not by content. If "The capital of France is" and "The capital of Japan is"
route to the same experts at the same positions, experts are position-coded.
If they route differently, experts are content-coded.

This experiment:
1. Captures complete routing maps (all positions, all layers)
2. Computes position selectivity per expert
3. Measures cross-prompt consistency at matched structural positions
4. Ablates the most position-selective experts to test causal impact

Run: python experiments/expert_function_classification/position_analysis.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
# Prompts: grouped by structure for cross-prompt comparison
# =============================================================================

PROMPT_GROUPS = {
    "capital_of_X": [
        {"text": "The capital of France is", "expected": "Paris"},
        {"text": "The capital of Japan is", "expected": "Tokyo"},
        {"text": "The capital of Australia is", "expected": "Canberra"},
        {"text": "The capital of Germany is", "expected": "Berlin"},
    ],
    "symbol_for_X": [
        {"text": "The chemical symbol for gold is", "expected": "Au"},
        {"text": "The chemical symbol for silver is", "expected": "Ag"},
        {"text": "The chemical symbol for iron is", "expected": "Fe"},
    ],
    "X_of_Y_is": [
        {"text": "The author of Romeo and Juliet is", "expected": "Shakespeare"},
        {"text": "The CEO of Microsoft is", "expected": "Nadella"},
        {"text": "The speed of light is approximately", "expected": "299"},
    ],
    "alt_structure": [
        # Same content as capital_of_X[0], different structure
        {"text": "France's capital city is", "expected": "Paris"},
        {"text": "What is the capital of France?", "expected": "Paris"},
        {"text": "Paris is the capital of", "expected": "France"},
    ],
    "different_types": [
        {"text": "Once upon a time there was a", "expected": None},
        {"text": "The opposite of hot is", "expected": "cold"},
        {"text": "If all cats are mammals then Fluffy is a", "expected": "mammal"},
    ],
}

TARGET_LAYERS = [8, 12, 16, 20]


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class PositionRouting:
    """Expert routing at one token position."""
    position_idx: int
    token: str
    expert_indices: list[int]
    weights: list[float]


@dataclass
class PromptRouting:
    """Full routing map for one prompt at one layer."""
    prompt: str
    layer_idx: int
    positions: list[PositionRouting]


@dataclass
class ExpertSelectivity:
    """Position selectivity score for one expert at one layer."""
    layer_idx: int
    expert_idx: int
    total_activations: int
    position_entropy: float  # Low = position specialist
    max_entropy: float  # Maximum possible entropy
    selectivity: float  # 1 - (entropy / max_entropy); 1=specialist, 0=generalist
    preferred_region: str  # "start", "middle", "end", "last_token"
    last_token_fraction: float  # Fraction of activations at the last token


@dataclass
class CrossPromptOverlap:
    """Expert overlap between two prompts at matched positions."""
    group_name: str
    prompt_a: str
    prompt_b: str
    layer_idx: int
    position_overlaps: list[dict]  # [{pos, overlap_count, total_slots}]
    mean_overlap: float  # Average Jaccard index across matched positions
    last_token_overlap: float  # Jaccard at the last token specifically


# =============================================================================
# Experiment
# =============================================================================


class PositionAnalysis:
    """Analyze whether MoE experts specialize by token position."""

    def __init__(self):
        self.router = None
        self.model = None
        self.tokenizer = None

    async def setup(self):
        from chuk_lazarus.introspection.moe import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        self.router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = self.router._model
        self.tokenizer = self.router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        info = self.router._info
        logger.info(
            f"Model loaded: {info.num_experts} experts, "
            f"{len(info.moe_layers)} MoE layers, "
            f"top-{info.num_experts_per_tok}"
        )

    def _generate(self, prompt: str, max_tokens: int = 30) -> str:
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

    # -----------------------------------------------------------------
    # Phase 1: Capture routing
    # -----------------------------------------------------------------

    async def _capture_prompt_routing(
        self, prompt: str
    ) -> dict[int, PromptRouting]:
        """Capture expert routing at all positions for one prompt."""
        weights_list = await self.router.capture_router_weights(
            prompt, layers=TARGET_LAYERS
        )

        result = {}
        for lw in weights_list:
            positions = []
            for pos in lw.positions:
                positions.append(PositionRouting(
                    position_idx=pos.position_idx,
                    token=pos.token,
                    expert_indices=list(pos.expert_indices),
                    weights=list(pos.weights),
                ))
            result[lw.layer_idx] = PromptRouting(
                prompt=prompt,
                layer_idx=lw.layer_idx,
                positions=positions,
            )
        return result

    # -----------------------------------------------------------------
    # Phase 2: Position selectivity
    # -----------------------------------------------------------------

    def _compute_selectivity(
        self,
        all_routing: dict[str, dict[int, PromptRouting]],
    ) -> dict[int, list[ExpertSelectivity]]:
        """Compute position selectivity for each expert at each layer."""
        results: dict[int, list[ExpertSelectivity]] = {}

        for layer_idx in TARGET_LAYERS:
            # Collect: for each expert, which relative positions does it appear at?
            expert_positions: dict[int, list[float]] = defaultdict(list)
            expert_last_token: dict[int, int] = Counter()
            expert_total: dict[int, int] = Counter()

            for prompt_text, layer_routing in all_routing.items():
                if layer_idx not in layer_routing:
                    continue
                routing = layer_routing[layer_idx]
                n_positions = len(routing.positions)
                if n_positions == 0:
                    continue

                for pos in routing.positions:
                    rel_pos = pos.position_idx / max(n_positions - 1, 1)
                    is_last = pos.position_idx == n_positions - 1

                    for expert_idx in pos.expert_indices:
                        expert_positions[expert_idx].append(rel_pos)
                        expert_total[expert_idx] += 1
                        if is_last:
                            expert_last_token[expert_idx] += 1

            # Compute entropy for each expert's position distribution
            n_bins = 4  # [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
            max_entropy = math.log2(n_bins)

            layer_selectivity = []
            for expert_idx in range(32):
                positions = expert_positions.get(expert_idx, [])
                total = expert_total.get(expert_idx, 0)

                if total == 0:
                    layer_selectivity.append(ExpertSelectivity(
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        total_activations=0,
                        position_entropy=max_entropy,
                        max_entropy=max_entropy,
                        selectivity=0.0,
                        preferred_region="none",
                        last_token_fraction=0.0,
                    ))
                    continue

                # Bin positions
                bins = [0] * n_bins
                for p in positions:
                    bin_idx = min(int(p * n_bins), n_bins - 1)
                    bins[bin_idx] += 1

                # Compute entropy
                entropy = 0.0
                for count in bins:
                    if count > 0:
                        prob = count / total
                        entropy -= prob * math.log2(prob)

                selectivity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

                # Find preferred region
                region_names = ["start", "early_mid", "late_mid", "end"]
                preferred = region_names[bins.index(max(bins))]

                last_frac = expert_last_token.get(expert_idx, 0) / total

                layer_selectivity.append(ExpertSelectivity(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    total_activations=total,
                    position_entropy=entropy,
                    max_entropy=max_entropy,
                    selectivity=selectivity,
                    preferred_region=preferred,
                    last_token_fraction=last_frac,
                ))

            results[layer_idx] = sorted(
                layer_selectivity, key=lambda s: s.selectivity, reverse=True
            )

        return results

    # -----------------------------------------------------------------
    # Phase 3: Cross-prompt consistency
    # -----------------------------------------------------------------

    def _compute_cross_prompt_overlap(
        self,
        all_routing: dict[str, dict[int, PromptRouting]],
    ) -> list[CrossPromptOverlap]:
        """For same-structure prompt pairs, measure expert overlap at matched positions."""
        results = []

        for group_name, prompts in PROMPT_GROUPS.items():
            texts = [p["text"] for p in prompts]
            # Compare all pairs within group
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    text_a, text_b = texts[i], texts[j]
                    if text_a not in all_routing or text_b not in all_routing:
                        continue

                    for layer_idx in TARGET_LAYERS:
                        routing_a = all_routing[text_a].get(layer_idx)
                        routing_b = all_routing[text_b].get(layer_idx)
                        if not routing_a or not routing_b:
                            continue

                        # Match positions by relative index
                        n_a = len(routing_a.positions)
                        n_b = len(routing_b.positions)
                        n_match = min(n_a, n_b)

                        position_overlaps = []
                        for k in range(n_match):
                            # Compare from the END (last token aligned)
                            pos_a = routing_a.positions[n_a - n_match + k]
                            pos_b = routing_b.positions[n_b - n_match + k]

                            set_a = set(pos_a.expert_indices)
                            set_b = set(pos_b.expert_indices)
                            intersection = len(set_a & set_b)
                            union = len(set_a | set_b)
                            jaccard = intersection / union if union > 0 else 0

                            position_overlaps.append({
                                "pos_from_end": n_match - 1 - k,
                                "token_a": pos_a.token,
                                "token_b": pos_b.token,
                                "experts_a": list(set_a),
                                "experts_b": list(set_b),
                                "overlap": intersection,
                                "jaccard": jaccard,
                            })

                        mean_jaccard = (
                            sum(p["jaccard"] for p in position_overlaps)
                            / len(position_overlaps)
                            if position_overlaps else 0
                        )
                        last_jaccard = (
                            position_overlaps[-1]["jaccard"]
                            if position_overlaps else 0
                        )

                        results.append(CrossPromptOverlap(
                            group_name=group_name,
                            prompt_a=text_a,
                            prompt_b=text_b,
                            layer_idx=layer_idx,
                            position_overlaps=position_overlaps,
                            mean_overlap=mean_jaccard,
                            last_token_overlap=last_jaccard,
                        ))

        return results

    # -----------------------------------------------------------------
    # Phase 4: Position-selective expert ablation
    # -----------------------------------------------------------------

    async def _ablate_selective_experts(
        self,
        selectivity: dict[int, list[ExpertSelectivity]],
    ) -> list[dict]:
        """Ablate the most position-selective experts and test impact."""
        results = []

        # Focus on L16 since we know the most about it
        layer_idx = 16
        experts_by_sel = selectivity.get(layer_idx, [])
        if not experts_by_sel:
            return results

        # Take top-3 most selective and bottom-3 least selective
        top_selective = experts_by_sel[:3]
        bottom_selective = [e for e in experts_by_sel if e.total_activations > 0][-3:]

        test_prompts = [
            {"text": "The capital of France is", "expected": "Paris"},
            {"text": "The chemical symbol for gold is", "expected": "Au"},
            {"text": "The CEO of Microsoft is", "expected": "Nadella"},
            {"text": "Once upon a time there was a", "expected": None},
        ]

        for expert_sel in top_selective + bottom_selective:
            expert_idx = expert_sel.expert_idx
            category = (
                "high_selectivity"
                if expert_sel.selectivity > 0.1
                else "low_selectivity"
            )

            for prompt in test_prompts:
                text, _ = await self.router.generate_with_ablation(
                    prompt["text"],
                    [expert_idx],
                    max_tokens=30,
                    layers=[layer_idx],
                )
                text = text.strip()

                correct = None
                if prompt["expected"]:
                    correct = prompt["expected"].lower() in text.lower()

                results.append({
                    "expert_idx": expert_idx,
                    "expert_id": f"L{layer_idx}E{expert_idx}",
                    "selectivity": expert_sel.selectivity,
                    "preferred_region": expert_sel.preferred_region,
                    "last_token_fraction": expert_sel.last_token_fraction,
                    "category": category,
                    "prompt": prompt["text"],
                    "expected": prompt["expected"],
                    "ablated_text": text[:100],
                    "correct": correct,
                })

                status = ""
                if correct is not None:
                    status = "OK" if correct else "LOST"
                logger.info(
                    f"    L{layer_idx}E{expert_idx} "
                    f"(sel={expert_sel.selectivity:.2f}, "
                    f"pref={expert_sel.preferred_region:>9}) | "
                    f"{prompt['text'][:30]:30} | {status:4} | "
                    f"{text[:40]}"
                )

        return results

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        await self.setup()

        # Flatten all prompts
        all_prompts = []
        for group_prompts in PROMPT_GROUPS.values():
            for p in group_prompts:
                all_prompts.append(p)

        # =====================================================================
        # Phase 1: Capture routing
        # =====================================================================
        logger.info("\nPhase 1: Capturing routing for all prompts")
        all_routing: dict[str, dict[int, PromptRouting]] = {}

        for i, prompt in enumerate(all_prompts):
            text = prompt["text"]
            routing = await self._capture_prompt_routing(text)
            all_routing[text] = routing

            # Show tokens and L16 experts at last position
            l16 = routing.get(16)
            if l16 and l16.positions:
                tokens = [p.token for p in l16.positions]
                last_experts = l16.positions[-1].expert_indices
                logger.info(
                    f"  [{i+1:2}/{len(all_prompts)}] "
                    f"{text[:45]:45} | "
                    f"tokens={len(tokens)} | "
                    f"L16 last={last_experts}"
                )

        # =====================================================================
        # Phase 2: Position selectivity
        # =====================================================================
        logger.info("\nPhase 2: Computing position selectivity")
        selectivity = self._compute_selectivity(all_routing)

        for layer_idx in TARGET_LAYERS:
            layer_sel = selectivity[layer_idx]
            top5 = layer_sel[:5]
            logger.info(f"\n  Layer {layer_idx} - Top 5 position-selective experts:")
            for s in top5:
                logger.info(
                    f"    E{s.expert_idx:>2}: selectivity={s.selectivity:.3f} "
                    f"entropy={s.position_entropy:.2f}/{s.max_entropy:.2f} "
                    f"pref={s.preferred_region:>9} "
                    f"last_token={s.last_token_fraction:.2f} "
                    f"activations={s.total_activations}"
                )

        # =====================================================================
        # Phase 3: Cross-prompt consistency
        # =====================================================================
        logger.info("\nPhase 3: Cross-prompt position consistency")
        overlaps = self._compute_cross_prompt_overlap(all_routing)

        # Summarize by group and layer
        group_layer_overlaps: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        group_layer_last: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for o in overlaps:
            group_layer_overlaps[o.group_name][o.layer_idx].append(
                o.mean_overlap
            )
            group_layer_last[o.group_name][o.layer_idx].append(
                o.last_token_overlap
            )

        logger.info("\n  Mean Jaccard overlap (same structure, different content):")
        logger.info(f"  {'Group':>20} | {'L8':>6} | {'L12':>6} | {'L16':>6} | {'L20':>6}")
        logger.info(f"  {'-'*20}-|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}")
        for group_name in PROMPT_GROUPS:
            vals = []
            for layer_idx in TARGET_LAYERS:
                layer_vals = group_layer_overlaps[group_name].get(layer_idx, [])
                avg = sum(layer_vals) / len(layer_vals) if layer_vals else 0
                vals.append(f"{avg:.3f}")
            logger.info(
                f"  {group_name:>20} | "
                + " | ".join(f"{v:>6}" for v in vals)
            )

        logger.info("\n  Last-token Jaccard overlap:")
        logger.info(f"  {'Group':>20} | {'L8':>6} | {'L12':>6} | {'L16':>6} | {'L20':>6}")
        logger.info(f"  {'-'*20}-|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}")
        for group_name in PROMPT_GROUPS:
            vals = []
            for layer_idx in TARGET_LAYERS:
                layer_vals = group_layer_last[group_name].get(layer_idx, [])
                avg = sum(layer_vals) / len(layer_vals) if layer_vals else 0
                vals.append(f"{avg:.3f}")
            logger.info(
                f"  {group_name:>20} | "
                + " | ".join(f"{v:>6}" for v in vals)
            )

        # =====================================================================
        # Phase 4: Ablation of position-selective experts
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("Phase 4: Ablating position-selective vs non-selective experts")
        logger.info("=" * 70)
        ablation_results = await self._ablate_selective_experts(selectivity)

        # =====================================================================
        # Save and print
        # =====================================================================
        results = self._build_results(
            all_routing, selectivity, overlaps, ablation_results
        )
        self._save_results(results)
        self._print_summary(selectivity, overlaps, ablation_results)
        return results

    def _build_results(
        self,
        all_routing: dict,
        selectivity: dict,
        overlaps: list,
        ablation_results: list,
    ) -> dict[str, Any]:
        # Serialize routing
        routing_data = {}
        for prompt_text, layer_routing in all_routing.items():
            routing_data[prompt_text] = {}
            for layer_idx, pr in layer_routing.items():
                routing_data[prompt_text][str(layer_idx)] = [
                    {
                        "position": p.position_idx,
                        "token": p.token,
                        "experts": p.expert_indices,
                        "weights": p.weights,
                    }
                    for p in pr.positions
                ]

        # Serialize selectivity
        sel_data = {}
        for layer_idx, layer_sel in selectivity.items():
            sel_data[str(layer_idx)] = [
                {
                    "expert_idx": s.expert_idx,
                    "selectivity": s.selectivity,
                    "entropy": s.position_entropy,
                    "preferred_region": s.preferred_region,
                    "last_token_fraction": s.last_token_fraction,
                    "total_activations": s.total_activations,
                }
                for s in layer_sel
            ]

        # Serialize overlaps
        overlap_data = [
            {
                "group": o.group_name,
                "prompt_a": o.prompt_a,
                "prompt_b": o.prompt_b,
                "layer_idx": o.layer_idx,
                "mean_overlap": o.mean_overlap,
                "last_token_overlap": o.last_token_overlap,
                "positions": o.position_overlaps,
            }
            for o in overlaps
        ]

        return {
            "metadata": {
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "experiment": "position_analysis",
                "num_prompts": sum(
                    len(v) for v in PROMPT_GROUPS.values()
                ),
                "target_layers": TARGET_LAYERS,
            },
            "routing": routing_data,
            "selectivity": sel_data,
            "cross_prompt_overlap": overlap_data,
            "ablation": ablation_results,
        }

    def _save_results(self, results: dict) -> None:
        output_path = (
            Path(__file__).parent
            / "results"
            / f"position_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")

    def _print_summary(
        self,
        selectivity: dict[int, list[ExpertSelectivity]],
        overlaps: list[CrossPromptOverlap],
        ablation_results: list[dict],
    ) -> None:
        print("\n" + "=" * 70)
        print("POSITION ANALYSIS - SUMMARY")
        print("=" * 70)

        # Position selectivity summary
        print("\n--- Expert Position Selectivity ---")
        for layer_idx in TARGET_LAYERS:
            layer_sel = selectivity.get(layer_idx, [])
            active = [s for s in layer_sel if s.total_activations > 0]
            avg_sel = (
                sum(s.selectivity for s in active) / len(active)
                if active else 0
            )
            high_sel = sum(1 for s in active if s.selectivity > 0.1)
            last_tok_specialists = sum(
                1 for s in active if s.last_token_fraction > 0.3
            )

            print(
                f"  L{layer_idx}: avg_selectivity={avg_sel:.3f} | "
                f"high_sel(>0.1)={high_sel}/32 | "
                f"last_token_specialists={last_tok_specialists}/32"
            )

            # Top 3
            for s in layer_sel[:3]:
                print(
                    f"    E{s.expert_idx:>2}: sel={s.selectivity:.3f} "
                    f"pref={s.preferred_region:>9} "
                    f"last_tok={s.last_token_fraction:.2f}"
                )

        # Cross-prompt overlap summary
        print("\n--- Cross-Prompt Position Consistency ---")
        print("(Higher = more position-coded, lower = more content-coded)")

        # Aggregate by group type
        same_struct_overlaps = []
        diff_struct_overlaps = []
        for o in overlaps:
            if o.group_name in ("capital_of_X", "symbol_for_X"):
                same_struct_overlaps.append(o.mean_overlap)
            elif o.group_name == "alt_structure":
                diff_struct_overlaps.append(o.mean_overlap)

        if same_struct_overlaps:
            avg = sum(same_struct_overlaps) / len(same_struct_overlaps)
            print(f"  Same structure, different content: {avg:.3f} Jaccard")
        if diff_struct_overlaps:
            avg = sum(diff_struct_overlaps) / len(diff_struct_overlaps)
            print(f"  Different structure, same content:  {avg:.3f} Jaccard")

        if same_struct_overlaps and diff_struct_overlaps:
            s = sum(same_struct_overlaps) / len(same_struct_overlaps)
            d = sum(diff_struct_overlaps) / len(diff_struct_overlaps)
            if s > d + 0.05:
                print(
                    "  → Experts are MORE position/structure-coded than content-coded"
                )
            elif d > s + 0.05:
                print(
                    "  → Experts are MORE content-coded than position/structure-coded"
                )
            else:
                print("  → Mixed: experts respond to both position and content")

        # Last-token overlap
        same_last = []
        diff_last = []
        for o in overlaps:
            if o.group_name in ("capital_of_X", "symbol_for_X"):
                same_last.append(o.last_token_overlap)
            elif o.group_name == "alt_structure":
                diff_last.append(o.last_token_overlap)

        if same_last:
            print(
                f"\n  Last-token overlap (same structure): "
                f"{sum(same_last)/len(same_last):.3f}"
            )
        if diff_last:
            print(
                f"  Last-token overlap (diff structure): "
                f"{sum(diff_last)/len(diff_last):.3f}"
            )

        # Ablation summary
        print("\n--- Position-Selective Expert Ablation (L16) ---")
        if ablation_results:
            high_sel = [r for r in ablation_results if r["category"] == "high_selectivity"]
            low_sel = [r for r in ablation_results if r["category"] == "low_selectivity"]

            for category, results_list in [
                ("High selectivity", high_sel),
                ("Low selectivity", low_sel),
            ]:
                factual = [r for r in results_list if r["correct"] is not None]
                correct = sum(1 for r in factual if r["correct"])
                total = len(factual)
                rate = correct / total if total else 0
                print(
                    f"  {category:>18}: "
                    f"{correct}/{total} facts preserved ({rate:.0%})"
                )

        # Key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        # Determine dominant coding
        if same_struct_overlaps and diff_struct_overlaps:
            s_avg = sum(same_struct_overlaps) / len(same_struct_overlaps)
            d_avg = sum(diff_struct_overlaps) / len(diff_struct_overlaps)

            print(f"  Same-structure overlap:  {s_avg:.3f}")
            print(f"  Diff-structure overlap:  {d_avg:.3f}")

            if s_avg > 0.5:
                print(
                    "\n  FINDING: Experts are strongly position-coded."
                )
                print(
                    "  The same experts fire at the same structural positions "
                    "regardless of content."
                )
                print(
                    "  Expert routing is driven by WHERE a token sits in the "
                    "sequence, not WHAT it represents."
                )
            elif s_avg > d_avg:
                print(
                    "\n  FINDING: Experts show moderate position-coding."
                )
                print(
                    "  Structural position matters more than semantic content "
                    "for expert selection."
                )
            else:
                print(
                    "\n  FINDING: Experts are primarily content-coded."
                )
                print(
                    "  Expert selection depends on what the token represents, "
                    "not where it appears."
                )

        print("=" * 70)


async def main():
    experiment = PositionAnalysis()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
