#!/usr/bin/env python3
"""Attention Head Ablation Experiment.

Hypothesis (Chris): Facts are stored as key-value associations in mid-layer
attention heads (L8-L12). Ablating specific KV head groups should break
factual recall while preserving other capabilities.

GQA architecture of GPT-OSS 20B:
  - 64 query heads, 8 KV heads (8:1 ratio)
  - Each KV head group = 8 query heads sharing the same K,V
  - Head dim = 64, hidden size = 2880

Phases:
  1. Baseline: Generate answers for 8 factual prompts
  2. KV Head Group Scan: For each target layer, ablate each KV group (0-7)
  3. Progressive: Escalate 1→2→4→8 KV groups at most impactful layers
  4. Full Layer: Ablate ALL 64 query heads at single layers

Ablation method: Class-level monkey-patching of GptOssAttention.__call__.
At the target layer, we replace mx.fast.scaled_dot_product_attention with
manual SDPA that applies a head mask to zero out ablated query heads.

Note: Manual SDPA omits attention sinks (learned per-head biases). This is
a minor discrepancy vs. the normal forward pass, but the ablation effect
(zeroing entire heads) is orders of magnitude larger.

Run:
  python experiments/expert_function_classification/attention_head_ablation.py
  python experiments/expert_function_classification/attention_head_ablation.py --layer 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──

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

# Mid-layers (user hypothesis: L8-L12) plus early and late for comparison
TARGET_LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]

MAX_TOKENS = 30
NUM_KV_HEADS = 8
NUM_QUERY_HEADS = 64
GQA_RATIO = NUM_QUERY_HEADS // NUM_KV_HEADS  # 8


# ── Data Structures ──


@dataclass
class HeadAblationResult:
    """Result of ablating one KV head group for one prompt at one layer."""

    layer_idx: int
    kv_head_idx: int
    query_heads_ablated: list[int]
    prompt: str
    expected: str
    baseline_text: str
    ablated_text: str
    fact_preserved: bool
    output_changed: bool


@dataclass
class ProgressiveResult:
    """Result of progressive KV head group ablation."""

    layer_idx: int
    num_kv_groups_ablated: int
    kv_groups_ablated: list[int]
    prompt: str
    expected: str
    baseline_text: str
    ablated_text: str
    fact_preserved: bool


@dataclass
class FullLayerResult:
    """Result of ablating ALL attention heads at one layer."""

    layer_idx: int
    prompt: str
    expected: str
    baseline_text: str
    ablated_text: str
    fact_preserved: bool


@dataclass
class LayerSummary:
    """Summary of head ablation effects at one layer."""

    layer_idx: int
    kv_head_results: dict[int, dict]
    most_impactful_kv_head: int
    max_fact_break_rate: float
    any_fact_broken: bool


# ── Main Experiment ──


class AttentionHeadAblation:
    """Ablate attention heads to find where factual knowledge is retrieved."""

    def __init__(self, target_layers: list[int] | None = None):
        self.target_layers = target_layers or TARGET_LAYERS
        self.router = None
        self.model = None
        self.tokenizer = None
        self.baselines: dict[str, str] = {}
        self.scan_results: list[HeadAblationResult] = []
        self.progressive_results: list[ProgressiveResult] = []
        self.full_layer_results: list[FullLayerResult] = []
        self.layer_summaries: list[LayerSummary] = []

        # Ablation state (controlled via _generate_with_head_ablation)
        self._ablation_target_layer: int | None = None
        self._ablation_query_heads: set[int] = set()
        self._original_attn_call = None
        self._attn_class = None

    async def setup(self):
        """Load model via ExpertRouter."""
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model via ExpertRouter...")
        self.router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = self.router._model
        self.tokenizer = self.router._tokenizer

        # Get attention class for monkey-patching
        self._attn_class = type(self.model.model.layers[0].self_attn)
        self._original_attn_call = self._attn_class.__call__

        # Verify architecture
        attn = self.model.model.layers[0].self_attn
        logger.info(f"  Query heads: {attn.num_heads}")
        logger.info(f"  KV heads: {attn.num_kv_heads}")
        logger.info(f"  Head dim: {attn.head_dim}")
        logger.info(f"  GQA ratio: {attn.num_heads // attn.num_kv_heads}:1")
        logger.info(f"  Layers: {len(self.model.model.layers)}")

    # ── Generation ──

    def _generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        """Generate text token-by-token (greedy, temperature=0)."""
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

    def _generate_with_head_ablation(
        self,
        prompt: str,
        target_layer: int,
        query_heads: list[int],
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Generate with specific query heads ablated at one layer.

        Uses class-level monkey-patching of GptOssAttention.__call__.
        At the target layer, replaces mx.fast.scaled_dot_product_attention
        with manual SDPA + head ablation mask.
        """
        # Set ablation state for the patched __call__
        self._ablation_target_layer = target_layer
        self._ablation_query_heads = set(query_heads)

        experiment = self  # Closure reference
        original_call = self._original_attn_call

        def patched_attn_call(
            attn_self: Any,
            x: mx.array,
            mask: mx.array | str | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            """Attention forward with head ablation at target layer."""
            # Non-target layers: use original forward
            if attn_self.layer_idx != experiment._ablation_target_layer:
                return original_call(attn_self, x, mask, cache)
            if not experiment._ablation_query_heads:
                return original_call(attn_self, x, mask, cache)

            batch, seq_len, _ = x.shape

            # ── QKV projections ──
            q = attn_self.q_proj(x)
            k = attn_self.k_proj(x)
            v = attn_self.v_proj(x)

            # Reshape to (batch, num_heads, seq_len, head_dim)
            q = q.reshape(batch, seq_len, attn_self.num_heads, attn_self.head_dim)
            q = q.transpose(0, 2, 1, 3)  # (batch, 64, seq, 64)
            k = k.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)
            k = k.transpose(0, 2, 1, 3)  # (batch, 8, seq, 64)
            v = v.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)
            v = v.transpose(0, 2, 1, 3)  # (batch, 8, seq, 64)

            # ── RoPE (with cache offset) ──
            if cache is not None:
                offset = cache[0].shape[2]
                q = attn_self.rope(q, offset=offset)
                k = attn_self.rope(k, offset=offset)
            else:
                q = attn_self.rope(q)
                k = attn_self.rope(k)

            # ── KV cache update ──
            if cache is not None:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            new_cache = (k, v)

            # ── Repeat KV heads for manual SDPA ──
            n_rep = attn_self.num_heads // attn_self.num_kv_heads
            if n_rep > 1:
                k_exp = mx.repeat(k, n_rep, axis=1)  # (batch, 64, kv_seq, 64)
                v_exp = mx.repeat(v, n_rep, axis=1)
            else:
                k_exp = k
                v_exp = v

            # ── Manual scaled dot-product attention ──
            attn_weights = (q @ k_exp.transpose(0, 1, 3, 2)) * attn_self.scale

            # Causal mask for prefill (seq_len > 1, no cache)
            if isinstance(mask, str) and mask == "causal":
                causal = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
                causal = causal.astype(attn_weights.dtype)
                attn_weights = attn_weights + causal
            elif mask is not None and not isinstance(mask, str):
                attn_weights = attn_weights + mask
            # mask=None during decode (single token with cache): no mask needed

            attn_weights = mx.softmax(attn_weights, axis=-1)
            attn_output = attn_weights @ v_exp  # (batch, 64, seq, 64)

            # ── HEAD ABLATION: zero out specified query heads ──
            heads_set = experiment._ablation_query_heads
            mask_vals = [
                0.0 if i in heads_set else 1.0 for i in range(attn_self.num_heads)
            ]
            head_mask = mx.array(mask_vals).reshape(1, attn_self.num_heads, 1, 1)
            head_mask = head_mask.astype(attn_output.dtype)
            attn_output = attn_output * head_mask

            # ── Reshape and output projection ──
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
            attn_output = attn_self.o_proj(attn_output)

            return attn_output, new_cache

        try:
            self._attn_class.__call__ = patched_attn_call
            result = self._generate(prompt, max_tokens)
        finally:
            self._attn_class.__call__ = self._original_attn_call

        return result

    # ── Helpers ──

    @staticmethod
    def _kv_group_to_query_heads(kv_head_idx: int) -> list[int]:
        """Convert KV head index to its 8 query head indices."""
        return list(range(kv_head_idx * GQA_RATIO, (kv_head_idx + 1) * GQA_RATIO))

    @staticmethod
    def _check_fact(text: str, expected: str) -> bool:
        """Check if expected keyword appears in generated text."""
        return expected.lower() in text.lower()

    # ── Phase 1: Baselines ──

    async def generate_baselines(self):
        """Generate baseline outputs for all facts."""
        logger.info("=" * 60)
        logger.info("PHASE 1: Generating baselines")
        logger.info("=" * 60)

        loop = asyncio.get_event_loop()
        for fact in FACTS:
            text = await loop.run_in_executor(None, self._generate, fact["prompt"])
            self.baselines[fact["prompt"]] = text
            preserved = self._check_fact(text, fact["expected"])
            logger.info(f"  {fact['prompt']}: {text[:60]}")
            logger.info(f"    Expected '{fact['expected']}': {'PASS' if preserved else 'FAIL'}")

        valid = sum(
            1
            for f in FACTS
            if self._check_fact(self.baselines[f["prompt"]], f["expected"])
        )
        logger.info(f"\nBaseline: {valid}/{len(FACTS)} facts correctly generated")
        return valid

    # ── Phase 2: KV Head Group Scan ──

    async def kv_head_scan(self):
        """Ablate each KV head group at each target layer."""
        n_passes = len(self.target_layers) * NUM_KV_HEADS * len(FACTS)
        logger.info("=" * 60)
        logger.info("PHASE 2: KV Head Group Scan")
        logger.info(f"  Layers: {self.target_layers}")
        logger.info(f"  KV heads per layer: {NUM_KV_HEADS}")
        logger.info(f"  Facts: {len(FACTS)}")
        logger.info(f"  Total generation passes: {n_passes}")
        logger.info("=" * 60)

        loop = asyncio.get_event_loop()

        for layer_idx in self.target_layers:
            logger.info(f"\n  Layer {layer_idx}:")

            for kv_head in range(NUM_KV_HEADS):
                query_heads = self._kv_group_to_query_heads(kv_head)
                facts_broken = 0
                outputs_changed = 0
                broken_facts = []

                for fact in FACTS:
                    baseline = self.baselines[fact["prompt"]]
                    ablated = await loop.run_in_executor(
                        None,
                        self._generate_with_head_ablation,
                        fact["prompt"],
                        layer_idx,
                        query_heads,
                    )
                    mx.eval(mx.zeros(1))  # Force evaluation

                    fact_preserved = self._check_fact(ablated, fact["expected"])
                    output_changed = ablated.strip() != baseline.strip()

                    if not fact_preserved and self._check_fact(
                        baseline, fact["expected"]
                    ):
                        facts_broken += 1
                        broken_facts.append(fact["expected"])
                    if output_changed:
                        outputs_changed += 1

                    self.scan_results.append(
                        HeadAblationResult(
                            layer_idx=layer_idx,
                            kv_head_idx=kv_head,
                            query_heads_ablated=query_heads,
                            prompt=fact["prompt"],
                            expected=fact["expected"],
                            baseline_text=baseline,
                            ablated_text=ablated,
                            fact_preserved=fact_preserved,
                            output_changed=output_changed,
                        )
                    )

                status = f"KV{kv_head}: {facts_broken}/{len(FACTS)} broken"
                if broken_facts:
                    status += f" [{', '.join(broken_facts)}]"
                status += f", {outputs_changed}/{len(FACTS)} changed"
                logger.info(f"    {status}")

    # ── Phase 3: Progressive Ablation ──

    async def progressive_ablation(self):
        """Escalate KV group ablation at layers of interest."""
        logger.info("=" * 60)
        logger.info("PHASE 3: Progressive KV Head Group Ablation")
        logger.info("=" * 60)

        # Find layers with fact breakage from Phase 2
        layer_break_counts: dict[int, int] = defaultdict(int)
        for r in self.scan_results:
            if not r.fact_preserved and self._check_fact(
                r.baseline_text, r.expected
            ):
                layer_break_counts[r.layer_idx] += 1

        if not layer_break_counts:
            # No facts broke: test progressive at hypothesized layers
            logger.info(
                "  No facts broken in Phase 2. "
                "Testing progressive at L8, L10, L12."
            )
            target_progressive = [8, 10, 12]
        else:
            sorted_layers = sorted(layer_break_counts.items(), key=lambda x: -x[1])
            target_progressive = [l for l, _ in sorted_layers[:3]]
            logger.info(f"  Layers with fact breakage: {dict(sorted_layers)}")

        logger.info(f"  Target layers: {target_progressive}")

        loop = asyncio.get_event_loop()
        escalation = [1, 2, 4, 8]

        for layer_idx in target_progressive:
            logger.info(f"\n  Layer {layer_idx}:")

            for n_groups in escalation:
                # Ablate KV groups 0..n-1
                kv_groups = list(range(n_groups))
                query_heads: list[int] = []
                for kg in kv_groups:
                    query_heads.extend(self._kv_group_to_query_heads(kg))

                facts_preserved = 0
                for fact in FACTS:
                    baseline = self.baselines[fact["prompt"]]
                    ablated = await loop.run_in_executor(
                        None,
                        self._generate_with_head_ablation,
                        fact["prompt"],
                        layer_idx,
                        query_heads,
                    )
                    mx.eval(mx.zeros(1))

                    preserved = self._check_fact(ablated, fact["expected"])
                    if preserved:
                        facts_preserved += 1

                    self.progressive_results.append(
                        ProgressiveResult(
                            layer_idx=layer_idx,
                            num_kv_groups_ablated=n_groups,
                            kv_groups_ablated=kv_groups,
                            prompt=fact["prompt"],
                            expected=fact["expected"],
                            baseline_text=baseline,
                            ablated_text=ablated,
                            fact_preserved=preserved,
                        )
                    )

                pct = facts_preserved / len(FACTS) * 100
                logger.info(
                    f"    {n_groups}/8 KV groups ({n_groups * GQA_RATIO}/64 Q heads): "
                    f"{facts_preserved}/{len(FACTS)} preserved ({pct:.0f}%)"
                )

    # ── Phase 4: Full Layer Attention Ablation ──

    async def full_layer_ablation(self):
        """Ablate ALL 64 query heads at single layers."""
        logger.info("=" * 60)
        logger.info("PHASE 4: Full Layer Attention Ablation")
        logger.info("  (All 8/8 KV groups = all 64 query heads zeroed)")
        logger.info("=" * 60)

        loop = asyncio.get_event_loop()
        all_query_heads = list(range(NUM_QUERY_HEADS))

        for layer_idx in self.target_layers:
            facts_preserved = 0
            for fact in FACTS:
                baseline = self.baselines[fact["prompt"]]
                ablated = await loop.run_in_executor(
                    None,
                    self._generate_with_head_ablation,
                    fact["prompt"],
                    layer_idx,
                    all_query_heads,
                )
                mx.eval(mx.zeros(1))

                preserved = self._check_fact(ablated, fact["expected"])
                if preserved:
                    facts_preserved += 1

                self.full_layer_results.append(
                    FullLayerResult(
                        layer_idx=layer_idx,
                        prompt=fact["prompt"],
                        expected=fact["expected"],
                        baseline_text=baseline,
                        ablated_text=ablated,
                        fact_preserved=preserved,
                    )
                )

            pct = facts_preserved / len(FACTS) * 100
            logger.info(
                f"  Layer {layer_idx}: "
                f"{facts_preserved}/{len(FACTS)} preserved ({pct:.0f}%)"
            )

    # ── Analysis ──

    def _build_layer_summaries(self):
        """Summarize Phase 2 results by layer."""
        grouped: dict[tuple[int, int], list[HeadAblationResult]] = defaultdict(list)
        for r in self.scan_results:
            grouped[(r.layer_idx, r.kv_head_idx)].append(r)

        layers: dict[int, dict[int, dict]] = defaultdict(dict)
        for (layer_idx, kv_head), results in grouped.items():
            baseline_valid = [
                r for r in results if self._check_fact(r.baseline_text, r.expected)
            ]
            if baseline_valid:
                broken = [r for r in baseline_valid if not r.fact_preserved]
                break_rate = len(broken) / len(baseline_valid)
            else:
                break_rate = 0.0

            changed = [r for r in results if r.output_changed]
            change_rate = len(changed) / len(results) if results else 0.0
            facts_broken = [
                r.expected
                for r in results
                if not r.fact_preserved
                and self._check_fact(r.baseline_text, r.expected)
            ]

            layers[layer_idx][kv_head] = {
                "fact_break_rate": break_rate,
                "change_rate": change_rate,
                "facts_broken": facts_broken,
            }

        for layer_idx in sorted(layers):
            kv_results = layers[layer_idx]
            max_break = max(d["fact_break_rate"] for d in kv_results.values())
            most_impactful = max(
                kv_results.keys(), key=lambda k: kv_results[k]["fact_break_rate"]
            )
            any_broken = any(d["fact_break_rate"] > 0 for d in kv_results.values())

            self.layer_summaries.append(
                LayerSummary(
                    layer_idx=layer_idx,
                    kv_head_results=kv_results,
                    most_impactful_kv_head=most_impactful,
                    max_fact_break_rate=max_break,
                    any_fact_broken=any_broken,
                )
            )

    # ── Output ──

    def _save_results(self) -> Path:
        """Save all results to JSON."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"attention_head_ablation_{timestamp}.json"

        output = {
            "metadata": {
                "experiment": "attention_head_ablation",
                "model": "openai/gpt-oss-20b",
                "timestamp": timestamp,
                "target_layers": self.target_layers,
                "num_kv_heads": NUM_KV_HEADS,
                "num_query_heads": NUM_QUERY_HEADS,
                "gqa_ratio": GQA_RATIO,
                "max_tokens": MAX_TOKENS,
                "ablation_method": "class-level monkey-patch of GptOssAttention.__call__",
                "sdpa_note": "manual SDPA omits attention sinks (minor discrepancy)",
            },
            "baselines": {
                f["prompt"]: {
                    "text": self.baselines.get(f["prompt"], ""),
                    "expected": f["expected"],
                    "fact_present": self._check_fact(
                        self.baselines.get(f["prompt"], ""), f["expected"]
                    ),
                }
                for f in FACTS
            },
            "phase2_kv_scan": [
                {
                    "layer_idx": r.layer_idx,
                    "kv_head_idx": r.kv_head_idx,
                    "query_heads_ablated": r.query_heads_ablated,
                    "prompt": r.prompt,
                    "expected": r.expected,
                    "baseline_text": r.baseline_text,
                    "ablated_text": r.ablated_text,
                    "fact_preserved": r.fact_preserved,
                    "output_changed": r.output_changed,
                }
                for r in self.scan_results
            ],
            "phase3_progressive": [
                {
                    "layer_idx": r.layer_idx,
                    "num_kv_groups_ablated": r.num_kv_groups_ablated,
                    "kv_groups_ablated": r.kv_groups_ablated,
                    "prompt": r.prompt,
                    "expected": r.expected,
                    "baseline_text": r.baseline_text,
                    "ablated_text": r.ablated_text,
                    "fact_preserved": r.fact_preserved,
                }
                for r in self.progressive_results
            ],
            "phase4_full_layer": [
                {
                    "layer_idx": r.layer_idx,
                    "prompt": r.prompt,
                    "expected": r.expected,
                    "baseline_text": r.baseline_text,
                    "ablated_text": r.ablated_text,
                    "fact_preserved": r.fact_preserved,
                }
                for r in self.full_layer_results
            ],
            "layer_summaries": [
                {
                    "layer_idx": s.layer_idx,
                    "kv_head_results": {
                        str(k): v for k, v in s.kv_head_results.items()
                    },
                    "most_impactful_kv_head": s.most_impactful_kv_head,
                    "max_fact_break_rate": s.max_fact_break_rate,
                    "any_fact_broken": s.any_fact_broken,
                }
                for s in self.layer_summaries
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")
        return output_path

    def _print_summary(self):
        """Print experiment summary."""
        print("\n" + "=" * 70)
        print("ATTENTION HEAD ABLATION RESULTS")
        print("=" * 70)

        # Baselines
        valid = sum(
            1
            for f in FACTS
            if self._check_fact(self.baselines.get(f["prompt"], ""), f["expected"])
        )
        print(f"\nBaselines: {valid}/{len(FACTS)} facts correct")

        # Phase 2: KV Head Scan
        print("\n--- Phase 2: KV Head Group Scan ---")
        print(
            f"{'Layer':>6} | {'Max Break%':>10} | {'Best KV':>8} | "
            f"{'Output Change%':>14} | {'Any Broken':>10}"
        )
        print("-" * 60)
        for s in self.layer_summaries:
            avg_change = sum(
                d["change_rate"] for d in s.kv_head_results.values()
            ) / len(s.kv_head_results)
            print(
                f"  L{s.layer_idx:>3} | {s.max_fact_break_rate:>9.0%} | "
                f"  KV{s.most_impactful_kv_head:>4} | "
                f"     {avg_change:>8.0%}     | "
                f"{'YES' if s.any_fact_broken else 'no':>10}"
            )

        # Phase 3: Progressive
        if self.progressive_results:
            print("\n--- Phase 3: Progressive KV Head Ablation ---")
            prog_grouped: dict[tuple[int, int], list[ProgressiveResult]] = defaultdict(
                list
            )
            for r in self.progressive_results:
                prog_grouped[(r.layer_idx, r.num_kv_groups_ablated)].append(r)

            layers = sorted(set(r.layer_idx for r in self.progressive_results))
            header = f"{'Layer':>6} |"
            for n in [1, 2, 4, 8]:
                header += f" {n}KV({n * GQA_RATIO}Q) |"
            print(header)
            print("-" * (8 + 4 * 12))

            for layer_idx in layers:
                row = f"  L{layer_idx:>3} |"
                for n in [1, 2, 4, 8]:
                    results = prog_grouped.get((layer_idx, n), [])
                    if results:
                        preserved = sum(1 for r in results if r.fact_preserved)
                        row += f"   {preserved}/{len(results)}    |"
                    else:
                        row += f"     --    |"
                print(row)

        # Phase 4: Full Layer
        if self.full_layer_results:
            print("\n--- Phase 4: Full Layer Attention Ablation ---")
            full_grouped: dict[int, list[FullLayerResult]] = defaultdict(list)
            for r in self.full_layer_results:
                full_grouped[r.layer_idx].append(r)

            for layer_idx in sorted(full_grouped):
                results = full_grouped[layer_idx]
                preserved = sum(1 for r in results if r.fact_preserved)
                pct = preserved / len(results) * 100
                print(
                    f"  L{layer_idx:>3}: {preserved}/{len(results)} "
                    f"preserved ({pct:.0f}%)"
                )

        # Key findings
        print("\n--- KEY FINDINGS ---")
        any_broken_p2 = any(s.any_fact_broken for s in self.layer_summaries)
        if any_broken_p2:
            broken_layers = [
                s.layer_idx for s in self.layer_summaries if s.any_fact_broken
            ]
            print(f"Phase 2: Facts broken at layers {broken_layers}")
            print("  -> Attention heads at these layers are CAUSAL for fact retrieval")

            # Which KV heads matter?
            for s in self.layer_summaries:
                if s.any_fact_broken:
                    for kv, info in s.kv_head_results.items():
                        if info["fact_break_rate"] > 0:
                            print(
                                f"  -> L{s.layer_idx} KV{kv}: "
                                f"{info['fact_break_rate']:.0%} break rate, "
                                f"broke: {info['facts_broken']}"
                            )
        else:
            print(
                "Phase 2: NO facts broken by single KV head group ablation "
                "at any layer"
            )
            print(
                "  -> Factual knowledge is NOT localized to individual KV head groups"
            )

        # Phase 4 findings
        if self.full_layer_results:
            full_any_broken = False
            for layer_idx in sorted(full_grouped):
                results = full_grouped[layer_idx]
                preserved = sum(1 for r in results if r.fact_preserved)
                if preserved < len(results):
                    full_any_broken = True
                    print(
                        f"Phase 4: L{layer_idx} full ablation broke "
                        f"{len(results) - preserved}/{len(results)} facts"
                    )
            if not full_any_broken:
                print(
                    "Phase 4: NO facts broken even with full layer attention ablation"
                )
                print(
                    "  -> Attention at individual layers is NOT necessary "
                    "for fact retrieval"
                )

    # ── Orchestration ──

    async def run(self) -> Path:
        """Run the full experiment."""
        await self.setup()

        valid = await self.generate_baselines()
        if valid < 6:
            logger.warning(
                f"Only {valid}/8 baselines correct. Results may be unreliable."
            )

        await self.kv_head_scan()
        self._build_layer_summaries()

        await self.progressive_ablation()
        await self.full_layer_ablation()

        self._print_summary()
        output_path = self._save_results()

        return output_path


async def main():
    parser = argparse.ArgumentParser(
        description="Attention head ablation for factual knowledge"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Run only on a single layer (e.g. --layer 10)",
    )
    args = parser.parse_args()

    if args.layer is not None:
        target_layers = [args.layer]
    else:
        target_layers = None

    experiment = AttentionHeadAblation(target_layers=target_layers)
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
