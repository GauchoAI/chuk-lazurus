#!/usr/bin/env python3
"""Attention Pattern at Emergence Layers.

Residual fact emergence showed facts crystallize at L20-21, with first
signal at L15. Layer skip showed L20-21 are important but not irreplaceable.

This experiment asks: what is attention DOING at the emergence layers?
Specifically, does the final token (prediction position) attend to the
entity token (France, gold, Microsoft) when facts first appear?

If attention focuses on the entity at emergence, that's the mechanistic
link: attention performs the "lookup" and writes the fact to the residual
stream. Experts then refine but don't originate the factual content.

Method:
  For each fact prompt, we:
  1. Identify the entity token position (e.g., "France" in "The capital of France is")
  2. Monkey-patch GptOssAttention to capture attention weights from Q, K, V
  3. At each layer (L0-L23), extract the final token's attention over all positions
  4. Measure how much attention the entity token receives vs other tokens
  5. Compare entity attention at pre-emergence layers vs emergence layers

Run: python experiments/expert_function_classification/attention_at_emergence.py
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
        "entity": "France",
    },
    {
        "prompt": "The chemical symbol for gold is",
        "expected_keyword": "Au",
        "entity": "gold",
    },
    {
        "prompt": "The author of Romeo and Juliet is",
        "expected_keyword": "Shakespeare",
        "entity": "Romeo",  # Multi-token entity; track first token
    },
    {
        "prompt": "The CEO of Microsoft is",
        "expected_keyword": "Nadella",
        "entity": "Microsoft",
    },
    {
        "prompt": "The capital of Japan is",
        "expected_keyword": "Tokyo",
        "entity": "Japan",
    },
    {
        "prompt": "The chemical symbol for silver is",
        "expected_keyword": "Ag",
        "entity": "silver",
    },
    {
        "prompt": "The capital of Australia is",
        "expected_keyword": "Canberra",
        "entity": "Australia",
    },
]


class AttentionAtEmergence:
    """Capture attention patterns at fact emergence layers."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._attn_class = None
        self._original_attn_call = None
        self._captured_weights: dict[int, mx.array] = {}  # layer -> [batch, heads, q_len, kv_len]

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        # Get the attention class for monkey-patching
        sample_layer = self.model.model.layers[0]
        self._attn_class = type(sample_layer.self_attn)
        self._original_attn_call = self._attn_class.__call__

        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded: {self.num_layers} layers. Ready.")

    def _find_entity_position(self, prompt: str, entity: str) -> int | None:
        """Find the token position of the entity in the prompt.

        Returns the position of the first token of the entity string.
        """
        prompt_ids = self.tokenizer.encode(prompt)

        # Encode the entity with and without space prefix
        for candidate in [f" {entity}", entity]:
            try:
                entity_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            except TypeError:
                entity_ids = self.tokenizer.encode(candidate)

            if not entity_ids:
                continue

            # Find the first entity token in the prompt tokens
            first_entity_id = entity_ids[0]
            for pos, tid in enumerate(prompt_ids):
                if tid == first_entity_id:
                    return pos

        return None

    def _capture_attention_forward(
        self, prompt: str
    ) -> dict[int, mx.array]:
        """Run forward pass capturing attention weights at all layers.

        Monkey-patches GptOssAttention to manually compute attention weights
        from Q, K, V (since mx.fast.scaled_dot_product_attention is fused
        and doesn't expose them).

        Returns: {layer_idx: attention_weights} where weights have shape
        [num_kv_groups, q_len, kv_len] (averaged across heads within each
        KV group for memory efficiency).
        """
        captured: dict[int, mx.array] = {}
        experiment = self
        original_call = self._original_attn_call

        def patched_attn(
            attn_self: Any,
            x: mx.array,
            mask: mx.array | str | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            batch, seq_len, _ = x.shape

            # Compute Q, K, V (same as original)
            q = attn_self.q_proj(x)
            k = attn_self.k_proj(x)
            v = attn_self.v_proj(x)

            q = q.reshape(batch, seq_len, attn_self.num_heads, attn_self.head_dim)
            k = k.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)
            v = v.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)

            q = q.transpose(0, 2, 1, 3)  # [B, H, S, D]
            k = k.transpose(0, 2, 1, 3)  # [B, KV, S, D]
            v = v.transpose(0, 2, 1, 3)

            # Apply RoPE
            if cache is not None:
                q = attn_self.rope(q, offset=cache[0].shape[2])
                k = attn_self.rope(k, offset=cache[0].shape[2])
            else:
                q = attn_self.rope(q)
                k = attn_self.rope(k)

            # Update cache
            if cache is not None:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            new_cache = (k, v)

            # Compute attention weights manually for capture
            # Expand KV heads for GQA: [B, H, S, D] @ [B, KV, D, S] with head grouping
            num_groups = attn_self.num_heads // attn_self.num_kv_heads  # 64/8 = 8
            k_expanded = mx.repeat(k, num_groups, axis=1)  # [B, H, S, D]

            # Compute scores: [B, H, S_q, S_k]
            scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * attn_self.scale

            # Apply mask
            if mask is not None and not isinstance(mask, str):
                scores = scores + mask

            # Softmax to get weights
            weights = mx.softmax(scores, axis=-1)  # [B, H, S_q, S_k]

            # Store: average across heads within each KV group for compactness
            # [B, H, S_q, S_k] -> [B, KV_groups, S_q, S_k] by reshaping and averaging
            weights_grouped = weights.reshape(
                batch, attn_self.num_kv_heads, num_groups, seq_len, -1
            )
            weights_avg = mx.mean(weights_grouped, axis=2)  # [B, KV, S_q, S_k]

            # Store the captured weights
            layer_idx = attn_self.layer_idx
            captured[layer_idx] = mx.stop_gradient(weights_avg[0])  # Drop batch dim

            # Still use the fused kernel for the actual output (for correctness)
            output = mx.fast.scaled_dot_product_attention(
                q, new_cache[0], new_cache[1],
                scale=attn_self.scale,
                mask=mask,
                sinks=attn_self.sinks,
            )
            output = output.transpose(0, 2, 1, 3)
            output = output.reshape(batch, seq_len, -1)
            output = attn_self.o_proj(output)

            return output, new_cache

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        try:
            self._attn_class.__call__ = patched_attn
            self.model(input_ids)
            mx.eval(list(captured.values()))
        finally:
            self._attn_class.__call__ = self._original_attn_call

        return captured

    async def analyze_fact(self, fact: dict) -> dict[str, Any]:
        """Analyze attention patterns for one fact."""
        prompt = fact["prompt"]
        entity = fact["entity"]
        keyword = fact["expected_keyword"]

        logger.info(f"\n  Fact: {prompt} (entity='{entity}')")

        # Find entity position
        entity_pos = self._find_entity_position(prompt, entity)
        prompt_tokens = self.tokenizer.encode(prompt)
        seq_len = len(prompt_tokens)
        last_pos = seq_len - 1

        if entity_pos is None:
            logger.warning(f"    Could not find entity '{entity}' in tokens")
            return {"prompt": prompt, "error": f"entity '{entity}' not found"}

        # Decode tokens for reference
        token_strs = [self.tokenizer.decode([tid]) for tid in prompt_tokens]
        logger.info(
            f"    Tokens: {token_strs}"
        )
        logger.info(
            f"    Entity '{entity}' at position {entity_pos} "
            f"(token: '{token_strs[entity_pos]}')"
        )
        logger.info(f"    Last token at position {last_pos} (prediction point)")

        # Capture attention weights at all layers
        loop = asyncio.get_event_loop()
        captured = await loop.run_in_executor(
            None, self._capture_attention_forward, prompt,
        )

        # Extract: for each layer, how much does the last token attend to the entity?
        layer_entity_attention = {}
        layer_max_attention = {}
        layer_attention_distribution = {}

        for layer_idx in sorted(captured.keys()):
            weights = captured[layer_idx]  # [KV_groups, S_q, S_k]
            # Get final token's attention across all KV groups
            # Average across KV groups for a single attention score per position
            final_token_attn = mx.mean(weights[:, last_pos, :], axis=0)  # [S_k]
            final_token_attn_list = final_token_attn.tolist()

            entity_attn = float(final_token_attn[entity_pos])
            max_attn = float(mx.max(final_token_attn))
            max_pos = int(mx.argmax(final_token_attn))

            layer_entity_attention[layer_idx] = entity_attn
            layer_max_attention[layer_idx] = {
                "value": max_attn,
                "position": max_pos,
                "token": token_strs[max_pos] if max_pos < len(token_strs) else "?",
            }
            layer_attention_distribution[layer_idx] = {
                str(pos): round(float(final_token_attn[pos]), 6)
                for pos in range(seq_len)
            }

        # Log key layers
        for layer_idx in [0, 8, 15, 19, 20, 21, 22, 23]:
            if layer_idx in layer_entity_attention:
                ea = layer_entity_attention[layer_idx]
                ma = layer_max_attention[layer_idx]
                logger.info(
                    f"    L{layer_idx:>2}: entity_attn={ea:.4f}, "
                    f"max={ma['value']:.4f} at pos {ma['position']} "
                    f"('{ma['token']}')"
                )

        return {
            "prompt": prompt,
            "entity": entity,
            "entity_position": entity_pos,
            "entity_token": token_strs[entity_pos],
            "expected_keyword": keyword,
            "seq_len": seq_len,
            "tokens": token_strs,
            "entity_attention_by_layer": {
                str(k): round(v, 6) for k, v in layer_entity_attention.items()
            },
            "max_attention_by_layer": {
                str(k): v for k, v in layer_max_attention.items()
            },
            "full_distribution_by_layer": {
                str(k): v for k, v in layer_attention_distribution.items()
            },
        }

    def _compute_summary(self, fact_results: list[dict]) -> dict[str, Any]:
        """Compute aggregate attention statistics."""
        valid = [r for r in fact_results if "error" not in r]

        # Average entity attention by layer
        avg_entity_attn: dict[int, list[float]] = defaultdict(list)
        for r in valid:
            for layer_str, attn in r["entity_attention_by_layer"].items():
                avg_entity_attn[int(layer_str)].append(attn)

        avg_curve = {
            layer: round(sum(vals) / len(vals), 6)
            for layer, vals in sorted(avg_entity_attn.items())
        }

        # Find peak entity attention layer
        if avg_curve:
            peak_layer = max(avg_curve, key=avg_curve.get)
            peak_value = avg_curve[peak_layer]
        else:
            peak_layer = None
            peak_value = None

        # Pre-emergence vs emergence vs post-emergence averages
        pre = [avg_curve.get(l, 0) for l in range(0, 15)]
        emergence = [avg_curve.get(l, 0) for l in range(15, 22)]
        post = [avg_curve.get(l, 0) for l in range(22, 24)]

        avg_pre = sum(pre) / len(pre) if pre else 0
        avg_emergence = sum(emergence) / len(emergence) if emergence else 0
        avg_post = sum(post) / len(post) if post else 0

        return {
            "num_facts": len(valid),
            "avg_entity_attention_by_layer": avg_curve,
            "peak_entity_attention": {
                "layer": peak_layer,
                "value": peak_value,
            },
            "phase_averages": {
                "pre_emergence_L0_L14": round(avg_pre, 6),
                "emergence_L15_L21": round(avg_emergence, 6),
                "post_emergence_L22_L23": round(avg_post, 6),
            },
            "interpretation": (
                f"Entity attention peaks at L{peak_layer} ({peak_value:.4f}). "
                f"Pre-emergence avg: {avg_pre:.4f}, "
                f"emergence avg: {avg_emergence:.4f}, "
                f"post-emergence avg: {avg_post:.4f}."
            ),
        }

    def _print_summary(self, summary: dict, fact_results: list[dict]):
        valid = [r for r in fact_results if "error" not in r]

        print("\n" + "=" * 90)
        print("ATTENTION PATTERN AT EMERGENCE LAYERS - RESULTS")
        print("=" * 90)

        # Per-fact entity attention at key layers
        print(
            f"\n{'Prompt':<36} | {'Entity':<12} | "
            f"{'L0':>6} | {'L8':>6} | {'L15':>6} | {'L19':>6} | "
            f"{'L20':>6} | {'L21':>6} | {'L22':>6} | {'L23':>6}"
        )
        print("-" * 130)

        for r in valid:
            prompt_short = r["prompt"][:34]
            entity = r["entity"][:10]
            attn = r["entity_attention_by_layer"]
            cols = []
            for l in ["0", "8", "15", "19", "20", "21", "22", "23"]:
                val = attn.get(l, 0)
                cols.append(f"{val:.4f}")
            print(
                f"{prompt_short:<36} | {entity:<12} | "
                + " | ".join(f"{c:>6}" for c in cols)
            )

        # Average entity attention curve
        print("\n" + "-" * 90)
        print("AVERAGE ENTITY ATTENTION BY LAYER (final token -> entity token)")
        print("-" * 90)

        curve = summary["avg_entity_attention_by_layer"]
        max_val = max(curve.values()) if curve else 1
        for layer in sorted(curve.keys()):
            val = curve[layer]
            bar_len = int((val / max(max_val, 0.001)) * 40)
            bar = "#" * bar_len
            marker = ""
            if layer == 15:
                marker = "  <- first fact signal"
            elif layer in (20, 21):
                marker = "  <- fact emergence"
            print(f"  L{layer:>2}: {val:>8.5f} |{bar}{marker}")

        # Phase comparison
        print("\n" + "-" * 90)
        print("PHASE AVERAGES")
        print("-" * 90)
        phases = summary["phase_averages"]
        print(f"  Pre-emergence  (L0-L14):  {phases['pre_emergence_L0_L14']:.5f}")
        print(f"  Emergence      (L15-L21): {phases['emergence_L15_L21']:.5f}")
        print(f"  Post-emergence (L22-L23): {phases['post_emergence_L22_L23']:.5f}")

        ratio = (
            phases["emergence_L15_L21"] / phases["pre_emergence_L0_L14"]
            if phases["pre_emergence_L0_L14"] > 0 else float("inf")
        )
        print(f"\n  Emergence/Pre-emergence ratio: {ratio:.1f}x")

        # Where does the last token attend most? (not just entity)
        print("\n" + "-" * 90)
        print("MOST-ATTENDED TOKEN BY LAYER (per fact)")
        print("-" * 90)

        for r in valid:
            prompt_short = r["prompt"][:35]
            entity_pos = r["entity_position"]
            print(f"\n  {prompt_short} (entity at pos {entity_pos}):")
            for l in ["15", "19", "20", "21", "22", "23"]:
                ma = r["max_attention_by_layer"].get(l)
                if ma:
                    is_entity = " <- ENTITY" if ma["position"] == entity_pos else ""
                    print(
                        f"    L{l:>2}: max={ma['value']:.4f} at pos {ma['position']} "
                        f"('{ma['token']}'){is_entity}"
                    )

        # Key findings
        print("\n" + "=" * 90)
        print("KEY FINDINGS")
        print("=" * 90)

        peak = summary["peak_entity_attention"]
        print(f"\n  Entity attention peaks at: L{peak['layer']} ({peak['value']:.4f})")

        if ratio > 2.0:
            print(
                f"\n  Attention to entity INCREASES {ratio:.1f}x at emergence layers."
            )
            print(
                "  This confirms: attention performs the fact 'lookup' by focusing"
            )
            print(
                "  on the entity token at exactly the layers where facts crystallize"
            )
            print(
                "  in the residual stream."
            )
        elif ratio > 1.0:
            print(
                f"\n  Modest increase ({ratio:.1f}x) in entity attention at emergence."
            )
            print(
                "  Fact crystallization may involve distributed attention rather than"
            )
            print(
                "  a focused entity lookup."
            )
        else:
            print(
                "\n  Entity attention does NOT increase at emergence layers."
            )
            print(
                "  Fact crystallization uses a different mechanism than direct"
            )
            print(
                "  entity-token attention."
            )

        print("=" * 90)

    def _save_results(self, results: dict) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"attention_at_emergence_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
        return output_path

    async def run(self):
        await self.setup()

        logger.info("=" * 70)
        logger.info("ATTENTION PATTERN AT EMERGENCE LAYERS")
        logger.info(f"  Facts: {len(FACTS)}")
        logger.info("=" * 70)

        fact_results = []
        for fact in FACTS:
            result = await self.analyze_fact(fact)
            fact_results.append(result)

        summary = self._compute_summary(fact_results)

        output = {
            "metadata": {
                "experiment": "attention_at_emergence",
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "num_facts": len(FACTS),
                "description": (
                    "Captures attention weights at all layers to measure how "
                    "much the final (prediction) token attends to the entity "
                    "token at emergence layers (L15-L21). Tests whether "
                    "attention performs the fact 'lookup'."
                ),
                "method": (
                    "Monkey-patches GptOssAttention to compute attention weights "
                    "from Q, K, V before the fused SDPA kernel. Averages across "
                    "heads within each KV group (8 KV groups, 8 heads per group). "
                    "Reports the mean across all KV groups."
                ),
                "prior_results": {
                    "fact_emergence_avg_top1": "L20.8 (logit lens)",
                    "fact_emergence_first_signal": "L15 (~5%)",
                    "layer_skip_L20_L21": "5/7 facts survive",
                },
            },
            "summary": summary,
            "fact_results": fact_results,
        }

        self._save_results(output)
        self._print_summary(summary, fact_results)


async def main():
    experiment = AttentionAtEmergence()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
