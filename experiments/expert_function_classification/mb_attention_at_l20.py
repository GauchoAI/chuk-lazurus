#!/usr/bin/env python3
"""Memory Bank Attention at L20 Experiment.

The memory bank injection point experiment revealed a striking puzzle:
at L20, bare prompts show 14.8% fact probability while MB prompts show
only 0.05% — despite the answer being explicitly in context.

This experiment captures attention weights at L19-L22 for both bare and
MB conditions, and asks: where is the final token attending at L20 in
the MB condition? Three hypotheses:

  1. Distraction: final token attends to instruction tokens ([Memory Bank],
     [End], "Using...answer:") rather than the answer "Paris"
  2. Interference: final token attends to "Paris" in MB but the
     representation is in the wrong subspace for retrieval
  3. Delayed integration: model ignores in-context "Paris" at L20
     entirely, only integrating at L21

Method:
  For each fact, run bare and MB prompts through the attention capture
  monkey-patch. Classify each token position into semantic regions:
    - MB_ANSWER: the answer token in the memory bank (e.g., "Paris")
    - MB_ENTITY: the entity token in the memory bank (e.g., "France")
    - MB_OTHER: other memory bank entry tokens
    - MB_DELIMITERS: [Memory Bank], [End Memory Bank]
    - INSTRUCTION: "Using the memory bank above, answer:"
    - QUERY: the query text (e.g., "The capital of France is")
    - QUERY_ENTITY: entity within the query
    - QUERY_COPULA: " is" at end of query
    - ANSWER_PREFIX: "Answer:" suffix

  Report attention to each region at L19, L20, L21, L22.

Run: python experiments/expert_function_classification/mb_attention_at_l20.py
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
        "mb_entry": "France | capital | Paris",
        "mb_answer": "Paris",
    },
    {
        "prompt": "The chemical symbol for gold is",
        "expected_keyword": "Au",
        "entity": "gold",
        "mb_entry": "Gold | chemical symbol | Au",
        "mb_answer": "Au",
    },
    {
        "prompt": "The author of Romeo and Juliet is",
        "expected_keyword": "Shakespeare",
        "entity": "Romeo",
        "mb_entry": "Romeo and Juliet | author | William Shakespeare",
        "mb_answer": "William",
    },
    {
        "prompt": "The CEO of Microsoft is",
        "expected_keyword": "Nadella",
        "entity": "Microsoft",
        "mb_entry": "Microsoft | CEO | Satya Nadella",
        "mb_answer": "Sat",
    },
    {
        "prompt": "The capital of Japan is",
        "expected_keyword": "Tokyo",
        "entity": "Japan",
        "mb_entry": "Japan | capital | Tokyo",
        "mb_answer": "Tokyo",
    },
    {
        "prompt": "The chemical symbol for silver is",
        "expected_keyword": "Ag",
        "entity": "silver",
        "mb_entry": "Silver | chemical symbol | Ag",
        "mb_answer": "Ag",
    },
    {
        "prompt": "The capital of Australia is",
        "expected_keyword": "Canberra",
        "entity": "Australia",
        "mb_entry": "Australia | capital | Canberra",
        "mb_answer": "Canberra",
    },
]

# Focus on these layers (emergence zone)
FOCUS_LAYERS = [18, 19, 20, 21, 22, 23]


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


class MBAttentionAtL20:
    """Compare attention patterns between bare and MB conditions at L20."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._attn_class = None
        self._original_attn_call = None

    async def setup(self):
        from chuk_lazarus.introspection.moe.expert_router import ExpertRouter

        logger.info("Loading model: openai/gpt-oss-20b")
        router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        self.model = router._model
        self.tokenizer = router._tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        sample_layer = self.model.model.layers[0]
        self._attn_class = type(sample_layer.self_attn)
        self._original_attn_call = self._attn_class.__call__

        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded: {self.num_layers} layers. Ready.")

    def _capture_attention_forward(self, prompt: str) -> dict[int, mx.array]:
        """Run forward pass capturing attention weights at focus layers.

        Returns {layer_idx: weights} where weights shape is
        [num_kv_groups, q_len, kv_len], averaged across heads in each group.
        """
        captured: dict[int, mx.array] = {}
        original_call = self._original_attn_call
        focus = set(FOCUS_LAYERS)

        def patched_attn(
            attn_self: Any,
            x: mx.array,
            mask: mx.array | str | None = None,
            cache: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
            batch, seq_len, _ = x.shape

            q = attn_self.q_proj(x)
            k = attn_self.k_proj(x)
            v = attn_self.v_proj(x)

            q = q.reshape(batch, seq_len, attn_self.num_heads, attn_self.head_dim)
            k = k.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)
            v = v.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)

            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            if cache is not None:
                q = attn_self.rope(q, offset=cache[0].shape[2])
                k = attn_self.rope(k, offset=cache[0].shape[2])
            else:
                q = attn_self.rope(q)
                k = attn_self.rope(k)

            if cache is not None:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            new_cache = (k, v)

            layer_idx = attn_self.layer_idx

            # Only compute attention weights for focus layers
            if layer_idx in focus:
                num_groups = attn_self.num_heads // attn_self.num_kv_heads
                k_expanded = mx.repeat(k, num_groups, axis=1)
                scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * attn_self.scale
                if mask is not None and not isinstance(mask, str):
                    scores = scores + mask
                weights = mx.softmax(scores, axis=-1)
                weights_grouped = weights.reshape(
                    batch, attn_self.num_kv_heads, num_groups, seq_len, -1
                )
                weights_avg = mx.mean(weights_grouped, axis=2)
                captured[layer_idx] = mx.stop_gradient(weights_avg[0])

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

    def _find_token_positions(self, token_ids: list[int], search_str: str) -> list[int]:
        """Find positions of tokens that decode to contain search_str."""
        positions = []
        for i, tid in enumerate(token_ids):
            decoded = self.tokenizer.decode([tid])
            if search_str.lower() in decoded.lower():
                positions.append(i)
        return positions

    def _classify_mb_regions(
        self, mb_prompt: str, fact: dict
    ) -> dict[str, list[int]]:
        """Classify each token position in the MB prompt into semantic regions.

        Returns dict of region_name -> list of token positions.
        """
        token_ids = self.tokenizer.encode(mb_prompt)
        token_strs = [self.tokenizer.decode([tid]) for tid in token_ids]
        n = len(token_ids)

        # Build the full text position-by-position to find region boundaries
        # Strategy: tokenize sub-parts to find boundaries
        all_mb_entries = [f["mb_entry"] for f in FACTS]
        mb_block = "\n".join(f"- {entry}" for entry in all_mb_entries)
        mb_header = "[Memory Bank]\n"
        mb_footer = "\n[End Memory Bank]\n\n"
        instruction = f"Using the memory bank above, answer: "
        query = fact["prompt"]
        answer_suffix = "\nAnswer:"

        # Tokenize each section to find approximate boundaries
        header_ids = self.tokenizer.encode(mb_header, add_special_tokens=False)
        block_ids = self.tokenizer.encode(mb_block, add_special_tokens=False)
        footer_ids = self.tokenizer.encode(mb_footer, add_special_tokens=False)
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_suffix, add_special_tokens=False)

        # Approximate positions (tokenization of parts may differ from whole)
        # Use cumulative lengths
        header_end = len(header_ids)
        block_end = header_end + len(block_ids)
        footer_end = block_end + len(footer_ids)
        instruction_end = footer_end + len(instruction_ids)
        query_end = instruction_end + len(query_ids)

        regions: dict[str, list[int]] = {
            "mb_delimiters": [],
            "mb_answer": [],
            "mb_entity": [],
            "mb_other": [],
            "instruction": [],
            "query_entity": [],
            "query_copula": [],
            "query_other": [],
            "answer_prefix": [],
        }

        # Find answer and entity positions within MB block
        mb_answer_str = fact["mb_answer"]
        mb_entity_str = fact["entity"]

        # First pass: assign regions by position ranges
        for i in range(n):
            if i < header_end:
                regions["mb_delimiters"].append(i)
            elif i < block_end:
                regions["mb_other"].append(i)  # default; override below
            elif i < footer_end:
                regions["mb_delimiters"].append(i)
            elif i < instruction_end:
                regions["instruction"].append(i)
            elif i < query_end:
                regions["query_other"].append(i)  # default; override below
            else:
                regions["answer_prefix"].append(i)

        # Second pass: find specific tokens within MB block
        # Search for answer token in MB entries region
        for i in regions["mb_other"][:]:
            decoded = token_strs[i].strip()
            if decoded and mb_answer_str.lower().startswith(decoded.lower()):
                regions["mb_other"].remove(i)
                regions["mb_answer"].append(i)
            elif decoded and mb_entity_str.lower() in decoded.lower():
                regions["mb_other"].remove(i)
                regions["mb_entity"].append(i)

        # Third pass: find entity and copula in query
        for i in regions["query_other"][:]:
            decoded = token_strs[i]
            if fact["entity"].lower() in decoded.lower():
                regions["query_other"].remove(i)
                regions["query_entity"].append(i)
            elif decoded.strip() == "is":
                regions["query_other"].remove(i)
                regions["query_copula"].append(i)

        return regions

    def _classify_bare_regions(
        self, bare_prompt: str, fact: dict
    ) -> dict[str, list[int]]:
        """Classify token positions in bare prompt."""
        token_ids = self.tokenizer.encode(bare_prompt)
        token_strs = [self.tokenizer.decode([tid]) for tid in token_ids]
        n = len(token_ids)

        regions: dict[str, list[int]] = {
            "query_entity": [],
            "query_copula": [],
            "query_other": [],
        }

        for i in range(n):
            decoded = token_strs[i]
            if fact["entity"].lower() in decoded.lower():
                regions["query_entity"].append(i)
            elif decoded.strip() == "is":
                regions["query_copula"].append(i)
            else:
                regions["query_other"].append(i)

        return regions

    def _region_attention(
        self,
        weights: mx.array,
        regions: dict[str, list[int]],
        last_pos: int,
    ) -> dict[str, float]:
        """Compute attention from last position to each region.

        weights shape: [kv_groups, seq_len, seq_len]
        Returns {region_name: total_attention_to_region}.
        """
        # Average across KV groups for final-token attention
        final_attn = mx.mean(weights[:, last_pos, :], axis=0)  # [seq_len]
        final_attn_list = final_attn.tolist()

        result = {}
        for region_name, positions in regions.items():
            if positions:
                total = sum(final_attn_list[p] for p in positions if p < len(final_attn_list))
                result[region_name] = round(total, 6)
            else:
                result[region_name] = 0.0

        return result

    async def analyze_fact(self, fact: dict) -> dict[str, Any]:
        """Compare attention in bare vs MB conditions for one fact."""
        prompt = fact["prompt"]
        entity = fact["entity"]
        mb_answer_str = fact["mb_answer"]

        logger.info(f"\n  Fact: {prompt} (entity='{entity}', answer='{mb_answer_str}')")

        # Build prompts
        bare_prompt = prompt
        all_mb_entries = [f["mb_entry"] for f in FACTS]
        mb_prompt = build_memory_bank_prompt(prompt, all_mb_entries)

        bare_tokens = self.tokenizer.encode(bare_prompt)
        mb_tokens = self.tokenizer.encode(mb_prompt)
        bare_strs = [self.tokenizer.decode([t]) for t in bare_tokens]
        mb_strs = [self.tokenizer.decode([t]) for t in mb_tokens]

        logger.info(f"    Bare: {len(bare_tokens)} tokens")
        logger.info(f"    MB:   {len(mb_tokens)} tokens")

        # Classify regions
        bare_regions = self._classify_bare_regions(bare_prompt, fact)
        mb_regions = self._classify_mb_regions(mb_prompt, fact)

        # Log region mapping
        for rname, positions in mb_regions.items():
            if positions:
                sample = [mb_strs[p] for p in positions[:3]]
                logger.info(f"    MB region '{rname}': {len(positions)} tokens, e.g. {sample}")

        # Capture attention for both conditions
        loop = asyncio.get_event_loop()
        bare_attn = await loop.run_in_executor(
            None, self._capture_attention_forward, bare_prompt
        )
        mb_attn = await loop.run_in_executor(
            None, self._capture_attention_forward, mb_prompt
        )

        bare_last = len(bare_tokens) - 1
        mb_last = len(mb_tokens) - 1

        # Compute region attention at each focus layer
        layer_results = {}
        for layer_idx in FOCUS_LAYERS:
            bare_layer = {}
            mb_layer = {}

            if layer_idx in bare_attn:
                bare_layer = self._region_attention(
                    bare_attn[layer_idx], bare_regions, bare_last
                )
            if layer_idx in mb_attn:
                mb_layer = self._region_attention(
                    mb_attn[layer_idx], mb_regions, mb_last
                )

            layer_results[layer_idx] = {
                "bare": bare_layer,
                "mb": mb_layer,
            }

            # Log comparison
            logger.info(f"    L{layer_idx}:")
            if bare_layer:
                qe = bare_layer.get("query_entity", 0)
                qc = bare_layer.get("query_copula", 0)
                qo = bare_layer.get("query_other", 0)
                logger.info(f"      Bare:  entity={qe:.4f}  copula={qc:.4f}  other={qo:.4f}")
            if mb_layer:
                mba = mb_layer.get("mb_answer", 0)
                mbe = mb_layer.get("mb_entity", 0)
                mbo = mb_layer.get("mb_other", 0)
                mbd = mb_layer.get("mb_delimiters", 0)
                ins = mb_layer.get("instruction", 0)
                qe = mb_layer.get("query_entity", 0)
                qc = mb_layer.get("query_copula", 0)
                qo = mb_layer.get("query_other", 0)
                ap = mb_layer.get("answer_prefix", 0)
                logger.info(
                    f"      MB:    mb_answer={mba:.4f}  mb_entity={mbe:.4f}  "
                    f"mb_other={mbo:.4f}  delims={mbd:.4f}"
                )
                logger.info(
                    f"             instruction={ins:.4f}  q_entity={qe:.4f}  "
                    f"q_copula={qc:.4f}  q_other={qo:.4f}  answer_pfx={ap:.4f}"
                )

        # Also get per-position attention at L20 for detailed analysis
        l20_per_position = {}
        if 20 in mb_attn:
            final_attn_l20 = mx.mean(mb_attn[20][:, mb_last, :], axis=0).tolist()
            # Top-5 positions by attention at L20
            indexed = [(i, v) for i, v in enumerate(final_attn_l20)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            l20_top5 = []
            for pos, val in indexed[:10]:
                tok = mb_strs[pos] if pos < len(mb_strs) else "?"
                l20_top5.append({"position": pos, "token": tok, "attention": round(val, 6)})
                logger.info(f"    L20 top attn: pos={pos} token='{tok}' attn={val:.4f}")
            l20_per_position = {
                "top_10": l20_top5,
                "full_distribution": {
                    str(i): round(v, 6) for i, v in enumerate(final_attn_l20)
                },
            }

        return {
            "prompt": prompt,
            "entity": entity,
            "mb_answer": mb_answer_str,
            "bare_tokens": bare_strs,
            "mb_tokens": mb_strs,
            "bare_regions": {k: v for k, v in bare_regions.items()},
            "mb_regions": {k: v for k, v in mb_regions.items()},
            "layer_results": {
                str(k): v for k, v in layer_results.items()
            },
            "l20_mb_detail": l20_per_position,
        }

    def _compute_summary(self, fact_results: list[dict]) -> dict[str, Any]:
        """Compute aggregate attention statistics."""
        valid = [r for r in fact_results if "error" not in r]

        # Average region attention by layer for MB condition
        avg_mb_regions: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        avg_bare_regions: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in valid:
            for layer_str, layer_data in r["layer_results"].items():
                layer_idx = int(layer_str)
                for region, attn in layer_data.get("mb", {}).items():
                    avg_mb_regions[layer_idx][region].append(attn)
                for region, attn in layer_data.get("bare", {}).items():
                    avg_bare_regions[layer_idx][region].append(attn)

        # Compute averages
        avg_mb = {}
        for layer_idx in sorted(avg_mb_regions.keys()):
            avg_mb[layer_idx] = {
                region: round(sum(vals) / len(vals), 6)
                for region, vals in avg_mb_regions[layer_idx].items()
            }

        avg_bare = {}
        for layer_idx in sorted(avg_bare_regions.keys()):
            avg_bare[layer_idx] = {
                region: round(sum(vals) / len(vals), 6)
                for region, vals in avg_bare_regions[layer_idx].items()
            }

        # Key comparison: MB attention to answer token at L20 vs L21
        mb_answer_l20 = avg_mb.get(20, {}).get("mb_answer", 0)
        mb_answer_l21 = avg_mb.get(21, {}).get("mb_answer", 0)
        mb_instruction_l20 = avg_mb.get(20, {}).get("instruction", 0)
        mb_delimiters_l20 = avg_mb.get(20, {}).get("mb_delimiters", 0)

        return {
            "num_facts": len(valid),
            "avg_mb_attention_by_layer": avg_mb,
            "avg_bare_attention_by_layer": avg_bare,
            "l20_diagnosis": {
                "mb_answer_attention_l20": mb_answer_l20,
                "mb_answer_attention_l21": mb_answer_l21,
                "mb_instruction_attention_l20": mb_instruction_l20,
                "mb_delimiters_attention_l20": mb_delimiters_l20,
                "interpretation": (
                    "Distraction" if mb_instruction_l20 + mb_delimiters_l20 > 0.3
                    else "Delayed integration" if mb_answer_l20 < 0.02
                    else "Interference" if mb_answer_l20 > 0.05
                    else "Unknown"
                ),
            },
        }

    def _print_summary(self, summary: dict, fact_results: list[dict]):
        valid = [r for r in fact_results if "error" not in r]

        print("\n" + "=" * 110)
        print("MEMORY BANK ATTENTION AT L20 - RESULTS")
        print("=" * 110)

        # Average MB attention by region and layer
        print("\n" + "-" * 110)
        print("AVERAGE MB CONDITION: ATTENTION BY REGION AND LAYER")
        print("-" * 110)

        regions = [
            "mb_answer", "mb_entity", "mb_other", "mb_delimiters",
            "instruction", "query_entity", "query_copula", "query_other", "answer_prefix",
        ]
        header = f"{'Layer':>5}"
        for r in regions:
            short = r.replace("mb_", "mb:").replace("query_", "q:")
            header += f" | {short:>12}"
        print(header)
        print("-" * 110)

        avg_mb = summary["avg_mb_attention_by_layer"]
        for layer_idx in sorted(avg_mb.keys()):
            row = f"L{layer_idx:>3}:"
            for r in regions:
                val = avg_mb[layer_idx].get(r, 0)
                row += f" | {val:>12.4f}"
            marker = ""
            if layer_idx == 20:
                marker = "  <- L20 DIP"
            print(f"{row}{marker}")

        # Average bare attention for comparison
        print("\n" + "-" * 110)
        print("AVERAGE BARE CONDITION: ATTENTION BY REGION AND LAYER")
        print("-" * 110)

        bare_regions = ["query_entity", "query_copula", "query_other"]
        header = f"{'Layer':>5}"
        for r in bare_regions:
            short = r.replace("query_", "q:")
            header += f" | {short:>12}"
        print(header)
        print("-" * 60)

        avg_bare = summary["avg_bare_attention_by_layer"]
        for layer_idx in sorted(avg_bare.keys()):
            row = f"L{layer_idx:>3}:"
            for r in bare_regions:
                val = avg_bare[layer_idx].get(r, 0)
                row += f" | {val:>12.4f}"
            print(row)

        # L20 diagnosis
        print("\n" + "-" * 110)
        print("L20 DIAGNOSIS")
        print("-" * 110)

        diag = summary["l20_diagnosis"]
        print(f"\n  At L20, the final token in MB condition attends to:")
        print(f"    Answer token in MB ('Paris' etc): {diag['mb_answer_attention_l20']:.4f}")
        print(f"    Instruction text:                  {diag['mb_instruction_attention_l20']:.4f}")
        print(f"    MB delimiters:                     {diag['mb_delimiters_attention_l20']:.4f}")
        print(f"\n  At L21 (where MB catches up):")
        print(f"    Answer token in MB:                {diag['mb_answer_attention_l21']:.4f}")

        print(f"\n  Diagnosis: {diag['interpretation']}")

        # Per-fact L20 top attention targets
        print("\n" + "-" * 110)
        print("PER-FACT: TOP ATTENTION TARGETS AT L20 (MB CONDITION)")
        print("-" * 110)

        for r in valid:
            prompt_short = r["prompt"][:35]
            print(f"\n  {prompt_short} (answer='{r['mb_answer']}'):")
            if "l20_mb_detail" in r and r["l20_mb_detail"]:
                for entry in r["l20_mb_detail"]["top_10"][:5]:
                    tok = entry["token"].replace("\n", "\\n")
                    print(f"    pos {entry['position']:>3}: '{tok:<15}' attn={entry['attention']:.4f}")

        # Entity attention comparison: bare vs MB
        print("\n" + "-" * 110)
        print("ENTITY ATTENTION COMPARISON (final token -> entity in query)")
        print("-" * 110)
        print(f"\n  {'Layer':>5} | {'Bare q_entity':>14} | {'MB q_entity':>14} | {'MB mb_answer':>14}")
        print("  " + "-" * 60)

        for layer_idx in FOCUS_LAYERS:
            bare_qe = avg_bare.get(layer_idx, {}).get("query_entity", 0)
            mb_qe = avg_mb.get(layer_idx, {}).get("query_entity", 0)
            mb_ans = avg_mb.get(layer_idx, {}).get("mb_answer", 0)
            print(f"  L{layer_idx:>3}: {bare_qe:>14.4f} | {mb_qe:>14.4f} | {mb_ans:>14.4f}")

        # Key findings
        print("\n" + "=" * 110)
        print("KEY FINDINGS")
        print("=" * 110)

        interpretation = diag["interpretation"]
        if interpretation == "Distraction":
            print(
                "\n  The L20 dip is caused by DISTRACTION."
            )
            print(
                "  At L20, the MB condition attends heavily to instruction tokens and"
            )
            print(
                "  delimiters rather than the answer or entity. The explicit context"
            )
            print(
                "  draws attention away from the retrieval pathway."
            )
        elif interpretation == "Delayed integration":
            print(
                "\n  The L20 dip is caused by DELAYED INTEGRATION."
            )
            print(
                "  At L20, the model doesn't attend to the MB answer token."
            )
            print(
                "  The in-context answer is effectively ignored until L21."
            )
            print(
                "  The model only integrates the explicit context after"
            )
            print(
                "  completing its own retrieval computation."
            )
        elif interpretation == "Interference":
            print(
                "\n  The L20 dip is caused by INTERFERENCE."
            )
            print(
                "  At L20, the model attends to the MB answer but the"
            )
            print(
                "  representation interferes with the retrieval pathway."
            )

        print("=" * 110)

    def _save_results(self, results: dict) -> Path:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"mb_attention_at_l20_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
        return output_path

    async def run(self):
        await self.setup()

        logger.info("=" * 70)
        logger.info("MEMORY BANK ATTENTION AT L20")
        logger.info(f"  Facts: {len(FACTS)}")
        logger.info(f"  Focus layers: {FOCUS_LAYERS}")
        logger.info(f"  Comparing: bare vs memory bank attention patterns")
        logger.info("=" * 70)

        fact_results = []
        for fact in FACTS:
            result = await self.analyze_fact(fact)
            fact_results.append(result)

        summary = self._compute_summary(fact_results)

        output = {
            "metadata": {
                "experiment": "mb_attention_at_l20",
                "model": "openai/gpt-oss-20b",
                "timestamp": datetime.now().isoformat(),
                "num_facts": len(FACTS),
                "focus_layers": FOCUS_LAYERS,
                "description": (
                    "Compares attention patterns between bare and MB prompts "
                    "at L19-L22. Diagnoses why MB probability dips to 0.05% "
                    "at L20 while bare shows 14.8%. Tests distraction vs "
                    "interference vs delayed integration hypotheses."
                ),
                "prior_results": {
                    "mb_injection_point": "MB does NOT shift emergence; L20 probability dip",
                    "attention_at_emergence": "Entity attention peaks L19/L21; 1.3x increase",
                    "bare_l20_probability": "0.1479",
                    "mb_l20_probability": "0.0005",
                },
            },
            "summary": summary,
            "fact_results": fact_results,
        }

        self._save_results(output)
        self._print_summary(summary, fact_results)


async def main():
    experiment = MBAttentionAtL20()
    await experiment.run()


if __name__ == "__main__":
    asyncio.run(main())
