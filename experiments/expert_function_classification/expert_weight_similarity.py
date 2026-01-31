#!/usr/bin/env python3
"""Expert Weight Similarity Analysis.

Compares weight matrices of MoE experts within and across position classes.
No inference required - just tensor loading and cosine similarity.

Key question: Is the position-class redundancy (Part 11-12) due to
weight-level duplication (>0.9 = same weights learned N times) or
functional overlap (<0.5 = different weights, similar effect)?

GPT-OSS 20B expert structure (per layer):
  - 32 experts, each with gate_up_proj and down_proj
  - Weights in MXFP4 format: blocks (uint32), scales (uint8), biases (bfloat16)
  - gate_up_proj: (5760, 360) blocks + (5760, 90) scales + (5760,) bias
  - down_proj: (2880, 360) blocks + (2880, 90) scales + (2880,) bias

Comparison methods:
  1. Bias vectors (unquantized bfloat16) - direct weight comparison
  2. Scale vectors (uint8 magnitude structure) - weight group magnitudes
  3. Dequantized weights via quantized_matmul - full weight comparison

Run: python experiments/expert_function_classification/expert_weight_similarity.py
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Position class assignments at L16 from Part 11/12
L16_POSITION_CLASSES = {
    "start": [7, 8, 13, 20, 24, 25],
    "early_mid": [5, 17, 19, 21],
    "late_mid": [1, 12, 22],
    "end": [0, 2, 4, 10, 11, 14, 26, 27, 29, 30, 31],
    "none": [3, 6, 9, 15, 16, 18, 23, 28],  # 0 activations on test prompts
}

TARGET_LAYERS = [8, 12, 16, 20]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def load_model():
    """Load the model weights."""
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import (
        detect_model_family,
        get_family_info,
    )

    model_id = "openai/gpt-oss-20b"
    logger.info(f"Loading model: {model_id}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

    logger.info("Model loaded.")
    return model


def get_expert_bias_vector(experts, expert_idx: int) -> np.ndarray:
    """Get concatenated bias vector for an expert (unquantized)."""
    gate_up_bias = np.array(experts.gate_up_proj_bias[expert_idx].astype(mx.float32))
    down_bias = np.array(experts.down_proj_bias[expert_idx].astype(mx.float32))
    return np.concatenate([gate_up_bias, down_bias])


def get_expert_scale_vector(experts, expert_idx: int) -> np.ndarray:
    """Get concatenated flattened scale vector for an expert."""
    gate_up_scales = np.array(
        experts.gate_up_proj_scales[expert_idx].astype(mx.float32)
    ).flatten()
    down_scales = np.array(
        experts.down_proj_scales[expert_idx].astype(mx.float32)
    ).flatten()
    return np.concatenate([gate_up_scales, down_scales])


def dequantize_expert_weights(experts, expert_idx: int) -> np.ndarray:
    """Dequantize expert weights using quantized_matmul with identity.

    Returns a flattened vector of the full dequantized weight matrices.
    gate_up: I(2880) @ W.T -> (2880, 5760) transposed weight matrix
    down: I(2880) @ W.T -> (2880, 2880) transposed weight matrix (input is intermediate_size)
    """
    hidden_size = experts.hidden_size
    intermediate_size = experts.intermediate_size

    # Dequantize gate_up_proj: (hidden_size -> 2 * intermediate_size)
    identity_h = mx.eye(hidden_size, dtype=mx.bfloat16)
    gate_up_blocks = experts.gate_up_proj_blocks[expert_idx]
    gate_up_scales = experts.gate_up_proj_scales[expert_idx]

    gate_up_w_t = mx.quantized_matmul(
        identity_h,
        gate_up_blocks,
        scales=gate_up_scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    mx.eval(gate_up_w_t)

    # Dequantize down_proj: (intermediate_size -> hidden_size)
    identity_i = mx.eye(intermediate_size, dtype=mx.bfloat16)
    down_blocks = experts.down_proj_blocks[expert_idx]
    down_scales = experts.down_proj_scales[expert_idx]

    down_w_t = mx.quantized_matmul(
        identity_i,
        down_blocks,
        scales=down_scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    mx.eval(down_w_t)

    # Flatten and concatenate
    gate_up_flat = np.array(gate_up_w_t.astype(mx.float32)).flatten()
    down_flat = np.array(down_w_t.astype(mx.float32)).flatten()

    return np.concatenate([gate_up_flat, down_flat])


def compute_pairwise_similarities(
    experts,
    expert_indices: list[int],
    method: str = "bias",
) -> list[tuple[int, int, float]]:
    """Compute pairwise cosine similarity between experts."""
    # Get vectors
    vectors = {}
    for idx in expert_indices:
        if method == "bias":
            vectors[idx] = get_expert_bias_vector(experts, idx)
        elif method == "scale":
            vectors[idx] = get_expert_scale_vector(experts, idx)
        elif method == "dequantized":
            vectors[idx] = dequantize_expert_weights(experts, idx)
        else:
            raise ValueError(f"Unknown method: {method}")

    results = []
    for i, j in combinations(expert_indices, 2):
        sim = cosine_similarity(vectors[i], vectors[j])
        results.append((i, j, sim))

    return results


def analyze_layer(model, layer_idx: int, position_classes: dict[str, list[int]]):
    """Analyze expert weight similarity at one layer."""
    layer = model.model.layers[layer_idx]
    experts = layer.mlp.experts

    logger.info(f"\n{'='*60}")
    logger.info(f"Layer {layer_idx}")
    logger.info(f"{'='*60}")

    results = {}

    for method in ["bias", "scale"]:
        logger.info(f"\n  Method: {method}")
        method_results = {}

        # Within-class similarities
        for cls_name, cls_experts in position_classes.items():
            if len(cls_experts) < 2:
                logger.info(f"    {cls_name}: <2 experts, skipping")
                method_results[cls_name] = {
                    "n_experts": len(cls_experts),
                    "n_pairs": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "pairs": [],
                }
                continue

            sims = compute_pairwise_similarities(experts, cls_experts, method)
            sim_vals = [s for _, _, s in sims]

            method_results[cls_name] = {
                "n_experts": len(cls_experts),
                "n_pairs": len(sims),
                "mean": float(np.mean(sim_vals)),
                "std": float(np.std(sim_vals)),
                "min": float(np.min(sim_vals)),
                "max": float(np.max(sim_vals)),
                "pairs": [(i, j, float(s)) for i, j, s in sims],
            }

            logger.info(
                f"    {cls_name:>10} ({len(cls_experts):>2} experts, "
                f"{len(sims):>3} pairs): "
                f"mean={np.mean(sim_vals):.3f}, "
                f"std={np.std(sim_vals):.3f}, "
                f"range=[{np.min(sim_vals):.3f}, {np.max(sim_vals):.3f}]"
            )

        # Cross-class similarities (sample: compare 1st member of each class)
        all_active_experts = []
        for cls_name, cls_experts in position_classes.items():
            if cls_name != "none" and cls_experts:
                all_active_experts.extend(cls_experts)

        if len(all_active_experts) >= 2:
            cross_sims = compute_pairwise_similarities(
                experts, all_active_experts, method
            )
            cross_vals = [s for _, _, s in cross_sims]

            method_results["_cross_class"] = {
                "n_experts": len(all_active_experts),
                "n_pairs": len(cross_sims),
                "mean": float(np.mean(cross_vals)),
                "std": float(np.std(cross_vals)),
                "min": float(np.min(cross_vals)),
                "max": float(np.max(cross_vals)),
            }

            logger.info(
                f"    {'CROSS-CLASS':>10} ({len(all_active_experts):>2} experts, "
                f"{len(cross_sims):>3} pairs): "
                f"mean={np.mean(cross_vals):.3f}, "
                f"std={np.std(cross_vals):.3f}"
            )

        # None-class vs active experts
        none_experts = position_classes.get("none", [])
        if len(none_experts) >= 2:
            none_sims = compute_pairwise_similarities(experts, none_experts, method)
            none_vals = [s for _, _, s in none_sims]

            method_results["none"] = {
                "n_experts": len(none_experts),
                "n_pairs": len(none_sims),
                "mean": float(np.mean(none_vals)),
                "std": float(np.std(none_vals)),
                "min": float(np.min(none_vals)),
                "max": float(np.max(none_vals)),
            }

            logger.info(
                f"    {'none':>10} ({len(none_experts):>2} experts, "
                f"{len(none_sims):>3} pairs): "
                f"mean={np.mean(none_vals):.3f}, "
                f"std={np.std(none_vals):.3f}"
            )

        results[method] = method_results

    # Dequantized comparison for a subset (expensive)
    logger.info(f"\n  Method: dequantized (full weight comparison)")
    logger.info(f"  Computing for 'end' class ({len(position_classes['end'])} experts)...")

    end_experts = position_classes["end"]
    if len(end_experts) >= 2:
        deq_sims = compute_pairwise_similarities(experts, end_experts, "dequantized")
        deq_vals = [s for _, _, s in deq_sims]

        results["dequantized_end"] = {
            "n_experts": len(end_experts),
            "n_pairs": len(deq_sims),
            "mean": float(np.mean(deq_vals)),
            "std": float(np.std(deq_vals)),
            "min": float(np.min(deq_vals)),
            "max": float(np.max(deq_vals)),
            "pairs": [(i, j, float(s)) for i, j, s in deq_sims],
        }

        logger.info(
            f"    end (dequantized): "
            f"mean={np.mean(deq_vals):.3f}, "
            f"std={np.std(deq_vals):.3f}, "
            f"range=[{np.min(deq_vals):.3f}, {np.max(deq_vals):.3f}]"
        )

    # Also dequantize start class for comparison
    start_experts = position_classes["start"]
    if len(start_experts) >= 2:
        logger.info(
            f"  Computing for 'start' class ({len(start_experts)} experts)..."
        )
        start_sims = compute_pairwise_similarities(
            experts, start_experts, "dequantized"
        )
        start_vals = [s for _, _, s in start_sims]

        results["dequantized_start"] = {
            "n_experts": len(start_experts),
            "n_pairs": len(start_sims),
            "mean": float(np.mean(start_vals)),
            "std": float(np.std(start_vals)),
            "min": float(np.min(start_vals)),
            "max": float(np.max(start_vals)),
            "pairs": [(i, j, float(s)) for i, j, s in start_sims],
        }

        logger.info(
            f"    start (dequantized): "
            f"mean={np.mean(start_vals):.3f}, "
            f"std={np.std(start_vals):.3f}, "
            f"range=[{np.min(start_vals):.3f}, {np.max(start_vals):.3f}]"
        )

    # Cross-class dequantized (sample: 3 end + 3 start)
    cross_sample = end_experts[:3] + start_experts[:3]
    if len(cross_sample) >= 4:
        logger.info("  Computing cross-class dequantized (3 end + 3 start)...")
        cross_deq_sims = compute_pairwise_similarities(
            experts, cross_sample, "dequantized"
        )
        # Separate within vs cross
        end_set = set(end_experts[:3])
        start_set = set(start_experts[:3])
        within_vals = [
            s
            for i, j, s in cross_deq_sims
            if (i in end_set and j in end_set) or (i in start_set and j in start_set)
        ]
        cross_vals = [
            s
            for i, j, s in cross_deq_sims
            if (i in end_set) != (j in end_set)
        ]

        results["dequantized_cross_sample"] = {
            "within_mean": float(np.mean(within_vals)) if within_vals else None,
            "cross_mean": float(np.mean(cross_vals)) if cross_vals else None,
            "within_pairs": len(within_vals),
            "cross_pairs": len(cross_vals),
        }

        logger.info(
            f"    within-class: {np.mean(within_vals):.3f} ({len(within_vals)} pairs)"
        )
        logger.info(
            f"    cross-class:  {np.mean(cross_vals):.3f} ({len(cross_vals)} pairs)"
        )

    return results


def print_summary(all_results: dict[int, dict]):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("EXPERT WEIGHT SIMILARITY ANALYSIS")
    print("=" * 70)

    for layer_idx, results in sorted(all_results.items()):
        print(f"\n--- Layer {layer_idx} ---")

        for method in ["bias", "scale"]:
            if method not in results:
                continue
            print(f"\n  {method.upper()} similarity:")
            print(f"  {'Class':>12} | {'N':>3} | {'Pairs':>5} | {'Mean':>6} | {'Std':>6} | {'Range':>15}")
            print(f"  {'-'*12}-+-{'-'*3}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*15}")

            for cls_name, cls_data in results[method].items():
                if cls_name.startswith("_"):
                    cls_name = "ALL-CROSS"
                if cls_data.get("mean") is None:
                    print(f"  {cls_name:>12} | {cls_data['n_experts']:>3} | {'--':>5} | {'--':>6} | {'--':>6} | {'--':>15}")
                    continue
                rng = f"[{cls_data['min']:.3f}, {cls_data['max']:.3f}]"
                print(
                    f"  {cls_name:>12} | {cls_data['n_experts']:>3} | "
                    f"{cls_data['n_pairs']:>5} | {cls_data['mean']:>6.3f} | "
                    f"{cls_data['std']:>6.3f} | {rng:>15}"
                )

        # Dequantized results
        for key in ["dequantized_end", "dequantized_start"]:
            if key in results:
                cls_name = key.replace("dequantized_", "")
                d = results[key]
                print(
                    f"\n  DEQUANTIZED ({cls_name}): "
                    f"mean={d['mean']:.3f}, std={d['std']:.3f}, "
                    f"range=[{d['min']:.3f}, {d['max']:.3f}]"
                )

        if "dequantized_cross_sample" in results:
            d = results["dequantized_cross_sample"]
            print(
                f"  DEQUANTIZED cross-class: "
                f"within={d['within_mean']:.3f}, cross={d['cross_mean']:.3f}"
            )

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Use L16 bias results for interpretation
    if 16 in all_results and "bias" in all_results[16]:
        bias = all_results[16]["bias"]
        end_mean = bias.get("end", {}).get("mean")
        cross_mean = bias.get("_cross_class", {}).get("mean")

        if end_mean is not None:
            if end_mean > 0.9:
                print(
                    f"\n  'end' class mean similarity: {end_mean:.3f} (>0.9)"
                )
                print("  -> WEIGHT-LEVEL DUPLICATION: Experts are literally the same")
                print("     weights learned multiple times. MoE training is wasteful.")
                print("     Merging: pick any one expert per position class.")
            elif end_mean > 0.7:
                print(
                    f"\n  'end' class mean similarity: {end_mean:.3f} (0.7-0.9)"
                )
                print("  -> HIGH SIMILARITY: Experts are very similar but not identical.")
                print("     Merging: averaging weights should work well.")
            elif end_mean > 0.5:
                print(
                    f"\n  'end' class mean similarity: {end_mean:.3f} (0.5-0.7)"
                )
                print("  -> MODERATE SIMILARITY: Experts overlap but are distinct.")
                print("     Merging: averaging may lose information. SVD-based")
                print("     compression might preserve more structure.")
            else:
                print(
                    f"\n  'end' class mean similarity: {end_mean:.3f} (<0.5)"
                )
                print("  -> LOW SIMILARITY: Functional redundancy, not weight duplication.")
                print("     Experts do different things at the same position.")
                print("     Merging: likely destructive. Different approach needed.")

        if cross_mean is not None and end_mean is not None:
            ratio = end_mean / cross_mean if cross_mean > 0 else float("inf")
            print(
                f"\n  Within-class/cross-class ratio: {ratio:.2f} "
                f"({end_mean:.3f} / {cross_mean:.3f})"
            )
            if ratio > 1.5:
                print("  -> Position classes ARE meaningful groupings at the weight level")
            elif ratio > 1.1:
                print("  -> Position classes show SOME weight-level structure")
            else:
                print("  -> Position classes are NOT reflected in weight similarity")


def main():
    model = load_model()

    all_results = {}

    # Primary analysis at L16 (where we have position class data)
    all_results[16] = analyze_layer(model, 16, L16_POSITION_CLASSES)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"expert_weight_similarity_{timestamp}.json"

    output = {
        "metadata": {
            "experiment": "expert_weight_similarity",
            "model": "openai/gpt-oss-20b",
            "timestamp": timestamp,
            "position_classes": L16_POSITION_CLASSES,
            "methods": ["bias", "scale", "dequantized"],
        },
        "results": {str(k): v for k, v in all_results.items()},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
