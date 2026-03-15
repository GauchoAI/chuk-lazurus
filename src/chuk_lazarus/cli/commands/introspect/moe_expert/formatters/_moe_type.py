"""MoE type analysis formatters for MoE expert CLI output."""

from __future__ import annotations

from ......introspection.moe import MoEType, MoETypeAnalysis
from ._base import format_header


def format_moe_type_result(result: MoETypeAnalysis) -> str:
    """Format MoE type analysis for display.

    Args:
        result: MoE type analysis result.

    Returns:
        Formatted output string.
    """
    type_labels = {
        MoEType.UPCYCLED: "UPCYCLED",
        MoEType.PRETRAINED_MOE: "PRETRAINED_MOE",
        MoEType.UNKNOWN: "UNKNOWN",
        # Legacy support
        MoEType.PSEUDO: "UPCYCLED",
        MoEType.NATIVE: "PRETRAINED_MOE",
    }
    type_label = type_labels.get(result.moe_type, "UNKNOWN")

    compressible = "Yes" if result.is_compressible else "No"
    compression = f"{result.estimated_compression:.1f}x" if result.is_compressible else "N/A"

    # Training origin description
    if result.moe_type == MoEType.UPCYCLED:
        origin_desc = "Dense model converted to MoE (upcycling)"
    elif result.moe_type == MoEType.PRETRAINED_MOE:
        origin_desc = "Trained as MoE from scratch"
    else:
        origin_desc = "Training origin unclear"

    lines = [
        format_header("MOE TYPE ANALYSIS"),
        f"Model:  {result.model_id}",
        f"Layer:  {result.layer_idx}",
        f"Type:   {type_label}",
        f"Origin: {origin_desc}",
        f"Confidence: {result.confidence:.0%} ({result.confidence_label})",
        "",
        "Evidence:",
        f"  Gate Rank:         {result.gate.effective_rank_95:>4} / {result.gate.max_rank:<4} ({result.gate.rank_ratio * 100:>5.1f}%)",
        f"  Up Rank:           {result.up.effective_rank_95:>4} / {result.up.max_rank:<4} ({result.up.rank_ratio * 100:>5.1f}%)",
        f"  Down Rank:         {result.down.effective_rank_95:>4} / {result.down.max_rank:<4} ({result.down.rank_ratio * 100:>5.1f}%)",
        f"  Cosine Similarity: {result.mean_cosine_similarity:.3f} (+/- {result.std_cosine_similarity:.3f})",
    ]

    # Add training signals if available
    if result.training_signals:
        lines.extend(
            [
                "",
                "Training Signals:",
                f"  Expert Similarity:    {result.training_signals.expert_similarity:.4f}",
                f"  Rank Ratio:           {result.training_signals.rank_ratio:.4f}",
                f"  Norm Variance:        {result.training_signals.expert_norm_variance:.4f}",
                f"  Upcycled Score:       {result.training_signals.upcycled_score:.2f}",
                f"  Pretrained Score:     {result.training_signals.pretrained_score:.2f}",
            ]
        )

    lines.extend(
        [
            "",
            "Compression:",
            f"  Compressible:      {compressible}",
            f"  Estimated Ratio:   {compression}",
            "=" * 70,
        ]
    )
    return "\n".join(lines)


def format_moe_type_comparison(r1: MoETypeAnalysis, r2: MoETypeAnalysis) -> str:
    """Format side-by-side MoE type comparison.

    Args:
        r1: First model's analysis.
        r2: Second model's analysis.

    Returns:
        Formatted comparison table.
    """

    def _type_str(r: MoETypeAnalysis) -> str:
        return {
            MoEType.UPCYCLED: "UPCYCLED",
            MoEType.PRETRAINED_MOE: "PRETRAINED",
            MoEType.UNKNOWN: "UNKNOWN",
            # Legacy support
            MoEType.PSEUDO: "UPCYCLED",
            MoEType.NATIVE: "PRETRAINED",
        }.get(r.moe_type, "UNKNOWN")

    def _compress_str(r: MoETypeAnalysis) -> str:
        return f"Yes ({r.estimated_compression:.1f}x)" if r.is_compressible else "No"

    def _confidence_str(r: MoETypeAnalysis) -> str:
        return f"{r.confidence:.0%}"

    # Truncate model names for table (use last path component)
    name1 = r1.model_id.split("/")[-1][:14]
    name2 = r2.model_id.split("/")[-1][:14]

    lines = [
        format_header("MOE TYPE COMPARISON"),
        f"+-----------------------+{'-' * 16}+{'-' * 16}+",
        f"| {'Metric':<21} | {name1:<14} | {name2:<14} |",
        f"+-----------------------+{'-' * 16}+{'-' * 16}+",
        f"| {'Type':<21} | {_type_str(r1):<14} | {_type_str(r2):<14} |",
        f"| {'Confidence':<21} | {_confidence_str(r1):<14} | {_confidence_str(r2):<14} |",
        f"| {'Gate Rank':<21} | {r1.gate.effective_rank_95:>4}/{r1.gate.max_rank:<9} | {r2.gate.effective_rank_95:>4}/{r2.gate.max_rank:<9} |",
        f"| {'Gate Rank %':<21} | {r1.gate.rank_ratio * 100:>13.1f}% | {r2.gate.rank_ratio * 100:>13.1f}% |",
        f"| {'Cosine Similarity':<21} | {r1.mean_cosine_similarity:>14.3f} | {r2.mean_cosine_similarity:>14.3f} |",
        f"| {'Compressible':<21} | {_compress_str(r1):<14} | {_compress_str(r2):<14} |",
        f"+-----------------------+{'-' * 16}+{'-' * 16}+",
    ]
    return "\n".join(lines)
