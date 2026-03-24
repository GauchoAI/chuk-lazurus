"""Generation-related formatters for MoE expert CLI output."""

from __future__ import annotations

from ......introspection.moe import (
    ExpertChatResult,
    ExpertComparisonResult,
    TopKVariationResult,
)
from ._base import format_header


def format_chat_result(
    result: ExpertChatResult,
    model_id: str,
    moe_type: str,
    *,
    verbose: bool = False,
) -> str:
    """Format chat result for display.

    Args:
        result: Chat result from ExpertRouter.
        model_id: Model identifier.
        moe_type: Type of MoE architecture.
        verbose: Whether to include detailed statistics.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header(f"CHAT WITH EXPERT {result.expert_idx}"),
        f"Model: {model_id}",
        f"MoE type: {moe_type}",
        "",
        f"Prompt: {result.prompt}",
        "",
        "Response:",
        result.response,
    ]

    if verbose:
        lines.extend(
            [
                "",
                "Statistics:",
                f"  Tokens generated: {result.stats.tokens_generated}",
                f"  Layers modified: {result.stats.layers_modified}",
                f"  Prompt tokens: {result.stats.prompt_tokens}",
            ]
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def format_comparison_result(
    result: ExpertComparisonResult,
    model_id: str,
    *,
    verbose: bool = False,
) -> str:
    """Format comparison result for display.

    Args:
        result: Comparison result from ExpertRouter.
        model_id: Model identifier.
        verbose: Whether to include detailed statistics.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("EXPERT COMPARISON"),
        f"Model: {model_id}",
        f"Prompt: {result.prompt}",
        "",
    ]

    for expert_result in result.expert_results:
        lines.append(f"--- Expert {expert_result.expert_idx} ---")
        lines.append(expert_result.response)
        if verbose:
            lines.append(f"  (tokens: {expert_result.stats.tokens_generated})")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_topk_result(result: TopKVariationResult, model_id: str) -> str:
    """Format top-k variation result for display.

    Args:
        result: Top-k variation result.
        model_id: Model identifier.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header(f"TOP-K EXPERIMENT - Using k={result.k_value} (default: {result.default_k})"),
        f"Model: {model_id}",
        f"Prompt: {result.prompt}",
        "",
        f"Normal (k={result.default_k}): {result.normal_response}",
        f"Modified (k={result.k_value}): {result.response}",
        "",
    ]

    if result.response != result.normal_response:
        lines.append("** OUTPUTS DIFFER **")
    else:
        lines.append("Outputs are identical")

    lines.append("=" * 70)
    return "\n".join(lines)
