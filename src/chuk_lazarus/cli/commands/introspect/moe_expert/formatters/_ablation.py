"""Ablation-related formatters for MoE expert CLI output."""

from __future__ import annotations

from ._base import format_header


def format_ablation_result(
    normal_output: str,
    ablated_output: str,
    expert_indices: list[int],
    prompt: str,
    model_id: str,
) -> str:
    """Format ablation result for display.

    Args:
        normal_output: Output without ablation.
        ablated_output: Output with ablation.
        expert_indices: Experts that were ablated.
        prompt: The input prompt.
        model_id: Model identifier.

    Returns:
        Formatted output string.
    """
    experts_str = ", ".join(str(e) for e in expert_indices)
    lines = [
        format_header(f"ABLATION - Expert(s) {experts_str}"),
        f"Model: {model_id}",
        f"Prompt: {prompt}",
        "",
        f"Normal:  {normal_output}",
        f"Ablated: {ablated_output}",
        "",
    ]

    if normal_output != ablated_output:
        lines.append("** OUTPUTS DIFFER - Expert(s) had an effect! **")
    else:
        lines.append("Outputs are identical - Expert(s) had no effect")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_entropy_analysis(
    entropies: list[tuple[int, float, float]],
    model_id: str,
    prompt: str,
) -> str:
    """Format routing entropy analysis for display.

    Args:
        entropies: List of (layer_idx, mean_entropy, normalized_entropy) tuples.
        model_id: Model identifier.
        prompt: The analyzed prompt.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("ROUTING ENTROPY ANALYSIS"),
        f"Model: {model_id}",
        f"Prompt: {prompt}",
        "",
        "Layer  Mean Entropy  Normalized",
        "-" * 35,
    ]

    for layer_idx, mean_ent, norm_ent in entropies:
        bar = "#" * int(norm_ent * 20)
        lines.append(f"  {layer_idx:3d}    {mean_ent:6.3f}       {norm_ent:.3f} {bar}")

    lines.append("=" * 70)
    return "\n".join(lines)
