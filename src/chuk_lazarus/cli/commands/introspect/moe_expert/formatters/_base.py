"""Base formatting utilities for MoE expert CLI output."""

from __future__ import annotations

from ......introspection.moe import MoEModelInfo


def format_header(title: str, width: int = 70) -> str:
    """Format a section header.

    Args:
        title: Header title.
        width: Total width of the header line.

    Returns:
        Formatted header string.
    """
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def format_subheader(title: str, width: int = 70) -> str:
    """Format a subsection header.

    Args:
        title: Header title.
        width: Total width of the header line.

    Returns:
        Formatted subheader string.
    """
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def format_model_info(info: MoEModelInfo, model_id: str) -> str:
    """Format model information for display.

    Args:
        info: MoE model information.
        model_id: Model identifier.

    Returns:
        Formatted model info string.
    """
    lines = [
        f"Model: {model_id}",
        f"  Architecture: {info.architecture.value}",
        f"  Total layers: {info.total_layers}",
        f"  MoE layers: {len(info.moe_layers)}",
        f"  Experts per layer: {info.num_experts}",
        f"  Experts per token: {info.num_experts_per_tok}",
    ]
    if info.has_shared_expert:
        lines.append("  Has shared expert: Yes")
    return "\n".join(lines)
