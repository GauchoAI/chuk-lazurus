"""Routing-related formatters for MoE expert CLI output."""

from __future__ import annotations

from ......introspection.moe import (
    CoactivationAnalysis,
    ExpertTaxonomy,
    LayerRouterWeights,
)
from ._base import format_header, format_subheader


def format_router_weights(
    weights: list[LayerRouterWeights],
    model_id: str,
    prompt: str,
) -> str:
    """Format router weights for display.

    Args:
        weights: Router weights from capture.
        model_id: Model identifier.
        prompt: The analyzed prompt.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("ROUTER WEIGHTS"),
        f"Model: {model_id}",
        f"Prompt: {prompt}",
        "",
    ]

    for layer_weights in weights:
        lines.append(f"Layer {layer_weights.layer_idx}:")
        for pos in layer_weights.positions:
            experts_str = ", ".join(
                f"E{e}({w:.3f})" for e, w in zip(pos.expert_indices, pos.weights)
            )
            lines.append(f"  [{pos.position_idx}] '{pos.token}': {experts_str}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_coactivation(
    analysis: CoactivationAnalysis,
    model_id: str,
    layer_idx: int,
) -> str:
    """Format co-activation analysis for display.

    Args:
        analysis: Co-activation analysis result.
        model_id: Model identifier.
        layer_idx: Layer that was analyzed.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header(f"CO-ACTIVATION ANALYSIS - Layer {layer_idx}"),
        f"Model: {model_id}",
        f"Total activations: {analysis.total_activations}",
        "",
        "Top Expert Pairs:",
    ]

    for pair in analysis.top_pairs[:10]:
        lines.append(
            f"  E{pair.expert_a} + E{pair.expert_b}: "
            f"{pair.coactivation_count} times ({pair.coactivation_rate:.1%})"
        )

    if analysis.generalist_experts:
        lines.append("")
        lines.append(f"Generalist experts: {list(analysis.generalist_experts)}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_taxonomy(taxonomy: ExpertTaxonomy, *, verbose: bool = False) -> str:
    """Format full expert taxonomy for display.

    Args:
        taxonomy: Expert taxonomy result.
        verbose: Whether to include all details.

    Returns:
        Formatted output string.
    """
    from collections import Counter

    lines = [
        format_header("EXPERT TAXONOMY"),
        f"Model: {taxonomy.model_id}",
        f"Layers: {taxonomy.num_layers}",
        f"Experts per layer: {taxonomy.num_experts}",
        f"Total experts analyzed: {len(taxonomy.expert_identities)}",
    ]

    # Group by layer
    by_layer: dict[int, list] = {}
    for identity in taxonomy.expert_identities:
        if identity.layer_idx not in by_layer:
            by_layer[identity.layer_idx] = []
        by_layer[identity.layer_idx].append(identity)

    # Overall statistics
    all_categories = Counter(identity.primary_category for identity in taxonomy.expert_identities)
    all_roles = Counter(identity.role.value for identity in taxonomy.expert_identities)

    # Group categories by type for summary
    code_cats = {k: v for k, v in all_categories.items() if k.startswith("code:")}
    structure_cats = {
        k: v
        for k, v in all_categories.items()
        if k
        in (
            "bracket",
            "operator",
            "punctuation",
            "identifier",
            "constant",
            "variable",
            "short_identifier",
        )
    }
    lang_cats = {
        k: v
        for k, v in all_categories.items()
        if k in ("function_word", "capitalized", "content", "whitespace", "number")
    }
    total_experts = len(taxonomy.expert_identities)

    lines.append("")
    lines.append("Category Summary:")

    # Code keywords summary
    if code_cats:
        code_total = sum(code_cats.values())
        code_pct = code_total / total_experts * 100
        code_details = ", ".join(
            f"{k.split(':')[1]}({v})" for k, v in sorted(code_cats.items(), key=lambda x: -x[1])[:5]
        )
        lines.append(f"  Code Keywords:    {code_total:4d} ({code_pct:5.1f}%) [{code_details}]")

    # Structure tokens
    if structure_cats:
        struct_total = sum(structure_cats.values())
        struct_pct = struct_total / total_experts * 100
        struct_details = ", ".join(
            f"{k}({v})" for k, v in sorted(structure_cats.items(), key=lambda x: -x[1])[:4]
        )
        lines.append(
            f"  Code Structure:   {struct_total:4d} ({struct_pct:5.1f}%) [{struct_details}]"
        )

    # Language tokens
    if lang_cats:
        lang_total = sum(lang_cats.values())
        lang_pct = lang_total / total_experts * 100
        lang_details = ", ".join(
            f"{k}({v})" for k, v in sorted(lang_cats.items(), key=lambda x: -x[1])[:4]
        )
        lines.append(f"  Language/Other:   {lang_total:4d} ({lang_pct:5.1f}%) [{lang_details}]")

    lines.append("")
    lines.append("Detailed Category Distribution:")
    for cat, count in all_categories.most_common():
        pct = count / total_experts * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {cat:<20} {count:4d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("Role Distribution: ")
    role_parts = [f"{role}: {count}" for role, count in all_roles.most_common()]
    lines[-1] += ", ".join(role_parts)

    # High-confidence specialists (notable experts)
    specialists = [
        e for e in taxonomy.expert_identities if e.role.value == "specialist" and e.confidence > 0.6
    ]
    if specialists:
        specialists.sort(key=lambda e: e.confidence, reverse=True)
        lines.append("")
        lines.append(format_subheader("HIGH-CONFIDENCE SPECIALISTS"))
        for exp in specialists[:20]:  # Show top 20
            tokens_str = ""
            if exp.top_tokens:
                tokens_str = f" tokens: {', '.join(repr(t) for t in exp.top_tokens[:3])}"
            lines.append(
                f"  L{exp.layer_idx:02d} E{exp.expert_idx:02d}: "
                f"{exp.primary_category:<15} "
                f"({exp.confidence:5.1%} conf, {exp.activation_rate:5.1%} act)"
                f"{tokens_str}"
            )
        if len(specialists) > 20:
            lines.append(f"  ... and {len(specialists) - 20} more specialists")

    # Per-layer summaries
    lines.append("")
    lines.append(format_subheader("LAYER SUMMARIES"))

    for layer_idx in sorted(by_layer.keys()):
        layer_experts = by_layer[layer_idx]
        layer_categories = Counter(e.primary_category for e in layer_experts)
        layer_specialists = sum(1 for e in layer_experts if e.role.value == "specialist")
        avg_confidence = sum(e.confidence for e in layer_experts) / len(layer_experts)

        # Top 2 categories for this layer
        top_cats = layer_categories.most_common(2)
        top_cats_str = ", ".join(f"{cat}({cnt})" for cat, cnt in top_cats)

        lines.append(
            f"  Layer {layer_idx:2d}: "
            f"{len(layer_experts):2d} experts, "
            f"{layer_specialists:2d} specialists, "
            f"avg conf {avg_confidence:.1%}, "
            f"top: {top_cats_str}"
        )

    # Detailed per-layer breakdown (verbose only)
    if verbose:
        lines.append("")
        lines.append(format_subheader("DETAILED LAYER BREAKDOWN"))

        for layer_idx in sorted(by_layer.keys()):
            layer_experts = by_layer[layer_idx]
            # Sort by confidence descending
            layer_experts.sort(key=lambda e: e.confidence, reverse=True)

            lines.append(f"\n  Layer {layer_idx}:")
            for exp in layer_experts:
                tokens_str = ""
                if exp.top_tokens:
                    tokens_str = f" [{', '.join(repr(t) for t in exp.top_tokens[:3])}]"
                role_marker = "★" if exp.role.value == "specialist" else "○"
                lines.append(
                    f"    {role_marker} E{exp.expert_idx:02d}: "
                    f"{exp.primary_category:<15} "
                    f"{exp.confidence:5.1%} conf, {exp.activation_rate:5.1%} act"
                    f"{tokens_str}"
                )

    if taxonomy.patterns:
        lines.append("")
        lines.append(format_subheader("DISCOVERED PATTERNS"))
        for pattern in taxonomy.patterns[:20]:
            tokens = ", ".join(f"'{t}'" for t in pattern.trigger_tokens[:3])
            lines.append(
                f"  E{pattern.expert_idx}@L{pattern.layer_idx}: {pattern.pattern_type} - {tokens}"
            )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
