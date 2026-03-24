"""Output formatters for MoE expert CLI commands.

Provides consistent, structured output formatting for all MoE expert actions.
Separates presentation logic from business logic.
"""

from ._ablation import format_ablation_result, format_entropy_analysis
from ._base import format_header, format_model_info, format_subheader
from ._compression import (
    format_overlay_result,
    format_storage_estimate,
    format_verification_result,
)
from ._generation import (
    format_chat_result,
    format_comparison_result,
    format_topk_result,
)
from ._moe_type import format_moe_type_comparison, format_moe_type_result
from ._routing import format_coactivation, format_router_weights, format_taxonomy
from ._visualization import format_orthogonality_ascii

__all__ = [
    "format_ablation_result",
    "format_chat_result",
    "format_coactivation",
    "format_comparison_result",
    "format_entropy_analysis",
    "format_header",
    "format_model_info",
    "format_moe_type_comparison",
    "format_moe_type_result",
    "format_orthogonality_ascii",
    "format_overlay_result",
    "format_router_weights",
    "format_storage_estimate",
    "format_subheader",
    "format_taxonomy",
    "format_topk_result",
    "format_verification_result",
]
