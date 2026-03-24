"""MoE compression via SVD overlay representation.

Implements the overlay compression strategy for pseudo-MoE models:
    expert_i = base + U_i @ S_i @ V_i^T

Where base is the mean expert and U_i, S_i, V_i are truncated SVD factors
of the low-rank delta (expert_i - base).

This provides significant compression for pseudo-MoE models (8x typical)
while preserving quality (<1% reconstruction error).
"""

from ._models import (
    CompressionConfig,
    CompressionResult,
    OverlayRepresentation,
    ProjectionOverlay,
    ReconstructionError,
    ReconstructionVerification,
    StorageEstimate,
)
from ._overlay_experts import OverlayExperts
from ._service import MoECompressionService

__all__ = [
    "CompressionConfig",
    "CompressionResult",
    "MoECompressionService",
    "OverlayExperts",
    "OverlayRepresentation",
    "ProjectionOverlay",
    "ReconstructionError",
    "ReconstructionVerification",
    "StorageEstimate",
]
