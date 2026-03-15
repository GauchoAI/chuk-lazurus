"""Pydantic models for MoE compression via SVD overlay representation.

Implements the overlay compression strategy for pseudo-MoE models:
    expert_i = base + U_i @ S_i @ V_i^T

Where base is the mean expert and U_i, S_i, V_i are truncated SVD factors
of the low-rank delta (expert_i - base).

This provides significant compression for pseudo-MoE models (8x typical)
while preserving quality (<1% reconstruction error).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProjectionOverlay(BaseModel):
    """Overlay representation for a single projection type."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(description="Projection name: gate, up, or down")
    shape: tuple[int, int] = Field(description="(out_features, in_features)")
    rank: int = Field(ge=1, description="Truncation rank used")
    num_experts: int = Field(ge=1, description="Number of experts")

    # Storage metrics
    original_bytes: int = Field(ge=0, description="Original storage in bytes")
    compressed_bytes: int = Field(ge=0, description="Compressed storage in bytes")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else 1.0


class OverlayRepresentation(BaseModel):
    """Complete overlay representation for a layer's experts."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    layer_idx: int = Field(ge=0, description="Layer index")
    num_experts: int = Field(ge=1, description="Number of experts")

    # Per-projection overlays
    gate: ProjectionOverlay = Field(description="Gate projection overlay")
    up: ProjectionOverlay = Field(description="Up projection overlay")
    down: ProjectionOverlay = Field(description="Down projection overlay")

    # Ranks used
    gate_rank: int = Field(ge=1, description="Rank used for gate projection")
    up_rank: int = Field(ge=1, description="Rank used for up projection")
    down_rank: int = Field(ge=1, description="Rank used for down projection")

    @property
    def total_original_bytes(self) -> int:
        """Total original storage in bytes."""
        return self.gate.original_bytes + self.up.original_bytes + self.down.original_bytes

    @property
    def total_compressed_bytes(self) -> int:
        """Total compressed storage in bytes."""
        return self.gate.compressed_bytes + self.up.compressed_bytes + self.down.compressed_bytes

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        return (
            self.total_original_bytes / self.total_compressed_bytes
            if self.total_compressed_bytes > 0
            else 1.0
        )


class ReconstructionError(BaseModel):
    """Reconstruction error metrics for a projection."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Projection name")
    mean_relative_error: float = Field(ge=0.0, description="Mean relative error across experts")
    max_relative_error: float = Field(ge=0.0, description="Max relative error across experts")
    mean_mse: float = Field(ge=0.0, description="Mean squared error")


class ReconstructionVerification(BaseModel):
    """Verification results for overlay reconstruction."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    layer_idx: int = Field(ge=0, description="Layer index")

    # Per-projection errors
    gate: ReconstructionError = Field(description="Gate projection errors")
    up: ReconstructionError = Field(description="Up projection errors")
    down: ReconstructionError = Field(description="Down projection errors")

    # Output-level verification
    mean_output_error: float = Field(ge=0.0, description="Mean output relative error")
    max_output_error: float = Field(ge=0.0, description="Max output relative error")

    # Ranks used
    gate_rank: int = Field(ge=1, description="Rank used for gate")
    up_rank: int = Field(ge=1, description="Rank used for up")
    down_rank: int = Field(ge=1, description="Rank used for down")

    @property
    def passed(self) -> bool:
        """Whether reconstruction quality is acceptable (<1% error)."""
        return self.max_output_error < 0.01

    @property
    def overall_weight_error(self) -> float:
        """Mean weight error across all projections."""
        return (
            self.gate.mean_relative_error
            + self.up.mean_relative_error
            + self.down.mean_relative_error
        ) / 3


class StorageEstimate(BaseModel):
    """Storage estimate for overlay compression."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    num_layers: int = Field(ge=1, description="Number of MoE layers")
    num_experts: int = Field(ge=1, description="Number of experts per layer")

    # Storage in MB
    original_mb: float = Field(ge=0.0, description="Original storage in MB")
    compressed_mb: float = Field(ge=0.0, description="Compressed storage in MB")

    # Ranks used
    gate_rank: int = Field(ge=1, description="Rank for gate projection")
    up_rank: int = Field(ge=1, description="Rank for up projection")
    down_rank: int = Field(ge=1, description="Rank for down projection")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.original_mb / self.compressed_mb if self.compressed_mb > 0 else 1.0

    @property
    def savings_mb(self) -> float:
        """Storage savings in MB."""
        return self.original_mb - self.compressed_mb


class CompressionConfig(BaseModel):
    """Configuration for compressed model format."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Original model identifier")
    num_layers: int = Field(ge=1, description="Number of MoE layers")
    num_experts: int = Field(ge=1, description="Number of experts per layer")
    moe_layer_indices: list[int] = Field(description="Indices of MoE layers in original model")

    # Projection dimensions
    gate_shape: tuple[int, int] = Field(description="(out_features, in_features) for gate")
    up_shape: tuple[int, int] = Field(description="(out_features, in_features) for up")
    down_shape: tuple[int, int] = Field(description="(out_features, in_features) for down")

    # Compression ranks
    gate_rank: int = Field(ge=1, description="Rank for gate projection")
    up_rank: int = Field(ge=1, description="Rank for up projection")
    down_rank: int = Field(ge=1, description="Rank for down projection")

    # Bias info
    has_biases: bool = Field(default=False, description="Whether biases are stored separately")

    # Storage stats
    original_bytes: int = Field(ge=0, description="Original expert storage in bytes")
    compressed_bytes: int = Field(ge=0, description="Compressed storage in bytes")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else 1.0


class CompressionResult(BaseModel):
    """Result of model compression."""

    model_config = ConfigDict(frozen=True)

    output_path: str = Field(description="Path to compressed model directory")
    config: CompressionConfig = Field(description="Compression configuration")
    mean_reconstruction_error: float = Field(ge=0.0, description="Mean reconstruction error")
    max_reconstruction_error: float = Field(ge=0.0, description="Max reconstruction error")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.config.compression_ratio

    @property
    def original_mb(self) -> float:
        """Original size in MB."""
        return self.config.original_bytes / (1024 * 1024)

    @property
    def compressed_mb(self) -> float:
        """Compressed size in MB."""
        return self.config.compressed_bytes / (1024 * 1024)
