"""Overlay experts for efficient inference using compressed MoE representation."""

from __future__ import annotations

import logging
from pathlib import Path

from ._models import CompressionConfig

logger = logging.getLogger(__name__)


class OverlayExperts:
    """Efficient expert computation using overlay representation.

    Instead of storing full expert weights, stores:
    - base: Mean expert weight (shared)
    - U, V: Low-rank factors per expert
    - biases: Per-expert biases (if model has them)

    Reconstruction: expert_i = base + U_i @ V_i

    Usage:
        experts = OverlayExperts.load("gpt-oss-20b-overlay")
        weight = experts.get_expert_weight(layer=0, projection="gate", expert=5)
        # or for efficient inference:
        output = experts.apply_expert(layer=0, projection="gate", expert=5, x=hidden)
    """

    def __init__(
        self,
        config: CompressionConfig,
        base_weights: dict,
        delta_weights: dict,
        biases: dict | None = None,
    ) -> None:
        """Initialize from loaded weights."""
        self.config = config
        self._base = base_weights
        self._deltas = delta_weights
        self._biases = biases or {}

    @classmethod
    def load(cls, path: str | Path) -> OverlayExperts:
        """Load compressed model from disk."""
        import mlx.core as mx

        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = CompressionConfig.model_validate_json(config_path.read_text())

        # Load weights
        base_path = path / "base_weights.safetensors"
        deltas_path = path / "deltas.safetensors"

        if not base_path.exists():
            raise FileNotFoundError(f"Base weights not found: {base_path}")
        if not deltas_path.exists():
            raise FileNotFoundError(f"Deltas not found: {deltas_path}")

        base_weights = mx.load(str(base_path))
        delta_weights = mx.load(str(deltas_path))

        # Load biases if available
        biases = None
        biases_path = path / "biases.safetensors"
        if biases_path.exists():
            biases = mx.load(str(biases_path))
            logger.info(f"Loaded biases: {list(biases.keys())}")

        logger.info(
            f"Loaded compressed model: {config.num_layers} layers, "
            f"{config.num_experts} experts, {config.compression_ratio:.1f}x compression"
        )

        return cls(config, base_weights, delta_weights, biases)

    def get_expert_weight(
        self,
        layer: int,
        projection: str,
        expert: int,
    ):
        """Reconstruct full expert weight matrix.

        Args:
            layer: Layer index (from moe_layer_indices)
            projection: "gate", "up", or "down"
            expert: Expert index

        Returns:
            Reconstructed weight matrix: base + U @ V
        """
        import mlx.core as mx

        base_key = f"layer_{layer}_{projection}_base"
        u_key = f"layer_{layer}_{projection}_expert_{expert}_U"
        v_key = f"layer_{layer}_{projection}_expert_{expert}_V"

        if base_key not in self._base:
            raise KeyError(f"Base weight not found: {base_key}")
        if u_key not in self._deltas:
            raise KeyError(f"Delta U not found: {u_key}")

        base = self._base[base_key]
        U = self._deltas[u_key]
        V = self._deltas[v_key]

        # Reconstruct: base + U @ V
        weight = base + U @ V
        mx.eval(weight)

        return weight

    def apply_expert(
        self,
        layer: int,
        projection: str,
        expert: int,
        x,
    ):
        """Apply expert to input efficiently using low-rank factorization.

        Instead of: y = x @ (base + U @ V).T + bias
        Computes:   y = x @ base.T + (x @ V.T) @ U.T + bias

        This is more efficient when rank << min(in_dim, out_dim).

        Args:
            layer: Layer index
            projection: "gate", "up", or "down"
            expert: Expert index
            x: Input tensor of shape (..., in_dim)

        Returns:
            Output tensor of shape (..., out_dim)
        """

        base_key = f"layer_{layer}_{projection}_base"
        u_key = f"layer_{layer}_{projection}_expert_{expert}_U"
        v_key = f"layer_{layer}_{projection}_expert_{expert}_V"

        base = self._base[base_key]
        U = self._deltas[u_key]  # (out_dim, rank)
        V = self._deltas[v_key]  # (rank, in_dim)

        # Efficient low-rank application
        # y = x @ base.T + x @ V.T @ U.T
        base_out = x @ base.T
        delta_out = (x @ V.T) @ U.T
        out = base_out + delta_out

        # Apply bias if available
        # Biases are stored per-expert: (num_experts, out_dim)
        bias_key = f"{projection}_bias"
        if bias_key in self._biases:
            bias = self._biases[bias_key][expert]  # (out_dim,)
            out = out + bias

        return out

    @property
    def num_layers(self) -> int:
        """Number of MoE layers."""
        return self.config.num_layers

    @property
    def num_experts(self) -> int:
        """Number of experts per layer."""
        return self.config.num_experts

    @property
    def moe_layer_indices(self) -> list[int]:
        """Original layer indices that are MoE layers."""
        return self.config.moe_layer_indices

    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        total = 0
        for w in self._base.values():
            total += w.nbytes
        for w in self._deltas.values():
            total += w.nbytes
        return total / (1024 * 1024)
