"""
ArchitectureConfig — per-model routing and injection parameters.

These values are DISCOVERED empirically, not formula-derived:
  retrieval_layer  — the layer whose H4 copy head routes facts (e.g. 29 for Gemma 4B)
  query_head       — the copy head index within that layer (e.g. 4 for Gemma 4B)
  injection_layer  — the layer where 1D subspace injection is applied (e.g. 30 for Gemma 4B)

Discovery method: SVD of W_q @ W_k^T per layer/head reveals the copy head as the
head with a near-rank-1 W_q@W_k^T (entity copying pattern). This is the same
experiment that found query_head=4 in a9704704.

Known validated values:
  Gemma 3 4B (34 layers, hidden=2560): retrieval_layer=29, query_head=4, injection_layer=30

For any other model family or size: use discover() or run `lazarus context calibrate-arch`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ArchitectureNotCalibrated(Exception):
    """Raised when no validated ArchitectureConfig exists for a model.

    Call ArchitectureConfig.discover(backbone) or run:
        lazarus context calibrate-arch --model <model_id> --checkpoint <path>
    """

    def __init__(self, model_type: str, num_layers: int) -> None:
        self.model_type = model_type
        self.num_layers = num_layers
        super().__init__(
            f"No validated ArchitectureConfig for model_type={model_type!r} "
            f"with num_layers={num_layers}.\n"
            f"Run: lazarus context calibrate-arch --model <model_id>\n"
            f"  or call: ArchitectureConfig.discover(backbone)"
        )


# ---------------------------------------------------------------------------
# ArchitectureConfig
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureConfig:
    """Routing and injection parameters for a specific model architecture.

    All three fields are empirically validated — never formula-derived.
    """

    retrieval_layer: int
    """Layer index of the copy head used for K-space routing."""

    query_head: int
    """Query head index of the copy head within retrieval_layer."""

    injection_layer: int
    """Layer where 1D subspace injection is applied (typically retrieval_layer + 1)."""

    # -----------------------------------------------------------------------
    # Class-level registry of validated configs
    # -----------------------------------------------------------------------

    # Keyed by (model_type, num_layers) — ClassVar excluded from dataclass fields
    _KNOWN: ClassVar[dict[tuple[str, int], "ArchitectureConfig"]] = {}

    def __post_init__(self) -> None:
        # Validate basic sanity
        if self.retrieval_layer < 0:
            raise ValueError(f"retrieval_layer must be >= 0, got {self.retrieval_layer}")
        if self.query_head < 0:
            raise ValueError(f"query_head must be >= 0, got {self.query_head}")
        if self.injection_layer < 0:
            raise ValueError(f"injection_layer must be >= 0, got {self.injection_layer}")

    # -----------------------------------------------------------------------
    # Factory: from model config (known values only)
    # -----------------------------------------------------------------------

    @classmethod
    def from_model_config(cls, config) -> "ArchitectureConfig":
        """Return validated ArchitectureConfig for a known model config.

        Raises ArchitectureNotCalibrated for any model not yet validated.
        Use discover(backbone) for unknown models.
        """
        model_type = getattr(config, "model_type", "").lower()
        num_layers = getattr(config, "num_hidden_layers", -1)

        # Normalise Gemma variant spellings
        if model_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
            model_type = "gemma"

        key = (model_type, num_layers)
        if key in cls._KNOWN:
            return cls._KNOWN[key]

        raise ArchitectureNotCalibrated(model_type, num_layers)

    # -----------------------------------------------------------------------
    # Factory: discover from behavioral analysis (not yet implemented)
    # -----------------------------------------------------------------------

    @classmethod
    def discover(cls, backbone, verbose: bool = False) -> "ArchitectureConfig":
        """Discover retrieval_layer, query_head, injection_layer via behavioral analysis.

        The copy head (e.g. L29 H4 for Gemma 4B) is a BEHAVIORAL property — it is
        the head whose attention output at entity token positions most closely matches
        the entity's embedding direction. This cannot be found from weight matrices
        alone (W_q@W_k^T SVD finds structurally low-rank heads, not copy heads).

        The correct discovery procedure requires:
        1. A set of example documents containing known entities
        2. Forward passes through each layer
        3. Per-head correlation between attention output and entity token embeddings
        4. The head with highest correlation is the copy head

        This is the approach used in experiment a9704704 to identify L29 H4 for
        Gemma 4B. The full procedure will be implemented as:
            lazarus context calibrate-arch --model <model_id> --examples <file>

        Raises
        ------
        ArchitectureNotCalibrated
            Always — discovery from weights alone is not reliable.
            Use `lazarus context calibrate-arch` or add values to ArchitectureConfig._KNOWN.
        """
        num_layers = len(backbone.adapted_layers)
        raise ArchitectureNotCalibrated(
            "unknown (discover() not yet implemented)",
            num_layers,
        )

    # -----------------------------------------------------------------------
    # Serialisation (for manifest storage)
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "retrieval_layer": self.retrieval_layer,
            "query_head": self.query_head,
            "injection_layer": self.injection_layer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureConfig":
        return cls(
            retrieval_layer=int(d["retrieval_layer"]),
            query_head=int(d["query_head"]),
            injection_layer=int(d["injection_layer"]),
        )

    def __repr__(self) -> str:
        return (
            f"ArchitectureConfig("
            f"retrieval_layer={self.retrieval_layer}, "
            f"query_head={self.query_head}, "
            f"injection_layer={self.injection_layer})"
        )


# ---------------------------------------------------------------------------
# Register validated configs
# ---------------------------------------------------------------------------

# Gemma 3 4B-IT (34 layers) — empirically validated in experiment a9704704
# L29 H4 is the copy head; 1D injection at L30 gives KL=0.000031
ArchitectureConfig._KNOWN[("gemma", 34)] = ArchitectureConfig(
    retrieval_layer=29,
    query_head=4,
    injection_layer=30,
)

# Gemma 3 1B-IT (26 layers) — discovered via calibrate_arch.py (2026-03-19)
# L17 H0 is the copy head (causal Δ=+0.113, next best H2 +0.043, 2.6× margin)
# Cosine-proxy calibration had found L18 H3 — causal ablation corrected this.
# Zeroing H0: Voltara 23%→3%, sell 82%→61%, Dravenport 30%→17%
# H1 is a suppressor (negative Δ=−0.196): zeroing it increases P(answer)
ArchitectureConfig._KNOWN[("gemma", 26)] = ArchitectureConfig(
    retrieval_layer=17,
    query_head=0,
    injection_layer=18,
)


# SmolLM2 360M-Instruct (32 layers, llama family) — discovered via calibrate_arch.py (2026-03-20)
# L30 H6 is the copy head (score=+0.018, mean Δ=+0.054, 33% coverage)
# H8 is a suppressor (Δ=−0.050): zeroing it increases P(answer) — mirrors Gemma 1B H1 pattern
# H6 and H8 share KV head 2 (heads 6-8, n_rep=3) — copy/suppressor pair in same KV group
# Lower coverage vs Gemma (33% vs 67%) may reflect more distributed copy mechanism at 360M scale
ArchitectureConfig._KNOWN[("llama", 32)] = ArchitectureConfig(
    retrieval_layer=30,
    query_head=6,
    injection_layer=31,
)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = ["ArchitectureConfig", "ArchitectureNotCalibrated"]
