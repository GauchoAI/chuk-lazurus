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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

# User-level config file for custom / discovered architectures
_USER_CONFIG_DIR = Path.home() / ".chuk_lazarus"
_USER_ARCH_FILE = _USER_CONFIG_DIR / "arch_configs.json"


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

    All fields are empirically validated — never formula-derived.
    The geometry fields (kv_head, head_dim, hidden_dim, k_dim) are
    populated from the model when available, or from the registry.
    """

    retrieval_layer: int
    """Layer index of the copy head used for K-space routing."""

    query_head: int
    """Query head index of the copy head within retrieval_layer."""

    injection_layer: int
    """Layer where 1D subspace injection is applied (typically retrieval_layer + 1)."""

    kv_head: int = -1
    """KV head index (GQA grouping). Derived from query_head at construction."""

    head_dim: int = 0
    """Attention head dimension (K-vector size before projection)."""

    hidden_dim: int = 0
    """Residual stream dimension."""

    k_dim: int = 0
    """K-vector dimension (= head_dim for standard attention)."""

    threshold_multiplier: float = 2.0
    """Adaptive threshold: mean_score × this value."""

    inject_coefficient: float = 10.0
    """Injection coefficient multiplier. 2× for per-token donors; 10× for cold queries
    with generic donor prompts (weaker alignment needs higher gain)."""

    entries_per_window: int = 8
    """Number of injection entries extracted per window during build."""

    crystal_layer: int = -1
    """Layer where crystallised residuals are captured/injected for persistent
    injection. Defaults to injection_layer when not explicitly set. L24-L32
    all work equally; L30 maximises model computation (L0-L29) while providing
    reliable injection."""

    window_size: int = 512
    """Token window size for knowledge store prefill."""

    # -----------------------------------------------------------------------
    # Class-level registry of validated configs
    # -----------------------------------------------------------------------

    # Keyed by (model_type, num_layers) — ClassVar excluded from dataclass fields
    _KNOWN: ClassVar[dict[tuple[str, int], ArchitectureConfig]] = {}

    def __post_init__(self) -> None:
        # Validate basic sanity
        if self.retrieval_layer < 0:
            raise ValueError(f"retrieval_layer must be >= 0, got {self.retrieval_layer}")
        if self.query_head < 0:
            raise ValueError(f"query_head must be >= 0, got {self.query_head}")
        if self.injection_layer < 0:
            raise ValueError(f"injection_layer must be >= 0, got {self.injection_layer}")
        # k_dim defaults to head_dim when not explicitly set
        if self.k_dim == 0 and self.head_dim > 0:
            object.__setattr__(self, "k_dim", self.head_dim)
        # crystal_layer defaults to injection_layer
        if self.crystal_layer < 0:
            object.__setattr__(self, "crystal_layer", self.injection_layer)

    # -----------------------------------------------------------------------
    # Factory: from model config (known values only)
    # -----------------------------------------------------------------------

    @classmethod
    def from_model_config(cls, config) -> ArchitectureConfig:
        """Return validated ArchitectureConfig for a known model config.

        Raises ArchitectureNotCalibrated for any model not yet validated.
        Use discover(backbone) for unknown models, or for_model() for
        a graceful fallback that returns None instead of raising.
        """
        model_type = getattr(config, "model_type", "").lower()
        num_layers = getattr(config, "num_hidden_layers", -1)

        # Normalise Gemma variant spellings
        if model_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
            model_type = "gemma"

        key = (model_type, num_layers)
        if key in cls._KNOWN:
            return cls._KNOWN[key]

        # Check user config file
        user_config = cls._load_user_config(model_type, num_layers)
        if user_config is not None:
            return user_config

        raise ArchitectureNotCalibrated(model_type, num_layers)

    @classmethod
    def for_model(cls, config) -> ArchitectureConfig | None:
        """Return ArchitectureConfig if available, None otherwise.

        Graceful alternative to from_model_config() — callers can degrade
        instead of crashing for uncalibrated models.

        Usage::

            ac = ArchitectureConfig.for_model(model.config)
            if ac is None:
                print("Vec injection unavailable — model not calibrated")
                # fall back to plain generation
        """
        try:
            return cls.from_model_config(config)
        except ArchitectureNotCalibrated:
            return None

    # -----------------------------------------------------------------------
    # Factory: discover from behavioral analysis (not yet implemented)
    # -----------------------------------------------------------------------

    @classmethod
    def discover(cls, backbone, verbose: bool = False) -> ArchitectureConfig:
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
    # User config file (~/.chuk_lazarus/arch_configs.json)
    # -----------------------------------------------------------------------

    @classmethod
    def _load_user_config(
        cls,
        model_type: str,
        num_layers: int,
    ) -> ArchitectureConfig | None:
        """Load from user config file if it exists and has a matching entry."""
        if not _USER_ARCH_FILE.exists():
            return None
        try:
            data = json.loads(_USER_ARCH_FILE.read_text())
            key = f"{model_type}:{num_layers}"
            if key in data:
                return cls.from_dict(data[key])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return None

    def save_to_user_config(self, model_type: str, num_layers: int) -> Path:
        """Persist this config to ~/.chuk_lazarus/arch_configs.json.

        Merges with existing entries. Returns the path written.
        """
        _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        data: dict = {}
        if _USER_ARCH_FILE.exists():
            try:
                data = json.loads(_USER_ARCH_FILE.read_text())
            except (json.JSONDecodeError, TypeError):
                data = {}

        key = f"{model_type}:{num_layers}"
        data[key] = self.to_dict()
        _USER_ARCH_FILE.write_text(json.dumps(data, indent=2) + "\n")
        return _USER_ARCH_FILE

    # -----------------------------------------------------------------------
    # Serialisation (for manifest storage)
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "retrieval_layer": self.retrieval_layer,
            "query_head": self.query_head,
            "injection_layer": self.injection_layer,
        }
        if self.kv_head >= 0:
            d["kv_head"] = self.kv_head
        if self.head_dim > 0:
            d["head_dim"] = self.head_dim
        if self.hidden_dim > 0:
            d["hidden_dim"] = self.hidden_dim
        if self.k_dim > 0:
            d["k_dim"] = self.k_dim
        if self.threshold_multiplier != 2.0:
            d["threshold_multiplier"] = self.threshold_multiplier
        if self.inject_coefficient != 2.0:
            d["inject_coefficient"] = self.inject_coefficient
        if self.entries_per_window != 8:
            d["entries_per_window"] = self.entries_per_window
        if self.crystal_layer >= 0 and self.crystal_layer != self.injection_layer:
            d["crystal_layer"] = self.crystal_layer
        if self.window_size != 512:
            d["window_size"] = self.window_size
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ArchitectureConfig:
        return cls(
            retrieval_layer=int(d["retrieval_layer"]),
            query_head=int(d["query_head"]),
            injection_layer=int(d["injection_layer"]),
            kv_head=int(d.get("kv_head", -1)),
            head_dim=int(d.get("head_dim", 0)),
            hidden_dim=int(d.get("hidden_dim", 0)),
            k_dim=int(d.get("k_dim", 0)),
            threshold_multiplier=float(d.get("threshold_multiplier", 2.0)),
            inject_coefficient=float(d.get("inject_coefficient", 2.0)),
            entries_per_window=int(d.get("entries_per_window", 8)),
            crystal_layer=int(d.get("crystal_layer", -1)),
            window_size=int(d.get("window_size", 512)),
        )

    def with_geometry(self, *, kv_head: int, head_dim: int, hidden_dim: int) -> ArchitectureConfig:
        """Return a copy with geometry fields populated from the model."""
        return ArchitectureConfig(
            retrieval_layer=self.retrieval_layer,
            query_head=self.query_head,
            injection_layer=self.injection_layer,
            kv_head=kv_head,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            k_dim=head_dim,
            threshold_multiplier=self.threshold_multiplier,
            inject_coefficient=self.inject_coefficient,
            entries_per_window=self.entries_per_window,
            crystal_layer=self.crystal_layer,
            window_size=self.window_size,
        )

    def __repr__(self) -> str:
        parts = [
            f"retrieval_layer={self.retrieval_layer}",
            f"query_head={self.query_head}",
            f"injection_layer={self.injection_layer}",
        ]
        if self.kv_head >= 0:
            parts.append(f"kv_head={self.kv_head}")
        if self.hidden_dim > 0:
            parts.append(f"hidden_dim={self.hidden_dim}")
        if self.k_dim > 0:
            parts.append(f"k_dim={self.k_dim}")
        return f"ArchitectureConfig({', '.join(parts)})"


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
