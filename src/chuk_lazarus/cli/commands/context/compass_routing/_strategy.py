"""Routing strategy enum."""

from __future__ import annotations

from enum import Enum


class RoutingStrategy(str, Enum):
    """Available compass routing strategies."""

    BM25 = "bm25"
    TWOPASS = "twopass"
    ATTENTION = "attention"
    DEFLECTION = "deflection"
    PREVIEW = "preview"
    HYBRID = "hybrid"
    COMPASS = "compass"
    DIRECTED = "directed"  # query-directed projection — the query IS the basis
    GUIDED = "guided"      # compass × token overlap — both model-internal
    DARKSPACE = "darkspace" # dual-score: compass + directed in 16D PCA
    CONTRASTIVE = "contrastive"  # query-specific subspace discovery at runtime
    GEOMETRIC = "geometric"      # compass + contrastive fused — both model geometry
    QK = "qk"                    # model's own Q/K attention projections — the dark space
    ITERATIVE = "iterative"      # multi-round compass navigation — model reads, compass shifts
    RESIDUAL = "residual"  # legacy: mean-centered cosine similarity
