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
    GUIDED = "guided"  # compass × token overlap — both model-internal
    DARKSPACE = "darkspace"  # dual-score: compass + directed in 16D PCA
    CONTRASTIVE = "contrastive"  # query-specific subspace discovery at runtime
    GEOMETRIC = "geometric"  # compass + contrastive fused — both model geometry
    QK = "qk"  # model's own Q/K attention projections — the dark space
    ITERATIVE = "iterative"  # multi-round compass navigation — model reads, compass shifts
    PROBE = "probe"  # probe-driven navigation — grounding probe controls everything
    UNIFIED = "unified"  # three-probe architecture — dark space decides everything
    KV_ROUTE = "kv_route"  # L29 H4 Q·K routing — model's own fact-addressing mechanism
    RESIDUAL = "residual"  # legacy: mean-centered cosine similarity
    SPARSE = "sparse"  # BM25 over pre-extracted keyword index (Mode 5 hybrid routing)
    MODE7 = "mode7"  # unified dark space router — auto-classifies and dispatches
    TEMPORAL = "temporal"  # temporal stride — evenly spaced windows for global/timeline queries
