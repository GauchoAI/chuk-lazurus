"""Pydantic types for vector injection (Experiment 2bd41b18)."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class SourceType(IntEnum):
    """Origin of a knowledge entry."""

    DOCUMENT = 0  # extracted during prefill
    GENERATED = 1  # extracted from model's own output


class VecInjectMatch(BaseModel):
    """One fact retrieved from the index, ready for vector injection.

    Attributes
    ----------
    token_id    : Vocabulary index of the answer token.
    coefficient : c = dot(R_L30, embed(token_id)) extracted during prefill.
    score       : Q·K cosine similarity — higher means more relevant.
    window_id   : Source window (for provenance / logging).
    position    : Token position within the window.
    distinctive : True when the answer token is distinctive enough for 1D
                  subspace injection.  False means the token starts with a
                  common 1-3 char prefix (" P", " St", " V", …) — the model
                  cannot distinguish it from other P/St/V-prefixed words.
                  Callers should use full-residual injection or window replay
                  for non-distinctive facts.
    source_id   : Source identifier — window ID for document entries, turn
                  number for generated entries.
    source_type : Origin: document (from prefill) or generated (from conversation).
    """

    token_id: int
    coefficient: float
    score: float
    window_id: int
    position: int
    distinctive: bool = True  # default True for legacy indexes without the flag
    source_id: int = 0
    source_type: SourceType = SourceType.DOCUMENT


class VecInjectResult(BaseModel):
    """Outcome of a VecInjectProvider.retrieve() call.

    Attributes
    ----------
    matches            : Retrieved facts sorted by descending score.
    retrieval_ms       : Wall time spent in retrieval (forward pass + matmul).
    injection_layer    : Layer index where vec_inject_all() should be applied.
    routing_confident  : True when top match score exceeds the provider's
                         confidence threshold.  False signals the caller to
                         fall back to window replay rather than inject.
    top_score          : Cosine score of the best match (0.0 if no matches).
    routing_stage      : Which tier resolved the query: "kspace" (Stage 1,
                         Q·K adaptive), "h4" (Stage 2, H4 copy-head output),
                         or "fallback" (no confident match).
    """

    matches: list[VecInjectMatch] = Field(default_factory=list)
    retrieval_ms: float = 0.0
    injection_layer: int = 30
    routing_confident: bool = True
    top_score: float = 0.0
    routing_stage: str = "kspace"  # "kspace" | "h4" | "fallback"


class VecInjectMeta(BaseModel):
    """Typed metadata read from a vec_inject.npz index file.

    Attributes
    ----------
    retrieval_layer : Layer from which K vectors were extracted (typically 29).
    kv_head         : KV head used for K extraction (maps from query_head).
    query_head      : Query head that addresses facts (4 for Gemma 4B).
    injection_layer : Layer where vector injection is applied (typically 30).
    """

    retrieval_layer: int
    kv_head: int
    query_head: int
    injection_layer: int
