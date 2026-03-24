"""Vector injection (vec_inject) — Experiment 2bd41b18.

The model's L29 H4 head copies a scalar projection of the answer token's
embedding direction into the residual stream.  The full information content
of one retrieved fact is 12 bytes:

    token_id   : int32   — the answer token
    coefficient: float32 — c = dot(R_L30, embed(token_id))

This is distinct from Mode 6 KV injection (full KV cache per window).
Vec injection stores only the 1D directional signature per fact position.

Public surface
--------------
Types:
    VecInjectMatch   — one retrieved fact ready for injection
    VecInjectResult  — retrieval outcome from any provider
    VecInjectMeta    — typed metadata from a vec_inject.npz file

Protocol:
    VecInjectProvider — async retrieve + inject interface

Primitives:
    vec_inject      — add one fact's directional component to residual h
    vec_inject_all  — apply all matches from VecInjectResult in one call

Providers:
    LocalVecInjectProvider — file-backed provider (vec_inject.npz or kv_route_index.npz)

Index format constants:
    VecInjectMetaKey   — NPZ scalar keys (no magic strings)
    VecInjectWindowKey — NPZ per-window array key helpers
    VEC_INJECT_FILE    — canonical filename "vec_inject.npz"

Usage
-----
    from chuk_lazarus.inference.context.vec_inject import (
        vec_inject_all,
        LocalVecInjectProvider,
    )

    provider = await LocalVecInjectProvider.load(checkpoint_dir, kv_gen)
    result   = await provider.retrieve(query_ids, query_text, top_k=5)

    # At result.injection_layer in the forward pass:
    h = vec_inject_all(h, result.matches, embed_matrix)
"""

from ._primitives import vec_inject, vec_inject_all
from ._protocol import VecInjectProvider
from ._types import SourceType, VecInjectMatch, VecInjectMeta, VecInjectResult
from .providers import (
    KV_ROUTE_FILE,
    VEC_INJECT_FILE,
    LocalVecInjectProvider,
    VecInjectMetaKey,
    VecInjectWindowKey,
)

__all__ = [
    # Types
    "SourceType",
    "VecInjectMatch",
    "VecInjectResult",
    "VecInjectMeta",
    # Protocol
    "VecInjectProvider",
    # Primitives
    "vec_inject",
    "vec_inject_all",
    # Providers
    "LocalVecInjectProvider",
    # Index format constants
    "VecInjectMetaKey",
    "VecInjectWindowKey",
    "VEC_INJECT_FILE",
    "KV_ROUTE_FILE",
]
