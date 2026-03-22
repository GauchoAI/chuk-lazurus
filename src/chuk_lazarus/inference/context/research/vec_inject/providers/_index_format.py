"""Canonical NPZ key layout for vec_inject.npz and kv_route_index.npz.

All serialisation and deserialisation code imports key names from here.
No magic strings anywhere else in the codebase.
"""

from __future__ import annotations

from enum import StrEnum


class VecInjectMetaKey(StrEnum):
    """Top-level scalar keys stored in every index file."""

    LAYER = "layer"
    KV_HEAD = "kv_head"
    QUERY_HEAD = "query_head"
    INJECT_LAYER = "inject_layer"


class VecInjectWindowKey:
    """Per-window array keys.

    Usage: VecInjectWindowKey.k_vecs(3) → "w3/k_vecs"
    """

    @staticmethod
    def k_vecs(wid: int) -> str:
        return f"w{wid}/k_vecs"

    @staticmethod
    def token_ids(wid: int) -> str:
        return f"w{wid}/token_ids"

    @staticmethod
    def coefs(wid: int) -> str:
        return f"w{wid}/coefs"

    @staticmethod
    def positions(wid: int) -> str:
        return f"w{wid}/positions"

    @staticmethod
    def distinctive(wid: int) -> str:
        """int32 flag per fact: 1 = distinctive token (safe for 1D injection),
        0 = common prefix token (caller should use full-residual or replay)."""
        return f"w{wid}/distinctive"

    @staticmethod
    def v_vecs(wid: int) -> str:
        """V vectors at the copy head's KV group — the content side of
        synthetic KV injection. Paired with k_vecs for attention at L29."""
        return f"w{wid}/v_vecs"

    @staticmethod
    def h4_vecs(wid: int) -> str:
        """H4 attention output vectors (2560D) for Stage-2 routing.

        Stored as float16 (5 KB/fact).  Absent in legacy indexes built before
        routing-wall-breakers experiment (2026-03-19) — callers must check.
        """
        return f"w{wid}/h4_vecs"

    @staticmethod
    def source_ids(wid: int) -> str:
        """Source identifier per fact: window_id for document, turn for generated."""
        return f"w{wid}/source_ids"

    @staticmethod
    def source_types(wid: int) -> str:
        """uint8 per fact: 0=document, 1=generated."""
        return f"w{wid}/source_types"

    @staticmethod
    def flat(wid: int) -> str:
        """Legacy flat key used by kv_route_index.npz (no sub-paths)."""
        return f"w{wid}"

    @staticmethod
    def window_id_from_key(key: str) -> int | None:
        """Parse window id from a 'wN' or 'wN/...' key.  None if not a window key."""
        part = key.split("/")[0]
        if part.startswith("w") and part[1:].isdigit():
            return int(part[1:])
        return None


# ── Knowledge store top-level keys (not per-window) ──────────────────


class KnowledgeStoreKey(StrEnum):
    """Top-level keys for knowledge_store.npz."""

    RESIDUAL = "residual"  # (1, hidden_dim) float32


# ── Canonical file names ──────────────────────────────────────────────

VEC_INJECT_FILE = "vec_inject.npz"
KV_ROUTE_FILE = "kv_route_index.npz"  # legacy routing-only index
KNOWLEDGE_STORE_FILE = "knowledge_store.npz"
