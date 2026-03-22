"""
KV store checkpoint — save/resume a KVDirectGenerator state.

Directory layout
----------------
<checkpoint_dir>/
  meta.json       — CheckpointMeta (model_id, seq_len, status, ...)
  kv.npz          — K,V tensors; keys l{i}_k and l{i}_v for layer i
  tokens.bin      — full source token IDs, uint32 little-endian

Status values: "partial" (mid-prefill, resumable) | "complete" (full prefill done)
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..kv_generator import KVStore


class ContextCheckpointFile(str, Enum):
    """File names within a context checkpoint directory."""

    META = "meta.json"
    KV = "kv.npz"
    TOKENS = "tokens.bin"


class ContextCheckpointStatus(str, Enum):
    """Prefill completion status."""

    PARTIAL = "partial"
    COMPLETE = "complete"


class CheckpointMeta(BaseModel):
    """Metadata stored alongside a KV checkpoint."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="HuggingFace model ID or local path")
    seq_len: int = Field(..., ge=0, description="Number of tokens prefilled so far")
    total_tokens: int = Field(..., ge=0, description="Total tokens in the source")
    num_layers: int = Field(..., ge=1, description="Number of transformer layers")
    num_kv_heads: int = Field(..., ge=1, description="Number of KV heads per layer")
    head_dim: int = Field(..., ge=1, description="Head dimension")
    chunk_size: int = Field(..., ge=1, description="Chunk size used during prefill")
    source_hash: str = Field(..., description="SHA-256 hex digest of the source bytes")
    status: ContextCheckpointStatus = Field(..., description="partial or complete")
    created_at: str = Field(..., description="ISO-8601 timestamp of first save")
    updated_at: str = Field(..., description="ISO-8601 timestamp of last update")

    @property
    def chunks_done(self) -> int:
        return math.ceil(self.seq_len / self.chunk_size) if self.chunk_size else 0

    @property
    def chunks_total(self) -> int:
        return math.ceil(self.total_tokens / self.chunk_size) if self.chunk_size else 0

    @property
    def is_complete(self) -> bool:
        return self.status == ContextCheckpointStatus.COMPLETE


class KVCheckpoint:
    """Save and load KVDirectGenerator state to/from a directory."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def save(
        cls,
        path: str | Path,
        kv_store: KVStore,
        token_ids: list[int],
        meta: CheckpointMeta,
    ) -> None:
        """
        Write kv.npz, tokens.bin, and meta.json atomically.

        Writes data files first, meta last so a partial write never looks complete.
        """
        ckpt = Path(path)
        ckpt.mkdir(parents=True, exist_ok=True)

        # 1. K,V arrays (bfloat16 preserved by mx.savez)
        arrays: dict[str, mx.array] = {}
        for i, (k, v) in enumerate(kv_store):
            arrays[f"l{i}_k"] = k
            arrays[f"l{i}_v"] = v
        mx.savez(str(ckpt / ContextCheckpointFile.KV.value), **arrays)
        mx.eval()  # flush before writing meta

        # 2. Token IDs (uint32 — supports vocab sizes up to 4 B, e.g. Gemma 256 K)
        token_bytes = struct.pack(f"<{len(token_ids)}I", *token_ids)
        (ckpt / ContextCheckpointFile.TOKENS.value).write_bytes(token_bytes)

        # 3. Metadata last (commit point)
        (ckpt / ContextCheckpointFile.META.value).write_text(meta.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> tuple[KVStore, list[int], CheckpointMeta]:
        """
        Load kv_store, token_ids, and metadata from a checkpoint directory.

        Returns (kv_store, token_ids, meta).
        """
        ckpt = Path(path)
        cls._validate(ckpt)

        meta = cls.load_meta(ckpt)
        if meta is None:
            raise FileNotFoundError(f"meta.json not found in {ckpt}")

        # Load KV arrays
        raw: dict[str, mx.array] = dict(mx.load(str(ckpt / ContextCheckpointFile.KV.value)))
        kv_store: KVStore = [(raw[f"l{i}_k"], raw[f"l{i}_v"]) for i in range(meta.num_layers)]

        # Load token IDs (uint32 — 4 bytes each)
        data = (ckpt / ContextCheckpointFile.TOKENS.value).read_bytes()
        n = len(data) // 4
        token_ids = list(struct.unpack(f"<{n}I", data[: n * 4]))

        return kv_store, token_ids, meta

    @classmethod
    def load_meta(cls, path: str | Path) -> CheckpointMeta | None:
        """Fast metadata-only load. Returns None if no checkpoint exists."""
        meta_path = Path(path) / ContextCheckpointFile.META.value
        if not meta_path.exists():
            return None
        raw = json.loads(meta_path.read_text())
        return CheckpointMeta.model_validate(raw)

    @classmethod
    def is_resumable(cls, path: str | Path, source_hash: str) -> bool:
        """
        Return True if path holds a partial checkpoint for the same source.
        """
        meta = cls.load_meta(path)
        if meta is None:
            return False
        return meta.status == ContextCheckpointStatus.PARTIAL and meta.source_hash == source_hash

    @staticmethod
    def source_hash(data: bytes) -> str:
        """Return SHA-256 hex digest of source bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _now() -> str:
        """Current time as ISO-8601 string."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def _validate(cls, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")
        if not (path / ContextCheckpointFile.META.value).exists():
            raise FileNotFoundError(
                f"{ContextCheckpointFile.META.value} not found in {path}. "
                "Checkpoint may be incomplete or corrupt."
            )


__all__ = [
    "CheckpointMeta",
    "ContextCheckpointFile",
    "ContextCheckpointStatus",
    "KVCheckpoint",
]
