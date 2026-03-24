"""
CheckpointLibrary — a pre-filled knowledge library loaded from disk.

A library is a directory written by tools/prefill_library.py containing:

    manifest.json          — metadata (name, model, window_size, ...)
    checkpoints.npz        — per-window, per-layer K,V tensors (last position)
    tokens.bin             — all token IDs, uint32 little-endian
    windows.json           — per-window metadata (offsets, token counts, previews)

Loading is instant (no prefill).  Once loaded, any window's K,V tensors
and token IDs are available for replay via UnlimitedContextEngine.

Design principles
-----------------
  - Pydantic models for all typed data (LibraryManifest, WindowMeta)
  - Enums for all file names and format versions — no magic strings
  - Async-native: from_path_async() for non-blocking loading
  - Model-agnostic: no Gemma-specific assumptions

File sizes (270M model, 512 token windows)
  checkpoints.npz  ~18 KB per window   (18 layers × 2 × 1 × 256 × 2 B)
  tokens.bin        ~1 KB per window    (~512 tokens × 2 B)
"""

from __future__ import annotations

import asyncio
import json
import struct
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Constants — no magic strings
# ---------------------------------------------------------------------------


class LibraryFile(str, Enum):
    """File names within a checkpoint library directory."""

    MANIFEST = "manifest.json"
    CHECKPOINTS = "checkpoints.npz"
    RESIDUALS = "residuals.npz"
    TOKENS = "tokens.bin"
    WINDOWS = "windows.json"

    def __str__(self) -> str:
        return self.value


class LibraryFormatVersion(str, Enum):
    """Supported library format versions."""

    V1 = "1.0"


# ---------------------------------------------------------------------------
# Typed data models — no dictionary goop
# ---------------------------------------------------------------------------


class WindowMeta(BaseModel):
    """Metadata for one window within a library."""

    model_config = ConfigDict(frozen=True)

    window_id: int = Field(..., ge=0, description="Zero-based window index")
    token_offset: int = Field(..., ge=0, description="Offset into tokens.bin (token units)")
    token_count: int = Field(..., ge=1, description="Number of tokens in this window")
    abs_offset: int = Field(..., ge=0, description="Absolute position of first token")
    preview: str = Field("", description="Decoded preview of first tokens")

    @property
    def abs_end(self) -> int:
        """Absolute position of the last token in this window."""
        return self.abs_offset + self.token_count - 1


class LibraryManifest(BaseModel):
    """Metadata for a pre-filled checkpoint library."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Human-readable library name")
    model_id: str = Field(..., description="HuggingFace model ID used for prefill")
    model_config_hash: str = Field(
        ..., description="Hash of model config for compatibility checking"
    )
    num_layers: int = Field(..., ge=1, description="Number of transformer layers")
    window_size: int = Field(..., ge=1, description="Nominal tokens per window")
    total_tokens: int = Field(..., ge=0, description="Total tokens across all windows")
    num_windows: int = Field(..., ge=0, description="Number of archived windows")
    checkpoint_bytes: int = Field(0, ge=0, description="Bytes used by checkpoints.npz")
    archive_bytes: int = Field(0, ge=0, description="Bytes used by tokens.bin")
    created_at: str = Field("", description="ISO-8601 creation timestamp")
    format_version: LibraryFormatVersion = Field(
        LibraryFormatVersion.V1, description="Library format version"
    )
    has_checkpoints: bool = Field(
        True,
        description=(
            "False for export-mode libraries: KV checkpoints not stored. "
            "Replay fallback uses fresh per-window prefill from tokens.bin instead of "
            "checkpoint extension (~1s slower per replay, affects ~15% of queries)."
        ),
    )
    arch_config: dict | None = Field(
        None,
        description=(
            "Serialised ArchitectureConfig: retrieval_layer, query_head, injection_layer. "
            "None for libraries built before ArchitectureConfig was introduced."
        ),
    )

    @property
    def total_bytes(self) -> int:
        """Combined on-disk size of checkpoint and token data."""
        return self.checkpoint_bytes + self.archive_bytes


# ---------------------------------------------------------------------------
# CheckpointLibrary
# ---------------------------------------------------------------------------


class CheckpointLibrary:
    """
    A pre-filled knowledge library loaded from disk.

    Constructors
    ------------
        CheckpointLibrary(path)                — synchronous (blocks on disk I/O)
        await CheckpointLibrary.from_path_async(path) — async, non-blocking

    Usage
    -----
        lib = CheckpointLibrary("libraries/gemma-3-270m-it-bf16/meridian")
        engine.load_library(lib)

        # Replay window 0 from this library:
        kv, abs_end = engine.replay_library_window(lib.name, window_id=0)

        # Locate which window contains a specific fact:
        wid = lib.find_window_for_term("Kael Dawnstrider", tokenizer)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._validate_directory()
        self.manifest: LibraryManifest = self._load_manifest()
        self.windows: list[WindowMeta] = self._load_windows()
        self._tokens: list[int] = self._load_tokens()
        self._checkpoints: dict[int, list[tuple[mx.array, mx.array]]] = self._load_checkpoints()
        self._residuals: dict[int, mx.array] = self._load_residuals()
        self._interval_residuals: dict[int, list[mx.array]] = self._load_interval_residuals()
        self._l26_interval_residuals: dict[int, list[mx.array]] = (
            self._load_l26_interval_residuals()
        )
        self._compass_basis: dict[str, mx.array] | None = self._load_compass_basis()
        self._kv_route_index: dict[str, mx.array] | None = self._load_kv_route_index()

    # ------------------------------------------------------------------
    # Properties — delegate to manifest for a clean public interface
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def num_windows(self) -> int:
        return self.manifest.num_windows

    @property
    def window_size(self) -> int:
        return self.manifest.window_size

    @property
    def total_tokens(self) -> int:
        return self.manifest.total_tokens

    @property
    def has_checkpoints(self) -> bool:
        return self.manifest.has_checkpoints

    @property
    def arch_config(self):
        """Return ArchitectureConfig for this library, or None if not stored.

        Returns an ArchitectureConfig instance if the manifest has arch_config data,
        otherwise None (caller should use ArchitectureConfig.from_model_config() or
        ArchitectureConfig.discover() to obtain one).
        """
        if self.manifest.arch_config is None:
            return None
        from ..knowledge.config import ArchitectureConfig

        return ArchitectureConfig.from_dict(self.manifest.arch_config)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_directory(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Library directory not found: {self._path}\n"
                f"Run tools/prefill_library.py to create it."
            )
        if not (self._path / LibraryFile.MANIFEST).exists():
            raise FileNotFoundError(
                f"{LibraryFile.MANIFEST} not found in {self._path}. "
                f"Library may be incomplete or corrupt."
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_manifest(self) -> LibraryManifest:
        raw = json.loads((self._path / LibraryFile.MANIFEST).read_text())
        return LibraryManifest.model_validate(raw)

    def _load_windows(self) -> list[WindowMeta]:
        raw: list[dict] = json.loads((self._path / LibraryFile.WINDOWS).read_text())
        return [WindowMeta.model_validate(w) for w in raw]

    def _load_tokens(self) -> list[int]:
        tokens_path = self._path / LibraryFile.TOKENS
        if not tokens_path.exists():
            raise FileNotFoundError(f"{LibraryFile.TOKENS} not found in {self._path}")
        data = tokens_path.read_bytes()
        n = len(data) // 4  # uint32 — 4 bytes per token
        return list(struct.unpack(f"<{n}I", data[: n * 4]))

    def _load_checkpoints(self) -> dict[int, list[tuple[mx.array, mx.array]]]:
        ckpt_path = self._path / LibraryFile.CHECKPOINTS
        if not ckpt_path.exists():
            return {}  # darkspace libraries have no checkpoints
        # mx.load returns a lazy dict — arrays materialised only on access.
        raw = mx.load(str(ckpt_path))
        # Check if checkpoint data actually exists (darkspace may write empty file)
        if "w0_l0_k" not in raw:
            return {}
        result: dict[int, list[tuple[mx.array, mx.array]]] = {}
        for wid in range(self.manifest.num_windows):
            kv_pairs = [
                (raw[f"w{wid}_l{li}_k"], raw[f"w{wid}_l{li}_v"])
                for li in range(self.manifest.num_layers)
            ]
            mx.eval(*[t for pair in kv_pairs for t in pair])
            result[wid] = kv_pairs
        return result

    def _load_residuals(self) -> dict[int, mx.array]:
        """Load per-window residual vectors (optional — older libraries may lack them)."""
        res_path = self._path / LibraryFile.RESIDUALS
        if not res_path.exists():
            return {}
        raw = mx.load(str(res_path))
        return {
            wid: raw[f"w{wid}_residual"]
            for wid in range(self.manifest.num_windows)
            if f"w{wid}_residual" in raw
        }

    def _load_interval_residuals(self) -> dict[int, list[mx.array]]:
        """Load per-window interval residuals (optional — newer libraries only)."""
        res_path = self._path / "interval_residuals.npz"
        if not res_path.exists():
            return {}
        raw = mx.load(str(res_path))
        result: dict[int, list[mx.array]] = {}
        for wid in range(self.manifest.num_windows):
            samples = []
            si = 0
            while f"w{wid}_s{si}" in raw:
                samples.append(raw[f"w{wid}_s{si}"])
                si += 1
            if samples:
                result[wid] = samples
        return result

    def _load_l26_interval_residuals(self) -> dict[int, list[mx.array]]:
        """Load per-window commitment-layer interval residuals (compass routing data)."""
        res_path = self._path / "compass_residuals.npz"
        if not res_path.exists():
            return {}
        raw = mx.load(str(res_path))
        result: dict[int, list[mx.array]] = {}
        for wid in range(self.manifest.num_windows):
            samples = []
            si = 0
            while f"w{wid}_s{si}" in raw:
                samples.append(raw[f"w{wid}_s{si}"])
                si += 1
            if samples:
                result[wid] = samples
        return result

    def _load_compass_basis(self) -> dict[str, mx.array] | None:
        """Load pre-computed PCA basis for compass routing."""
        basis_path = self._path / "compass_basis.npz"
        if not basis_path.exists():
            return None
        return mx.load(str(basis_path))

    def _load_kv_route_index(self) -> dict[str, mx.array] | None:
        """Load K-vector routing index (L29 H4 K vectors at fact positions)."""
        index_path = self._path / "kv_route_index.npz"
        if not index_path.exists():
            return None
        return mx.load(str(index_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_model(self, model_id: str, config_hash: str) -> bool:
        """Return True if this library was built for the given model."""
        return self.manifest.model_id == model_id and self.manifest.model_config_hash == config_hash

    def get_window_tokens(self, window_id: int) -> list[int]:
        """Return token IDs for window_id."""
        w = self.windows[window_id]
        return self._tokens[w.token_offset : w.token_offset + w.token_count]

    def get_checkpoint(self, window_id: int) -> list[tuple[mx.array, mx.array]]:
        """
        Return last-position K,V tensors for window_id.

        Each tuple is (K, V) where shape is (1, num_kv_heads, 1, head_dim).
        """
        return self._checkpoints[window_id]

    @property
    def has_residuals(self) -> bool:
        """True if this library contains per-window residual vectors."""
        return len(self._residuals) > 0

    def get_residual(self, window_id: int) -> mx.array:
        """
        Return the residual stream vector at this window's boundary.

        Shape: (1, 1, hidden_size) — the pre-final-norm Markov state.
        """
        return self._residuals[window_id]

    @property
    def has_interval_residuals(self) -> bool:
        """True if this library contains per-window interval residuals."""
        return len(self._interval_residuals) > 0

    @property
    def interval_samples_per_window(self) -> int:
        """Number of interval residual samples per window."""
        if not self._interval_residuals:
            return 0
        first_wid = next(iter(self._interval_residuals))
        return len(self._interval_residuals[first_wid])

    def get_interval_residuals(self, window_id: int) -> list[mx.array]:
        """Return list of interval residual vectors for a window.

        Each element has shape (1, hidden_size).
        """
        return self._interval_residuals[window_id]

    @property
    def has_compass(self) -> bool:
        """True if this library has compass routing data (L-layer residuals + PCA basis)."""
        return len(self._l26_interval_residuals) > 0 and self._compass_basis is not None

    @property
    def compass_layer(self) -> int | None:
        """The transformer layer used for compass residuals."""
        if self._compass_basis is None:
            return None
        return int(self._compass_basis["compass_layer"].item())

    @property
    def compass_pc_start(self) -> int | None:
        """First PC of the content subspace."""
        if self._compass_basis is None:
            return None
        return int(self._compass_basis["pc_start"].item())

    @property
    def compass_pc_end(self) -> int | None:
        """Last PC (exclusive) of the content subspace."""
        if self._compass_basis is None:
            return None
        return int(self._compass_basis["pc_end"].item())

    @property
    def has_sparse_index(self) -> bool:
        """True if this library has a sparse semantic index (Mode 5)."""
        return (self._path / "sparse_index.json").exists()

    @property
    def is_darkspace(self) -> bool:
        """True if this library uses dark space frame bank projections."""
        return (
            self._compass_basis is not None
            and "mode" in self._compass_basis
            and "".join(chr(int(c)) for c in self._compass_basis["mode"].tolist()) == "darkspace"
        )

    def get_compass_residuals(self, window_id: int) -> list[mx.array]:
        """Return compass-layer interval residuals for a window."""
        return self._l26_interval_residuals[window_id]

    def get_compass_basis(self) -> tuple[mx.array, mx.array, int, int]:
        """Return (mean_vector, basis_matrix, pc_start, pc_end) for compass projection.

        basis_matrix shape: (pc_end - pc_start, hidden_size) — the PCA rows to project into.
        """
        cb = self._compass_basis
        return (
            cb["mean"],
            cb["basis"],
            int(cb["pc_start"].item()),
            int(cb["pc_end"].item()),
        )

    @property
    def has_structural_basis(self) -> bool:
        """True if the compass basis includes structural PCs for removal."""
        return self._compass_basis is not None and "structural_basis" in self._compass_basis

    def get_structural_basis(self) -> mx.array:
        """Return structural PCA basis (PCs 0..pc_start-1) for projection removal.

        Shape: (pc_start, hidden_size).
        """
        return self._compass_basis["structural_basis"]

    # ------------------------------------------------------------------
    # K-vector routing index (L29 H4 Q·K routing)
    # ------------------------------------------------------------------

    @property
    def has_kv_route_index(self) -> bool:
        """True if this library has a K-vector routing index."""
        return self._kv_route_index is not None

    @property
    def kv_route_layer(self) -> int | None:
        """Layer used for K-vector extraction."""
        if self._kv_route_index is None:
            return None
        return int(self._kv_route_index["layer"].item())

    @property
    def kv_route_kv_head(self) -> int | None:
        """KV head used for K-vector extraction."""
        if self._kv_route_index is None:
            return None
        return int(self._kv_route_index["kv_head"].item())

    def get_kv_route_vectors(self) -> tuple[mx.array, list[tuple[int, int]]]:
        """Return (K_matrix, position_map) for K-vector routing.

        K_matrix: (N_total, head_dim) — all stored K vectors.
        position_map: list of (window_id, sample_idx) per row.
        """
        idx = self._kv_route_index
        all_k = []
        wid_map = []
        for wid in range(self.num_windows):
            key = f"w{wid}"
            if key in idx:
                k = idx[key]  # (n_facts, head_dim)
                for si in range(k.shape[0]):
                    all_k.append(k[si])
                    wid_map.append((wid, si))
        return mx.stack(all_k, axis=0) if all_k else mx.zeros((0,)), wid_map

    # ------------------------------------------------------------------
    # Full KV (Mode 6 — prefix caching)
    # ------------------------------------------------------------------

    @property
    def has_full_kv(self) -> bool:
        """True if this library has per-window full KV caches (Mode 6)."""
        kv_dir = self._path / "kv_full"
        return kv_dir.exists() and (kv_dir / "w0.npz").exists()

    @property
    def has_pre_rope_kv(self) -> bool:
        """True if this library has pre-RoPE KV caches (Mode 6 zero-compute)."""
        kv_dir = self._path / "kv_pre_rope"
        return kv_dir.exists() and (kv_dir / "w0.npz").exists()

    def get_pre_rope_kv(self, window_id: int) -> list[tuple[mx.array, mx.array]]:
        """Load pre-RoPE K + V for a window from disk.

        Returns list[L] of (K_pre_rope, V) per layer, where:
            K_pre_rope shape: (1, nkv, n_facts, head_dim) — post-norm, WITHOUT RoPE
            V shape: (1, nkv, n_facts, head_dim) — position-independent

        n_facts is typically ~8 (the most surprising positions per window).
        Apply RoPE at desired positions via kv_gen.inject_pre_rope_kv().
        """
        kv_path = self._path / "kv_pre_rope" / f"w{window_id}.npz"
        if not kv_path.exists():
            raise FileNotFoundError(
                f"Pre-RoPE KV not found for window {window_id}: {kv_path}\n"
                f"Re-run prefill with --store-kv-full to save per-window KV caches."
            )
        raw = mx.load(str(kv_path))
        kv_pairs = []
        li = 0
        while f"l{li}_k" in raw:
            k = raw[f"l{li}_k"]
            v = raw[f"l{li}_v"]
            kv_pairs.append((k, v))
            li += 1
        mx.eval(*[t for pair in kv_pairs for t in pair])
        return kv_pairs

    def get_pre_rope_positions(self, window_id: int) -> list[int]:
        """Return the original token positions saved for this window's pre-RoPE KV."""
        kv_path = self._path / "kv_pre_rope" / f"w{window_id}.npz"
        if not kv_path.exists():
            return []
        raw = mx.load(str(kv_path))
        if "positions" in raw:
            return raw["positions"].tolist()
        # Legacy: full positions (all positions saved)
        n = raw["l0_k"].shape[2]
        return list(range(n))

    def get_full_kv(self, window_id: int) -> list[tuple[mx.array, mx.array]]:
        """Load the full KV cache for a window from disk.

        Returns list[L] of (K, V) per layer, where:
            K shape: (1, num_kv_heads, seq_len, head_dim) — post-norm, post-RoPE
            V shape: (1, num_kv_heads, seq_len, head_dim)

        Lazy-loaded: only materializes the requested window's KV.
        """
        kv_path = self._path / "kv_full" / f"w{window_id}.npz"
        if not kv_path.exists():
            raise FileNotFoundError(
                f"Full KV not found for window {window_id}: {kv_path}\n"
                f"Re-run prefill with --store-kv-full to save per-window KV caches."
            )
        raw = mx.load(str(kv_path))
        kv_pairs = []
        li = 0
        while f"l{li}_k" in raw:
            k = raw[f"l{li}_k"]
            v = raw[f"l{li}_v"]
            kv_pairs.append((k, v))
            li += 1
        mx.eval(*[t for pair in kv_pairs for t in pair])
        return kv_pairs

    @property
    def full_kv_size_bytes(self) -> int:
        """Total size of full KV data on disk."""
        kv_dir = self._path / "kv_full"
        if not kv_dir.exists():
            return 0
        return sum(f.stat().st_size for f in kv_dir.glob("w*.npz"))

    def window_abs_range(self, window_id: int) -> tuple[int, int]:
        """Return (abs_start, abs_end) for a window."""
        w = self.windows[window_id]
        return w.abs_offset, w.abs_end

    def find_window_for_term(self, term: str, tokenizer) -> int | None:
        """
        Return the first window_id whose decoded text contains `term`.
        Returns None if not found in any window.
        """
        term_lower = term.lower()
        for wid in range(self.num_windows):
            text = tokenizer.decode(self.get_window_tokens(wid), skip_special_tokens=True)
            if term_lower in text.lower():
                return wid
        return None

    # ------------------------------------------------------------------
    # Async loading
    # ------------------------------------------------------------------

    @classmethod
    async def from_path_async(cls, path: str | Path) -> CheckpointLibrary:
        """Load a library asynchronously (non-blocking disk I/O)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls, path)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CheckpointLibrary("
            f"name={self.name!r}, "
            f"windows={self.num_windows}, "
            f"tokens={self.total_tokens}, "
            f"model={self.manifest.model_id!r})"
        )
