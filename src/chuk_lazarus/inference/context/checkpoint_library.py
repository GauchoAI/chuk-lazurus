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
        self._l26_interval_residuals: dict[int, list[mx.array]] = self._load_l26_interval_residuals()
        self._compass_basis: dict[str, mx.array] | None = self._load_compass_basis()

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
            raise FileNotFoundError(f"{LibraryFile.CHECKPOINTS} not found in {self._path}")
        raw: dict[str, mx.array] = dict(mx.load(str(ckpt_path)))
        return {
            wid: [
                (raw[f"w{wid}_l{li}_k"], raw[f"w{wid}_l{li}_v"])
                for li in range(self.manifest.num_layers)
            ]
            for wid in range(self.manifest.num_windows)
        }

    def _load_residuals(self) -> dict[int, mx.array]:
        """Load per-window residual vectors (optional — older libraries may lack them)."""
        res_path = self._path / LibraryFile.RESIDUALS
        if not res_path.exists():
            return {}
        raw: dict[str, mx.array] = dict(mx.load(str(res_path)))
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
        raw: dict[str, mx.array] = dict(mx.load(str(res_path)))
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
        raw: dict[str, mx.array] = dict(mx.load(str(res_path)))
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
        return dict(mx.load(str(basis_path)))

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
