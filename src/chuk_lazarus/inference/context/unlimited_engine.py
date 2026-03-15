"""
Unlimited Context Engine — Mode 4.

Extends Mode 3 (KV-Direct) with window checkpointing and replay to support
arbitrary context lengths.  The native context window is a sliding viewport;
any prior window can be replayed on demand for cross-window retrieval.

Three storage tiers
-------------------
  HOT   — Active KV-Direct window (bounded by device memory)
           Grows token-by-token within the current window.
           Evicted when the window closes.

  WARM  — Window boundary checkpoints (last-position K,V per layer)
           Saved at each window boundary.  Used to seed replay of
           later windows with correct prior state.

  COLD  — Token archive (token-ID list per window, ~2 bytes/token)
           Never evicted.  Source of truth for replay.

Window lifecycle
----------------
  1. process(tokens)     — extend active KV; auto-close at boundary
  2. _close_window()     — save checkpoint + archive; reset KV
  3. replay_window(id)   — re-run window tokens to reconstruct KV
  4. generate(query, ...) — optional replay + extend + greedy decode

Cross-window retrieval
----------------------
  replay_window(N=0) → prefill archived tokens from scratch
  replay_window(N>0) → extend archived tokens from checkpoint_{N-1}
                        (one checkpoint token at position abs_{N-1})

  Phase 1: two-window demo.  Replay window 0 (prefill from scratch),
           concatenate with current-window KV, extend with query.
  Phase 2: multi-window chaining.  Replay window N using checkpoint from
           window N-1; chain gives correct prior context at each boundary.

Storage for 1 M tokens (125 × 8 K windows, Gemma 4B):
  HOT  :   150 MB  (single window, fixed)
  WARM :   125 × 174 KB  ≈  21 MB
  COLD :   125 ×  16 KB  ≈   2 MB
  Total: ~173 MB  vs  ~56 GB standard KV  →  365× compression

Design principles
-----------------
  - Pydantic for all typed outputs (EngineStats)
  - Async-native: process_async / generate_async / generate_cross_library_async
  - No magic strings: LibrarySource dataclass replaces raw (str, int) tuples
  - Model-agnostic: KVGeneratorProtocol for the generator interface
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Per-layer (K, V) pair list — the format used by KVDirectGenerator
KVStore = list[tuple[mx.array, mx.array]]


# ---------------------------------------------------------------------------
# Generator protocol — model-agnostic interface
# ---------------------------------------------------------------------------


@runtime_checkable
class KVGeneratorProtocol(Protocol):
    """
    Structural protocol for any KV-direct generator.

    UnlimitedContextEngine accepts any object that satisfies this interface,
    making it model-agnostic.  KVDirectGenerator is the reference implementation.
    """

    def prefill(
        self,
        input_ids: mx.array,
    ) -> tuple[mx.array, KVStore]: ...

    def extend(
        self,
        new_token_ids: mx.array,
        kv_store: KVStore,
        abs_start: int,
    ) -> tuple[mx.array, KVStore]: ...

    def step_uncompiled(
        self,
        new_token_ids: mx.array,
        kv_store: KVStore,
        seq_len: int,
    ) -> tuple[mx.array, KVStore]: ...


# ---------------------------------------------------------------------------
# LibrarySource — typed replacement for raw (str, int) tuples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibrarySource:
    """
    Identifies a single (library, window) pair for retrieval.

    Use instead of bare (str, int) tuples to make call sites self-documenting:

        sources = [
            LibrarySource(library_name="The World of Meridian", window_id=0),
            LibrarySource(library_name="Resonance Engineering", window_id=0),
        ]
        engine.generate_cross_library(query_ids, sources=sources)
    """

    library_name: str
    window_id: int


# ---------------------------------------------------------------------------
# EngineStats — typed output for stats()
# ---------------------------------------------------------------------------


class EngineStats(BaseModel):
    """Storage and context statistics for UnlimitedContextEngine."""

    model_config = ConfigDict(frozen=True)

    total_tokens: int = Field(..., ge=0, description="Total tokens seen (archived + current)")
    archived_windows: int = Field(..., ge=0, description="Number of closed windows in the archive")
    current_window_id: int = Field(..., ge=0, description="ID of the active window")
    current_window_tokens: int = Field(..., ge=0, description="Tokens in the active window")
    checkpoint_bytes: int = Field(..., ge=0, description="Bytes used by warm checkpoint store")
    archive_bytes: int = Field(..., ge=0, description="Bytes used by cold token archive")
    cold_warm_bytes: int = Field(..., ge=0, description="Total warm + cold bytes")
    equivalent_kv_bytes: int = Field(
        ..., ge=0, description="What a standard KV cache would use for the same tokens"
    )
    compression_ratio: float = Field(
        ..., ge=0.0, description="equivalent_kv_bytes / cold_warm_bytes"
    )

    @property
    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"{self.archived_windows} windows / {self.total_tokens} tokens — "
            f"{self.compression_ratio:.0f}× compression vs full KV"
        )


# ---------------------------------------------------------------------------
# Warm tier — checkpoints
# ---------------------------------------------------------------------------


class CheckpointStore:
    """
    Per-window-boundary K,V snapshots (warm tier).

    Each checkpoint is the last-position K,V from the closed window, one
    (K, V) tuple per layer.  K and V carry their baked-in RoPE offsets.

    Bytes per checkpoint:
      num_layers × 2 × num_kv_heads × 1 × head_dim × 2  (bfloat16)
      270M : 18 × 2 × 1  × 256 × 2 =  18 KB
        4B : 34 × 2 × 4  × 320 × 2 = 174 KB
    """

    def __init__(self):
        self._kv: dict[int, list[tuple[mx.array, mx.array]]] = {}
        self._abs_pos: dict[int, int] = {}  # window_id → abs position of checkpoint token

    def save(
        self,
        window_id: int,
        kv_last: list[tuple[mx.array, mx.array]],
        abs_pos: int,
    ) -> None:
        self._kv[window_id] = kv_last
        self._abs_pos[window_id] = abs_pos

    def load(self, window_id: int) -> tuple[list[tuple[mx.array, mx.array]], int]:
        """Return (kv_last, abs_pos)."""
        return self._kv[window_id], self._abs_pos[window_id]

    def evict(self, window_ids: list[int]) -> None:
        """Remove checkpoints from memory (after they've been saved to disk)."""
        for wid in window_ids:
            self._kv.pop(wid, None)
            self._abs_pos.pop(wid, None)

    def __contains__(self, window_id: int) -> bool:
        return window_id in self._kv

    def __len__(self) -> int:
        return len(self._kv)

    def total_bytes(self) -> int:
        return sum(k.nbytes + v.nbytes for kv_list in self._kv.values() for k, v in kv_list)


# ---------------------------------------------------------------------------
# Residual store — window boundary Markov states
# ---------------------------------------------------------------------------


class ResidualStore:
    """
    Per-window-boundary residual stream vectors.

    Each residual is the pre-final-norm hidden state at the last token of
    the closed window: shape (1, 1, hidden_size).  This is the cumulative
    Markov state — it encodes the full context up to and including this
    window boundary.

    Bytes per residual (bfloat16):
      270M : 2304 × 2 =  4.5 KB
        4B : 3072 × 2 =  6.0 KB
    """

    def __init__(self):
        self._residuals: dict[int, mx.array] = {}

    def save(self, window_id: int, residual: mx.array) -> None:
        """Save the residual vector for a window boundary."""
        self._residuals[window_id] = residual

    def load(self, window_id: int) -> mx.array:
        """Return the residual vector for a window boundary."""
        return self._residuals[window_id]

    def evict(self, window_ids: list[int]) -> None:
        """Remove residuals from memory (after they've been saved to disk)."""
        for wid in window_ids:
            self._residuals.pop(wid, None)

    def __contains__(self, window_id: int) -> bool:
        return window_id in self._residuals

    def __len__(self) -> int:
        return len(self._residuals)

    def total_bytes(self) -> int:
        return sum(r.nbytes for r in self._residuals.values())


# ---------------------------------------------------------------------------
# Cold tier — token archive
# ---------------------------------------------------------------------------


class TokenArchive:
    """
    Per-window token-ID lists (cold tier).

    Append-only.  Never evicted.  Provides the raw token stream for replay.
    ~2 bytes per token (uint16 vocabulary IDs).
    """

    def __init__(self):
        self._tokens: dict[int, list[int]] = {}
        self._abs_offsets: dict[int, int] = {}  # window_id → abs start position

    def archive(
        self,
        window_id: int,
        token_ids: list[int],
        abs_offset: int,
    ) -> None:
        self._tokens[window_id] = token_ids
        self._abs_offsets[window_id] = abs_offset

    def retrieve(self, window_id: int) -> tuple[list[int], int]:
        """Return (token_ids, abs_offset)."""
        return self._tokens[window_id], self._abs_offsets[window_id]

    def __len__(self) -> int:
        return len(self._tokens)

    def total_tokens(self) -> int:
        return sum(len(t) for t in self._tokens.values())

    def total_bytes(self) -> int:
        return sum(len(t) * 2 for t in self._tokens.values())  # uint16 estimate


# ---------------------------------------------------------------------------
# Mode 4 engine
# ---------------------------------------------------------------------------


class UnlimitedContextEngine:
    """
    Mode 4: Unlimited context via checkpoint-chained window replay.

    Wraps KVDirectGenerator with automatic windowing, checkpointing,
    token archiving, and cross-window replay.

    Usage
    -----
        engine = UnlimitedContextEngine(rs_model, config, window_size=8192)

        # Feed a long document (auto-windows, checkpoints, archives)
        engine.process(token_ids)

        # Close any partial window to enable retrieval
        engine.flush()

        # Without replay — model sees only current-window context
        tokens = engine.generate(query_ids, max_new_tokens=50)

        # With replay — model sees window 0 context + current window + query
        tokens = engine.generate(query_ids, replay_window_ids=[0], max_new_tokens=50)

        print(engine.stats())

    Notes
    -----
    RoPE positions
        prefill()  uses positions 0..S-1 (offset=0, the GemmaAttention default).
        extend()   uses abs_start..abs_start+N-1 for new tokens; old K keeps
                   its baked-in positions.
        For window 0 (abs_offset=0), prefill and extend-from-empty are
        equivalent.  For windows N>0, we use extend with abs_start=abs_offset
        so that K positions align correctly across windows.

    Checkpoint duplication (Phase 2 replay)
        replay_window(N>0) primes from checkpoint_{N-1} (1 token at position
        abs_{N-1}).  When that KV is combined with kv_0 for multi-window
        retrieval, position abs_{N-1} appears in both kv_0 and the primed
        replay.  In practice this causes negligible attention bias (the model
        sees the same token twice) and does not affect retrieval accuracy for
        the demo.  Production systems should strip the leading checkpoint token
        before concatenation.

    Sliding window attention
        KVDirectGenerator treats all layers as global (no sliding window mask),
        matching CompiledRSGenerator.  This is fine for demos; production
        systems should clip k_old to the per-layer window size.
    """

    def __init__(
        self,
        rs_model,
        config,
        window_size: int = 8192,
        model_id: str = "",
        config_hash: str = "",
    ):
        from .kv_generator import make_kv_generator

        self.kv_gen = make_kv_generator(rs_model)
        self.config = config
        self.window_size = window_size
        self.model_id = model_id
        self.config_hash = config_hash

        self.checkpoints = CheckpointStore()
        self.residuals = ResidualStore()
        self.archive = TokenArchive()

        # --- Pre-filled knowledge libraries (load_library) ---
        self.libraries: dict[str, object] = {}

        # --- Active window state ---
        self.current_window_id: int = 0
        self.current_window_tokens: list[int] = []
        self.kv_store: list | None = None  # per-layer (K, V) for active window
        self.hot_len: int = 0  # tokens in kv_store
        self.abs_offset: int = 0  # abs position of first token in kv_store
        self._last_residual: mx.array | None = None  # residual at last token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, token_ids: list[int]) -> None:
        """
        Feed tokens into the engine.

        Tokens are batched into windows of self.window_size.  When a window
        fills, it is automatically checkpointed and archived; the next window
        begins immediately.
        """
        remaining = list(token_ids)
        while remaining:
            space = self.window_size - len(self.current_window_tokens)
            chunk = remaining[:space]
            remaining = remaining[space:]
            self._extend_current_window(chunk)
            if len(self.current_window_tokens) >= self.window_size:
                self._close_window()

    def flush(self) -> None:
        """
        Close any partial current window.

        Call before replay if the current window has not filled naturally.
        After flush, the partial window is in the archive and can be replayed.
        """
        if self.current_window_tokens:
            self._close_window()

    def replay_window(
        self,
        window_id: int,
    ) -> tuple[list[tuple[mx.array, mx.array]], int]:
        """
        Replay a historical window, reconstructing its full KV cache.

        Returns
        -------
        kv      : list of (K, V) per layer; K/V shape (1, nkv, S[+1], head_dim)
        abs_end : absolute position of the last token in this window
                  (use abs_end + 1 as abs_start when extending further)

        For window_id == 0 (no prior checkpoint):
            Runs kv_gen.prefill(tokens) — positions 0..S-1.

        For window_id > 0 (has prior checkpoint):
            Initialises from checkpoint_{window_id-1} (1 token at abs position
            abs_{N-1}), then extends with the archived tokens starting at
            abs_{N-1} + 1.  The resulting KV includes the checkpoint token as
            the first position, followed by the window's own tokens.
        """
        tokens, abs_offset = self.archive.retrieve(window_id)
        ids = mx.array(tokens)[None]  # (1, S)

        if window_id > 0 and (window_id - 1) in self.checkpoints:
            prior_kv, prior_abs = self.checkpoints.load(window_id - 1)
            logits, kv = self.kv_gen.extend(ids, prior_kv, abs_start=prior_abs + 1)
        else:
            # Window 0 or no checkpoint available: pure prefill (offset=0)
            logits, kv = self.kv_gen.prefill(ids)

        mx.eval(*[t for pair in kv for t in pair])
        abs_end = abs_offset + len(tokens) - 1
        return kv, abs_end

    def generate(
        self,
        query_token_ids: list[int],
        replay_window_ids: list[int] | None = None,
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """
        Generate tokens from a query, optionally replaying historical windows.

        Parameters
        ----------
        query_token_ids  : Token IDs of the query (prompt continuation).
        replay_window_ids: Window IDs to replay before generating.
                           Replayed windows are concatenated oldest-first,
                           followed by the current window's KV, followed by
                           the query tokens.
        max_new_tokens   : Upper bound on generated tokens.
        eos_token_id     : If provided, stop generation at this token.

        Returns
        -------
        List of generated token IDs (not including the query).
        """
        # Build combined context (replayed windows + current window) and
        # determine the correct abs_start for query tokens.
        #
        # Key insight: when replaying a historical window, the query should be
        # positioned immediately after the replayed content — not after all
        # tokens ever processed.  This keeps the RoPE relative-distance between
        # the query and the retrieved fact small (1..window_fact_tail), which
        # is essential for attention-based retrieval to work on small models.
        #
        # Example (multi-fact, 3×512 windows processed, replay window 0):
        #   self.abs_offset = 1536, hot_len = 0
        #   → naive abs_start = 1536  (distance to fact at pos 493: ~1043 — too far)
        #   replay_abs_end   = 511
        #   → corrected abs_start = 512  (distance to fact at pos 493: 19 — works)
        replay_parts: list[list[tuple[mx.array, mx.array]]] = []
        replay_abs_end = -1

        if replay_window_ids:
            for wid in sorted(replay_window_ids):
                w_kv, w_abs_end = self.replay_window(wid)
                replay_parts.append(w_kv)
                replay_abs_end = max(replay_abs_end, w_abs_end)

        # Combine: replayed windows + current (active) window
        parts = replay_parts[:]
        if self.kv_store is not None:
            parts.append(self.kv_store)

        context_kv = self._merge_kv_parts(parts)

        # abs_start: immediately after the last position of the context.
        # If replay is present, use replay_abs_end + 1 (so the query is
        # adjacent in RoPE space to the end of the replayed window).
        # If current window is also present, take the later of the two.
        if replay_abs_end >= 0:
            current_end = (self.abs_offset + self.hot_len - 1) if self.hot_len > 0 else -1
            abs_start = max(replay_abs_end, current_end) + 1
        else:
            abs_start = self.abs_offset + self.hot_len

        # Extend KV with query tokens
        q_ids = mx.array(query_token_ids)[None]
        logits, gen_kv = self.kv_gen.extend(q_ids, context_kv, abs_start=abs_start)
        mx.eval(logits)

        seq_len = abs_start + len(query_token_ids)

        # Autoregressive decode.
        # Use step_uncompiled rather than step (compiled) because the compiled
        # step function can mis-trace when called with a kv_store built from
        # extend() over a replayed context — the compiled graph sees unexpected
        # shapes and raises a broadcast error.  The per-token cost difference
        # is negligible for interactive use.
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_tok = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_tok)
            if eos_token_id is not None and next_tok == eos_token_id:
                break
            logits, gen_kv = self.kv_gen.step_uncompiled(mx.array([[next_tok]]), gen_kv, seq_len)
            mx.eval(logits)
            seq_len += 1

        return generated

    def stats(self) -> EngineStats:
        """
        Storage and context statistics.

        Includes compression ratio vs a hypothetical unbounded KV cache for
        the same total token count.
        """
        checkpoint_bytes = self.checkpoints.total_bytes()
        archive_bytes = self.archive.total_bytes()
        total_archived = self.archive.total_tokens()
        current_tokens = len(self.current_window_tokens)
        total_tokens = total_archived + current_tokens

        kv_bytes_per_token = (
            2
            * self.config.num_key_value_heads
            * self.config.head_dim
            * self.config.num_hidden_layers
            * 2  # bfloat16
        )
        equivalent_kv_bytes = total_tokens * kv_bytes_per_token
        cold_warm_bytes = checkpoint_bytes + archive_bytes

        return EngineStats(
            total_tokens=total_tokens,
            archived_windows=len(self.archive),
            current_window_id=self.current_window_id,
            current_window_tokens=current_tokens,
            checkpoint_bytes=checkpoint_bytes,
            archive_bytes=archive_bytes,
            cold_warm_bytes=cold_warm_bytes,
            equivalent_kv_bytes=equivalent_kv_bytes,
            compression_ratio=equivalent_kv_bytes / max(cold_warm_bytes, 1),
        )

    # ------------------------------------------------------------------
    # Pre-filled library API
    # ------------------------------------------------------------------

    def load_library(self, library) -> None:
        """
        Register a pre-filled CheckpointLibrary for cross-library retrieval.

        If model_id / config_hash are set on this engine, the library is
        verified to match.  A mismatch raises ValueError; omit or leave
        model_id/config_hash empty to skip verification.
        """
        if self.model_id and self.config_hash:
            if not library.verify_model(self.model_id, self.config_hash):
                built_for = library.manifest.model_id
                raise ValueError(
                    f"Library '{library.name}' was built for model "
                    f"'{built_for}', but this engine uses '{self.model_id}'. "
                    f"Rebuild the library with the correct model."
                )
        self.libraries[library.name] = library

    def replay_library_window(
        self,
        library_name: str,
        window_id: int,
    ) -> tuple[list[tuple[mx.array, mx.array]], int]:
        """
        Replay window `window_id` from a pre-filled library.

        Identical semantics to replay_window() but reads from the library's
        stored checkpoints and token archive rather than the engine's own.

        Returns (kv, abs_end) — same contract as replay_window().
        """
        lib = self.libraries[library_name]
        tokens = lib.get_window_tokens(window_id)
        ids = mx.array(tokens)[None]  # (1, S)

        if window_id > 0:
            prior_kv = lib.get_checkpoint(window_id - 1)
            prior_abs_start, prior_abs_end = lib.window_abs_range(window_id - 1)
            abs_start_for_window = prior_abs_end + 1
            logits, kv = self.kv_gen.extend(ids, prior_kv, abs_start=abs_start_for_window)
        else:
            logits, kv = self.kv_gen.prefill(ids)

        mx.eval(*[t for pair in kv for t in pair])
        _, abs_end = lib.window_abs_range(window_id)
        return kv, abs_end

    def generate_cross_library(
        self,
        query_token_ids: list[int],
        sources: list[LibrarySource],
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """
        Generate from a query with multiple library windows as context.

        Each LibrarySource is replayed independently.  All replayed KV caches
        are concatenated along the sequence axis, then the query is extended
        and decoded.

        The query is placed at abs_start = max(abs_end across all replayed
        windows) + 1, so its RoPE position is adjacent to the end of the
        last replayed window (keeping relative distance small for reliable
        retrieval on small models).

        Parameters
        ----------
        query_token_ids : Token IDs of the query prompt.
        sources         : List of LibrarySource(library_name, window_id) to replay.
                          Order determines the left-to-right concatenation
                          of the combined KV context.
        max_new_tokens  : Upper bound on generated tokens.
        eos_token_id    : Stop token (optional).

        Returns
        -------
        List of generated token IDs (not including the query).
        """
        parts: list[KVStore] = []
        max_abs_end = -1

        for source in sources:
            kv, abs_end = self.replay_library_window(source.library_name, source.window_id)
            parts.append(kv)
            max_abs_end = max(max_abs_end, abs_end)

        # Concatenate all replayed KV caches along the sequence axis
        context_kv = self._merge_kv_parts(parts)

        abs_start = max_abs_end + 1

        # Extend KV with query tokens
        q_ids = mx.array(query_token_ids)[None]
        logits, gen_kv = self.kv_gen.extend(q_ids, context_kv, abs_start=abs_start)
        mx.eval(logits)

        seq_len = abs_start + len(query_token_ids)

        # Autoregressive decode (uncompiled to avoid broadcast shape errors
        # with dynamically-sized kv_stores from multi-library concatenation)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_tok = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_tok)
            if eos_token_id is not None and next_tok == eos_token_id:
                break
            logits, gen_kv = self.kv_gen.step_uncompiled(mx.array([[next_tok]]), gen_kv, seq_len)
            mx.eval(logits)
            seq_len += 1

        return generated

    # ------------------------------------------------------------------
    # Async variants — non-blocking wrappers for the sync API
    # ------------------------------------------------------------------

    async def process_async(self, token_ids: list[int]) -> None:
        """Feed tokens asynchronously (runs process() in a thread pool)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.process, token_ids)

    async def flush_async(self) -> None:
        """Close any partial window asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.flush)

    async def generate_async(
        self,
        query_token_ids: list[int],
        replay_window_ids: list[int] | None = None,
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Generate asynchronously. See generate() for full documentation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                query_token_ids,
                replay_window_ids=replay_window_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            ),
        )

    async def generate_cross_library_async(
        self,
        query_token_ids: list[int],
        sources: list[LibrarySource],
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Generate cross-library asynchronously. See generate_cross_library()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_cross_library(
                query_token_ids,
                sources=sources,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_empty_kv(self) -> list[tuple[mx.array, mx.array]]:
        """
        Empty KV store (S=0) for use as initial context in extend().

        mx.concatenate([zeros(1,nkv,0,dh), k_new], axis=2) == k_new,
        so extend-from-empty is equivalent to prefill but supports an
        explicit abs_start offset.
        """
        nkv = self.config.num_key_value_heads
        dh = self.config.head_dim
        nl = self.config.num_hidden_layers
        return [
            (
                mx.zeros((1, nkv, 0, dh), dtype=mx.bfloat16),
                mx.zeros((1, nkv, 0, dh), dtype=mx.bfloat16),
            )
            for _ in range(nl)
        ]

    def _extend_current_window(self, token_ids: list[int]) -> None:
        """
        Extend the active window's KV store with new tokens.

        First chunk of a window:
          - Window 0: prefill from scratch.
          - Window N>0: seed from prior window's checkpoint so the Markov
            state chains across windows (cumulative, not isolated).

        Subsequent chunks within the same window:
          - Extend from existing kv_store at abs_start = abs_offset + hot_len.

        Always captures the residual at the last token position for the
        Markov state compass.  The most recent residual is saved when the
        window closes (_close_window).
        """
        if not token_ids:
            return

        ids = mx.array(token_ids)[None]

        if self.kv_store is None:
            if self.current_window_id == 0:
                logits, self.kv_store, self._last_residual = self.kv_gen.prefill_with_residual(ids)
            else:
                # Chain from prior window's checkpoint — cumulative Markov state
                prior_kv, _prior_abs = self.checkpoints.load(
                    self.current_window_id - 1
                )
                logits, self.kv_store, self._last_residual = self.kv_gen.extend_with_residual(
                    ids, prior_kv, abs_start=self.abs_offset
                )
        else:
            abs_start = self.abs_offset + self.hot_len
            logits, self.kv_store, self._last_residual = self.kv_gen.extend_with_residual(
                ids, self.kv_store, abs_start=abs_start
            )

        mx.eval(logits)
        self.hot_len += len(token_ids)
        self.current_window_tokens.extend(token_ids)

    def _close_window(self) -> None:
        """
        Checkpoint and archive the completed window, then reset active state.

        Checkpoint = last-position K,V slice per layer
          (shape per layer: K (1, nkv, 1, head_dim), V (1, nkv, 1, head_dim))
        Residual   = pre-final-norm hidden state at last token (Markov state)
        Archive    = token-ID list with absolute offset
        """
        if not self.current_window_tokens or self.kv_store is None:
            return

        # Checkpoint: extract last-position K,V (O(1) slice)
        last_kv = [(k[:, :, -1:, :], v[:, :, -1:, :]) for k, v in self.kv_store]
        abs_last = self.abs_offset + self.hot_len - 1
        mx.eval(*[t for pair in last_kv for t in pair])
        self.checkpoints.save(self.current_window_id, last_kv, abs_last)

        # Save residual (Markov state) at window boundary
        if hasattr(self, '_last_residual') and self._last_residual is not None:
            self.residuals.save(self.current_window_id, self._last_residual)

        # Archive token IDs
        self.archive.archive(
            self.current_window_id,
            self.current_window_tokens.copy(),
            self.abs_offset,
        )

        # Advance to next window
        self.abs_offset += len(self.current_window_tokens)
        self.current_window_id += 1
        self.current_window_tokens = []
        self.kv_store = None
        self.hot_len = 0
        self._last_residual = None

    def _merge_kv_parts(self, parts: list[KVStore]) -> KVStore:
        """
        Concatenate per-layer KV stores along the sequence dimension (axis=2).

        Returns an empty KV store when parts is empty so that extend() still
        works (concatenating zeros with new keys is a no-op).
        """
        if not parts:
            return self._make_empty_kv()
        if len(parts) == 1:
            return parts[0]
        merged = parts[0]
        for part in parts[1:]:
            merged = [
                (mx.concatenate([k1, k2], axis=2), mx.concatenate([v1, v2], axis=2))
                for (k1, v1), (k2, v2) in zip(merged, part)
            ]
        return merged
