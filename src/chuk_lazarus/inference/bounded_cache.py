"""
Bounded KV cache with residual stream backing store.

Three-tier state management for memory-constrained inference:

  Tier 1 — HOT   : Active-generation state, bounded by device memory budget.
                    Mode "kv": standard KV cache, O(1) per token, full speed.
                    Mode "rs": per-layer residuals + compiled RS step,
                               2-2.6× slower, dark inference always available.

  Tier 2 — WARM  : Residual checkpoints + dark residuals.
                    Checkpoints: per-layer residual at layer C every N tokens.
                    Dark residuals: configurable layers, always in device memory.

  Tier 3 — COLD  : Token IDs. 4 bytes/token. Unlimited history. Never evicted.

Memory budget
-------------
  Mode "kv" : budget limits KV cache size.
    max_kv_tokens = budget / (2 × kv_heads × head_dim × 2 × num_layers)

  Mode "rs" : budget limits per-layer residual size.
    max_rs_tokens = budget / (hidden_dim × 2 × num_layers)

  Residual is 1.0-2.25× LARGER than KV cache per token for Gemma.
  So RS mode fits fewer tokens within the same budget.

  The tradeoff: RS mode gets dark residuals at zero extra cost during generation.
  KV mode is faster (200 tok/s vs 70-100 tok/s) but needs extra RS passes for dark.

Eviction: when window_size > max_tokens, window slides forward.
Old tokens leave the model's attention. Token IDs stay in cold storage.

Checkpoint rebuild (infrastructure in place, layer-split TODO):
  See GemmaModel.forward_from_residual for the primitives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """
    Residual snapshot at a layer boundary for a prefix of the conversation.

    token_position : absolute token index up to which this checkpoint is valid.
    layer_idx      : residual is the output of layer (layer_idx - 1).
    residual       : mx.array shape (1, checkpoint_len, hidden_dim).
    """
    token_position: int
    layer_idx: int
    residual: mx.array
    created_at: float = field(default_factory=time.monotonic)

    @property
    def bytes(self) -> int:
        return self.residual.nbytes


@dataclass
class ConversationState:
    """Three-tier conversation state. Mutated by the engine each turn."""

    # --- Tier 1: hot ---
    # KV mode: list of (K, V) per layer
    # RS mode:  list of (seq_len, hidden) residual per layer
    hot_state: list | None = None
    hot_token_count: int = 0
    window_start: int = 0
    hot_start: int = 0  # absolute token index of first token in hot_state (RS mode)

    # --- Tier 2: warm ---
    checkpoints: list[Checkpoint] = field(default_factory=list)
    dark_residuals: dict[int, mx.array] = field(default_factory=dict)

    # --- Tier 3: cold ---
    token_ids: list[int] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def window_token_ids(self) -> list[int]:
        return self.token_ids[self.window_start:]

    @property
    def window_size(self) -> int:
        return self.total_tokens - self.window_start


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BoundedKVEngine:
    """
    Inference engine with bounded hot-tier state and residual stream backing store.

    Parameters
    ----------
    std_model : GemmaForCausalLM
    rs_model  : GemmaResidualStreamForCausalLM
    config    : GemmaConfig
    budget_bytes : int
        Maximum bytes for the hot tier (KV cache or per-layer residuals).
    generation_mode : "kv" or "rs"
        "kv" — standard KV cache, full generation speed, ~200 tok/s.
              Dark residuals require extra RS passes after generation.
        "rs" — compiled RS generator, 2-2.6× slower generation,
               dark residuals captured at zero extra cost during generation.
    checkpoint_interval : int
        Store a residual checkpoint every N total tokens.
    checkpoint_layer : int or None
        Layer to snapshot (default: num_layers // 2).
    dark_layers : list[int] or None
        Layers to always capture for probe/inject.
    """

    def __init__(
        self,
        std_model,
        rs_model,
        config,
        budget_bytes: int,
        generation_mode: str = "kv",
        checkpoint_interval: int = 256,
        checkpoint_layer: int | None = None,
        dark_layers: list[int] | None = None,
    ):
        assert generation_mode in ("kv", "rs", "kv_direct"), \
            f"generation_mode must be 'kv', 'rs', or 'kv_direct', got {generation_mode!r}"

        self.std    = std_model
        self.rs     = rs_model
        self.config = config
        self.budget = budget_bytes
        self.mode   = generation_mode

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_layer    = checkpoint_layer or config.num_hidden_layers // 2
        self.dark_layers         = dark_layers or []

        # Per-token hot-state bytes (all layers)
        _kv_bytes = (
            2 * config.num_key_value_heads * config.head_dim * 2
            * config.num_hidden_layers
        )
        _rs_bytes = config.hidden_size * 2 * config.num_hidden_layers

        if self.mode == "kv":
            self._bytes_per_token = _kv_bytes
        elif self.mode == "rs":
            self._bytes_per_token = _rs_bytes
        else:  # kv_direct — same cost as standard KV
            self._bytes_per_token = _kv_bytes

        self.max_hot_tokens = budget_bytes // self._bytes_per_token

        # Compiled RS generator — always initialised (used for dark residuals
        # in KV and kv_direct modes, and as the hot-tier engine in RS mode)
        from .rs_generator import CompiledRSGenerator
        self._rs_gen = CompiledRSGenerator(rs_model, config)

        # KV-direct generator — used as the hot-tier engine in kv_direct mode
        from .kv_generator import KVDirectGenerator
        self._kv_gen = KVDirectGenerator(rs_model, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_conversation(self) -> ConversationState:
        return ConversationState()

    def process_turn(
        self,
        state: ConversationState,
        new_token_ids: list[int],
        max_new_tokens: int = 64,
    ) -> tuple[list[int], ConversationState, dict]:
        """Process one conversation turn. Returns (generated, state, stats)."""
        t0 = time.perf_counter()

        state.token_ids.extend(new_token_ids)
        # Reserve headroom for tokens we're about to generate so the hot tier
        # stays within budget after generation, not just before it.
        state = self._enforce_budget(state, headroom=max_new_tokens)

        path, state, last_out = self._ensure_hot_state(state)

        t_gen = time.perf_counter()
        if self.mode == "kv":
            generated, state = self._generate_kv(state, last_out, max_new_tokens)
        elif self.mode == "rs":
            generated, state = self._generate_rs(state, last_out, max_new_tokens)
        else:
            generated, state = self._generate_kv_direct(state, last_out, max_new_tokens)
        gen_ms = (time.perf_counter() - t_gen) * 1000

        if self.dark_layers:
            if self.mode in ("kv", "kv_direct"):
                state = self._capture_dark_residuals(state)
            # RS mode: dark residuals already in hot_state; extract them
            # (they're already captured as part of the RS forward pass per layer)

        total_ms = (time.perf_counter() - t0) * 1000
        return generated, state, {
            **self.memory_report(state),
            "path":             path,
            "generated_tokens": len(generated),
            "gen_ms":           gen_ms,
            "total_ms":         total_ms,
            "tok_per_sec":      len(generated) / (gen_ms / 1000) if gen_ms > 0 else 0,
        }

    def memory_report(self, state: ConversationState) -> dict:
        hot_bytes  = self._bytes_per_token * state.hot_token_count if state.hot_state is not None else 0
        ckpt_bytes = sum(c.bytes for c in state.checkpoints)
        dark_bytes = sum(r.nbytes for r in state.dark_residuals.values())
        id_bytes   = state.total_tokens * 4

        return {
            "mode":             self.mode,
            "total_tokens":     state.total_tokens,
            "window_start":     state.window_start,
            "window_size":      state.window_size,
            "hot_token_count":  state.hot_token_count,
            "hot_bytes":        hot_bytes,
            "checkpoint_count": len(state.checkpoints),
            "checkpoint_bytes": ckpt_bytes,
            "dark_layer_count": len(state.dark_residuals),
            "dark_bytes":       dark_bytes,
            "token_id_bytes":   id_bytes,
            "budget_bytes":     self.budget,
            "budget_used_pct":  100 * hot_bytes / self.budget if self.budget > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _enforce_budget(self, state: ConversationState, headroom: int = 0) -> ConversationState:
        effective_max = max(1, self.max_hot_tokens - headroom)
        if state.window_size <= effective_max:
            return state

        new_window_start = state.total_tokens - effective_max
        if new_window_start <= state.window_start:
            return state

        state.window_start = new_window_start

        # KV mode: can't slide KV cache, must rebuild from scratch
        # RS / kv_direct mode: hot_state slides incrementally — don't clear it here
        if self.mode == "kv":
            state.hot_state       = None
            state.hot_token_count = 0

        state.checkpoints = [
            c for c in state.checkpoints
            if c.token_position > new_window_start
        ]
        return state

    # ------------------------------------------------------------------
    # Hot-state management (mode-dispatched)
    # ------------------------------------------------------------------

    def _ensure_hot_state(
        self, state: ConversationState
    ) -> tuple[str, ConversationState, Any]:
        """Dispatch to KV, RS, or kv_direct hot-state management."""
        if self.mode == "kv":
            return self._ensure_kv(state)
        elif self.mode == "rs":
            return self._ensure_rs(state)
        else:
            return self._ensure_kv_direct(state)

    # ---- KV mode ----

    def _ensure_kv(
        self, state: ConversationState
    ) -> tuple[str, ConversationState, Any]:
        """
        Bring KV cache up to date with the full active window.

        Hot:  extend existing cache with uncached tokens, one at a time.
              (The existing GemmaModel mask assumes seq_len=1 with KV cache;
               batch incremental prefill has a mask shape mismatch.)
        Warm: checkpoint exists — falls back to cold rebuild until layer-split done.
        Cold: full prefill from token IDs.
        """
        if state.hot_state is not None:
            uncached = state.window_token_ids[state.hot_token_count:]
            cache    = state.hot_state
            out      = None

            for tok in uncached:
                ids = mx.array([[tok]])
                out = self.std(ids, cache=cache)
                mx.eval(out.logits)
                cache = out.cache
                state.hot_token_count += 1

            if out is None:
                # Re-run last token so we have fresh logits
                trimmed = [
                    (k[:, :, :-1, :], v[:, :, :-1, :]) if k is not None else None
                    for k, v in cache
                ]
                last = mx.array([[state.window_token_ids[-1]]])
                out  = self.std(last, cache=trimmed)
                mx.eval(out.logits)
                cache = out.cache

            state.hot_state = cache
            return "hot", state, out

        best = self._best_checkpoint(state)
        if best is not None:
            state, out = self._cold_rebuild_kv(state)   # TODO: warm rebuild
            return "warm", state, out

        state, out = self._cold_rebuild_kv(state)
        return "cold", state, out

    def _cold_rebuild_kv(
        self, state: ConversationState
    ) -> tuple[ConversationState, Any]:
        ids = mx.array(state.window_token_ids)[None]
        out = self.std(ids)
        mx.eval(out.logits)
        state.hot_state       = out.cache
        state.hot_token_count = state.window_size
        return state, out

    # ---- RS mode ----

    def _ensure_rs(
        self, state: ConversationState
    ) -> tuple[str, ConversationState, Any]:
        """
        Bring RS hot state up to date with the full active window.

        Hot path (incremental):
          1. Slide stored residuals to evict tokens before new window_start.
          2. Extend with any uncached tokens (new user turn).
          O(new_tokens × window) instead of O(window²) for re-prefill.

        Cold path: full prefill from window token IDs.

        Path labels reflect window state (not cache hit/miss):
          grow  — window still growing, no eviction yet
          full  — window at budget capacity, oldest tokens evicted each turn
        """
        path = "full" if state.window_start > 0 else "grow"

        if state.hot_state is not None:
            stored = state.hot_state

            # Step 1: slide out tokens evicted by budget enforcement
            evict_count = state.window_start - state.hot_start
            if evict_count > 0:
                stored = self._rs_gen.slide(stored, evict_count)
                state.hot_token_count = max(0, state.hot_token_count - evict_count)
                state.hot_start = state.window_start

            # Step 2: extend with uncached tokens (new turn's input)
            uncached = state.window_token_ids[state.hot_token_count:]
            if not uncached:
                raise RuntimeError("RS hot path: no uncached tokens after slide — logic error")
            abs_start = state.hot_start + state.hot_token_count
            logits, stored = self._rs_gen.extend(
                mx.array(uncached)[None], stored, abs_start
            )
            mx.eval(logits)
            state.hot_token_count = state.window_size

            state.hot_state = stored
            state.hot_start = state.window_start
            # abs_seq_len = position of the NEXT token to generate (0-indexed)
            abs_seq_len = state.hot_start + state.hot_token_count
            return path, state, (logits, stored, abs_seq_len)

        # Cold path: full prefill from window token IDs
        logits, stored = self._rs_gen.prefill(
            mx.array(state.window_token_ids)[None]
        )
        mx.eval(logits)
        state.hot_state       = stored
        state.hot_token_count = state.window_size
        state.hot_start       = state.window_start

        # Capture dark residuals from stored (they're per-layer residuals)
        if self.dark_layers:
            for layer_idx in self.dark_layers:
                if layer_idx < len(stored):
                    mx.eval(stored[layer_idx])
                    state.dark_residuals[layer_idx] = stored[layer_idx]

        # abs_seq_len = position of the NEXT token to generate (0-indexed)
        abs_seq_len = state.hot_start + state.hot_token_count
        return path, state, (logits, stored, abs_seq_len)

    # ---- kv_direct mode ----

    def _ensure_kv_direct(
        self, state: ConversationState
    ) -> tuple[str, ConversationState, Any]:
        """
        Bring kv_direct hot state up to date with the full active window.

        Identical logic to _ensure_rs but uses KVDirectGenerator.
        hot_state is list[tuple[K, V]] instead of list[residual].
        Slides in place on eviction — no full rebuild needed.
        """
        path = "full" if state.window_start > 0 else "grow"

        if state.hot_state is not None:
            stored = state.hot_state

            # Slide out tokens evicted by budget enforcement
            evict_count = state.window_start - state.hot_start
            if evict_count > 0:
                stored = self._kv_gen.slide(stored, evict_count)
                state.hot_token_count = max(0, state.hot_token_count - evict_count)
                state.hot_start = state.window_start

            # Extend with uncached tokens (new turn's input)
            uncached = state.window_token_ids[state.hot_token_count:]
            if not uncached:
                raise RuntimeError("kv_direct hot path: no uncached tokens after slide")
            abs_start = state.hot_start + state.hot_token_count
            logits, stored = self._kv_gen.extend(
                mx.array(uncached)[None], stored, abs_start
            )
            mx.eval(logits)
            state.hot_token_count = state.window_size
            state.hot_state = stored
            state.hot_start = state.window_start
            abs_seq_len = state.hot_start + state.hot_token_count
            return path, state, (logits, stored, abs_seq_len)

        # Cold path: full prefill from window token IDs
        logits, stored = self._kv_gen.prefill(
            mx.array(state.window_token_ids)[None]
        )
        mx.eval(logits)
        state.hot_state       = stored
        state.hot_token_count = state.window_size
        state.hot_start       = state.window_start
        abs_seq_len = state.hot_start + state.hot_token_count
        return path, state, (logits, stored, abs_seq_len)

    # ------------------------------------------------------------------
    # Generation (mode-dispatched)
    # ------------------------------------------------------------------

    def _generate_kv(
        self,
        state: ConversationState,
        last_out: Any,
        max_new_tokens: int,
    ) -> tuple[list[int], ConversationState]:
        cache     = last_out.cache
        generated = []

        for _ in range(max_new_tokens):
            next_tok = int(mx.argmax(last_out.logits[0, -1, :]))
            generated.append(next_tok)
            state.token_ids.append(next_tok)
            state.hot_token_count += 1

            if self.checkpoint_interval > 0 and state.total_tokens % self.checkpoint_interval == 0:
                state = self._store_checkpoint(state)

            ids      = mx.array([[next_tok]])
            last_out = self.std(ids, cache=cache)
            mx.eval(last_out.logits)
            cache    = last_out.cache

        state.hot_state = cache
        return generated, state

    def _generate_rs(
        self,
        state: ConversationState,
        rs_state: Any,
        max_new_tokens: int,
    ) -> tuple[list[int], ConversationState]:
        logits, stored, abs_seq_len = rs_state
        # abs_seq_len = absolute position of the NEXT token to generate (0-indexed)
        generated = []

        for _ in range(max_new_tokens):
            next_tok = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_tok)
            state.token_ids.append(next_tok)
            state.hot_token_count += 1

            if self.checkpoint_interval > 0 and state.total_tokens % self.checkpoint_interval == 0:
                state = self._store_checkpoint(state)

            logits, stored = self._rs_gen.step(
                mx.array([[next_tok]]), stored, abs_seq_len
            )
            mx.eval(logits)
            abs_seq_len += 1

        state.hot_state = stored
        # Update dark residuals from final stored state
        if self.dark_layers:
            for layer_idx in self.dark_layers:
                if layer_idx < len(stored):
                    state.dark_residuals[layer_idx] = stored[layer_idx]

        return generated, state

    def _generate_kv_direct(
        self,
        state: ConversationState,
        kv_state: Any,
        max_new_tokens: int,
    ) -> tuple[list[int], ConversationState]:
        logits, stored, abs_seq_len = kv_state
        generated = []

        for _ in range(max_new_tokens):
            next_tok = int(mx.argmax(logits[0, -1, :]))
            generated.append(next_tok)
            state.token_ids.append(next_tok)
            state.hot_token_count += 1

            if self.checkpoint_interval > 0 and state.total_tokens % self.checkpoint_interval == 0:
                state = self._store_checkpoint(state)

            logits, stored = self._kv_gen.step(
                mx.array([[next_tok]]), stored, abs_seq_len
            )
            mx.eval(logits)
            abs_seq_len += 1

        state.hot_state = stored
        return generated, state

    # ------------------------------------------------------------------
    # Checkpoint and dark residual capture
    # ------------------------------------------------------------------

    def _best_checkpoint(self, state: ConversationState) -> Checkpoint | None:
        min_useful = max(1, state.window_size // 4)
        valid = [
            c for c in state.checkpoints
            if c.token_position > state.window_start
            and (c.token_position - state.window_start) >= min_useful
        ]
        return max(valid, key=lambda c: c.token_position) if valid else None

    def _store_checkpoint(self, state: ConversationState) -> ConversationState:
        ids     = mx.array(state.window_token_ids)[None]
        partial = self.rs.forward_to_layer(ids, self.checkpoint_layer)
        mx.eval(partial.residual)

        ckpt = Checkpoint(
            token_position=state.total_tokens,
            layer_idx=self.checkpoint_layer,
            residual=partial.residual,
        )
        state.checkpoints = [
            c for c in state.checkpoints if c.layer_idx != self.checkpoint_layer
        ]
        state.checkpoints.append(ckpt)
        return state

    def _capture_dark_residuals(self, state: ConversationState) -> ConversationState:
        """Capture dark residuals using RS model (KV mode only — RS mode captures during generation)."""
        ids = mx.array(state.window_token_ids)[None]
        for layer_idx in self.dark_layers:
            partial = self.rs.forward_to_layer(ids, layer_idx + 1)
            mx.eval(partial.residual)
            state.dark_residuals[layer_idx] = partial.residual
        return state
