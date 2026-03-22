"""
KV-direct generator — model-agnostic stateful inference.

Architectural observation (experiment a9704704):
  The RS generator stored per-layer residuals (hidden_size D) and recomputed K,V
  at every step — O(S × hidden × head_dim) wasted matmuls.

  This generator stores K,V directly after they are computed at prefill time.
  At step time only Q, K_new, V_new are computed for the single new token.

    Memory  : identical to standard KV cache (2 × nkv × head_dim × S × L)
    Speed   : approaches KV-cached speed — no K,V recompute for old tokens
    Quality : bit-exact with standard KV (K,V are the same tensors)

Model-agnostic design
---------------------
  KVDirectGenerator accepts any ModelBackboneProtocol. Architecture-specific
  details (4-norm vs 2-norm blocks, clip_residual vs plain add, q_norm/k_norm,
  embedding scale, sliding-window masks) are handled by the backbone adapter.

  Built-in adapters:
    GemmaBackboneAdapter  — wraps GemmaResidualStreamForCausalLM
    LlamaBackboneAdapter  — wraps LlamaForCausalLM / Mistral

  Factory:
    make_kv_generator(model)           — auto-detects family
    KVDirectGenerator.from_gemma_rs(rs_model, config)
    KVDirectGenerator.from_llama(llama_model)

Lifecycle
---------
  Turn 1: prefill(130 tokens)  → kv_store: list[L] of (K, V)
           generate 40 tokens via step() → kv_store grows
  Turn 2: extend(80 tokens, kv_store, abs_start=170)
           generate more tokens via step_uncompiled()
  Turn N: budget hit — slide(kv_store, evict_n) → O(1) slice, no recompute
"""

from __future__ import annotations

import mlx.core as mx

from .protocols import ModelBackboneProtocol

# Dtype for attention masks — bfloat16 matches weight precision.
_MASK_DTYPE = mx.bfloat16

from .protocols import KVStore


def _run_layer(
    layer, h: mx.array, mask: mx.array | None, B: int, S: int, offset: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run one transformer layer: attention + FFN. Returns (h_out, k, v).

    This is the standard forward path used by prefill variants. Methods that
    need manual attention (weight capture) or concatenated KV (extend/step)
    implement their own layer loops.
    """
    x = layer.pre_attn_norm(h)
    q, k, v = layer.project_qkv(x, B, S, offset=offset)

    k_rpt = mx.repeat(k, layer.n_rep, axis=1) if layer.n_rep > 1 else k
    v_rpt = mx.repeat(v, layer.n_rep, axis=1) if layer.n_rep > 1 else v

    attn_out = mx.fast.scaled_dot_product_attention(
        q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask
    )
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    attn_out = layer.output_project(attn_out)

    h = layer.residual_add_attn(h, attn_out)
    h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))
    return h, k, v


class KVDirectGenerator:
    """
    Incremental generator that stores K,V per layer.

    Accepts any ModelBackboneProtocol — Gemma, Llama, Mistral, etc.

    All methods assume batch_size=1:
        input_ids : (1, S) int32
        kv_store  : list[num_layers] of (K, V) tuples
        K.shape   = (1, num_kv_heads, seq_len, head_dim)  — post-norm, post-RoPE
        V.shape   = (1, num_kv_heads, seq_len, head_dim)
        logits    : (1, S, vocab_size)
        residual  : (1, S, hidden_size) or (1, 1, hidden_size) for single-position
    """

    def __init__(self, backbone: ModelBackboneProtocol) -> None:
        self.backbone = backbone
        self._step = mx.compile(self._raw_step, shapeless=True)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_gemma_rs(cls, rs_model, config=None) -> KVDirectGenerator:
        """
        Construct from a GemmaResidualStreamForCausalLM.

        The `config` argument is accepted but unused — the adapter reads it
        directly from the model. Kept for call-site compatibility.
        """
        from .adapters.gemma_adapter import GemmaBackboneAdapter

        return cls(GemmaBackboneAdapter(rs_model))

    @classmethod
    def from_llama(cls, llama_model) -> KVDirectGenerator:
        """Construct from a LlamaForCausalLM."""
        from .adapters.llama_adapter import LlamaBackboneAdapter

        return cls(LlamaBackboneAdapter(llama_model))

    # ------------------------------------------------------------------
    # Prefill — full forward from token IDs
    # ------------------------------------------------------------------

    def prefill(
        self,
        input_ids: mx.array,  # (1, S)
    ) -> tuple[mx.array, KVStore]:
        """
        Full forward pass. Returns (logits, kv_store).

        kv_store[i] = (K, V) for layer i:
            K shape (1, nkv, S, head_dim)  — post-norm, post-RoPE
            V shape (1, nkv, S, head_dim)
        """
        logits, kv_store, _ = self._prefill_core(input_ids)
        return logits, kv_store

    def prefill_with_residual(
        self,
        input_ids: mx.array,  # (1, S)
    ) -> tuple[mx.array, KVStore, mx.array]:
        """
        Full forward pass returning residual at last position.

        Returns (logits, kv_store, residual_last) where residual_last is
        the pre-final-norm hidden state at the last token: shape (1, 1, hidden_size).
        This is the Markov state — the cumulative context signal.
        """
        return self._prefill_core(input_ids)

    def _prefill_core(
        self,
        input_ids: mx.array,
    ) -> tuple[mx.array, KVStore, mx.array]:
        """Core prefill returning (logits, kv_store, residual_last)."""
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)
        kv_store: KVStore = []

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)
            h, k, v = _run_layer(layer, h, mask, B, S)
            kv_store.append((k, v))

        # Capture residual at last position BEFORE final norm
        residual_last = h[:, -1:, :]  # (1, 1, hidden_size)

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        mx.eval(logits, residual_last, *[t for pair in kv_store for t in pair])
        return logits, kv_store, residual_last

    def prefill_to_layer(
        self,
        input_ids: mx.array,  # (1, S)
        target_layer: int = 26,
        sample_positions: list[int] | None = None,
        initial_residual: mx.array | None = None,  # (1, 1, hidden) — chained context
    ) -> mx.array:
        """Forward pass through target_layer, return residuals at sampled positions.

        Runs layers 0..target_layer only (saves compute vs full forward).

        Parameters
        ----------
        target_layer : Layer index to stop at (inclusive).
        sample_positions : Token positions to extract. If None, returns all positions.
        initial_residual : If provided, prepend as position 0. All tokens
            attend to it via causal mask. This chains document context across
            windows — the Markov property guarantees completeness.

        Returns
        -------
        Residuals at sampled positions: shape (1, S_total, hidden_size)
        where S_total = S + (1 if initial_residual else 0).
        """
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)

        if initial_residual is not None:
            h = mx.concatenate([initial_residual, h], axis=1)

        S_total = h.shape[1]

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)
            h, _, _ = _run_layer(layer, h, mask, B, S_total)

            if i == target_layer:
                break

        if sample_positions is not None:
            h = h[:, sample_positions, :]

        mx.eval(h)
        return h

    def prefill_pages(
        self,
        input_ids: mx.array,  # (1, S)
        n_pages: int = 8,
    ) -> tuple[mx.array, KVStore, list[tuple[mx.array, mx.array]]]:
        """Prefill and return pre-RoPE K,V at evenly-spaced page positions.

        Returns (logits, kv_store, pages) where:
          - kv_store: standard post-RoPE KV (for generation if needed)
          - pages: list of n_pages tuples, each containing:
              per_layer list of (K_pre_rope, V) at that page position.

        The pre-RoPE K can be re-positioned via apply_rope() at injection time.
        V is position-independent — no RoPE applied to values.
        """
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)
        kv_store: KVStore = []

        # Determine page positions
        positions = [int(i * (S - 1) / max(n_pages - 1, 1)) for i in range(n_pages)]

        # Per-layer storage for pages: pages_per_layer[layer][page_idx] = (k_pre, v)
        pages_per_layer: list[list[tuple[mx.array, mx.array]]] = []

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)

            # Get pre-RoPE K for page storage
            x = layer.pre_attn_norm(h)
            _q_pre, k_pre, v_all = layer.project_qkv_pre_rope(x, B, S)

            # Apply RoPE for the actual forward pass
            q, k, v = layer.project_qkv(x, B, S, offset=0)

            k_rpt = mx.repeat(k, layer.n_rep, axis=1) if layer.n_rep > 1 else k
            v_rpt = mx.repeat(v, layer.n_rep, axis=1) if layer.n_rep > 1 else v

            attn_out = mx.fast.scaled_dot_product_attention(
                q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)
            attn_out = layer.output_project(attn_out)

            h = layer.residual_add_attn(h, attn_out)
            h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))

            kv_store.append((k, v))

            # Extract pre-RoPE K and V at page positions
            layer_pages = []
            for pi in positions:
                k_page = k_pre[:, :, pi : pi + 1, :]  # (1, nkv, 1, dh) pre-RoPE
                v_page = v_all[:, :, pi : pi + 1, :]  # (1, nkv, 1, dh)
                layer_pages.append((k_page, v_page))
            pages_per_layer.append(layer_pages)

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)

        # Restructure: pages[page_idx] = list of (k_pre, v) per layer
        pages = []
        for pi in range(n_pages):
            page_kv = [(pages_per_layer[li][pi]) for li in range(len(backbone.adapted_layers))]
            pages.append(page_kv)

        eval_targets = [logits]
        eval_targets.extend(t for pair in kv_store for t in pair)
        for page in pages:
            eval_targets.extend(t for k, v in page for t in (k, v))
        mx.eval(*eval_targets)

        return logits, kv_store, pages

    def inject_pages(
        self,
        pages: list[list[tuple[mx.array, mx.array]]],
        target_offsets: list[int],
    ) -> KVStore:
        """Build a KV store by applying RoPE to pre-RoPE pages at target positions.

        Args:
            pages: list of N pages, each = list of (K_pre_rope, V) per layer
            target_offsets: list of N position offsets for RoPE

        Returns:
            KVStore with N positions, RoPE applied at target_offsets.
        """
        backbone = self.backbone
        kv_store: KVStore = []

        for li, layer in enumerate(backbone.adapted_layers):
            k_parts = []
            v_parts = []
            for pi, page in enumerate(pages):
                k_pre, v = page[li]
                # Apply RoPE at the target position
                k_roped = layer.apply_rope(k_pre, offset=target_offsets[pi])
                k_parts.append(k_roped)
                v_parts.append(v)
            k_all = mx.concatenate(k_parts, axis=2)  # (1, nkv, N, dh)
            v_all = mx.concatenate(v_parts, axis=2)
            kv_store.append((k_all, v_all))

        mx.eval(*[t for pair in kv_store for t in pair])
        return kv_store

    def prefill_from_layer(
        self,
        residuals: mx.array,  # (1, N, hidden_size) — pre-computed L26 states
        start_layer: int = 26,
    ) -> tuple[mx.array, KVStore]:
        """Build KV cache by processing pre-computed residuals through upper layers.

        Takes stored L26 residuals and runs them through layers start_layer..end.
        Layers 0..start_layer-1 get empty KV (no entries).
        Layers start_layer..end get real KV computed from the residuals.

        The model sees the injected content only in its upper layers —
        where content routing, tone, and entity signals live (L24-L28+).
        Lower layers handle token-level processing which was already done
        during prefill.

        Returns (logits, kv_store) where kv_store has real entries at
        upper layers and empty entries at lower layers.
        """
        backbone = self.backbone
        B, N, _D = residuals.shape
        h = residuals
        kv_store: KVStore = []

        for i, layer in enumerate(backbone.adapted_layers):
            if i < start_layer:
                # Zero KV for lower layers — same shape as upper layers
                # so extend() masks work, but content is zeros (no signal).
                nkv = layer.num_kv_heads
                dh = layer.head_dim
                zero_k = mx.zeros((B, nkv, N, dh))
                zero_v = mx.zeros((B, nkv, N, dh))
                kv_store.append((zero_k, zero_v))
                continue

            mask = backbone.prefill_mask(i, h)
            h, k, v = _run_layer(layer, h, mask, B, N)
            kv_store.append((k, v))

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        mx.eval(logits, *[t for pair in kv_store for t in pair])
        return logits, kv_store

    def prefill_interval_residuals(
        self,
        input_ids: mx.array,  # (1, S)
        n_samples: int = 8,
    ) -> tuple[mx.array, KVStore, mx.array]:
        """Prefill and return residuals at evenly-spaced interior positions.

        Returns (logits, kv_store, interval_residuals) where
        interval_residuals has shape (1, n_samples, hidden_size) —
        the pre-final-norm hidden states at n_samples evenly-spaced
        positions through the sequence.
        """
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)
        kv_store: KVStore = []

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)
            h, k, v = _run_layer(layer, h, mask, B, S)
            kv_store.append((k, v))

        # Extract residuals at evenly-spaced positions BEFORE final norm
        # h shape: (B, S, hidden_size)
        positions = [int(i * (S - 1) / max(n_samples - 1, 1)) for i in range(n_samples)]
        interval_residuals = h[:, positions, :]  # (1, n_samples, hidden_size)

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        mx.eval(logits, interval_residuals, *[t for pair in kv_store for t in pair])
        return logits, kv_store, interval_residuals

    # ------------------------------------------------------------------
    # Prefill with attention weight capture (L26 routing)
    # ------------------------------------------------------------------

    def prefill_with_attention(
        self,
        input_ids: mx.array,  # (1, S)
        capture_layers: set[int] | None = None,
    ) -> tuple[mx.array, KVStore, dict[int, mx.array]]:
        """Prefill with manual attention at specified layers to capture weights.

        Returns (logits, kv_store, attention_weights) where
        attention_weights[layer_idx] has shape (1, num_heads, S, S) in float32.
        """
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)
        kv_store: KVStore = []

        if capture_layers is None:
            capture_layers = {26}

        captured_weights: dict[int, mx.array] = {}

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)
            x = layer.pre_attn_norm(h)
            q, k, v = layer.project_qkv(x, B, S, offset=0)

            k_rpt = mx.repeat(k, layer.n_rep, axis=1) if layer.n_rep > 1 else k
            v_rpt = mx.repeat(v, layer.n_rep, axis=1) if layer.n_rep > 1 else v

            if i in capture_layers:
                scores = (q @ k_rpt.transpose(0, 1, 3, 2)) * layer.attn_scale
                scores = scores.astype(mx.float32)
                if mask is not None:
                    scores = scores + mask.astype(mx.float32)
                weights = mx.softmax(scores, axis=-1)
                captured_weights[i] = weights
                attn_out = weights.astype(v_rpt.dtype) @ v_rpt
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)
            else:
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask
                )
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)

            attn_out = layer.output_project(attn_out)
            h = layer.residual_add_attn(h, attn_out)
            h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))
            kv_store.append((k, v))

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)

        eval_targets = [logits]
        eval_targets.extend(t for pair in kv_store for t in pair)
        eval_targets.extend(captured_weights.values())
        mx.eval(*eval_targets)

        return logits, kv_store, captured_weights

    # ------------------------------------------------------------------
    # Pre-RoPE KV capture (Mode 6 — save for position-independent caching)
    # ------------------------------------------------------------------

    def prefill_pre_rope(
        self,
        input_ids: mx.array,  # (1, S)
    ) -> tuple[mx.array, KVStore, KVStore, mx.array]:
        """Prefill returning both post-RoPE KV and pre-RoPE K + V.

        Returns (logits, kv_store, pre_rope_kv, residual_last) where:
          - kv_store: standard post-RoPE KV (for continued generation)
          - pre_rope_kv: list[L] of (K_pre_rope, V) per layer
            K_pre_rope is post-norm, post-q/k-norm, but WITHOUT RoPE
            V is position-independent (no RoPE on values)
          - residual_last: pre-final-norm hidden state at last token

        Save pre_rope_kv to disk. At load time, apply RoPE at desired
        contiguous positions via inject_pre_rope_kv().
        """
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)
        kv_store: KVStore = []
        pre_rope_kv: KVStore = []

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)
            x = layer.pre_attn_norm(h)

            # Get both pre-RoPE and post-RoPE projections
            _q_pre, k_pre, v = layer.project_qkv_pre_rope(x, B, S)
            q, k, _v = layer.project_qkv(x, B, S, offset=0)

            k_rpt = mx.repeat(k, layer.n_rep, axis=1) if layer.n_rep > 1 else k
            v_rpt = mx.repeat(v, layer.n_rep, axis=1) if layer.n_rep > 1 else v

            attn_out = mx.fast.scaled_dot_product_attention(
                q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, -1)
            attn_out = layer.output_project(attn_out)

            h = layer.residual_add_attn(h, attn_out)
            h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))

            kv_store.append((k, v))
            pre_rope_kv.append((k_pre, v))

        residual_last = h[:, -1:, :]
        h = backbone.final_norm(h)
        logits = backbone.unembed(h)

        eval_targets = [logits, residual_last]
        eval_targets.extend(t for pair in kv_store for t in pair)
        eval_targets.extend(t for pair in pre_rope_kv for t in pair)
        mx.eval(*eval_targets)
        return logits, kv_store, pre_rope_kv, residual_last

    def inject_pre_rope_kv(
        self,
        windows_kv: list[KVStore],
        window_sizes: list[int],
        offset: int = 0,
    ) -> KVStore:
        """Build a KV store from pre-RoPE K,V with contiguous RoPE positions.

        Takes multiple windows' pre-RoPE KV and applies RoPE so positions
        are contiguous: window 0 gets offset..offset+S0-1, etc.

        Args:
            windows_kv: list of N windows, each is list[L] of (K_pre_rope, V)
            window_sizes: list of N sequence lengths (tokens per window)
            offset: starting RoPE position (e.g., after preamble)

        Returns:
            KVStore with all windows concatenated, RoPE at contiguous positions.
        """
        backbone = self.backbone
        kv_store: KVStore = []

        # Compute position offsets for each window
        offsets = []
        pos = offset
        for size in window_sizes:
            offsets.append(pos)
            pos += size

        for li, layer in enumerate(backbone.adapted_layers):
            k_parts = []
            v_parts = []
            for wi, wkv in enumerate(windows_kv):
                k_pre, v = wkv[li]
                # Apply RoPE at contiguous offset
                k_roped = layer.apply_rope(k_pre, offset=offsets[wi])
                k_parts.append(k_roped)
                v_parts.append(v)
            k_all = mx.concatenate(k_parts, axis=2)
            v_all = mx.concatenate(v_parts, axis=2)
            kv_store.append((k_all, v_all))

        mx.eval(*[t for pair in kv_store for t in pair])
        return kv_store

    # ------------------------------------------------------------------
    # Extend — N new tokens batched against existing K,V store
    # ------------------------------------------------------------------

    def extend(
        self,
        new_token_ids: mx.array,  # (1, N)
        kv_store: KVStore,  # list[L] of (K, V) each (1, nkv, S, dh)
        abs_start: int,  # absolute position of first new token
    ) -> tuple[mx.array, KVStore]:
        """
        Process N new tokens against existing K,V store in one batched pass.

        New tokens attend to stored positions (respecting sliding window per
        layer) plus each other causally.

        Returns (logits_for_N_tokens, extended_kv_store).
        extended_kv_store[i] has K,V of shape (1, nkv, S+N, head_dim).
        """
        logits, kv_store, _, _ = self._extend_core(new_token_ids, kv_store, abs_start)
        return logits, kv_store

    def extend_with_residual(
        self,
        new_token_ids: mx.array,
        kv_store: KVStore,
        abs_start: int,
    ) -> tuple[mx.array, KVStore, mx.array]:
        """
        Extend returning residual at last new-token position.

        Returns (logits, extended_kv_store, residual_last) where residual_last
        is the pre-final-norm hidden state at the last new token: (1, 1, hidden_size).
        """
        logits, kv_store, residual_last, _ = self._extend_core(new_token_ids, kv_store, abs_start)
        return logits, kv_store, residual_last

    def extend_to_layer(
        self,
        new_token_ids: mx.array,  # (1, N)
        kv_store: KVStore,
        abs_start: int,
        target_layer: int,
    ) -> tuple[mx.array, KVStore, mx.array]:
        """Extend and capture hidden state at target_layer (last new-token position).

        Same forward pass as extend() but also captures the residual stream
        after layer target_layer before continuing to the final output.

        Returns (logits, extended_kv_store, layer_h) where layer_h is the
        hidden state at the last new token after target_layer:
        shape (1, 1, hidden_size).

        Use this instead of extend() + a separate prefill_to_layer() call when
        you need an intermediate residual — it halves the compute cost.
        """
        logits, kv_store, _, layer_h = self._extend_core(
            new_token_ids, kv_store, abs_start, capture_layer=target_layer
        )
        return logits, kv_store, layer_h

    def _extend_core(
        self,
        new_token_ids: mx.array,
        kv_store: KVStore,
        abs_start: int,
        capture_layer: int | None = None,
    ) -> tuple[mx.array, KVStore, mx.array, mx.array | None]:
        """Core extend returning (logits, extended_kv_store, residual_last, layer_h).

        layer_h is the hidden state at capture_layer (last new-token position),
        or None if capture_layer is None.
        """
        backbone = self.backbone
        B, N = new_token_ids.shape
        S = kv_store[0][0].shape[2]

        h = backbone.embed(new_token_ids)

        # Causal mask among new tokens — same for all layers
        causal_new = mx.triu(mx.full((N, N), -1e9, dtype=_MASK_DTYPE), k=1)  # (N, N)

        sw = backbone.sliding_window

        # Global mask: all S stored positions visible
        global_mask = mx.concatenate([mx.zeros((N, S), dtype=_MASK_DTYPE), causal_new], axis=-1)[
            None, None
        ]  # (1, 1, N, S+N)

        # Sliding-window mask: only last sw stored positions visible
        if sw is not None and S > sw:
            sw_stored = mx.concatenate(
                [
                    mx.full((N, S - sw), -1e9, dtype=_MASK_DTYPE),
                    mx.zeros((N, sw), dtype=_MASK_DTYPE),
                ],
                axis=-1,
            )
        else:
            sw_stored = mx.zeros((N, S), dtype=_MASK_DTYPE)
        sw_mask = mx.concatenate([sw_stored, causal_new], axis=-1)[None, None]

        new_kv_store: KVStore = []
        layer_h_captured: mx.array | None = None

        for i, layer in enumerate(backbone.adapted_layers):
            k_old, v_old = kv_store[i]

            x = layer.pre_attn_norm(h)
            q, k_new, v_new = layer.project_qkv(x, B, N, offset=abs_start)

            k_all = mx.concatenate([k_old, k_new], axis=2)  # (B, nkv, S+N, dh)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            k_rpt = mx.repeat(k_all, layer.n_rep, axis=1) if layer.n_rep > 1 else k_all
            v_rpt = mx.repeat(v_all, layer.n_rep, axis=1) if layer.n_rep > 1 else v_all

            mask_i = global_mask if backbone.is_global_layer(i) else sw_mask
            attn_out = mx.fast.scaled_dot_product_attention(
                q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask_i
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)
            attn_out = layer.output_project(attn_out)

            h = layer.residual_add_attn(h, attn_out)
            h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))

            new_kv_store.append((k_all, v_all))

            if capture_layer is not None and i == capture_layer:
                layer_h_captured = h[:, -1:, :]  # (1, 1, hidden_size)

        # Capture residual at last position BEFORE final norm
        residual_last = h[:, -1:, :]  # (1, 1, hidden_size)

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        eval_targets = [logits, residual_last, *[t for pair in new_kv_store for t in pair]]
        if layer_h_captured is not None:
            eval_targets.append(layer_h_captured)
        mx.eval(*eval_targets)
        return logits, new_kv_store, residual_last, layer_h_captured

    # ------------------------------------------------------------------
    # Extend with attention weight capture
    # ------------------------------------------------------------------

    def extend_with_attention_weights(
        self,
        new_token_ids: mx.array,  # (1, N)
        kv_store: KVStore,  # list[L] of (K, V) each (1, nkv, S, dh)
        abs_start: int,
        capture_layers: set[int] | None = None,
    ) -> tuple[mx.array, KVStore, dict[int, mx.array]]:
        """
        Extend with manual attention at specified layers to capture weights.

        Parameters
        ----------
        capture_layers : set of layer indices to capture attention weights from.
                         If None, captures from all global layers.

        Returns
        -------
        (logits, extended_kv_store, attention_weights)
        attention_weights[layer_idx] has shape (1, num_heads, N, S+N) in float32.
        """
        backbone = self.backbone
        B, N = new_token_ids.shape
        S = kv_store[0][0].shape[2]

        h = backbone.embed(new_token_ids)

        # Causal mask among new tokens
        causal_new = mx.triu(mx.full((N, N), -1e9, dtype=_MASK_DTYPE), k=1)

        sw = backbone.sliding_window

        # Global mask: all S stored positions visible
        global_mask = mx.concatenate([mx.zeros((N, S), dtype=_MASK_DTYPE), causal_new], axis=-1)[
            None, None
        ]

        # Sliding-window mask
        if sw is not None and S > sw:
            sw_stored = mx.concatenate(
                [
                    mx.full((N, S - sw), -1e9, dtype=_MASK_DTYPE),
                    mx.zeros((N, sw), dtype=_MASK_DTYPE),
                ],
                axis=-1,
            )
        else:
            sw_stored = mx.zeros((N, S), dtype=_MASK_DTYPE)
        sw_mask = mx.concatenate([sw_stored, causal_new], axis=-1)[None, None]

        # Default: capture all global layers
        if capture_layers is None:
            capture_layers = {
                i for i in range(len(backbone.adapted_layers)) if backbone.is_global_layer(i)
            }

        new_kv_store: KVStore = []
        captured_weights: dict[int, mx.array] = {}

        for i, layer in enumerate(backbone.adapted_layers):
            k_old, v_old = kv_store[i]
            x = layer.pre_attn_norm(h)
            q, k_new, v_new = layer.project_qkv(x, B, N, offset=abs_start)

            k_all = mx.concatenate([k_old, k_new], axis=2)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            k_rpt = mx.repeat(k_all, layer.n_rep, axis=1) if layer.n_rep > 1 else k_all
            v_rpt = mx.repeat(v_all, layer.n_rep, axis=1) if layer.n_rep > 1 else v_all

            mask_i = global_mask if backbone.is_global_layer(i) else sw_mask

            if i in capture_layers:
                # Manual attention to capture weights
                # q: (B, num_heads, N, head_dim)
                # k_rpt: (B, num_heads, S+N, head_dim)
                scores = (q @ k_rpt.transpose(0, 1, 3, 2)) * layer.attn_scale
                scores = scores.astype(mx.float32)
                scores = scores + mask_i.astype(mx.float32)
                weights = mx.softmax(scores, axis=-1)  # (B, num_heads, N, S+N)
                captured_weights[i] = weights

                attn_out = weights.astype(v_rpt.dtype) @ v_rpt
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)
            else:
                # Fused path — no weight capture
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k_rpt, v_rpt, scale=layer.attn_scale, mask=mask_i
                )
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)

            attn_out = layer.output_project(attn_out)
            h = layer.residual_add_attn(h, attn_out)
            h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))
            new_kv_store.append((k_all, v_all))

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)

        eval_targets = [logits]
        eval_targets.extend(t for pair in new_kv_store for t in pair)
        eval_targets.extend(captured_weights.values())
        mx.eval(*eval_targets)

        return logits, new_kv_store, captured_weights

    # ------------------------------------------------------------------
    # Slide — evict oldest tokens from K,V store
    # ------------------------------------------------------------------

    @staticmethod
    def slide(kv_store: KVStore, evict_count: int) -> KVStore:
        """
        Drop the first evict_count tokens from every layer's K,V store.
        O(1) — just a slice, no recomputation.
        """
        return [(k[:, :, evict_count:, :], v[:, :, evict_count:, :]) for k, v in kv_store]

    # ------------------------------------------------------------------
    # Chunked prefill — progressive with resume support
    # ------------------------------------------------------------------

    def prefill_chunked(
        self,
        input_ids: mx.array,  # (1, N) — tokens to prefill
        chunk_size: int = 512,
        abs_start: int = 0,  # absolute position of first token (for resume)
        kv_store: KVStore | None = None,  # existing store to extend (for resume)
    ):
        """
        Chunked prefill generator.

        Yields (tokens_done, tokens_total, last_logits, kv_store) after each chunk,
        where tokens_done is relative to the start of this call (not abs_start).

        The caller should:
          - Save a checkpoint between yields for Ctrl+C safety.
          - Pass abs_start + tokens_done as the seq_len to step() afterwards.

        On the first call (no resume): abs_start=0, kv_store=None.
        On resume: abs_start=seq_len_so_far, kv_store=loaded_kv_store.
        """
        _, S = input_ids.shape
        current_kv: KVStore = kv_store if kv_store is not None else []
        offset = 0
        first_chunk = kv_store is None  # True when starting fresh

        while offset < S:
            end = min(offset + chunk_size, S)
            chunk = input_ids[:, offset:end]
            abs_offset = abs_start + offset

            if first_chunk:
                last_logits, current_kv = self.prefill(chunk)
                first_chunk = False
            else:
                last_logits, current_kv = self.extend(chunk, current_kv, abs_start=abs_offset)

            offset = end
            yield offset, S, last_logits, current_kv

    # ------------------------------------------------------------------
    # Single-token step (compiled)
    # ------------------------------------------------------------------

    def _raw_step(
        self,
        new_token_ids: mx.array,  # (1, 1)
        kv_store: KVStore,  # list[L] of (K, V) each (1, nkv, S, dh)
        seq_len: int,  # absolute position of the new token
    ) -> tuple[mx.array, KVStore]:
        """
        Process one new token against stored K,V.

        For each layer:
          - K_old, V_old from kv_store[i] — no recomputation
          - Q_new, K_new, V_new from new token only (1-position matmuls)
          - Fused attention over [K_old; K_new], [V_old; V_new]
          - FFN on new token only
          - Append K_new, V_new to store
        """
        backbone = self.backbone
        B = 1

        h = backbone.embed(new_token_ids)
        new_kv_store: KVStore = []

        sw = backbone.sliding_window

        for i, layer in enumerate(backbone.adapted_layers):
            k_old, v_old = kv_store[i]

            x = layer.pre_attn_norm(h)
            q, k_new, v_new = layer.project_qkv(x, B, 1, offset=seq_len)

            k_all = mx.concatenate([k_old, k_new], axis=2)  # (B, nkv, S+1, dh)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            k_rpt = mx.repeat(k_all, layer.n_rep, axis=1) if layer.n_rep > 1 else k_all
            v_rpt = mx.repeat(v_all, layer.n_rep, axis=1) if layer.n_rep > 1 else v_all

            # Apply sliding-window mask for non-global layers when KV exceeds window
            is_global = backbone.is_global_layer(i)
            total = k_all.shape[2]

            if (not is_global) and sw is not None and total > sw:
                step_mask = mx.concatenate(
                    [
                        mx.full((1, 1, 1, total - sw), -1e9, dtype=_MASK_DTYPE),
                        mx.zeros((1, 1, 1, sw), dtype=_MASK_DTYPE),
                    ],
                    axis=-1,
                )
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k_rpt, v_rpt, scale=layer.attn_scale, mask=step_mask
                )
            else:
                attn_out = mx.fast.scaled_dot_product_attention(
                    q, k_rpt, v_rpt, scale=layer.attn_scale
                )

            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
            attn_out = layer.output_project(attn_out)

            h = layer.residual_add_attn(h, attn_out)
            h = layer.residual_add_ffn(h, layer.ffn(layer.pre_ffn_norm(h)))

            new_kv_store.append((k_all, v_all))

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        return logits, new_kv_store

    def step(
        self,
        new_token_ids: mx.array,
        kv_store: KVStore,
        seq_len: int,
    ) -> tuple[mx.array, KVStore]:
        """Compiled single-token step. See _raw_step for details."""
        return self._step(new_token_ids, kv_store, seq_len)

    def step_uncompiled(
        self,
        new_token_ids: mx.array,
        kv_store: KVStore,
        seq_len: int,
    ) -> tuple[mx.array, KVStore]:
        """
        Uncompiled single-token step.

        Use instead of step() when the kv_store was built from extend() over a
        dynamically-sized replayed context — mx.compile(shapeless=True) can
        mis-trace in that case and raise broadcast shape errors.
        """
        return self._raw_step(new_token_ids, kv_store, seq_len)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def kv_bytes(self, seq_len: int) -> int:
        """Bytes used by K,V store for seq_len tokens."""
        layer = self.backbone.adapted_layers[0]
        num_layers = len(self.backbone.adapted_layers)
        return 2 * layer.num_kv_heads * layer.head_dim * seq_len * num_layers * 2

    def residual_equivalent_bytes(self, seq_len: int) -> int:
        """Bytes that an RS residual store would use for the same seq_len."""
        num_layers = len(self.backbone.adapted_layers)
        return self.backbone.hidden_size * seq_len * num_layers * 2


def make_kv_generator(model, config=None) -> KVDirectGenerator:
    """
    Factory: create a KVDirectGenerator for any supported model.

    Auto-detects the model family from the class name.

    Examples::

        gen = make_kv_generator(rs_model)        # GemmaResidualStreamForCausalLM
        gen = make_kv_generator(llama_model)     # LlamaForCausalLM
        gen = make_kv_generator(mistral_model)   # Mistral (Llama-family)

    Pass a pre-built ModelBackboneProtocol to KVDirectGenerator() directly
    for families not yet covered here.
    """
    cls_name = type(model).__name__
    if "Gemma" in cls_name:
        return KVDirectGenerator.from_gemma_rs(model)
    if "Llama" in cls_name or "Mistral" in cls_name:
        return KVDirectGenerator.from_llama(model)
    raise ValueError(
        f"Cannot auto-detect adapter for {cls_name!r}. "
        "Pass a pre-built ModelBackboneProtocol to KVDirectGenerator() directly, "
        "or implement an adapter in chuk_lazarus.inference.context.adapters."
    )
