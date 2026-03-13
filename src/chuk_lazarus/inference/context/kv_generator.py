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

# Type alias for the per-layer KV store
KVStore = list[tuple[mx.array, mx.array]]


class KVDirectGenerator:
    """
    Incremental generator that stores K,V per layer.

    Accepts any ModelBackboneProtocol — Gemma, Llama, Mistral, etc.

    Storage per layer: (K, V) where
        K.shape = (batch, num_kv_heads, seq_len, head_dim)
        V.shape = (batch, num_kv_heads, seq_len, head_dim)
    K and V are post-norm, post-RoPE — ready for scaled_dot_product_attention.
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
        backbone = self.backbone
        B, S = input_ids.shape
        h = backbone.embed(input_ids)
        kv_store: KVStore = []

        for i, layer in enumerate(backbone.adapted_layers):
            mask = backbone.prefill_mask(i, h)
            x = layer.pre_attn_norm(h)
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

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        mx.eval(logits, *[t for pair in kv_store for t in pair])
        return logits, kv_store

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

        h = backbone.final_norm(h)
        logits = backbone.unembed(h)
        mx.eval(logits, *[t for pair in new_kv_store for t in pair])
        return logits, new_kv_store

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
