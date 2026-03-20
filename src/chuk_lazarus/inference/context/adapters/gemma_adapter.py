"""
Gemma adapter implementing TransformerLayerProtocol / ModelBackboneProtocol.

Wraps GemmaResidualStreamForCausalLM (and its blocks) without modifying
any existing model code.

Gemma-specific details handled here:
- 4 RMSNorm layers per block
- clip_residual for residual adds
- per-head q_norm / k_norm before RoPE
- sqrt(hidden_size) embedding scale
- sliding-window attention with is_global_layer() config method
"""

from __future__ import annotations

import mlx.core as mx

from chuk_lazarus.models_v2.families.gemma.model import clip_residual


class GemmaLayerAdapter:
    """
    Adapts a single GemmaRSBlock to TransformerLayerProtocol.

    GemmaRSBlock norms:
        input_layernorm          → pre_attn_norm
        post_attention_layernorm → residual_add_attn (applied to delta)
        pre_feedforward_layernorm→ pre_ffn_norm
        post_feedforward_layernorm→ residual_add_ffn (applied to delta)
    """

    __slots__ = ("_block",)

    def __init__(self, block) -> None:
        self._block = block

    # --- Attention ---

    def pre_attn_norm(self, h: mx.array) -> mx.array:
        return self._block.input_layernorm(h)

    def project_qkv(
        self, x: mx.array, B: int, S: int, offset: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        attn = self._block.self_attn
        nq = attn.num_heads
        nkv = attn.num_kv_heads
        dh = attn.head_dim

        q = attn.q_proj(x).reshape(B, S, nq, dh).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)

        q = attn.q_norm(q)
        k = attn.k_norm(k)

        q = attn.rope(q, offset=offset)
        k = attn.rope(k, offset=offset)
        return q, k, v

    def project_qkv_pre_rope(
        self,
        x: mx.array,
        B: int,
        S: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Project Q, K, V with norms but WITHOUT RoPE.

        Returns pre-RoPE K,V for storage. RoPE is applied at injection time
        with the desired target positions via apply_rope().
        """
        attn = self._block.self_attn
        nq = attn.num_heads
        nkv = attn.num_kv_heads
        dh = attn.head_dim

        q = attn.q_proj(x).reshape(B, S, nq, dh).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)

        q = attn.q_norm(q)
        k = attn.k_norm(k)
        # No RoPE — caller applies it later with desired positions
        return q, k, v

    def apply_rope(self, x: mx.array, offset: int) -> mx.array:
        """Apply RoPE to pre-RoPE Q or K at the desired position offset."""
        return self._block.self_attn.rope(x, offset=offset)

    def output_project(self, attn_result: mx.array) -> mx.array:
        return self._block.self_attn.o_proj(attn_result)

    def residual_add_attn(self, h: mx.array, attn_out: mx.array) -> mx.array:
        return clip_residual(h, self._block.post_attention_layernorm(attn_out))

    # --- FFN ---

    def pre_ffn_norm(self, h: mx.array) -> mx.array:
        return self._block.pre_feedforward_layernorm(h)

    def ffn(self, x: mx.array) -> mx.array:
        return self._block.mlp(x)

    def residual_add_ffn(self, h: mx.array, ffn_out: mx.array) -> mx.array:
        return clip_residual(h, self._block.post_feedforward_layernorm(ffn_out))

    # --- Dimensions ---

    @property
    def num_heads(self) -> int:
        return self._block.self_attn.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._block.self_attn.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._block.self_attn.head_dim

    @property
    def n_rep(self) -> int:
        return self._block.self_attn.n_rep

    @property
    def attn_scale(self) -> float:
        return self._block.self_attn.scale


class GemmaBackboneAdapter:
    """
    Adapts GemmaForCausalLM or GemmaResidualStreamForCausalLM to ModelBackboneProtocol.

    Handles both model types — GemmaResidualStream has private helpers (_embed,
    _mask_for_layer, _unembed); GemmaModel uses embed_tokens/embedding_scale/
    _create_attention_mask/lm_head instead.
    """

    def __init__(self, causal_lm) -> None:
        """
        Args:
            causal_lm: GemmaForCausalLM or GemmaResidualStreamForCausalLM with loaded weights.
        """
        self._model = causal_lm
        self._backbone = causal_lm.model  # GemmaResidualStream

        self._adapted: list[GemmaLayerAdapter] = [
            GemmaLayerAdapter(block) for block in self._backbone.layers
        ]

    @property
    def adapted_layers(self) -> list[GemmaLayerAdapter]:
        return self._adapted

    def embed(self, input_ids: mx.array) -> mx.array:
        if hasattr(self._backbone, "_embed"):
            return self._backbone._embed(input_ids)
        # GemmaModel: embed_tokens + sqrt(hidden_size) scaling
        h = self._backbone.embed_tokens(input_ids)
        scale = mx.array(self._backbone.embedding_scale, dtype=mx.bfloat16).astype(h.dtype)
        return h * scale

    def unembed(self, h: mx.array) -> mx.array:
        if hasattr(self._model, "_unembed"):
            return self._model._unembed(h)
        # GemmaForCausalLM
        if self._model.tie_word_embeddings:
            return self._model.model.embed_tokens.as_linear(h)
        return self._model.lm_head(h)

    def final_norm(self, h: mx.array) -> mx.array:
        return self._backbone.norm(h)

    def prefill_mask(self, layer_idx: int, h: mx.array) -> mx.array | None:
        if hasattr(self._backbone, "_mask_for_layer"):
            return self._backbone._mask_for_layer(layer_idx, h)
        # GemmaModel fallback
        if h.shape[1] <= 1:
            return None
        is_global = self._backbone.config.is_global_layer(layer_idx)
        window = None if is_global else self._backbone.sliding_window
        return self._backbone._create_attention_mask(h, cache=None, window_size=window)

    def is_global_layer(self, layer_idx: int) -> bool:
        return self._backbone.config.is_global_layer(layer_idx)

    @property
    def sliding_window(self) -> int | None:
        return getattr(self._backbone.config, "sliding_window", None)

    @property
    def hidden_size(self) -> int:
        return self._backbone.config.hidden_size

    @property
    def embed_matrix(self) -> mx.array:
        """Token embedding weight matrix, shape (vocab_size, hidden_size)."""
        return self._backbone.embed_tokens.weight


__all__ = ["GemmaBackboneAdapter", "GemmaLayerAdapter"]
