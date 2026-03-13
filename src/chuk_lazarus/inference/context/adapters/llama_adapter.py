"""
Llama adapter implementing TransformerLayerProtocol / ModelBackboneProtocol.

Wraps LlamaForCausalLM (and its blocks) without modifying any existing model code.

Llama-specific details handled here:
- 2 RMSNorm layers per block (pre-attn, pre-FFN on full residual)
- Plain residual adds (no norm on deltas)
- No per-head q_norm / k_norm
- No embedding scale
- No sliding-window for standard Llama (all layers global)
  SlidingWindowAttention blocks are detected and treated as non-global
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LlamaLayerAdapter:
    """
    Adapts a single LlamaBlock to TransformerLayerProtocol.

    LlamaBlock norms:
        input_layernorm          → pre_attn_norm
        post_attention_layernorm → pre_ffn_norm (applied to full residual, not delta)

    Residual adds are plain addition (no norm applied to the delta).
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

        # Llama has no q_norm / k_norm — apply RoPE directly
        if attn.rope is not None:
            q = attn.rope(q, offset=offset)
            k = attn.rope(k, offset=offset)
        return q, k, v

    def output_project(self, attn_result: mx.array) -> mx.array:
        return self._block.self_attn.o_proj(attn_result)

    def residual_add_attn(self, h: mx.array, attn_out: mx.array) -> mx.array:
        # Plain add — Llama applies no norm to the attention delta
        return h + attn_out

    # --- FFN ---

    def pre_ffn_norm(self, h: mx.array) -> mx.array:
        # Llama's post_attention_layernorm is applied to the full residual before FFN
        return self._block.post_attention_layernorm(h)

    def ffn(self, x: mx.array) -> mx.array:
        return self._block.mlp(x)

    def residual_add_ffn(self, h: mx.array, ffn_out: mx.array) -> mx.array:
        # Plain add — Llama applies no norm to the FFN delta
        return h + ffn_out

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


class LlamaBackboneAdapter:
    """
    Adapts a LlamaForCausalLM to ModelBackboneProtocol.

    No embedding scale (unlike Gemma).
    Standard Llama: all layers are global (no sliding window).
    Mistral variants: blocks using SlidingWindowAttention are detected automatically.
    """

    def __init__(self, causal_lm) -> None:
        """
        Args:
            causal_lm: LlamaForCausalLM instance with loaded weights.
        """
        self._model = causal_lm
        self._backbone = causal_lm.model  # LlamaModel

        self._adapted: list[LlamaLayerAdapter] = [
            LlamaLayerAdapter(block) for block in self._backbone.layers
        ]

    @property
    def adapted_layers(self) -> list[LlamaLayerAdapter]:
        return self._adapted

    def embed(self, input_ids: mx.array) -> mx.array:
        # No embedding scale in Llama
        return self._backbone.embed_tokens(input_ids)

    def unembed(self, h: mx.array) -> mx.array:
        return self._model.lm_head(h)

    def final_norm(self, h: mx.array) -> mx.array:
        return self._backbone.norm(h)

    def prefill_mask(self, layer_idx: int, h: mx.array) -> mx.array | None:
        _, seq_len, _ = h.shape
        if seq_len <= 1:
            return None
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        return mask.astype(h.dtype)

    def is_global_layer(self, layer_idx: int) -> bool:
        """
        True for full causal attention; False for sliding-window layers.

        Standard Llama: always True.
        Mistral (SlidingWindowAttention on even layers): detected from block type.
        """
        from chuk_lazarus.models_v2.components.attention.sliding_window import (
            SlidingWindowAttention,
        )

        block = self._backbone.layers[layer_idx]
        return not isinstance(block.self_attn, SlidingWindowAttention)

    @property
    def sliding_window(self) -> int | None:
        return getattr(self._model.config, "sliding_window", None)

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size


__all__ = ["LlamaBackboneAdapter", "LlamaLayerAdapter"]
