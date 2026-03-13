"""
Protocols for model-agnostic KV-direct generation.

Any transformer architecture can work with KVDirectGenerator by implementing
these protocols — either directly or via a thin adapter wrapper.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class TransformerLayerProtocol(Protocol):
    """
    Per-layer interface for KVDirectGenerator.

    Normalises architecture differences:
    - Residual add semantics  (Gemma: clip_residual+norm delta, Llama: plain add)
    - Pre-FFN norm source     (Gemma: dedicated 4th norm, Llama: 2nd norm on full residual)
    - Per-head norms          (Gemma: q_norm/k_norm, Llama: none)
    - RoPE application with explicit absolute offset
    """

    def pre_attn_norm(self, h: mx.array) -> mx.array:
        """Pre-attention normalisation (input_layernorm for both Gemma and Llama)."""
        ...

    def project_qkv(
        self, x: mx.array, B: int, S: int, offset: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Project x → Q, K, V.

        Applies per-head norms (Gemma: q_norm/k_norm; Llama: none) and RoPE
        with the given absolute position offset.

        Returns (q, k, v):
            q: (B, num_heads,    S, head_dim)
            k: (B, num_kv_heads, S, head_dim)  — pre-GQA-repeat, post-RoPE
            v: (B, num_kv_heads, S, head_dim)
        """
        ...

    def output_project(self, attn_result: mx.array) -> mx.array:
        """Output projection (o_proj). attn_result: (B, S, num_heads*head_dim)."""
        ...

    def residual_add_attn(self, h: mx.array, attn_out: mx.array) -> mx.array:
        """
        Add attention output to the residual stream.

        Gemma: clip_residual(h, post_attention_layernorm(attn_out))
        Llama: h + attn_out
        """
        ...

    def pre_ffn_norm(self, h: mx.array) -> mx.array:
        """
        Normalise hidden state before the FFN.

        Gemma: pre_feedforward_layernorm(h)   — 3rd dedicated norm
        Llama: post_attention_layernorm(h)    — 2nd norm, applied to full residual
        """
        ...

    def ffn(self, x: mx.array) -> mx.array:
        """Feed-forward network (MLP)."""
        ...

    def residual_add_ffn(self, h: mx.array, ffn_out: mx.array) -> mx.array:
        """
        Add FFN output to the residual stream.

        Gemma: clip_residual(h, post_feedforward_layernorm(ffn_out))
        Llama: h + ffn_out
        """
        ...

    @property
    def num_heads(self) -> int:
        """Total query heads."""
        ...

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads (< num_heads for GQA)."""
        ...

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        ...

    @property
    def n_rep(self) -> int:
        """GQA repetition factor = num_heads // num_kv_heads."""
        ...

    @property
    def attn_scale(self) -> float:
        """Attention scale factor = head_dim ** -0.5."""
        ...


@runtime_checkable
class ModelBackboneProtocol(Protocol):
    """
    Top-level model interface for KVDirectGenerator.

    Wraps an entire CausalLM and provides a uniform interface regardless of
    the underlying architecture (Gemma, Llama, Mistral, ...).
    """

    @property
    def adapted_layers(self) -> list[TransformerLayerProtocol]:
        """Adapted transformer blocks, one per layer."""
        ...

    def embed(self, input_ids: mx.array) -> mx.array:
        """Token IDs → hidden states (B, S, hidden). Includes any embedding scale."""
        ...

    def unembed(self, h: mx.array) -> mx.array:
        """Final hidden → logits (B, S, vocab)."""
        ...

    def final_norm(self, h: mx.array) -> mx.array:
        """Final layer normalisation."""
        ...

    def prefill_mask(self, layer_idx: int, h: mx.array) -> mx.array | None:
        """
        Attention mask for prefill at layer_idx over the full sequence h.

        Returns None when h has only 1 token (no mask needed).
        Global layers → standard causal mask.
        Sliding-window layers → windowed causal mask.
        """
        ...

    def is_global_layer(self, layer_idx: int) -> bool:
        """True for full causal attention layers; False for sliding-window layers."""
        ...

    @property
    def sliding_window(self) -> int | None:
        """Sliding window size if the model uses SW attention, else None."""
        ...

    @property
    def hidden_size(self) -> int:
        """Hidden state dimension (used for memory accounting)."""
        ...


__all__ = ["ModelBackboneProtocol", "TransformerLayerProtocol"]
