"""
Compiled residual-stream single-token generator.

Strategy: store per-layer residuals from prefill (read-only).
For each new token:
  1. K_old, V_old ← stored_residuals[i] @ wk.T / wv.T  (one matmul per layer)
  2. Q, K_new, V_new from the new token embedding
  3. mx.fast.scaled_dot_product_attention over [K_old|K_new], [V_old|V_new]
  4. FFN on new token only

mx.compile sees the entire step and can fuse K,V projection into the attention
kernel — K and V are computed and consumed without materialising as separate tensors.

Memory during generation:
  num_layers × seq_len × hidden_dim × 2 bytes (per-layer residuals)
  vs KV cache: num_layers × seq_len × 2 × kv_heads × head_dim × 2 bytes

Ratio: hidden_dim / (2 × kv_heads × head_dim)
  270M: 640 / (2×1×256) = 1.25×  (residual LARGER)
  4B:   2560 / (2×4×256) = 1.25× (residual LARGER)
  12B:  3840 / (2×8×256) = 0.94× (residual SMALLER — RS wins)

Between turns: only token IDs are stored (4 bytes/token, 4608× smaller than KV).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class CompiledRSGenerator:
    """
    Single-token generator using compiled residual-stream forward.

    Use this as the hot-tier engine when:
    - Memory budget cannot fit a full KV cache for the active window
    - Dark inference (probe/inject) is needed during generation
    - The 2-2.6× generation overhead is acceptable

    Usage
    -----
        gen = CompiledRSGenerator(rs_model, config)

        # Prefill
        logits, stored = gen.prefill(input_ids)  # input_ids: (1, seq_len)

        # Generate
        seq_len = input_ids.shape[1]
        for _ in range(max_new_tokens):
            next_tok = mx.argmax(logits[0, -1, :])
            logits, stored = gen.step(mx.array([[next_tok]]), stored, seq_len)
            seq_len += 1
    """

    def __init__(self, rs_model, config):
        self.model  = rs_model
        self.config = config
        self._step  = mx.compile(self._raw_step, shapeless=True)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self, input_ids: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Full forward pass. Returns (logits, per_layer_residuals).

        per_layer_residuals[i] is the residual ENTERING layer i — i.e. the
        complete Markov state that determines K,V for layer i.
        """
        backbone = self.model.model

        h = backbone._embed(input_ids)
        layer_inputs: list[mx.array] = []

        for i, layer in enumerate(backbone.layers):
            layer_inputs.append(h)
            mask = backbone._mask_for_layer(i, h)
            h = layer(h, mask=mask)

        h = backbone.norm(h)
        logits = self.model._unembed(h)
        mx.eval(logits, *layer_inputs)
        return logits, layer_inputs

    # ------------------------------------------------------------------
    # Single-token step (compiled)
    # ------------------------------------------------------------------

    def _raw_step(
        self,
        new_token_ids: mx.array,            # (1, 1)
        stored_residuals: list[mx.array],   # list[L] of (1, seq_len, hidden)
        seq_len: int,                       # current sequence length (for RoPE offset)
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Process one new token.

        For each layer:
          K_old, V_old from stored_residuals[i] (big matmul, read-only data)
          Q, K_new, V_new from new token embed
          fused attention over [K_old|K_new], [V_old|V_new]
          FFN on new token only

        Returns (logits, updated_stored_residuals).
        """
        backbone = self.model.model

        h_new = backbone._embed(new_token_ids)  # (1, 1, hidden)
        new_layer_inputs: list[mx.array] = []

        for i, layer in enumerate(backbone.layers):
            attn  = layer.self_attn
            h_old = stored_residuals[i]  # (1, seq_len, hidden)

            # Normalise
            x_old = layer.input_layernorm(h_old)
            x_new = layer.input_layernorm(h_new)

            # K, V from stored residual (old positions)
            k_old = attn.k_proj(x_old)
            v_old = attn.v_proj(x_old)

            # Q, K, V for new token
            q_new = attn.q_proj(x_new)
            k_new = attn.k_proj(x_new)
            v_new = attn.v_proj(x_new)

            # Reshape to (batch, heads, seq, head_dim)
            B, S_old, _ = h_old.shape
            nq  = attn.num_heads
            nkv = attn.num_kv_heads
            dh  = attn.head_dim

            k_old = k_old.reshape(B, S_old, nkv, dh).transpose(0, 2, 1, 3)
            v_old = v_old.reshape(B, S_old, nkv, dh).transpose(0, 2, 1, 3)
            q_new = q_new.reshape(B, 1,     nq,  dh).transpose(0, 2, 1, 3)
            k_new = k_new.reshape(B, 1,     nkv, dh).transpose(0, 2, 1, 3)
            v_new = v_new.reshape(B, 1,     nkv, dh).transpose(0, 2, 1, 3)

            # Q/K norm (Gemma-specific)
            q_new = attn.q_norm(q_new)
            k_old = attn.k_norm(k_old)
            k_new = attn.k_norm(k_new)

            # RoPE: old positions at 0..seq_len-1, new token at seq_len
            k_old = attn.rope(k_old)
            q_new = attn.rope(q_new, offset=seq_len)
            k_new = attn.rope(k_new, offset=seq_len)

            # Concat old + new K, V
            k_all = mx.concatenate([k_old, k_new], axis=2)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            # GQA repeat
            if attn.n_rep > 1:
                k_all = mx.repeat(k_all, attn.n_rep, axis=1)
                v_all = mx.repeat(v_all, attn.n_rep, axis=1)

            # Fused attention (no causal mask — new token attends to all)
            attn_out = mx.fast.scaled_dot_product_attention(
                q_new, k_all, v_all, scale=attn.scale
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
            attn_out = attn.o_proj(attn_out)

            # Residual add
            h_new = h_new + layer.post_attention_layernorm(attn_out)

            # FFN on new token only
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h_new))
            h_new   = h_new + layer.post_feedforward_layernorm(ffn_out)

            # Append new token's pre-layer residual to stored block
            new_layer_inputs.append(
                mx.concatenate([stored_residuals[i], h_new], axis=1)
            )

        h_final = backbone.norm(h_new)
        logits  = self.model._unembed(h_final)
        return logits, new_layer_inputs

    def step(
        self,
        new_token_ids: mx.array,
        stored_residuals: list[mx.array],
        seq_len: int,
    ) -> tuple[mx.array, list[mx.array]]:
        """Compiled single-token step. See _raw_step for details."""
        return self._step(new_token_ids, stored_residuals, seq_len)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def residual_bytes(self, seq_len: int) -> int:
        """Bytes used by per-layer residuals for seq_len tokens."""
        return self.config.num_hidden_layers * seq_len * self.config.hidden_size * 2

    def kv_equivalent_bytes(self, seq_len: int) -> int:
        """KV cache bytes that would be used for the same seq_len."""
        return (
            2 * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * seq_len * self.config.head_dim * 2
        )
