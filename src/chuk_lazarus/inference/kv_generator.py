"""
KV-direct residual stream generator.

Architectural observation from experiment a9704704:

  The current RS generator stores per-layer residuals (hidden_size=2560D) and
  recomputes K,V at every step by running k_proj(layernorm(h_old)) over S positions.
  This is the source of the 3.3× overhead vs standard KV caching.

  This generator replaces that approach: store K,V directly after they are computed,
  reuse them at step time without recomputation. The result:

    Memory  : identical to standard KV cache (2 × nkv × head_dim × S × num_layers)
    Speed   : approaches KV-cached speed — no K,V recompute for old tokens at step time
    Quality : bit-exact with standard KV (K,V are the same tensors)

  For Gemma 4B: hidden=2560, nkv=4, head_dim=320 → same bytes/token as residual store.
  The memory is exactly equivalent; the speedup comes purely from skipping the matmuls.

Extension: rank-r compressed attention (next phase)
  Experiment a9704704 showed head_dim=320 is 21-32× over-provisioned.
  W_q @ W_k^T per head has effective rank 3-5 for 90% and 10-15 for 99% accuracy.
  A future LowRankKVGenerator can store K_compressed = V_k.T @ K (r-dim, r≈15)
  and transform queries to the same basis, cutting K storage by 21× with no retraining.
  That is the next experiment.

Lifecycle
---------
  Turn 1 : prefill(130 tokens)  → kv_store: list[L] of (K, V) each (1, nkv, 130, dh)
            generate 40 tokens via step() → kv_store grows to (1, nkv, 170, dh)

  Turn 2 : extend(80 tokens, kv_store, abs_start=170)
            → kv_store grows to (1, nkv, 250, dh)
            generate 40 tokens → (1, nkv, 290, dh)

  Turn N : budget hit — slide(kv_store, evict_n) → O(1) slice, no recompute

Note on sliding windows
-----------------------
  Gemma uses sliding window attention for non-global layers (window=512).
  This implementation treats all layers as global (matches CompiledRSGenerator).
  For production: slide k_old to window_size before concat in _raw_step.
"""

from __future__ import annotations

import mlx.core as mx


class KVDirectGenerator:
    """
    Incremental generator that stores K,V per layer instead of full residuals.

    Storage per layer: (K, V) where
        K.shape = (batch, num_kv_heads, seq_len, head_dim)
        V.shape = (batch, num_kv_heads, seq_len, head_dim)

    K and V are post-norm, post-RoPE — ready for scaled_dot_product_attention.
    At step time, only Q, K_new, V_new are computed for the new token.
    Old K,V are concatenated directly (no recompute).
    """

    def __init__(self, rs_model, config):
        self.model = rs_model
        self.config = config
        self._step = mx.compile(self._raw_step, shapeless=True)

    # ------------------------------------------------------------------
    # Prefill — full forward from token IDs
    # ------------------------------------------------------------------

    def prefill(
        self,
        input_ids: mx.array,                    # (1, S)
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """
        Full forward pass. Returns (logits, per_layer_kv).

        kv_store[i] = (K, V) for layer i:
            K shape (1, nkv, S, head_dim)  — post-norm, post-RoPE
            V shape (1, nkv, S, head_dim)  — post-proj
        """
        backbone = self.model.model
        B, S = input_ids.shape

        h = backbone._embed(input_ids)          # (B, S, hidden)
        kv_store: list[tuple[mx.array, mx.array]] = []

        for i, layer in enumerate(backbone.layers):
            mask = backbone._mask_for_layer(i, h)

            # Run attention manually to intercept K,V before discarding them
            x = layer.input_layernorm(h)
            attn_out, (k, v) = layer.self_attn(x, mask, cache=None)
            # k: (B, nkv, S, head_dim)  post-norm, post-RoPE
            # v: (B, nkv, S, head_dim)  post-proj

            h = h + layer.post_attention_layernorm(attn_out)
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            h = h + layer.post_feedforward_layernorm(ffn_out)

            kv_store.append((k, v))

        h = backbone.norm(h)
        logits = self.model._unembed(h)
        mx.eval(logits, *[t for pair in kv_store for t in pair])
        return logits, kv_store

    # ------------------------------------------------------------------
    # Extend — N new tokens batched against existing K,V store
    # ------------------------------------------------------------------

    def extend(
        self,
        new_token_ids: mx.array,                         # (1, N)
        kv_store: list[tuple[mx.array, mx.array]],       # list[L] of (K, V) each (1, nkv, S, dh)
        abs_start: int,                                  # absolute position of first new token
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """
        Process N new tokens against existing K,V store in one batched pass.

        New tokens attend to all S stored positions (fully visible) plus
        each other causally (token i sees tokens 0..i among the N new ones).

        Returns (logits_for_N_tokens, extended_kv_store).
        extended_kv_store[i] has K,V of shape (1, nkv, S+N, head_dim).
        """
        backbone = self.model.model
        B, N = new_token_ids.shape
        S = kv_store[0][0].shape[2]             # sequence length of stored K

        h = backbone._embed(new_token_ids)      # (B, N, hidden)

        # Causal mask: new tokens see all S stored + causal among themselves
        neg_inf    = mx.full((N, N), -1e9, dtype=mx.bfloat16)
        causal_new = mx.triu(neg_inf, k=1)       # (N, N) upper-tri = future
        stored_vis = mx.zeros((N, S), dtype=mx.bfloat16)
        mask = mx.concatenate([stored_vis, causal_new], axis=-1)[None, None]  # (1,1,N,S+N)

        new_kv_store: list[tuple[mx.array, mx.array]] = []

        for i, layer in enumerate(backbone.layers):
            k_old, v_old = kv_store[i]          # (B, nkv, S, head_dim)
            attn = layer.self_attn
            nq  = attn.num_heads
            nkv = attn.num_kv_heads
            dh  = attn.head_dim

            x_new = layer.input_layernorm(h)    # (B, N, hidden)

            q_new = attn.q_proj(x_new).reshape(B, N, nq,  dh).transpose(0, 2, 1, 3)
            k_new = attn.k_proj(x_new).reshape(B, N, nkv, dh).transpose(0, 2, 1, 3)
            v_new = attn.v_proj(x_new).reshape(B, N, nkv, dh).transpose(0, 2, 1, 3)

            q_new = attn.q_norm(q_new)
            k_new = attn.k_norm(k_new)

            # RoPE: new tokens at absolute positions abs_start..abs_start+N-1
            # k_old already has RoPE from when it was originally computed
            q_new = attn.rope(q_new, offset=abs_start)
            k_new = attn.rope(k_new, offset=abs_start)

            k_all = mx.concatenate([k_old, k_new], axis=2)  # (B, nkv, S+N, dh)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            if attn.n_rep > 1:
                k_rpt = mx.repeat(k_all, attn.n_rep, axis=1)
                v_rpt = mx.repeat(v_all, attn.n_rep, axis=1)
            else:
                k_rpt, v_rpt = k_all, v_all

            attn_out = mx.fast.scaled_dot_product_attention(
                q_new, k_rpt, v_rpt, scale=attn.scale, mask=mask
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)
            attn_out = attn.o_proj(attn_out)

            h = h + layer.post_attention_layernorm(attn_out)
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            h = h + layer.post_feedforward_layernorm(ffn_out)

            new_kv_store.append((k_all, v_all))

        h_norm = backbone.norm(h)
        logits  = self.model._unembed(h_norm)   # (B, N, vocab)

        mx.eval(logits, *[t for pair in new_kv_store for t in pair])
        return logits, new_kv_store

    # ------------------------------------------------------------------
    # Slide — evict oldest tokens from K,V store
    # ------------------------------------------------------------------

    @staticmethod
    def slide(
        kv_store: list[tuple[mx.array, mx.array]],
        evict_count: int,
    ) -> list[tuple[mx.array, mx.array]]:
        """
        Drop the first evict_count tokens from every layer's K,V store.
        O(1) — just a slice, no recomputation.
        """
        return [
            (k[:, :, evict_count:, :], v[:, :, evict_count:, :])
            for k, v in kv_store
        ]

    # ------------------------------------------------------------------
    # Single-token step (compiled)
    # ------------------------------------------------------------------

    def _raw_step(
        self,
        new_token_ids: mx.array,                         # (1, 1)
        kv_store: list[tuple[mx.array, mx.array]],       # list[L] of (K, V) each (1, nkv, S, dh)
        seq_len: int,                                    # absolute position of the new token
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """
        Process one new token against stored K,V.

        For each layer:
          - K_old, V_old from kv_store[i] — no recomputation, just use them
          - Q_new, K_new, V_new from new token only (1-position matmuls)
          - Fused attention over [K_old; K_new], [V_old; V_new]
          - FFN on new token only
          - Append K_new, V_new to store

        This eliminates the O(S × hidden × head_dim) k_proj/v_proj matmuls
        from the RS generator's step loop. Only O(1 × hidden × head_dim) remains.
        """
        backbone = self.model.model
        B = 1

        h_new = backbone._embed(new_token_ids)  # (B, 1, hidden)
        new_kv_store: list[tuple[mx.array, mx.array]] = []

        for i, layer in enumerate(backbone.layers):
            k_old, v_old = kv_store[i]          # (B, nkv, S, head_dim)
            attn = layer.self_attn
            nq  = attn.num_heads
            nkv = attn.num_kv_heads
            dh  = attn.head_dim

            x_new = layer.input_layernorm(h_new)

            q_new = attn.q_proj(x_new).reshape(B, 1, nq,  dh).transpose(0, 2, 1, 3)
            k_new = attn.k_proj(x_new).reshape(B, 1, nkv, dh).transpose(0, 2, 1, 3)
            v_new = attn.v_proj(x_new).reshape(B, 1, nkv, dh).transpose(0, 2, 1, 3)

            q_new = attn.q_norm(q_new)
            k_new = attn.k_norm(k_new)

            # RoPE only on new token — k_old already has RoPE from prefill/prior steps
            q_new = attn.rope(q_new, offset=seq_len)
            k_new = attn.rope(k_new, offset=seq_len)

            k_all = mx.concatenate([k_old, k_new], axis=2)  # (B, nkv, S+1, dh)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            if attn.n_rep > 1:
                k_rpt = mx.repeat(k_all, attn.n_rep, axis=1)
                v_rpt = mx.repeat(v_all, attn.n_rep, axis=1)
            else:
                k_rpt, v_rpt = k_all, v_all

            # No mask needed: single query, causal is guaranteed by seq order
            attn_out = mx.fast.scaled_dot_product_attention(
                q_new, k_rpt, v_rpt, scale=attn.scale
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
            attn_out = attn.o_proj(attn_out)

            h_new = h_new + layer.post_attention_layernorm(attn_out)
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h_new))
            h_new   = h_new + layer.post_feedforward_layernorm(ffn_out)

            new_kv_store.append((k_all, v_all))

        h_final = backbone.norm(h_new)
        logits  = self.model._unembed(h_final)
        return logits, new_kv_store

    def step(
        self,
        new_token_ids: mx.array,
        kv_store: list[tuple[mx.array, mx.array]],
        seq_len: int,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Compiled single-token step. See _raw_step for details."""
        return self._step(new_token_ids, kv_store, seq_len)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def kv_bytes(self, seq_len: int) -> int:
        """
        Bytes used by K,V store for seq_len tokens.

        For Gemma 4B: 2 × 4 × 320 × seq_len × 34 × 2 bytes
                     = same as standard KV cache
                     = same as full-residual RS store (since hidden = 2×nkv×head_dim)
        """
        return (
            2                                       # K and V
            * self.config.num_key_value_heads       # nkv heads
            * self.config.head_dim                  # head_dim
            * seq_len
            * self.config.num_hidden_layers         # all layers
            * 2                                     # bfloat16
        )

    def residual_equivalent_bytes(self, seq_len: int) -> int:
        """Bytes that RS residual store would use for the same seq_len."""
        return (
            self.config.hidden_size
            * seq_len
            * self.config.num_hidden_layers
            * 2                                     # bfloat16
        )
