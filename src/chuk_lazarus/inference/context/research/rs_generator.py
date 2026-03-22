"""
Compiled residual-stream generator — incremental hot path.

Three operations:

  prefill(input_ids)
    Full forward pass from token IDs. Stores per-layer pre-attention residuals.
    O(N²) attention. Run once at the start of a conversation or after eviction.

  extend(new_token_ids, stored, abs_start)
    Process N new tokens against existing stored residuals in one batched pass.
    Causal mask lets new tokens attend to all stored + causally to each other.
    O(N × (S+N)) attention. Run for each new user message when hot state exists.

  step(new_token_id, stored, seq_len)   [compiled]
    Process one new token. O(S) matmuls for K,V from stored residuals.
    Run for each generated token.

Lifecycle (incremental — no re-prefill between turns):

  Turn 1 : prefill(130 tokens)  → stored shape (1, 130, hidden) per layer
           generate 40 tokens via step() → stored grows to (1, 170, hidden)

  Turn 2 : extend(80 new tokens, stored, abs_start=170)
           → stored grows to (1, 250, hidden)
           generate 40 tokens → stored (1, 290, hidden)

  Turn 4 : budget hit — slide stored by evicting first K tokens (simple slice)
           extend(80 new tokens) → stored stays near budget

RoPE correctness:
  stored_residuals[i] contains pre-layer-i residuals for positions
  [window_start, window_start + stored_len).
  k_old rope offset = seq_len - S_old  (= window_start when S_old = seq_len - window_start)
  This is computed inside _raw_step from the passed seq_len and S_old shape.

Storage correctness:
  prefill stores layer_inputs[i]  = residual ENTERING layer i (pre-norm, pre-attention).
  step stores h_pre (= h_new before layer i runs)  — same convention.
  extend stores h_new_pre_layer[i] — same convention.
  All three are consistent: layernorm(stored[i]) gives the correct K,V input.
"""

from __future__ import annotations

import mlx.core as mx


class CompiledRSGenerator:
    """
    Incremental residual-stream generator.

    All methods assume batch_size=1:
        input_ids          : (1, S) int32
        stored_residuals   : list[num_layers] of (1, S, hidden_size)
        logits             : (1, S, vocab_size)

    Usage
    -----
        gen = CompiledRSGenerator(rs_model, config)

        # Turn 1 — full prefill
        logits, stored = gen.prefill(input_ids)   # input_ids: (1, S)
        seq_len = input_ids.shape[1]

        # Generate
        for _ in range(max_new_tokens):
            next_tok = mx.argmax(logits[0, -1, :])
            logits, stored = gen.step(mx.array([[next_tok]]), stored, seq_len)
            seq_len += 1

        # Turn 2 — incremental extend (no re-prefill)
        abs_start = seq_len          # absolute position of first new token
        logits, stored = gen.extend(new_input_ids, stored, abs_start)
        seq_len += new_input_ids.shape[1]

        # Eviction (when budget hit)
        evict_n = 40
        stored = gen.slide(stored, evict_n)
        # abs_start for subsequent extend calls = seq_len (unchanged — absolute positions)
    """

    def __init__(self, rs_model, config):
        self.model = rs_model
        self.config = config
        self._step = mx.compile(self._raw_step, shapeless=True)

    # ------------------------------------------------------------------
    # Prefill — full forward from token IDs
    # ------------------------------------------------------------------

    def prefill(self, input_ids: mx.array) -> tuple[mx.array, list[mx.array]]:
        """
        Full forward pass. Returns (logits, per_layer_residuals).

        stored[i] is the residual ENTERING layer i for all input positions.
        This is the pre-norm, pre-attention residual — the Markov state for K,V.
        """
        backbone = self.model.model

        h = backbone._embed(input_ids)
        layer_inputs: list[mx.array] = []

        for i, layer in enumerate(backbone.layers):
            layer_inputs.append(h)  # pre-layer-i residual
            mask = backbone._mask_for_layer(i, h)
            h = layer(h, mask=mask)

        h = backbone.norm(h)
        logits = self.model._unembed(h)
        mx.eval(logits, *layer_inputs)
        return logits, layer_inputs

    # ------------------------------------------------------------------
    # Extend — N new tokens batched against stored residuals
    # ------------------------------------------------------------------

    def extend(
        self,
        new_token_ids: mx.array,  # (1, N)
        stored_residuals: list[mx.array],  # list[L] of (1, S, hidden)
        abs_start: int,  # absolute position of first NEW token
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Process N new tokens against existing stored residuals in one pass.

        New tokens attend to:
          - All S stored positions (no mask — all visible)
          - Each other causally (token i sees tokens 0..i among the N new ones)

        Returns (logits_for_all_N_tokens, extended_stored_residuals).
        extended_stored[i] has shape (1, S+N, hidden).
        """
        backbone = self.model.model
        B, N = new_token_ids.shape
        S = stored_residuals[0].shape[1]

        h = backbone._embed(new_token_ids)  # (B, N, hidden)

        # Causal mask: (1, 1, N, S+N)
        # New token i can attend to all S stored + new tokens 0..i
        neg_inf = mx.full((N, N), -1e9, dtype=mx.bfloat16)
        causal_new = mx.triu(neg_inf, k=1)  # (N, N) upper tri = future
        stored_vis = mx.zeros((N, S), dtype=mx.bfloat16)  # (N, S) all visible
        mask = mx.concatenate([stored_vis, causal_new], axis=-1)[None, None]  # (1,1,N,S+N)

        new_pre: list[mx.array] = []  # pre-layer-i residuals for new tokens

        for i, layer in enumerate(backbone.layers):
            attn = layer.self_attn
            h_old = stored_residuals[i]  # (B, S, hidden)

            new_pre.append(h)  # save pre-layer-i h (correct residual to store)

            x_old = layer.input_layernorm(h_old)  # (B, S, hidden)
            x_new = layer.input_layernorm(h)  # (B, N, hidden)

            nq = attn.num_heads
            nkv = attn.num_kv_heads
            dh = attn.head_dim

            k_old = attn.k_proj(x_old).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
            v_old = attn.v_proj(x_old).reshape(B, S, nkv, dh).transpose(0, 2, 1, 3)
            q_new = attn.q_proj(x_new).reshape(B, N, nq, dh).transpose(0, 2, 1, 3)
            k_new = attn.k_proj(x_new).reshape(B, N, nkv, dh).transpose(0, 2, 1, 3)
            v_new = attn.v_proj(x_new).reshape(B, N, nkv, dh).transpose(0, 2, 1, 3)

            q_new = attn.q_norm(q_new)
            k_old = attn.k_norm(k_old)
            k_new = attn.k_norm(k_new)

            # RoPE: old tokens at abs positions (abs_start-S)..(abs_start-1)
            #        new tokens at abs positions abs_start..(abs_start+N-1)
            k_old = attn.rope(k_old, offset=abs_start - S)
            q_new = attn.rope(q_new, offset=abs_start)
            k_new = attn.rope(k_new, offset=abs_start)

            k_all = mx.concatenate([k_old, k_new], axis=2)  # (B, nkv, S+N, dh)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            if attn.n_rep > 1:
                k_all = mx.repeat(k_all, attn.n_rep, axis=1)
                v_all = mx.repeat(v_all, attn.n_rep, axis=1)

            attn_out = mx.fast.scaled_dot_product_attention(
                q_new, k_all, v_all, scale=attn.scale, mask=mask
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, -1)
            attn_out = attn.o_proj(attn_out)

            h = h + layer.post_attention_layernorm(attn_out)
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            h = h + layer.post_feedforward_layernorm(ffn_out)

        h_norm = backbone.norm(h)
        logits = self.model._unembed(h_norm)  # (B, N, vocab)

        # Extend stored with pre-layer residuals of new tokens
        extended = [
            mx.concatenate([stored_residuals[j], new_pre[j]], axis=1)
            for j in range(len(backbone.layers))
        ]

        mx.eval(logits, *extended)
        return logits, extended

    # ------------------------------------------------------------------
    # Slide — evict oldest tokens from stored residuals
    # ------------------------------------------------------------------

    @staticmethod
    def slide(
        stored_residuals: list[mx.array],
        evict_count: int,
    ) -> list[mx.array]:
        """
        Drop the first evict_count tokens from every layer's stored residuals.
        O(1) — just a slice, no recomputation.
        """
        return [s[:, evict_count:, :] for s in stored_residuals]

    # ------------------------------------------------------------------
    # Single-token step (compiled)
    # ------------------------------------------------------------------

    def _raw_step(
        self,
        new_token_ids: mx.array,  # (1, 1)
        stored_residuals: list[mx.array],  # list[L] of (1, S, hidden)
        seq_len: int,  # absolute position of the new token
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Process one new token against stored residuals.

        For each layer:
          - Save h_pre (pre-layer residual of new token) — this is what we store
          - K_old, V_old from stored_residuals[i] with correct RoPE offset
          - Q, K_new, V_new from new token
          - Fused attention; FFN on new token only
          - Append h_pre to stored block (not h_post — matches prefill convention)
        """
        backbone = self.model.model

        h_new = backbone._embed(new_token_ids)  # (1, 1, hidden)
        new_layer_inputs: list[mx.array] = []

        for i, layer in enumerate(backbone.layers):
            attn = layer.self_attn
            h_old = stored_residuals[i]  # (1, S, hidden)

            h_pre = h_new  # pre-layer-i residual for new token (store this)

            x_old = layer.input_layernorm(h_old)
            x_new = layer.input_layernorm(h_new)

            B, S_old, _ = h_old.shape
            nq = attn.num_heads
            nkv = attn.num_kv_heads
            dh = attn.head_dim

            k_old = attn.k_proj(x_old).reshape(B, S_old, nkv, dh).transpose(0, 2, 1, 3)
            v_old = attn.v_proj(x_old).reshape(B, S_old, nkv, dh).transpose(0, 2, 1, 3)
            q_new = attn.q_proj(x_new).reshape(B, 1, nq, dh).transpose(0, 2, 1, 3)
            k_new = attn.k_proj(x_new).reshape(B, 1, nkv, dh).transpose(0, 2, 1, 3)
            v_new = attn.v_proj(x_new).reshape(B, 1, nkv, dh).transpose(0, 2, 1, 3)

            q_new = attn.q_norm(q_new)
            k_old = attn.k_norm(k_old)
            k_new = attn.k_norm(k_new)

            # RoPE: old tokens at positions (seq_len - S_old)..(seq_len-1)
            #        new token at position seq_len
            k_old = attn.rope(k_old, offset=seq_len - S_old)
            q_new = attn.rope(q_new, offset=seq_len)
            k_new = attn.rope(k_new, offset=seq_len)

            k_all = mx.concatenate([k_old, k_new], axis=2)
            v_all = mx.concatenate([v_old, v_new], axis=2)

            if attn.n_rep > 1:
                k_all = mx.repeat(k_all, attn.n_rep, axis=1)
                v_all = mx.repeat(v_all, attn.n_rep, axis=1)

            attn_out = mx.fast.scaled_dot_product_attention(q_new, k_all, v_all, scale=attn.scale)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
            attn_out = attn.o_proj(attn_out)

            h_new = h_new + layer.post_attention_layernorm(attn_out)
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h_new))
            h_new = h_new + layer.post_feedforward_layernorm(ffn_out)

            # Store pre-layer-i residual (h_pre), not post-layer-i (h_new)
            new_layer_inputs.append(mx.concatenate([stored_residuals[i], h_pre], axis=1))

        h_final = backbone.norm(h_new)
        logits = self.model._unembed(h_final)
        return logits, new_layer_inputs

    def step(
        self,
        new_token_ids: mx.array,
        stored_residuals: list[mx.array],
        seq_len: int,
    ) -> tuple[mx.array, list[mx.array]]:
        """Compiled single-token step. See _raw_step for details."""
        return self._step(new_token_ids, stored_residuals, seq_len)

    def step_uncompiled(
        self,
        new_token_ids: mx.array,
        stored_residuals: list[mx.array],
        seq_len: int,
    ) -> tuple[mx.array, list[mx.array]]:
        """Uncompiled single-token step. Mirrors KVDirectGenerator.step_uncompiled."""
        return self._raw_step(new_token_ids, stored_residuals, seq_len)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def residual_bytes(self, seq_len: int) -> int:
        """Bytes used by per-layer residuals for seq_len tokens."""
        return self.config.num_hidden_layers * seq_len * self.config.hidden_size * 2

    def kv_equivalent_bytes(self, seq_len: int) -> int:
        """KV cache bytes that would be used for the same seq_len."""
        return (
            2
            * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * seq_len
            * self.config.head_dim
            * 2
        )
