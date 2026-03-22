"""Persistent 1D subspace injection at crystal_layer.

The injection formula adds the entity direction to the residual stream:

    h += coefficient * embed(token_id) / ||embed(token_id)||^2

This is applied at crystal_layer (default L30) during each generation step.
The coefficient is stored at 2x natural (phase transition at 1.5x).

The injection is self-terminating: after the entity tokens are in the model's
KV cache, agreement gating detects that the model predicts the same token
with and without injection, and stops injecting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from ._sampling import sample_token

if TYPE_CHECKING:
    from .config import ArchitectureConfig
    from .store import InjectionEntry

# Dtype for attention masks — must match kv_generator.py
_MASK_DTYPE = mx.bfloat16


# ── Core injection primitive ─────────────────────────────────────────


def inject_1d(
    residual: mx.array,
    token_id: int,
    coefficient: float,
    embed_matrix: mx.array,
) -> mx.array:
    """Add the 1D component along a token embedding direction.

    h += coefficient * embed(token_id) / ||embed(token_id)||^2

    Parameters
    ----------
    residual     : (..., hidden_dim) hidden state to modify.
    token_id     : Target token whose embedding direction is used.
    coefficient  : Injection magnitude (stored at 2x natural).
    embed_matrix : (vocab_size, hidden_dim) token embedding weights.

    Returns
    -------
    Modified residual with injection applied.
    """
    embed = embed_matrix[token_id]  # (hidden_dim,)
    embed_norm_sq = (embed * embed).sum()  # scalar
    direction = embed / embed_norm_sq  # e / ||e||^2
    return residual + coefficient * direction


# ── Step with injection ──────────────────────────────────────────────


def _step_with_injection(
    kv_gen,
    new_token_ids: mx.array,  # (1, 1)
    kv_store: list[tuple[mx.array, mx.array]],
    seq_len: int,
    inject_token_id: int,
    inject_coeff: float,
    crystal_layer: int,
) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
    """Single-token step with 1D injection at crystal_layer.

    Mirrors kv_gen._raw_step() but injects before crystal_layer processes.
    The injection modifies the hidden state, which then flows through
    crystal_layer and all subsequent layers — their K,V entries encode
    the injected entity direction. This is the persistent injection mechanism.
    """
    backbone = kv_gen.backbone
    embed_matrix = backbone.embed_matrix
    B = 1

    h = backbone.embed(new_token_ids)
    new_kv_store: list[tuple[mx.array, mx.array]] = []

    sw = backbone.sliding_window

    for i, layer in enumerate(backbone.adapted_layers):
        # Inject BEFORE crystal_layer processes the residual
        if i == crystal_layer:
            h = inject_1d(h, inject_token_id, inject_coeff, embed_matrix)

        k_old, v_old = kv_store[i]

        x = layer.pre_attn_norm(h)
        q, k_new, v_new = layer.project_qkv(x, B, 1, offset=seq_len)

        k_all = mx.concatenate([k_old, k_new], axis=2)
        v_all = mx.concatenate([v_old, v_new], axis=2)

        k_rpt = mx.repeat(k_all, layer.n_rep, axis=1) if layer.n_rep > 1 else k_all
        v_rpt = mx.repeat(v_all, layer.n_rep, axis=1) if layer.n_rep > 1 else v_all

        # Sliding-window mask for non-global layers
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


# ── Generate with persistent injection ───────────────────────────────


def generate_with_injection(
    kv_gen,
    prompt_ids: mx.array,  # (1, S)
    entries: list[InjectionEntry],
    config: ArchitectureConfig,
    max_tokens: int = 80,
    temperature: float = 0.0,
    stop_ids: set[int] | None = None,
) -> list[int]:
    """Autoregressive generation with persistent 1D injection.

    Generation proceeds in two phases:

    Phase 1 — Injection (len(entries) steps):
      At each step, the current entry's token direction is injected at
      crystal_layer. The injection biases the model toward the target
      token. After all entries are consumed, generation switches to
      Phase 2.

    Phase 2 — Free generation (remaining steps):
      Standard autoregressive decode without injection. The entity
      tokens are already in the KV cache from Phase 1.

    The first token uses two forward passes:
      Pass 1: prefill_to_layer + inject + prefill_from_layer → biased logits
      Pass 2: full prefill → clean KV cache for continuation

    Parameters
    ----------
    kv_gen      : KVDirectGenerator instance.
    prompt_ids  : (1, S) chat-templated prompt token IDs.
    entries     : Injection entries for the target fact, sorted by position.
    config      : ArchitectureConfig with crystal_layer.
    max_tokens  : Maximum tokens to generate.
    temperature : Sampling temperature. 0.0 = greedy.
    stop_ids    : Token IDs that terminate generation.

    Returns
    -------
    List of generated token IDs.
    """
    stop_ids = stop_ids or set()
    crystal_layer = config.crystal_layer
    embed_matrix = kv_gen.backbone.embed_matrix

    if not entries:
        # No entries — plain generation
        logits, kv_store = kv_gen.prefill(prompt_ids)
        mx.eval(logits)
        return _generate_plain_loop(kv_gen, logits, kv_store, prompt_ids.shape[1],
                                    max_tokens, temperature, stop_ids)

    # Sort entries by position for sequential injection
    sorted_entries = sorted(entries, key=lambda e: (e.fact_id, e.position_in_window))

    # ── First token: two-pass injection ──────────────────────────────
    first_entry = sorted_entries[0]

    # Pass 1: injection-biased first-token logits
    h = kv_gen.prefill_to_layer(prompt_ids, target_layer=crystal_layer - 1)
    # Inject at last position only
    h_last = h[:, -1:, :]
    h_last = inject_1d(h_last, first_entry.token_id, first_entry.coefficient, embed_matrix)
    h_injected = mx.concatenate([h[:, :-1, :], h_last], axis=1)
    inject_logits, _ = kv_gen.prefill_from_layer(h_injected, start_layer=crystal_layer)
    mx.eval(inject_logits)

    # Pass 2: full prefill for clean KV cache
    logits_full, kv_store = kv_gen.prefill(prompt_ids)
    mx.eval(logits_full)
    seq_len = prompt_ids.shape[1]

    # Sample first token from injection-biased logits
    first_logits = inject_logits[0, -1]
    first_token = sample_token(first_logits, temperature)
    if first_token in stop_ids:
        return [first_token]

    generated: list[int] = [first_token]

    # ── Phase 1: consecutive injection steps (no gap tokens) ─────────
    # Each injection step processes the PREVIOUS generated token with
    # the NEXT entry's injection. No intermediate sampling — the entries
    # fire at consecutive generation steps.
    prev_token = first_token
    for entry in sorted_entries[1:]:
        if len(generated) >= max_tokens:
            break

        # Process previous token with THIS entry's injection
        logits, kv_store = _step_with_injection(
            kv_gen,
            mx.array([[prev_token]]),
            kv_store,
            seq_len,
            inject_token_id=entry.token_id,
            inject_coeff=entry.coefficient,
            crystal_layer=crystal_layer,
        )
        seq_len += 1

        # Sample from injection-biased logits
        token = sample_token(logits[0, -1], temperature)
        if token in stop_ids:
            break
        generated.append(token)
        prev_token = token

    # ── Transition: one clean step to flush injection bias ───────────
    # The last step_with_injection left injection-biased logits.
    # Process the last generated token through clean layers (no injection)
    # so Phase 2 starts from un-biased predictions.
    if len(sorted_entries) == 1:
        # Only one entry — extend first token into clean KV
        logits, kv_store = kv_gen.extend(
            mx.array([[first_token]]), kv_store, abs_start=seq_len
        )
        seq_len += 1
    else:
        # Process last injected token cleanly to get un-biased logits
        logits, kv_store = kv_gen.step_uncompiled(
            mx.array([[prev_token]]), kv_store, seq_len=seq_len
        )
        seq_len += 1

    # ── Phase 2: free generation ─────────────────────────────────────
    for _ in range(max_tokens - len(generated)):
        last_logits = logits[0, -1]
        token = sample_token(last_logits, temperature)
        if token in stop_ids:
            break
        generated.append(token)
        logits, kv_store = kv_gen.step_uncompiled(
            mx.array([[token]]), kv_store, seq_len=seq_len
        )
        seq_len += 1

    return generated


def _generate_plain_loop(
    kv_gen,
    logits: mx.array,
    kv_store: list[tuple[mx.array, mx.array]],
    seq_len: int,
    max_tokens: int,
    temperature: float,
    stop_ids: set[int],
) -> list[int]:
    """Plain autoregressive generation loop (no injection)."""
    generated: list[int] = []
    for _ in range(max_tokens):
        last_logits = logits[0, -1]
        token = sample_token(last_logits, temperature)
        if token in stop_ids:
            break
        generated.append(token)
        logits, kv_store = kv_gen.step_uncompiled(
            mx.array([[token]]), kv_store, seq_len=seq_len
        )
        seq_len += 1
    return generated
