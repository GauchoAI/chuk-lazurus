"""Vector injection index extraction — per-fact c = dot(R_L30, embed(token)).

Experiment 2bd41b18: L29 H4 copies embed(answer_token) × scalar into the
residual stream.  The scalar is the injection coefficient c.  Stored per
fact position, (token_id, c) is 12 bytes — the complete information content
of a retrieved fact.

Output: vec_inject.npz in the checkpoint directory.

  w{N}/k_vecs    : (n_facts, head_dim) float16  — K vectors at L29 KV-head
  w{N}/token_ids : (n_facts,) int32             — answer token at each position
  w{N}/coefs     : (n_facts,) float32           — c = dot(R_L30, embed(token))
  w{N}/positions : (n_facts,) int32             — token position in window
  w{N}/h4_vecs   : (n_facts, hidden_size) float16 — H4 output for Stage-2 routing
  layer          : int                          — retrieval layer (29)
  kv_head        : int                          — KV head index
  query_head     : int                          — query head (4)
  inject_layer   : int                          — injection layer (30)

Two forward passes per window:
  Pass 1: prefill_to_layer(target_layer=29) → h entering L30
    K-projection on h → routing K vectors (Stage 1)
    dot(h[p], embed(T)) → injection coefficient c

  Pass 2: prefill_to_layer(target_layer=28) → h entering L29
    H4 causal attention + O_proj slice → H4 output vectors (Stage 2)
    Routing-wall-breakers experiment (2026-03-19): 4/4 at N=12 same-template,
    margins 2.4×–8.4×.  The copy head's output is the entity identity signal,
    isolated from the 7 structural heads that cause template crowding.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .....inference.context.vec_inject.providers import (
    VEC_INJECT_FILE,
    VecInjectMetaKey,
    VecInjectWindowKey,
)
from .._types import KVectorMode
from ._progress import phase_progress


def _is_distinctive_token(token_id: int, tokenizer) -> bool:
    """Return True if this token is distinctive enough for 1D injection.

    Tokens that are single common prefix characters (" P", " St", " V", etc.)
    fail because the model's prior for any word starting with that prefix
    overwhelms the small injected coefficient.

    A token is considered distinctive when its decoded string (stripped of
    leading whitespace) is at least 4 characters long.
    """
    if tokenizer is None:
        return True  # can't check — assume distinctive
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=True).lstrip()
        return len(text) >= 4
    except Exception:
        return True  # decode failed — assume distinctive


def extract_vec_inject_index(
    engine,
    output_path: Path,
    num_archived: int,
    config,
    retrieval_layer: int | None = None,
    query_head: int | None = None,
    inject_layer: int | None = None,
    lib=None,
    tokenizer=None,
    kvector_mode: KVectorMode = KVectorMode.SPARSE,
) -> None:
    """Extract K-routing vectors and injection coefficients in one forward pass.

    Parameters
    ----------
    retrieval_layer : Layer for K extraction. If None, resolved from ArchitectureConfig.
    query_head      : Query head that does fact-copying. If None, resolved from ArchitectureConfig.
    inject_layer    : Layer where injection is applied. If None, resolved from ArchitectureConfig.
    lib             : CheckpointLibrary — if provided, reads window tokens from it.
    tokenizer       : Optional tokenizer for token distinctiveness checks.
                      Facts with non-distinctive answer tokens (common 1-3 char
                      prefixes like " P", " St") are flagged in the index.
    kvector_mode    : Which positions to extract (sparse/interval/full).

    Output
    ------
    vec_inject.npz written to output_path.
    """
    from .....inference.context.arch_config import ArchitectureConfig

    kv_gen = engine.kv_gen
    backbone = kv_gen.backbone
    num_layers = len(backbone.adapted_layers)

    # Resolve retrieval_layer and query_head from arch config when not specified.
    # inject_layer is always resolved separately as retrieval_layer + 1 (see below).
    if retrieval_layer is None or query_head is None:
        ac = ArchitectureConfig.from_model_config(config)  # raises if not calibrated
        retrieval_layer = retrieval_layer if retrieval_layer is not None else ac.retrieval_layer
        query_head = query_head if query_head is not None else ac.query_head

    if retrieval_layer >= num_layers:
        retrieval_layer = num_layers - 1

    # inject_layer defaults to retrieval_layer + 1 when not specified.
    # This preserves "override retrieval_layer → inject_layer follows" semantics.
    if inject_layer is None:
        inject_layer = min(retrieval_layer + 1, num_layers - 1)

    layer_adapter = backbone.adapted_layers[retrieval_layer]
    n_rep = layer_adapter.n_rep
    nkv = layer_adapter.num_kv_heads
    kv_head_idx = min(query_head // n_rep, nkv - 1) if n_rep > 1 else min(query_head, nkv - 1)
    head_dim = layer_adapter.head_dim

    # SPARSE mode uses sparse_index.json positions (old keyword-based routing).
    # INTERVAL and FULL modes do not consult the sparse index.
    fact_positions: dict[int, list[int]] | None = None
    if kvector_mode == KVectorMode.SPARSE:
        fact_positions = _load_fact_positions(output_path, num_archived)

    mode_label = kvector_mode.value
    phase_label = (
        f"vec_inject L{retrieval_layer}→{inject_layer} H{query_head}→KV{kv_head_idx} [{mode_label}]"
    )
    result: dict[str, mx.array] = {}
    total_facts = 0

    t0 = time.monotonic()
    for wid in range(num_archived):
        if lib is not None:
            w_tokens = lib.get_window_tokens(wid)
        else:
            w_tokens, _ = engine.archive.retrieve(wid)
        w_ids = mx.array(w_tokens)[None]  # (1, S)
        B, S = w_ids.shape

        if kvector_mode == KVectorMode.FULL:
            positions = list(range(S))
        elif kvector_mode == KVectorMode.SPARSE and fact_positions and wid in fact_positions:
            positions = fact_positions[wid]
        else:
            # INTERVAL: 8 evenly-spaced positions, independent of sparse index
            n_samples = min(32, S)
            positions = [int(i * (S - 1) / max(n_samples - 1, 1)) for i in range(n_samples)]

        positions = [p for p in positions if 0 <= p < S]
        if not positions:
            continue

        # Single forward pass → h entering inject_layer
        h = kv_gen.prefill_to_layer(w_ids, target_layer=retrieval_layer)

        # K vectors at retrieval layer (for routing)
        x = layer_adapter.pre_attn_norm(h)
        _q, k, _v = layer_adapter.project_qkv(x, B, S, offset=0)
        k_head = k[:, kv_head_idx, :, :]  # (1, S, head_dim)
        k_at_pos = k_head[0, positions, :]  # (n_facts, head_dim)
        mx.eval(k_at_pos)

        # Injection coefficients: c = dot(h[p], embed(T))
        h_positions = h[0, positions, :]  # (n_facts, hidden_size)
        mx.eval(h_positions)
        h_np = np.array(h_positions.tolist(), dtype=np.float32)

        coefs_list: list[float] = []
        token_ids_list: list[int] = []
        distinctive_list: list[int] = []  # 1 = distinctive, 0 = needs fallback
        for i, pos in enumerate(positions):
            tok = w_tokens[pos]
            token_ids_list.append(tok)
            e_tok = backbone.embed(mx.array([[tok]]))[0, 0, :].astype(mx.float32)
            mx.eval(e_tok)
            e_np = np.array(e_tok.tolist(), dtype=np.float32)
            coefs_list.append(float(np.dot(h_np[i], e_np)))
            distinctive_list.append(1 if _is_distinctive_token(tok, tokenizer) else 0)

        k_np = np.array(k_at_pos.tolist(), dtype=np.float16)
        # Store as numpy arrays — np.savez has no keyword-argument limit
        # (mx.savez uses nanobind which caps at 1024; 725 windows × 5 = 3625 keys)
        result[VecInjectWindowKey.k_vecs(wid)] = k_np
        result[VecInjectWindowKey.token_ids(wid)] = np.array(token_ids_list, dtype=np.int32)
        result[VecInjectWindowKey.coefs(wid)] = np.array(coefs_list, dtype=np.float32)
        result[VecInjectWindowKey.positions(wid)] = np.array(positions, dtype=np.int32)
        result[VecInjectWindowKey.distinctive(wid)] = np.array(distinctive_list, dtype=np.int32)

        # ── Pass 2: H4 output vectors for Stage-2 routing ────────────
        # Requires h entering L29 (output of L28), not L29's output.
        if retrieval_layer > 0:
            h_pre = kv_gen.prefill_to_layer(w_ids, target_layer=retrieval_layer - 1)
            x_pre = layer_adapter.pre_attn_norm(h_pre)
            _q_h4, k_pre, v_pre = layer_adapter.project_qkv(x_pre, B, S, offset=0)

            kv_h = kv_head_idx
            dh = head_dim
            mask = backbone.prefill_mask(retrieval_layer, h_pre)

            # Causal SDPA over H4 only — (1, 1, S, dh)
            attn_h4 = mx.fast.scaled_dot_product_attention(
                _q_h4[:, query_head : query_head + 1, :, :],
                k_pre[:, kv_h : kv_h + 1, :, :],
                v_pre[:, kv_h : kv_h + 1, :, :],
                scale=layer_adapter.attn_scale,
                mask=mask,
            )
            h4_all = attn_h4[0, 0, :, :]  # (S, dh)

            # Project H4's slice through O_proj → hidden space (S, D)
            o_weight = layer_adapter._block.self_attn.o_proj.weight  # (D, nq*dh)
            h4_contrib = h4_all @ o_weight[:, query_head * dh : (query_head + 1) * dh].T

            h4_at_pos = h4_contrib[positions, :]  # (n_facts, D)
            mx.eval(h4_at_pos)
            h4_np = np.array(h4_at_pos.astype(mx.float16).tolist(), dtype=np.float16)
            result[VecInjectWindowKey.h4_vecs(wid)] = h4_np

        total_facts += len(positions)
        phase_progress(phase_label, wid + 1, num_archived, t0)

    print(file=sys.stderr)

    result[VecInjectMetaKey.LAYER] = np.array(retrieval_layer)
    result[VecInjectMetaKey.KV_HEAD] = np.array(kv_head_idx)
    result[VecInjectMetaKey.QUERY_HEAD] = np.array(query_head)
    result[VecInjectMetaKey.INJECT_LAYER] = np.array(inject_layer)

    # np.savez handles arbitrary key counts; mx.load reads numpy npz format correctly
    np.savez(str(output_path / VEC_INJECT_FILE), **result)

    hidden_size = backbone.hidden_size
    has_h4 = retrieval_layer > 0
    k_bytes = head_dim * 2          # float16
    h4_bytes = hidden_size * 2 if has_h4 else 0  # float16
    storage_bytes = total_facts * (k_bytes + 4 + 4 + 4 + h4_bytes)
    h4_note = f" + {hidden_size}D h4_vec" if has_h4 else ""
    print(
        f"  vec_inject [{mode_label}]: {total_facts} facts × "
        f"({head_dim}D k_vec + coef + tok_id{h4_note}) = "
        f"{storage_bytes / 1024:.1f} KB "
        f"(L{retrieval_layer}→{inject_layer} KV-H{kv_head_idx})",
        file=sys.stderr,
        flush=True,
    )


def _load_fact_positions(output_path: Path, num_archived: int) -> dict[int, list[int]] | None:
    """Load fact positions from sparse index if available."""
    sparse_path = output_path / "sparse_index.json"
    if not sparse_path.exists():
        return None

    import json

    raw = json.loads(sparse_path.read_text())
    entries = raw.get("entries", [])

    positions: dict[int, list[int]] = {}
    for entry in entries:
        wid = entry.get("window_id")
        if wid is None or wid >= num_archived:
            continue
        spans = entry.get("fact_spans", [])
        if spans:
            pos_list = [s["position"] for s in spans if "position" in s]
            if pos_list:
                positions[wid] = pos_list

    return positions if positions else None
