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
  layer          : int                          — retrieval layer (29)
  kv_head        : int                          — KV head index
  query_head     : int                          — query head (4)
  inject_layer   : int                          — injection layer (30)

Single forward pass per window:
  prefill_to_layer(target_layer=29) → h entering L30
  K-projection on h → routing K vectors
  dot(h[p], embed(T)) → injection coefficient c
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .._types import KVectorMode
from ._progress import phase_progress
from .....inference.context.vec_inject.providers import (
    VEC_INJECT_FILE,
    VecInjectMetaKey,
    VecInjectWindowKey,
)


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
    retrieval_layer: int = 29,
    query_head: int = 4,
    inject_layer: int | None = None,
    lib=None,
    tokenizer=None,
    kvector_mode: KVectorMode = KVectorMode.SPARSE,
) -> None:
    """Extract K-routing vectors and injection coefficients in one forward pass.

    Parameters
    ----------
    retrieval_layer : Layer for K extraction (default 29 for Gemma 4B).
    query_head      : Query head that does fact-copying (default 4).
    inject_layer    : Layer where injection is applied (default retrieval_layer+1).
    lib             : CheckpointLibrary — if provided, reads window tokens from it.
    tokenizer       : Optional tokenizer for token distinctiveness checks.
                      Facts with non-distinctive answer tokens (common 1-3 char
                      prefixes like " P", " St") are flagged in the index.
    kvector_mode    : Which positions to extract (sparse/interval/full).

    Output
    ------
    vec_inject.npz written to output_path.
    """
    kv_gen = engine.kv_gen
    backbone = kv_gen.backbone
    num_layers = len(backbone.adapted_layers)

    if retrieval_layer >= num_layers:
        retrieval_layer = num_layers - 1

    if inject_layer is None:
        inject_layer = min(retrieval_layer + 1, num_layers - 1)

    layer_adapter = backbone.adapted_layers[retrieval_layer]
    n_rep = layer_adapter.n_rep
    nkv = layer_adapter.num_kv_heads
    kv_head_idx = min(query_head // n_rep, nkv - 1) if n_rep > 1 else min(query_head, nkv - 1)
    head_dim = layer_adapter.head_dim

    fact_positions: dict[int, list[int]] | None = None
    if kvector_mode != KVectorMode.FULL:
        fact_positions = _load_fact_positions(output_path, num_archived)

    mode_label = kvector_mode.value
    phase_label = (
        f"vec_inject L{retrieval_layer}→{inject_layer} "
        f"H{query_head}→KV{kv_head_idx} [{mode_label}]"
    )
    result: dict[str, mx.array] = {}
    total_facts = 0

    t0 = time.monotonic()
    for wid in range(num_archived):
        if lib is not None:
            w_tokens = lib.get_window_tokens(wid)
        else:
            w_tokens, _ = engine.archive.retrieve(wid)
        w_ids = mx.array(w_tokens)[None]   # (1, S)
        B, S = w_ids.shape

        if kvector_mode == KVectorMode.FULL:
            positions = list(range(S))
        elif kvector_mode == KVectorMode.SPARSE and fact_positions and wid in fact_positions:
            positions = fact_positions[wid]
        else:
            n_samples = min(8, S)
            positions = [int(i * (S - 1) / max(n_samples - 1, 1)) for i in range(n_samples)]

        positions = [p for p in positions if 0 <= p < S]
        if not positions:
            continue

        # Single forward pass → h entering inject_layer
        h = kv_gen.prefill_to_layer(w_ids, target_layer=retrieval_layer)

        # K vectors at retrieval layer (for routing)
        x = layer_adapter.pre_attn_norm(h)
        _q, k, _v = layer_adapter.project_qkv(x, B, S, offset=0)
        k_head = k[:, kv_head_idx, :, :]   # (1, S, head_dim)
        k_at_pos = k_head[0, positions, :]  # (n_facts, head_dim)
        mx.eval(k_at_pos)

        # Injection coefficients: c = dot(h[p], embed(T))
        h_positions = h[0, positions, :]    # (n_facts, hidden_size)
        mx.eval(h_positions)
        h_np = np.array(h_positions.tolist(), dtype=np.float32)

        coefs_list: list[float] = []
        token_ids_list: list[int] = []
        distinctive_list: list[int] = []   # 1 = distinctive, 0 = needs fallback
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
        result[VecInjectWindowKey.k_vecs(wid)]       = k_np
        result[VecInjectWindowKey.token_ids(wid)]    = np.array(token_ids_list, dtype=np.int32)
        result[VecInjectWindowKey.coefs(wid)]        = np.array(coefs_list, dtype=np.float32)
        result[VecInjectWindowKey.positions(wid)]    = np.array(positions, dtype=np.int32)
        result[VecInjectWindowKey.distinctive(wid)]  = np.array(distinctive_list, dtype=np.int32)

        total_facts += len(positions)
        phase_progress(phase_label, wid + 1, num_archived, t0)

    print(file=sys.stderr)

    result[VecInjectMetaKey.LAYER]        = np.array(retrieval_layer)
    result[VecInjectMetaKey.KV_HEAD]      = np.array(kv_head_idx)
    result[VecInjectMetaKey.QUERY_HEAD]   = np.array(query_head)
    result[VecInjectMetaKey.INJECT_LAYER] = np.array(inject_layer)

    # np.savez handles arbitrary key counts; mx.load reads numpy npz format correctly
    np.savez(str(output_path / VEC_INJECT_FILE), **result)

    storage_bytes = total_facts * (head_dim * 2 + 4 + 4 + 4)
    print(
        f"  vec_inject [{mode_label}]: {total_facts} facts × "
        f"({head_dim}D k_vec + coef + tok_id) = "
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
