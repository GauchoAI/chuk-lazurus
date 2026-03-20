"""K-vector routing index extraction — L29 H4 K vectors at fact positions.

Extracts K vectors from the retrieval head (L29 KV-head-2 for Gemma 4B)
at positions in each window. These K vectors are the model's own
addressing mechanism — the same vectors that L29 H4 uses for Q·K matching
during attention.

Three coverage modes (--phases kvectors vs kvectors_full):
  sparse   — fact positions from sparse_index.json (surprise-guided)
  interval — 8 evenly-spaced samples per window (1.6% coverage, fallback)
  full     — every position, 100% coverage (~256KB/window, ~181MB for Apollo 11)

Storage: 256D bf16 per position = 512 bytes per fact.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .._types import KVectorMode
from ._progress import phase_progress


def extract_kv_route_index(
    engine,
    output_path: Path,
    num_archived: int,
    config,
    retrieval_layer: int | None = None,
    query_head: int | None = None,
    lib=None,
    kvector_mode: KVectorMode = KVectorMode.SPARSE,
) -> None:
    """Extract K vectors at the retrieval head for fact positions.

    Parameters
    ----------
    retrieval_layer : Layer for K extraction. If None, resolved from ArchitectureConfig.
    query_head : Query head that does the fact copying. If None, resolved from ArchitectureConfig.
                 Maps to KV head via query_head // n_rep.
    lib : CheckpointLibrary — if provided, reads tokens from it instead of engine.archive.

    Saves kv_route_index.npz with:
        w{N}: (n_facts, head_dim) K vectors for window N
        layer: int
        kv_head: int
    """
    import mlx.core as mx

    from .....inference.context.arch_config import ArchitectureConfig

    kv_gen = engine.kv_gen
    backbone = kv_gen.backbone
    num_layers = len(backbone.adapted_layers)

    # Resolve arch config if any param is unspecified
    if retrieval_layer is None or query_head is None:
        ac = ArchitectureConfig.from_model_config(config)  # raises if not calibrated
        retrieval_layer = retrieval_layer if retrieval_layer is not None else ac.retrieval_layer
        query_head = query_head if query_head is not None else ac.query_head

    if retrieval_layer >= num_layers:
        retrieval_layer = num_layers - 1

    layer_adapter = backbone.adapted_layers[retrieval_layer]
    n_rep = layer_adapter.n_rep
    nkv = layer_adapter.num_kv_heads
    kv_head_idx = min(query_head // n_rep, nkv - 1) if n_rep > 1 else min(query_head, nkv - 1)
    head_dim = layer_adapter.head_dim

    # Try to load fact positions from sparse index (skip for full mode)
    fact_positions = None
    if kvector_mode != KVectorMode.FULL:
        fact_positions = _load_fact_positions(output_path, num_archived)

    mode_label = kvector_mode.value
    phase_label = f"kv_route L{retrieval_layer} H{query_head}→KV{kv_head_idx} [{mode_label}]"
    kv_dict: dict[str, mx.array] = {}
    total_facts = 0

    t0 = time.monotonic()
    for wid in range(num_archived):
        if lib is not None:
            w_tokens = lib.get_window_tokens(wid)
        else:
            w_tokens, _ = engine.archive.retrieve(wid)
        w_ids = mx.array(w_tokens)[None]
        B, S = w_ids.shape

        # Determine positions to extract
        if kvector_mode == KVectorMode.FULL:
            positions = list(range(S))
        elif kvector_mode == KVectorMode.SPARSE and fact_positions and wid in fact_positions:
            positions = fact_positions[wid]
        else:
            # Fallback: interval sampling (32 positions)
            n_samples = min(32, S)
            positions = [int(i * (S - 1) / max(n_samples - 1, 1)) for i in range(n_samples)]

        # Clamp positions to valid range
        positions = [p for p in positions if 0 <= p < S]
        if not positions:
            continue

        # Forward to retrieval layer and extract K at the specific head
        h = kv_gen.prefill_to_layer(w_ids, target_layer=retrieval_layer)
        # h shape: (1, S, hidden_size)

        # Project through attention at the retrieval layer
        x = layer_adapter.pre_attn_norm(h)
        _q, k, _v = layer_adapter.project_qkv(x, B, S, offset=0)
        # k shape: (1, nkv, S, head_dim) — post-norm, post-RoPE

        # Extract K at the specific KV head and positions
        k_head = k[:, kv_head_idx, :, :]  # (1, S, head_dim)
        k_at_positions = k_head[0, positions, :]  # (n_facts, head_dim)
        mx.eval(k_at_positions)

        kv_dict[f"w{wid}"] = k_at_positions
        total_facts += len(positions)
        phase_progress(phase_label, wid + 1, num_archived, t0)

    print(file=sys.stderr)

    # Add metadata
    kv_dict["layer"] = mx.array(retrieval_layer)
    kv_dict["kv_head"] = mx.array(kv_head_idx)
    kv_dict["query_head"] = mx.array(query_head)

    mx.savez(str(output_path / "kv_route_index.npz"), **kv_dict)

    # Storage report
    storage_bytes = total_facts * head_dim * 2  # bf16
    print(
        f"  kv_route index [{mode_label}]: {total_facts} positions × {head_dim}D = "
        f"{storage_bytes / 1024:.1f} KB "
        f"(L{retrieval_layer} KV-head-{kv_head_idx})",
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
