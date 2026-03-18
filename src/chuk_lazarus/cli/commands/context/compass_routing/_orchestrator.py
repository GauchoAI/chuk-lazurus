"""Main compass_route() entry point — dispatches to scoring functions."""

from __future__ import annotations

import sys
import time

from ._strategy import RoutingStrategy
from ._bm25 import _bm25_score_windows
from ._geometric import (
    _compass_score_windows,
    _contrastive_score_windows,
    _deflection_score_windows,
    _directed_score_windows,
)
from ._model_based import (
    _PREVIEW_TOKENS,
    _attention_score_windows,
    _preview_score_windows,
    _qk_score_windows,
)
from ._composite import _darkspace_score_windows, _guided_score_windows
from ._legacy import _residual_cosine_score_windows
from ._kv_route import _kv_route_score_windows
from ._sparse import _sparse_score_windows
from ._temporal import _temporal_stride_windows


def compass_route(
    lib,
    kv_gen,
    prompt_ids: list[int],
    prompt_text: str,
    tokenizer,
    model_config=None,
    strategy: RoutingStrategy = RoutingStrategy.BM25,
    top_k: int = 3,
    bm25_shortlist: int = 10,
    exclude: set[int] | None = None,
    query_residual: "mx.array | None" = None,
    routing_layer: int = 29,
    routing_head: int = 4,
) -> list[int]:
    """Route a query to the most relevant windows.

    Parameters
    ----------
    lib             : CheckpointLibrary
    kv_gen          : KVDirectGenerator
    prompt_ids      : Encoded prompt token IDs
    prompt_text     : Raw prompt text (for BM25 tokenization)
    tokenizer       : Tokenizer (for decoding window tokens in BM25)
    model_config    : Model config (unused currently, reserved)
    strategy        : Which routing strategy to use
    top_k           : Number of windows to select
    bm25_shortlist  : Number of BM25 candidates for hybrid re-ranking
    exclude         : Window IDs to exclude from selection (already visited)
    query_residual  : Pre-computed L26 residual (e.g. from generation position).
                      When provided, scoring functions use this instead of
                      computing from prompt_ids. Enables generation-guided routing.

    Returns
    -------
    Sorted list of window IDs to replay.
    """
    t0 = time.time()
    last_wid = lib.num_windows - 1

    if strategy == RoutingStrategy.BM25:
        scores = _bm25_score_windows(lib, tokenizer, prompt_text)
        method_name = "BM25"

    elif strategy == RoutingStrategy.ATTENTION:
        scores = _attention_score_windows(lib, kv_gen, prompt_ids)
        num_captured = len([i for i in range(len(kv_gen.backbone.adapted_layers))
                           if kv_gen.backbone.is_global_layer(i)])
        method_name = f"attention ({lib.num_windows} checkpoints, {num_captured} global layers)"

    elif strategy == RoutingStrategy.PREVIEW:
        scores = _preview_score_windows(lib, kv_gen, prompt_ids)
        method_name = f"preview ({_PREVIEW_TOKENS}+{_PREVIEW_TOKENS} tok/window)"

    elif strategy == RoutingStrategy.DEFLECTION:
        scores = _deflection_score_windows(lib, kv_gen, prompt_ids)
        method_name = "residual deflection"

    elif strategy == RoutingStrategy.HYBRID:
        # Stage 1: BM25 pre-filter
        bm25_scores = _bm25_score_windows(lib, tokenizer, prompt_text)
        candidates = [
            wid for wid, s in bm25_scores[:bm25_shortlist] if s > 0.0
        ]

        if len(candidates) < top_k:
            # BM25 didn't find enough — fall back to preview on all windows
            scores = _preview_score_windows(lib, kv_gen, prompt_ids)
            method_name = f"preview (BM25 fallback, {lib.num_windows} windows)"
        else:
            # Stage 2: preview re-rank on shortlist
            scores = _preview_score_windows(
                lib, kv_gen, prompt_ids, candidate_wids=candidates
            )
            method_name = f"hybrid (BM25→{len(candidates)}→preview)"

    elif strategy == RoutingStrategy.COMPASS:
        if not lib.has_compass:
            print("  Warning: no compass data in library, falling back to BM25", file=sys.stderr)
            print("  Re-run prefill to generate compass data.", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _compass_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            pc_s, pc_e = lib.compass_pc_start, lib.compass_pc_end
            if lib.is_darkspace:
                method_name = f"darkspace (L{layer}, {pc_e}D frame bank)"
            elif lib.has_structural_basis:
                method_name = f"compass (L{layer}, structural PC 0-{pc_s-1} removed, full dark space)"
            else:
                method_name = f"compass (L{layer}, PC {pc_s}-{pc_e-1}, {pc_e-pc_s}D)"

    elif strategy == RoutingStrategy.QK:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _qk_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            method_name = f"Q/K attention (L{layer}, model's own routing)"

    elif strategy == RoutingStrategy.GEOMETRIC:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            # Convert pre-computed residual to numpy if provided
            _qvec_np = None
            if query_residual is not None:
                import numpy as np
                _qvec_np = np.array(
                    query_residual.reshape(-1).tolist(), dtype=np.float32
                )

            # Both scores from model's own geometry
            compass_scores = _compass_score_windows(
                lib, kv_gen, prompt_ids, query_vec_np=_qvec_np,
            )
            contrastive_scores = _contrastive_score_windows(
                lib, kv_gen, prompt_ids, tokenizer, query_vec_np=_qvec_np,
            )

            # Reciprocal rank fusion (RRF) — each strategy votes independently.
            # RRF score = 1/(k+rank_compass) + 1/(k+rank_contrastive)
            # k=60 is standard. Windows ranked high by EITHER strategy rise.
            _RRF_K = 30  # standard RRF constant; low values → winner-take-all
            compass_rank = {wid: rank for rank, (wid, _) in enumerate(compass_scores)}
            contrastive_rank = {wid: rank for rank, (wid, _) in enumerate(contrastive_scores)}
            scores = [
                (wid, 1.0 / (_RRF_K + compass_rank.get(wid, 999))
                     + 1.0 / (_RRF_K + contrastive_rank.get(wid, 999)))
                for wid in range(lib.num_windows)
            ]
            scores.sort(key=lambda x: -x[1])

            layer = lib.compass_layer
            method_name = f"geometric (compass + contrastive RRF, L{layer})"

    elif strategy == RoutingStrategy.CONTRASTIVE:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _contrastive_score_windows(lib, kv_gen, prompt_ids, tokenizer)
            layer = lib.compass_layer
            method_name = f"contrastive (L{layer}, query-specific 8D frame)"

    elif strategy == RoutingStrategy.DARKSPACE:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _darkspace_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            method_name = f"darkspace (coarse 16D PCA → fine L{layer} 2560D directed)"

    elif strategy == RoutingStrategy.GUIDED:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _guided_score_windows(lib, kv_gen, prompt_ids, tokenizer)
            layer = lib.compass_layer
            pc_s, pc_e = lib.compass_pc_start, lib.compass_pc_end
            method_name = f"guided (compass L{layer} PC {pc_s}-{pc_e-1} × token overlap)"

    elif strategy == RoutingStrategy.DIRECTED:
        if not lib.has_compass:
            print("  Warning: no compass data in library, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            scores = _directed_score_windows(lib, kv_gen, prompt_ids)
            layer = lib.compass_layer
            method_name = f"directed (L{layer}, query-defined 1D projection)"

    elif strategy == RoutingStrategy.KV_ROUTE:
        if not lib.has_compass:
            print("  Warning: no compass data, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no compass)"
        else:
            # L29 H4 Q·K routing — the model's own fact-addressing mechanism
            retrieval_layer = routing_layer
            retrieval_head = routing_head
            scores = _kv_route_score_windows(
                lib, kv_gen, prompt_ids,
                retrieval_layer=retrieval_layer,
                query_head=retrieval_head,
            )
            layer = lib.compass_layer
            method_name = (
                f"KV route (L{retrieval_layer} H{retrieval_head} Q·K, "
                f"compass residuals at L{layer})"
            )

    elif strategy == RoutingStrategy.SPARSE:
        if not lib.has_sparse_index:
            print("  Warning: no sparse index in library, falling back to BM25", file=sys.stderr)
            print("  Run prefill with --phases sparse to generate it.", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no sparse index)"
        else:
            scores = _sparse_score_windows(lib, prompt_text)
            method_name = "sparse keyword BM25"

    elif strategy == RoutingStrategy.RESIDUAL:
        if not lib.has_residuals:
            print("  Warning: no residuals in library, falling back to BM25", file=sys.stderr)
            scores = _bm25_score_windows(lib, tokenizer, prompt_text)
            method_name = "BM25 (fallback, no residuals)"
        else:
            scores = _residual_cosine_score_windows(lib, kv_gen, prompt_ids)
            method_name = "residual cosine (legacy)"

    elif strategy == RoutingStrategy.TEMPORAL:
        scores = _temporal_stride_windows(lib, k=top_k)
        method_name = f"temporal stride ({top_k} evenly spaced)"

    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")

    elapsed_ms = (time.time() - t0) * 1000

    # Filter out excluded windows before selection
    if exclude:
        scores = [(wid, s) for wid, s in scores if wid not in exclude]

    # Select top-k
    selected = [wid for wid, _ in scores[:top_k]]

    # Always include last window for continuity
    if last_wid not in selected:
        selected.append(last_wid)

    # Print routing table
    print(f"  Compass routing ({method_name}, {elapsed_ms:.0f}ms):", file=sys.stderr)
    show_n = max(top_k + 2, 5)
    for i, (wid, score) in enumerate(scores[:show_n]):
        marker = " *" if wid in selected else ""
        w = lib.windows[wid]
        print(
            f"    window {wid:>2} (score={score:+.4f}){marker}  {w.preview[:50]}",
            file=sys.stderr,
        )
        if i == top_k - 1 and top_k < len(scores):
            print(f"    {'─' * 60}", file=sys.stderr)

    # Score range
    all_scores = [s for _, s in scores]
    if all_scores:
        print(
            f"  Score range: min={min(all_scores):+.4f} max={max(all_scores):+.4f} "
            f"spread={max(all_scores) - min(all_scores):.2e}",
            file=sys.stderr,
        )

    # Return in replay order: continuity window first (far from prompt),
    # then routed windows in ascending score, so the best match is last
    # (closest to the prompt).  This matters for sliding-window attention
    # models where non-global layers only attend to nearby positions.
    score_map = {wid: s for wid, s in scores}
    continuity = [wid for wid in selected if wid == last_wid and wid not in {w for w, _ in scores[:top_k]}]
    routed = [wid for wid in selected if wid not in continuity]
    routed.sort(key=lambda wid: score_map.get(wid, -1e9))
    return continuity + routed
