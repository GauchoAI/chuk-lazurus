"""Build a knowledge store from a document.  Three-pass.

Pass 1 (fast, no model):
  Tokenise into 512-token windows.
  Compute IDF across all windows.
  Store unique token IDs per window (routing) + ordered lists (reconstruction).

Pass 2 (one forward per window):
  Chain boundary residuals (Markov property).
  Each boundary carries the cumulative state of all prior windows.
  Combined with token IDs, one forward pass reconstructs full state. KL=0.0.

Pass 3 (one generation per window):
  Model-native keyword expansion (5 topic tokens per window).
  Bridges vocabulary gaps: "baseball" for a window with "Philadelphia 7, Baltimore 3".
"""

from __future__ import annotations

import math
from typing import Callable

import mlx.core as mx

from .config import ArchitectureConfig
from .route import SparseKeywordIndex, TFIDFRouter, extract_window_keywords
from .store import InjectionEntry, KnowledgeStore


def streaming_prefill(
    kv_gen,
    document_tokens: list[int],
    config: ArchitectureConfig,
    tokenizer=None,
    progress_fn: Callable[[int, int], None] | None = None,
) -> KnowledgeStore:
    """Build a knowledge store from a document.  Three-pass.

    Parameters
    ----------
    kv_gen          : KVDirectGenerator instance.
    document_tokens : Full document as token IDs.
    config          : ArchitectureConfig with crystal_layer, window_size, etc.
    tokenizer       : Required for donor prompt construction and keywords.
    progress_fn     : Optional callback(window_id, num_windows).

    Returns
    -------
    KnowledgeStore with injection entries, TF-IDF data, and keyword index.
    """
    window_size = config.window_size
    num_windows = math.ceil(len(document_tokens) / window_size) if document_tokens else 0
    min_entries = config.entries_per_window
    inject_coefficient = config.inject_coefficient

    # ── Chunk the document into windows ──────────────────────────────
    windows: list[list[int]] = []
    for wid in range(num_windows):
        start = wid * window_size
        end = min(start + window_size, len(document_tokens))
        windows.append(document_tokens[start:end])

    # ── Pass 1: Token index + IDF + target selection (no model) ──────
    window_tokens: dict[int, set[int]] = {}
    window_token_lists: dict[int, list[int]] = {}
    keywords: dict[int, list[str]] = {}
    sparse_index = SparseKeywordIndex()

    for wid, chunk_ids in enumerate(windows):
        window_tokens[wid] = set(chunk_ids)
        window_token_lists[wid] = chunk_ids
        kws = extract_window_keywords(chunk_ids, tokenizer) if tokenizer else []
        keywords[wid] = kws
        sparse_index.add(wid, kws)

    idf = TFIDFRouter.compute_idf(window_tokens)

    # Select target tokens per window (rarest by IDF, dynamic count)
    # targets[wid] = [(token_id, position_in_window), ...]
    targets: dict[int, list[tuple[int, int]]] = {}
    for wid, chunk_ids in enumerate(windows):
        targets[wid] = _select_targets(chunk_ids, idf, min_k=min_entries)

    # ── Pass 2: Chain boundary residuals (the Markov chain) ────────────
    # Each boundary carries the cumulative state of all prior windows.
    # Combined with token IDs, one forward pass reconstructs full state.
    boundary_residual: mx.array | None = None
    boundaries: dict[int, mx.array] = {}
    residual_streams: dict[int, mx.array] = {}

    for wid, chunk_ids in enumerate(windows):
        w_ids = mx.array(chunk_ids)[None]
        h = kv_gen.prefill_to_layer(
            w_ids,
            target_layer=config.crystal_layer,
            initial_residual=boundary_residual,
        )

        # Store boundary for this window (the Markov chain link)
        boundary_residual = h[:, -1:, :]
        mx.eval(boundary_residual)
        boundaries[wid] = boundary_residual[0, 0, :]  # (hidden_dim,) float32

        # Optional: store full L30 residual stream for pre-cache (Mode B)
        offset = 1 if wid > 0 else 0  # skip chained boundary position
        stream = h[0, offset:, :]
        residual_streams[wid] = stream
        mx.eval(stream)

        del h

        if progress_fn:
            progress_fn(wid, num_windows)

    # ── Pass 3: Keyword expansion (one forward per window) ─────────
    # Generate 5 topic words per window using the model. These bridge
    # the vocabulary gap: "baseball" gets associated with window 170
    # even though the text says "Philadelphia 7, Baltimore 3."
    # ~20 bytes per window. Fixes the 4/5 → 5/5 routing accuracy.
    all_entries: list[InjectionEntry] = []
    fact_id_counter = 0

    if tokenizer is not None:
        from ._sampling import sample_token

        for wid, chunk_ids in enumerate(windows):
            window_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            # Completion prompt: model generates topic words
            topic_prompt = f"{window_text}\n\nTopics:"
            topic_ids = tokenizer.encode(topic_prompt, add_special_tokens=True)
            topic_mx = mx.array(topic_ids)[None]

            logits, kv_store = kv_gen.prefill(topic_mx)
            mx.eval(logits)
            seq_len = topic_mx.shape[1]

            # Generate 5 topic tokens
            for _ in range(5):
                token = sample_token(logits[0, -1], 0.0)
                # Add generated token to routing set
                window_tokens[wid].add(token)
                # Also add case/space variants for robust matching
                kw_text = tokenizer.decode([token]).strip()
                if kw_text and len(kw_text) >= 2:
                    kw_lower = kw_text.lower()
                    keywords[wid].append(kw_lower)
                    sparse_index.add(wid, [kw_lower])
                    # Add variant token IDs: " word", "word", "Word"
                    for variant in [kw_lower, f" {kw_lower}", kw_text, f" {kw_text}"]:
                        var_ids = tokenizer.encode(variant, add_special_tokens=False)
                        for vid in var_ids:
                            window_tokens[wid].add(vid)
                logits, kv_store = kv_gen.step_uncompiled(
                    mx.array([[token]]), kv_store, seq_len=seq_len)
                seq_len += 1

        # Recompute IDF with expanded token sets
        idf = TFIDFRouter.compute_idf(window_tokens)

    # Build injection entries from IDF-selected targets
    embed_matrix = kv_gen.backbone.embed_matrix
    for wid, chunk_ids in enumerate(windows):
        target_list = targets.get(wid, [])
        for token_id, pos_in_window in target_list:
            embed = embed_matrix[token_id]
            embed_norm_sq = (embed * embed).sum()
            # Use a simple coefficient from the embedding projection
            # (the focused context replay handles narrative, not the entries)
            natural_coeff = float(mx.linalg.norm(embed).item())
            stored_coeff = inject_coefficient * natural_coeff

            all_entries.append(InjectionEntry(
                token_id=token_id,
                coefficient=stored_coeff,
                window_id=wid,
                position_in_window=pos_in_window,
                fact_id=fact_id_counter,
            ))
            fact_id_counter += 1

    return KnowledgeStore(
        entries=all_entries,
        boundaries=boundaries,
        residual_streams=residual_streams,
        window_tokens=window_tokens,
        window_token_lists=window_token_lists,
        idf=idf,
        keywords=keywords,
        config=config,
        boundary_residual=boundary_residual,
        num_windows=num_windows,
        num_tokens=len(document_tokens),
    )


def _select_targets(
    chunk_ids: list[int],
    idf: dict[int, float],
    min_k: int = 8,
) -> list[tuple[int, int]]:
    """Select target tokens by IDF (rarest first), dynamic count.

    Returns [(token_id, position_in_window), ...] sorted by IDF descending.
    Deduplicates by token_id (first occurrence wins).
    Skips byte tokens (Gemma: IDs >= 236700) and position 0 (BOS).

    Dynamic count: max(min_k, number of df=1 tokens in window).
    """
    if not chunk_ids:
        return []

    # Score each position by its token's IDF
    seen: set[int] = set()
    scored: list[tuple[float, int, int]] = []  # (idf, token_id, position)

    for pos in range(1, len(chunk_ids)):  # skip position 0
        tid = chunk_ids[pos]
        if tid >= 236700:  # skip byte tokens
            continue
        if tid in seen:
            continue
        seen.add(tid)
        token_idf = idf.get(tid, 0.0)
        if token_idf > 0:
            scored.append((token_idf, tid, pos))

    # Sort by IDF descending
    scored.sort(reverse=True)

    # Dynamic count: at least min_k, but keep all df=1 tokens
    # df=1 tokens have the maximum IDF = log(N)
    if scored:
        max_idf = scored[0][0]
        # Count tokens within 0.01 of max IDF (all df=1)
        num_df1 = sum(1 for s, _, _ in scored if abs(s - max_idf) < 0.01)
        k = max(min_k, num_df1)
    else:
        k = min_k

    selected = scored[:k]
    selected_positions = {pos for _, _, pos in selected}

    # Entity gap filling: if two selected positions are separated by
    # 1-2 positions, the gap tokens are part of the same entity.
    # "John" (98) + "oyle" (100) → fill " C" (99).
    filled = _fill_entity_gaps(selected_positions, chunk_ids, max_gap=2)

    # Build final list: IDF-selected + gap-filled, sorted by position
    result: list[tuple[int, int]] = []
    for pos in sorted(filled):
        tid = chunk_ids[pos]
        result.append((tid, pos))

    return result


def _fill_entity_gaps(
    selected_positions: set[int],
    chunk_ids: list[int],
    max_gap: int = 2,
) -> set[int]:
    """Fill gaps between nearby selected positions.

    If positions 98 and 100 are both selected, position 99 is a gap
    within an entity. Fill it — the gap token (" C") is part of
    "Coyle" even though its IDF is too low for selection.
    """
    filled = set(selected_positions)
    sorted_pos = sorted(selected_positions)

    for i in range(len(sorted_pos) - 1):
        gap = sorted_pos[i + 1] - sorted_pos[i]
        if 1 < gap <= max_gap + 1:
            for p in range(sorted_pos[i] + 1, sorted_pos[i + 1]):
                if 0 < p < len(chunk_ids):  # bounds check
                    filled.add(p)

    return filled
