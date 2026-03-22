"""Build a knowledge store from a document.  Three-pass.

Pass 1 (fast, no model):
  Tokenise all windows → unique token sets.
  Compute IDF across all windows.
  Select target tokens per window (rarest by IDF, dynamic count).

Pass 2 (medium, one forward pass per window):
  Chain boundary residuals across all windows (Markov property).
  No coefficient extraction — just chaining.

Pass 3 (slow, one forward pass per window):
  For each window, build a context+query donor prompt.
  Forward the donor to crystal_layer.
  Extract coefficients from the ANSWER position (last token).
  The donor primes the model to predict the answer, so the
  coefficient points in the answer direction (91-100% P(target)
  vs ~0% from raw prefill).

The coefficient formula:
    natural_coeff = dot(donor_residual_last, embed(token_id)) / ||embed||^2
    stored_coeff  = inject_coefficient * natural_coeff   (default 2x)
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

    # ── Pass 2: Chain boundary residuals (one forward per window) ────
    boundary_residual: mx.array | None = None
    boundary_residuals: dict[int, mx.array | None] = {}

    for wid, chunk_ids in enumerate(windows):
        w_ids = mx.array(chunk_ids)[None]
        h = kv_gen.prefill_to_layer(
            w_ids,
            target_layer=config.crystal_layer,
            initial_residual=boundary_residual,
        )
        boundary_residual = h[:, -1:, :]
        mx.eval(boundary_residual)
        boundary_residuals[wid] = boundary_residual
        del h

        if progress_fn:
            progress_fn(wid, num_windows)

    # ── Pass 3: Donor coefficient extraction ────────────────────────
    # One forward pass per window with a generic donor prompt.
    # The donor primes the model to produce entity answers, giving
    # coefficients in the answer direction (91-100% P(target) at 2×).
    embed_matrix = kv_gen.backbone.embed_matrix
    all_entries: list[InjectionEntry] = []
    fact_id_counter = 0

    for wid, chunk_ids in enumerate(windows):
        target_list = targets.get(wid, [])
        if not target_list:
            continue

        # Build donor prompt: context + entity-priming question
        window_text = tokenizer.decode(chunk_ids, skip_special_tokens=True) if tokenizer else ""
        donor_prompt = (
            f"{window_text}\n\n"
            "Who or what is the most notable entity mentioned above? "
            "Answer with just the name."
        )
        donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False) if tokenizer else chunk_ids

        # Forward donor to crystal_layer — extract answer-direction residual
        donor_mx = mx.array(donor_ids)[None]
        h_donor = kv_gen.prefill_to_layer(
            donor_mx,
            target_layer=config.crystal_layer,
        )
        # Last position residual — where the answer direction lives
        r_last = h_donor[0, -1, :]  # (hidden_dim,)
        mx.eval(r_last)
        del h_donor

        # Extract coefficients for each target token
        for token_id, pos_in_window in target_list:
            embed = embed_matrix[token_id]
            embed_norm_sq = (embed * embed).sum()
            natural_coeff = (r_last * embed).sum() / (embed_norm_sq + 1e-8)
            stored_coeff = float(inject_coefficient * natural_coeff.item())

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
