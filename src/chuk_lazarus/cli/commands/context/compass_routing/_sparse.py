"""Sparse keyword index scoring — BM25 over pre-extracted keywords.

Like BM25 but runs on the pre-extracted keyword index instead of
decoding full window tokens. Much faster (no tokenizer decode) and
semantically richer (keywords are entity+context triplets).

Used for hybrid routing: sparse scoring selects windows, then
Mode 4 replays them for full-text generation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _sparse_score_windows(
    lib,
    query_text: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    """Score each window against the query using BM25 on sparse keywords.

    Returns list of (window_id, score) sorted descending by score.
    """
    # Load sparse index
    index_path = lib.path / "sparse_index.json"
    if not index_path.exists():
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    data = json.loads(index_path.read_text())
    if isinstance(data, dict):
        entries = data.get("entries", [])
    else:
        entries = data

    # Build per-window keyword map
    import re as _re
    window_keywords: dict[int, list[str]] = {}
    for entry in entries:
        wid = entry.get("window_id", -1)
        kws = entry.get("keywords", [])
        # Flatten keyword triplets into individual terms for matching
        terms: list[str] = []
        for kw in kws:
            for word in kw.lower().split():
                clean = _re.sub(r'[^\w]', '', word)
                if len(clean) > 1:
                    terms.append(clean)
        window_keywords[wid] = terms

    # Query terms (strip punctuation for matching)
    import re as _re
    query_terms = set()
    for word in query_text.lower().split():
        clean = _re.sub(r'[^\w]', '', word)
        if len(clean) > 1:
            query_terms.add(clean)

    if not query_terms:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    num_windows = lib.num_windows

    # Term frequencies per window
    window_tfs: list[dict[str, int]] = []
    window_lengths: list[int] = []
    for wid in range(num_windows):
        terms = window_keywords.get(wid, [])
        window_lengths.append(len(terms))
        tf: dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        window_tfs.append(tf)

    avg_dl = sum(window_lengths) / max(num_windows, 1)

    # Document frequency for query terms
    df: dict[str, int] = {}
    for term in query_terms:
        count = sum(1 for tf in window_tfs if term in tf)
        df[term] = count

    # Score each window
    scores: list[tuple[int, float]] = []
    for wid in range(num_windows):
        score = 0.0
        dl = window_lengths[wid]
        tf = window_tfs[wid]

        for term in query_terms:
            if term not in tf:
                continue
            term_tf = tf[term]
            term_df = df.get(term, 0)
            idf = math.log((num_windows - term_df + 0.5) / (term_df + 0.5) + 1.0)
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * dl / max(avg_dl, 0.001))
            score += idf * numerator / denominator

        scores.append((wid, score))

    scores.sort(key=lambda x: -x[1])
    return scores
