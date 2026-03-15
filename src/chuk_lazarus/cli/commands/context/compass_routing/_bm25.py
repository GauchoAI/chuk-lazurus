"""BM25 token-level scoring."""

from __future__ import annotations

import math


def _bm25_score_windows(
    lib,
    tokenizer,
    query_text: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    """Score each window against the query using BM25.

    Returns list of (window_id, score) sorted descending by score.
    """
    # Tokenize query into terms (simple whitespace + lowercased)
    query_terms = set(query_text.lower().split())
    if not query_terms:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

    # Build per-window term frequency and document frequency
    num_windows = lib.num_windows
    window_term_freqs: list[dict[str, int]] = []
    window_lengths: list[int] = []

    for wid in range(num_windows):
        tokens = lib.get_window_tokens(wid)
        text = tokenizer.decode(tokens, skip_special_tokens=True).lower()

        words = text.split()
        window_lengths.append(len(words))

        tf: dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        window_term_freqs.append(tf)

    avg_dl = sum(window_lengths) / max(num_windows, 1)

    # Document frequency for query terms
    df: dict[str, int] = {}
    for term in query_terms:
        count = 0
        for tf in window_term_freqs:
            if term in tf:
                count += 1
        df[term] = count

    # Score each window
    scores: list[tuple[int, float]] = []
    for wid in range(num_windows):
        score = 0.0
        dl = window_lengths[wid]
        tf = window_term_freqs[wid]

        for term in query_terms:
            if term not in tf:
                continue
            term_tf = tf[term]
            term_df = df.get(term, 0)

            # IDF with smoothing
            idf = math.log((num_windows - term_df + 0.5) / (term_df + 0.5) + 1.0)

            # BM25 term score
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator

        scores.append((wid, score))

    scores.sort(key=lambda x: -x[1])
    return scores
