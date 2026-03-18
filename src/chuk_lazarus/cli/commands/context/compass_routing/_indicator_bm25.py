"""BM25 scoring against fixed indicator term banks.

Unlike standard BM25 which scores against the user's query text,
indicator BM25 scores against predefined term banks for specific
content types (engagement markers, tension markers).

Validated experimentally:
  - Engagement indicators: 81.8% recall for humor/social content
  - Tension indicators: ~55% recall (40-50% semantic gap — many tense
    moments lack keywords, which is why probe re-ranking is essential)
"""

from __future__ import annotations

import math


# ── Indicator term banks (from Lazarus experiments) ────────────────────

ENGAGEMENT_INDICATORS = [
    "laughter", "music", "chuckle", "laughing", "laugh",
    "joke", "funny", "kidding", "birthday", "congratulations",
    "beautiful", "cool", "gee", "hey", "czar",
    "surprised", "amazing", "incredible", "fantastic",
]

TENSION_INDICATORS = [
    "alarm", "abort", "warning", "fuel", "seconds",
    "emergency", "caution", "critical", "problem", "malfunction",
    "failure", "danger", "anomaly", "off-nominal",
    "go/no-go", "hold", "scrub",
]


def _indicator_bm25_score_windows(
    lib,
    tokenizer,
    indicators: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    """BM25 scoring against a fixed indicator term bank.

    Same algorithm as _bm25_score_windows but with indicator terms
    instead of query text. Returns (window_id, score) sorted descending.
    """
    query_terms = set(t.lower() for t in indicators)
    if not query_terms:
        return [(wid, 0.0) for wid in range(lib.num_windows)]

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

    # Document frequency for indicator terms
    df: dict[str, int] = {}
    for term in query_terms:
        count = sum(1 for tf in window_term_freqs if term in tf)
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

            idf = math.log((num_windows - term_df + 0.5) / (term_df + 0.5) + 1.0)
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator

        scores.append((wid, score))

    scores.sort(key=lambda x: -x[1])
    return scores
