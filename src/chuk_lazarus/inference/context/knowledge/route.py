"""Query routing — TF-IDF token overlap and keyword matching.

Two routing strategies:
    TFIDFRouter   — model-native: query token IDs vs window token sets, weighted by IDF
    KeywordRouter — text-based: keyword phrase matching (wraps SparseKeywordIndex)

Both return a window_id (int) for the best-matching window.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..research._stopwords import FUNCTION_WORDS as _FUNCTION_WORDS
from ..research._stopwords import WORD_RE as _WORD_RE


# ── Keyword extraction helpers ───────────────────────────────────────


def _extract_keywords_from_text(text: str, max_keywords: int = 10) -> list[str]:
    """Extract content words from text, excluding function words."""
    words = _WORD_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        wl = w.lower()
        if wl not in _FUNCTION_WORDS and wl not in seen and len(wl) >= 2:
            seen.add(wl)
            result.append(wl)
            if len(result) >= max_keywords:
                break
    return result


def extract_window_keywords(
    token_ids: list[int],
    tokenizer,
    max_keywords: int = 5,
) -> list[str]:
    """Extract keywords from a window's tokens for the sparse index."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return _extract_keywords_from_text(text, max_keywords=max_keywords)


# ── Sparse keyword index ─────────────────────────────────────────────


@dataclass
class SparseKeywordIndex:
    """Keyword -> passage mapping for query-time routing.

    ~800 bytes total for a 370K-token document.
    """

    entries: dict[str, list[int]] = field(default_factory=dict)
    """keyword (lowercase) -> list of passage indices."""

    def add(self, passage_idx: int, keywords: list[str]) -> None:
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.entries:
                self.entries[kw_lower] = []
            if passage_idx not in self.entries[kw_lower]:
                self.entries[kw_lower].append(passage_idx)

    def query(self, query_text: str) -> list[int]:
        """Return passage indices matching any query keyword, ranked by hit count."""
        query_words = _extract_keywords_from_text(query_text)
        hits: dict[int, int] = {}
        for word in query_words:
            for idx in self.entries.get(word, []):
                hits[idx] = hits.get(idx, 0) + 1
        return sorted(hits, key=lambda i: hits[i], reverse=True)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.entries, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> SparseKeywordIndex:
        data = json.loads(path.read_text())
        return cls(entries=data)

    @property
    def num_keywords(self) -> int:
        return len(self.entries)


# ── TF-IDF Router ────────────────────────────────────────────────────


class TFIDFRouter:
    """Route queries by TF-IDF weighted token overlap.

    During build, each window's unique token IDs and a global IDF table
    are computed.  At query time, the query's token IDs are intersected
    with each window's token set, scored by IDF weight.
    """

    def __init__(
        self,
        window_tokens: dict[int, set[int]],
        idf: dict[int, float],
    ) -> None:
        self.window_tokens = window_tokens
        self.idf = idf

    def route(self, query_token_ids: list[int]) -> int | None:
        """Return the window_id with highest TF-IDF overlap score, or None."""
        if not self.window_tokens:
            return None
        query_set = set(query_token_ids)
        best_wid: int | None = None
        best_score: float = 0.0
        for wid, tokens in self.window_tokens.items():
            overlap = query_set & tokens
            score = sum(self.idf.get(t, 0.0) for t in overlap)
            if score > best_score:
                best_score = score
                best_wid = wid
        return best_wid

    def route_with_score(self, query_token_ids: list[int]) -> tuple[int | None, float]:
        """Return (window_id, score) for best match."""
        if not self.window_tokens:
            return None, 0.0
        query_set = set(query_token_ids)
        best_wid: int | None = None
        best_score: float = 0.0
        for wid, tokens in self.window_tokens.items():
            overlap = query_set & tokens
            score = sum(self.idf.get(t, 0.0) for t in overlap)
            if score > best_score:
                best_score = score
                best_wid = wid
        return best_wid, best_score

    # ── Build helpers ─────────────────────────────────────────────────

    @staticmethod
    def compute_idf(window_tokens: dict[int, set[int]]) -> dict[int, float]:
        """Compute IDF from window token sets.

        IDF(t) = log(N / df(t))  where df(t) = number of windows containing token t.
        """
        n_windows = len(window_tokens)
        if n_windows == 0:
            return {}
        token_df: dict[int, int] = {}
        for tokens in window_tokens.values():
            for t in tokens:
                token_df[t] = token_df.get(t, 0) + 1
        return {t: math.log(n_windows / df) for t, df in token_df.items()}


# ── Keyword Router ────────────────────────────────────────────────────


class KeywordRouter:
    """Route queries by keyword phrase matching.

    Wraps SparseKeywordIndex with the same interface as TFIDFRouter.
    """

    def __init__(self, keywords: dict[int, list[str]]) -> None:
        self._index = SparseKeywordIndex()
        for wid, kws in keywords.items():
            self._index.add(wid, kws)

    def route(self, query_text: str) -> int | None:
        """Return the window_id with most keyword hits, or None."""
        candidates = self._index.query(query_text)
        return candidates[0] if candidates else None

    @property
    def sparse_index(self) -> SparseKeywordIndex:
        return self._index
