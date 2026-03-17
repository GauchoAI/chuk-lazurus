"""
Sparse Semantic Index — Mode 5.

Accumulates keyword entries from document windows, renders them as a
prompt prefix for replay-free retrieval. Saves/loads to disk.

The key insight: extract only what the model DOESN'T know. The model's
own surprise (prediction rank) at each token is the signal. High surprise
= novel = extract. Low surprise = parametric = skip.

Minimum viable fact unit: entity + relation_TYPE + value = 3 content tokens.
"Zarkov city Voltara" retrieves at 100%. "Zarkov Voltara" fails at 48%.

Storage: ~800 bytes for a 370K-token document (triplet compression).
Latency: ~5-10ms (standard prefill of keyword prompt).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------
# Function words — stripped from keyword entries to save tokens
# -----------------------------------------------------------------------

FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "we", "you", "me", "him", "us", "my", "your", "our", "who",
    "what", "which", "when", "where", "how", "why",
})

# Common abbreviations that appear everywhere and carry no information
COMMON_ABBREVIATIONS = frozenset({
    "I", "OK", "AM", "PM", "US", "UK", "TV", "ID", "GO", "CC",
    "SC", "MS", "AN", "ON", "OR", "AT", "OF", "BY", "IF", "UP",
    "NO", "SO", "DO",
})

# -----------------------------------------------------------------------
# Stopwords — broader set for content word extraction
# -----------------------------------------------------------------------

STOPWORDS = frozenset({
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'in', 'of', 'at', 'to', 'for', 'with', 'by', 'on',
    'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me',
    'was', 'were', 'is', 'are', 'have', 'had', 'do', 'did',
    'and', 'or', 'but', 'so', 'because', 'if', 'when',
    'very', 'really', 'just', 'also', 'then', 'not', 'no',
    'be', 'been', 'being', 'has', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'must',
    'about', 'from', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'over',
    'here', 'there', 'where', 'how', 'what', 'which', 'who',
    'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'than',
    'too', 'out', 'up', 'down', 'off', 'over', 'now',
    'get', 'got', 'go', 'going', 'went', 'come', 'came',
    'make', 'made', 'take', 'took', 'give', 'gave',
    'say', 'said', 'tell', 'told', 'ask', 'asked',
    'think', 'thought', 'know', 'knew', 'see', 'saw',
    'want', 'need', 'like', 'look', 'looked',
    'well', 'back', 'way', 'right', 'good', 'new',
    'roger', 'copy', 'over', 'okay',  # transcript-specific
})


# -----------------------------------------------------------------------
# Content word extraction
# -----------------------------------------------------------------------

def extract_content_words(tokens: list[int], tokenizer) -> list[str]:
    """Extract content words from a window's token IDs.

    Decodes all tokens to text, splits into words, filters stopwords
    and short tokens. Handles multi-token words (e.g. "Namath" split
    into "Nam"+"ath") by working on the decoded text rather than
    individual tokens.

    Returns deduplicated list of lowercase content words.
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Split into words: mixed-case words (3+ chars) and all-caps (2+ chars)
    words = re.findall(r'[A-Za-z][a-z]{2,}|[A-Z]{2,}', text)

    content_words: list[str] = []
    seen: set[str] = set()
    for word in words:
        lower = word.lower()
        if lower in STOPWORDS:
            continue
        if len(lower) < 3:
            continue
        if lower not in seen:
            seen.add(lower)
            content_words.append(lower)

    return content_words


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class TokenClassification:
    """Per-token surprise classification."""
    position: int
    token: str
    rank: int
    category: str  # "parametric", "semi_parametric", "novel"


@dataclass
class EntityCandidate:
    """A detected entity/number/abbreviation candidate."""
    position: int
    token: str
    entity_type: str  # "named_entity", "numeric", "abbreviation", "high_surprise"
    surprise_rank: int = 0
    multi_token_end: int | None = None


@dataclass
class FactTriplet:
    """Entity + surrounding context tokens."""
    entity: str
    context: list[str]
    entity_type: str
    position: int
    surprise_rank: int = 0


@dataclass
class FactSpan:
    """A fact-bearing token span within a window."""
    position: int  # token index of the fact token
    radius: int = 5  # ±N context tokens around the fact

    @property
    def start(self) -> int:
        return max(0, self.position - self.radius)

    def end(self, window_size: int) -> int:
        return min(window_size, self.position + self.radius)

    def to_dict(self) -> dict:
        return {"position": self.position, "radius": self.radius}

    @classmethod
    def from_dict(cls, d: dict) -> FactSpan:
        return cls(position=d["position"], radius=d.get("radius", 5))


@dataclass
class SparseEntry:
    """One window's extracted keywords and fact spans."""
    window_id: int
    keywords: list[str] = field(default_factory=list)
    content_words: list[str] = field(default_factory=list)
    surprise_rank: int = 0
    fact_spans: list[FactSpan] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"window_id": self.window_id, "keywords": self.keywords}
        if self.content_words:
            d["content_words"] = self.content_words
        if self.surprise_rank > 0:
            d["surprise_rank"] = self.surprise_rank
        if self.fact_spans:
            d["fact_spans"] = [s.to_dict() for s in self.fact_spans]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SparseEntry:
        spans = [FactSpan.from_dict(s) for s in d.get("fact_spans", [])]
        return cls(
            window_id=d["window_id"],
            keywords=d.get("keywords", []),
            content_words=d.get("content_words", []),
            surprise_rank=d.get("surprise_rank", 0),
            fact_spans=spans,
        )


# -----------------------------------------------------------------------
# Surprise classifier
# -----------------------------------------------------------------------

class SurpriseClassifier:
    """Classify tokens by prediction rank from logits already computed."""

    def __init__(
        self,
        novel_threshold: int = 50,
        semi_threshold: int = 3,
    ):
        self.novel_threshold = novel_threshold
        self.semi_threshold = semi_threshold

    def classify_tokens(self, tokens: list[str], ranks: list[int]) -> list[TokenClassification]:
        """Classify each token by its prediction rank."""
        classifications = []
        for i, (token, rank) in enumerate(zip(tokens, ranks)):
            if rank <= self.semi_threshold:
                category = "parametric"
            elif rank <= self.novel_threshold:
                category = "semi_parametric"
            else:
                category = "novel"
            classifications.append(TokenClassification(
                position=i, token=token, rank=rank, category=category,
            ))
        return classifications


# -----------------------------------------------------------------------
# Entity extractor — heuristic extraction from text
# -----------------------------------------------------------------------

class EntityExtractor:
    """Extract entities, facts, and terms from window text.

    When surprise ranks are provided, only extracts novel/semi-parametric
    tokens. Without ranks, extracts all capitalised entities (fallback).
    """

    def __init__(
        self,
        max_keywords: int = 8,
        context_window: int = 2,
        novel_rank_threshold: int = 50,
        number_rank_threshold: int = 10,
        surprise_catch_all_rank: int = 200,
    ):
        self.max_keywords = max_keywords
        self.context_window = context_window
        self.novel_rank_threshold = novel_rank_threshold
        self.number_rank_threshold = number_rank_threshold
        self.surprise_catch_all_rank = surprise_catch_all_rank

    def extract(self, text: str, token_ranks: list[int] | None = None) -> list[str]:
        """Extract keyword triplets from text.

        Args:
            text: window text (decoded from tokens)
            token_ranks: per-token surprise ranks from logits (optional).
                         When provided, only novel tokens are extracted.

        Returns:
            List of keyword strings (entity + context triplets).
        """
        # Normalize whitespace
        text = " ".join(text.split())
        words = text.split()
        if not words:
            return []

        # Build per-word surprise mask if ranks provided
        has_surprise = token_ranks is not None and len(token_ranks) > 0
        # Note: token_ranks map to BPE tokens, not words. For word-level
        # extraction we use a simplified heuristic: mark word positions
        # near high-surprise token positions as interesting.

        keywords: list[str] = []
        seen: set[str] = set()

        def _add(kw: str) -> bool:
            kw_clean = " ".join(kw.split())
            if kw_clean and kw_clean.lower() not in seen and len(keywords) < self.max_keywords:
                seen.add(kw_clean.lower())
                keywords.append(kw_clean)
                return True
            return False

        # --- Rule 1: Named entities (capitalised, non-sentence-initial) ---
        for m in re.finditer(
            r'(?<![.!?\n]\s)(?<!\A)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b',
            text
        ):
            entity = m.group(1)
            if entity.lower() in FUNCTION_WORDS:
                continue
            # Grab ±context_window words
            start_char = m.start()
            end_char = m.end()
            before = text[:start_char].rstrip().split()
            after = text[end_char:].lstrip().split()
            ctx_before = [w for w in before[-self.context_window:]
                          if w.lower() not in FUNCTION_WORDS and len(w) > 1]
            ctx_after = [w for w in after[:self.context_window]
                         if w.lower() not in FUNCTION_WORDS and len(w) > 1]

            # Build triplet
            parts = []
            if ctx_before:
                parts.extend(ctx_before)
            parts.append(entity)
            if ctx_after:
                parts.extend(ctx_after)
            _add(" ".join(parts))

        # --- Rule 2: Numbers with context ---
        for m in re.finditer(r'\b(\d+(?:[.,]\d+)*)\b', text):
            n = m.group(1)
            if len(n) < 3 and '.' not in n:
                continue
            before = text[:m.start()].rstrip().split()
            ctx = [w for w in before[-2:] if w.lower() not in FUNCTION_WORDS and len(w) > 1]
            if ctx:
                _add(f"{' '.join(ctx)} {n}")
            else:
                _add(n)
            if len(keywords) >= self.max_keywords:
                break

        # --- Rule 3: Domain abbreviations ---
        for m in re.finditer(r'\b([A-Z]{2,}(?:[-/][A-Z]{2,})*)\b', text):
            a = m.group(1)
            if a in COMMON_ABBREVIATIONS or len(a) < 2:
                continue
            _add(a)
            if len(keywords) >= self.max_keywords:
                break

        return keywords[:self.max_keywords]

    def extract_with_surprise(
        self,
        text: str,
        token_texts: list[str],
        token_ranks: list[int],
    ) -> list[str]:
        """Extract keywords, filtering by surprise rank.

        Only extracts entities/numbers that the model found surprising.
        This filters out parametric noise (the model predicts "Roger",
        "Houston" easily — skip them) and keeps novel facts.

        Args:
            text: full window text
            token_texts: per-token decoded strings
            token_ranks: per-token prediction ranks from logits

        Returns:
            Keyword triplets for novel content only.
        """
        text = " ".join(text.split())

        # Find positions of novel tokens
        novel_positions: set[int] = set()
        for i, rank in enumerate(token_ranks):
            if rank > self.novel_rank_threshold:
                novel_positions.add(i)

        # Find the text spans that correspond to novel token positions
        # For simplicity, we check if an entity's text appears near a
        # novel token position.
        keywords: list[str] = []
        seen: set[str] = set()

        def _add(kw: str) -> bool:
            kw_clean = " ".join(kw.split())
            if kw_clean and kw_clean.lower() not in seen and len(keywords) < self.max_keywords:
                seen.add(kw_clean.lower())
                keywords.append(kw_clean)
                return True
            return False

        # Named entities — only if near a novel token
        for m in re.finditer(
            r'(?<![.!?\n]\s)(?<!\A)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b',
            text
        ):
            entity = m.group(1)
            if entity.lower() in FUNCTION_WORDS:
                continue

            # Check if any token near this character position is novel
            # Approximate: character position / avg_chars_per_token ≈ token position
            avg_chars = max(1, len(text) / max(1, len(token_ranks)))
            approx_token_pos = int(m.start() / avg_chars)
            is_near_novel = any(
                abs(approx_token_pos - np) < 5
                for np in novel_positions
            )
            if not is_near_novel and novel_positions:
                continue  # skip parametric entities

            before = text[:m.start()].rstrip().split()
            after = text[m.end():].lstrip().split()
            ctx_before = [w for w in before[-self.context_window:]
                          if w.lower() not in FUNCTION_WORDS and len(w) > 1]
            ctx_after = [w for w in after[:self.context_window]
                         if w.lower() not in FUNCTION_WORDS and len(w) > 1]
            parts = ctx_before + [entity] + ctx_after
            _add(" ".join(parts))

        # High-surprise numbers
        for m in re.finditer(r'\b(\d+(?:[.,]\d+)*)\b', text):
            n = m.group(1)
            if len(n) < 3:
                continue
            avg_chars = max(1, len(text) / max(1, len(token_ranks)))
            approx_pos = int(m.start() / avg_chars)
            if not any(abs(approx_pos - np) < 3 for np in novel_positions):
                continue
            before = text[:m.start()].rstrip().split()
            ctx = [w for w in before[-2:] if w.lower() not in FUNCTION_WORDS]
            _add(f"{' '.join(ctx)} {n}" if ctx else n)
            if len(keywords) >= self.max_keywords:
                break

        return keywords[:self.max_keywords]


# -----------------------------------------------------------------------
# Sparse Semantic Index
# -----------------------------------------------------------------------

class SparseSemanticIndex:
    """Accumulated keyword index across all document windows."""

    DEFAULT_TEMPLATE = (
        "Below is a keyword index of a document. "
        "Each line maps a window ID to key entities, facts, and terms "
        "from that section. If a keyword entry contains the answer "
        "to the question, state it directly."
    )

    def __init__(self, template: str | None = None):
        self.entries: list[SparseEntry] = []
        self.template = template or self.DEFAULT_TEMPLATE

    def add(self, entry: SparseEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def total_keywords(self) -> int:
        return sum(len(e.keywords) for e in self.entries)

    @property
    def non_empty_count(self) -> int:
        return sum(1 for e in self.entries if e.keywords)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_index(self, max_keywords: int | None = None) -> str:
        lines: list[str] = []
        for entry in self.entries:
            kws = entry.keywords
            if max_keywords is not None:
                kws = kws[:max_keywords]
            if kws:
                lines.append(f"W{entry.window_id}: {', '.join(kws)}")
        return "\n".join(lines)

    def render_prompt(
        self,
        query: str,
        max_keywords: int | None = None,
        chat_template: bool = True,
    ) -> str:
        index_text = self.render_index(max_keywords=max_keywords)
        if chat_template:
            return (
                f"<start_of_turn>user\n"
                f"{self.template}\n\n"
                f"Index:\n{index_text}\n\n"
                f"Question: {query}\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            return (
                f"{self.template}\n\n"
                f"Index:\n{index_text}\n\n"
                f"Question: {query}\n"
                f"Answer: "
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        data = {
            "template": self.template,
            "entries": [e.to_dict() for e in self.entries],
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | str) -> SparseSemanticIndex:
        path = Path(path)
        data = json.loads(path.read_text())
        if isinstance(data, list):
            entries = data
            template = None
        else:
            entries = data.get("entries", [])
            template = data.get("template")
        idx = cls(template=template)
        for d in entries:
            idx.add(SparseEntry.from_dict(d))
        return idx

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "num_entries": len(self.entries),
            "non_empty": self.non_empty_count,
            "total_keywords": self.total_keywords,
            "avg_keywords": (
                self.total_keywords / self.non_empty_count
                if self.non_empty_count > 0 else 0
            ),
            "est_tokens_full": self.total_keywords * 3,
            "est_tokens_triplet": min(self.non_empty_count * 3, self.total_keywords) * 3,
        }
