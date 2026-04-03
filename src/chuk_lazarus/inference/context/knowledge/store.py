"""Knowledge Store v10 — 8-byte injection entries.

Supersedes v9 (10 KB crystallised residuals per passage). Each fact is now
stored as a (token_id, coefficient) pair — 8 bytes per entry, 13 bytes with
metadata. The injection formula is applied at crystal_layer during generation.

Store format on disk:
    manifest.json       — metadata, config, version
    entries.npz         — injection entries (N x 5 structured fields)
    window_tokens.npz   — unique token IDs per window (for TF-IDF routing)
    idf.json            — IDF table (token_id -> float)
    keywords.json       — keyword phrases per window (optional)
    boundary_residual.npy — final boundary residual (optional, for chaining)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import ArchitectureConfig
from .cosine_router import CosineRouter, build_query_embedding, load_embeddings
from .offline_router import OfflineRouter, load_offline_router
from .synthetic_router import load_synthetic_embeddings
from .route import KeywordRouter, TFIDFRouter

# ── File constants ───────────────────────────────────────────────────

MANIFEST_FILE = "manifest.json"
ENTRIES_FILE = "entries.npz"
WINDOW_TOKENS_FILE = "window_tokens.npz"
WINDOW_TOKEN_LISTS_FILE = "window_token_lists.npz"
IDF_FILE = "idf.json"
KEYWORDS_FILE = "keywords.json"
BOUNDARY_RESIDUAL_FILE = "boundary_residual.npy"
BOUNDARIES_DIR = "boundaries"
RESIDUALS_DIR = "residuals"
STORE_VERSION = 12

# ── v9 file constants (for detection) ────────────────────────────────
_V9_PASSAGES_FILE = "passages.npz"


# ── InjectionEntry ───────────────────────────────────────────────────


@dataclass
class InjectionEntry:
    """A single injection target extracted during build.

    At generation time: h += coefficient * embed(token_id) / ||embed||^2
    """

    token_id: int  # uint32 — injection target AND routing key
    coefficient: float  # float32 — injection magnitude (stored at 2x)
    window_id: int  # uint16 — which window this came from
    position_in_window: int  # uint16 — position within the 512-token window
    fact_id: int  # uint16 — groups multi-token facts

    def to_tuple(self) -> tuple[int, float, int, int, int]:
        return (
            self.token_id,
            self.coefficient,
            self.window_id,
            self.position_in_window,
            self.fact_id,
        )


# ── Numpy structured dtype for serialisation ─────────────────────────

_ENTRY_DTYPE = np.dtype(
    [
        ("token_id", np.uint32),
        ("coefficient", np.float32),
        ("window_id", np.uint16),
        ("position_in_window", np.uint16),
        ("fact_id", np.uint16),
    ]
)


def _entries_to_numpy(entries: list[InjectionEntry]) -> np.ndarray:
    arr = np.empty(len(entries), dtype=_ENTRY_DTYPE)
    for i, e in enumerate(entries):
        arr[i] = (e.token_id, e.coefficient, e.window_id, e.position_in_window, e.fact_id)
    return arr


def _numpy_to_entries(arr: np.ndarray) -> list[InjectionEntry]:
    return [
        InjectionEntry(
            token_id=int(row["token_id"]),
            coefficient=float(row["coefficient"]),
            window_id=int(row["window_id"]),
            position_in_window=int(row["position_in_window"]),
            fact_id=int(row["fact_id"]),
        )
        for row in arr
    ]


# ── KnowledgeStore ───────────────────────────────────────────────────


@dataclass
class KnowledgeStore:
    """Persistent knowledge store for a single document (v10).

    8 bytes per fact (token_id + coefficient) instead of 10 KB per passage.
    Routing via TF-IDF token overlap or keyword matching.
    """

    entries: list[InjectionEntry]
    """All injection entries across all windows."""

    window_tokens: dict[int, set[int]]
    """Unique token IDs per window (for TF-IDF routing)."""

    window_token_lists: dict[int, list[int]]
    """Ordered token IDs per window (for position lookup at query time)."""

    idf: dict[int, float]
    """IDF table: token_id -> log(N/df)."""

    keywords: dict[int, list[str]]
    """Keyword phrases per window (for keyword routing)."""

    boundaries: dict[int, mx.array] = field(default_factory=dict)
    """Boundary residual per window (hidden_dim,) — the Markov chain."""

    config: ArchitectureConfig = field(default=None)
    """Architecture parameters used during build."""

    boundary_residual: mx.array | None = None
    """Final boundary residual for chained prefill resume: (1, 1, hidden_dim)."""

    residual_streams: dict[int, mx.array] | None = field(default=None, repr=False)
    """L30 residual streams per window (set during build, not persisted in memory)."""

    _store_path: Path | None = field(default=None, repr=False)
    """Path to store on disk (for lazy residual loading)."""

    num_windows: int = 0
    """Number of windows in the source document."""

    num_tokens: int = 0
    """Total tokens in the source document."""

    # Cached routers (built lazily)
    _tfidf_router: TFIDFRouter | None = field(default=None, repr=False)
    _keyword_router: KeywordRouter | None = field(default=None, repr=False)
    _cosine_router: CosineRouter | None = field(default=None, repr=False)
    _synthetic_router: CosineRouter | None = field(default=None, repr=False)
    _offline_router: OfflineRouter | None = field(default=None, repr=False)

    # ── Routing ───────────────────────────────────────────────────────

    def route(self, query_text: str, tokenizer=None, method: str = "auto") -> int | None:
        """Route query to best window. Returns window_id or None.

        Parameters
        ----------
        query_text : Natural language query.
        tokenizer  : Required for TF-IDF routing (to tokenize query).
        method     : "tfidf", "keyword", or "auto" (tries tfidf first).
        """
        if method == "auto":
            # Try TF-IDF first if tokenizer available, fall back to keyword
            if tokenizer is not None and self.window_tokens:
                wid = self._route_tfidf(query_text, tokenizer)
                if wid is not None:
                    return wid
            return self._route_keyword(query_text)
        elif method == "tfidf":
            if tokenizer is None:
                raise ValueError("TF-IDF routing requires a tokenizer")
            return self._route_tfidf(query_text, tokenizer)
        elif method == "keyword":
            return self._route_keyword(query_text)
        else:
            raise ValueError(f"Unknown routing method: {method!r}")

    def route_with_score(
        self, query_text: str, tokenizer=None, method: str = "tfidf"
    ) -> tuple[int | None, float]:
        """Route and return (window_id, score)."""
        if method == "tfidf":
            if tokenizer is None:
                raise ValueError("TF-IDF routing requires a tokenizer")
            router = self._get_tfidf_router(tokenizer)
            query_ids = tokenizer.encode(query_text, add_special_tokens=False)
            return router.route_with_score(query_ids)
        # keyword doesn't have a meaningful score
        wid = self._route_keyword(query_text)
        return wid, (1.0 if wid is not None else 0.0)

    def route_top_k(
        self,
        query_text: str,
        tokenizer,
        k: int = 3,
        expansion_ids: list[int] | None = None,
    ) -> list[int]:
        """Return top-k window IDs by TF-IDF score.

        Runs base query first, then expanded query.  Merges: base
        results have priority; expansion fills remaining slots with
        windows not already selected.  This way expansion helps when
        the base query misses (vocabulary gap) but can't overwrite
        good base matches.
        """
        router = self._get_tfidf_router(tokenizer)
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)

        # Base routing
        base_result = router.route(query_ids, top_k=k)
        if not isinstance(base_result, list):
            base_result = [base_result] if base_result is not None else []

        if not expansion_ids:
            return base_result[:k]

        # Two-pass weighted routing with disambiguation.
        #
        # Pass 1: Score each window as base_score + 0.3 * expansion_score.
        # Pass 2: For expansion tokens shared across top candidates, only the
        #         window with the highest base_score (confidence) keeps them.
        #         This resolves ambiguity without extra LLM calls.
        stopwords = router.stopword_ids
        base_set = set(query_ids) - stopwords
        exp_set = {t for t in expansion_ids if self.idf.get(t, 0.0) > 0 and t not in stopwords}
        exp_only = exp_set - base_set

        if not exp_only:
            return base_result[:k]

        # Pass 1: initial scoring
        window_scores: dict[int, tuple[float, float]] = {}  # wid → (base, exp)
        for wid, tokens in self.window_tokens.items():
            base_score = sum(self.idf.get(t, 0.0) for t in base_set & tokens)
            exp_score = sum(self.idf.get(t, 0.0) for t in exp_only & tokens)
            if base_score > 0 or exp_score > 0:
                window_scores[wid] = (base_score, exp_score)

        if not window_scores:
            return base_result[:k]

        # Find top candidates (2x k to have enough for disambiguation)
        initial = sorted(window_scores.items(),
                         key=lambda x: x[1][0] + 0.3 * x[1][1], reverse=True)
        candidates = [wid for wid, _ in initial[:k * 2]]

        # Pass 2: disambiguate shared expansion tokens
        # For each expansion token, find which candidate windows contain it.
        # If it appears in multiple candidates, only the one with the highest
        # base_score keeps it — the others surrender it.
        token_owner: dict[int, int] = {}  # token_id → winning wid
        for t in exp_only:
            contenders = []
            for wid in candidates:
                if t in self.window_tokens.get(wid, set()):
                    base_s = window_scores[wid][0]
                    contenders.append((base_s, wid))
            if contenders:
                # Highest base score wins. Ties broken by IDF relevance of
                # this token within the window (lower wid as tiebreaker).
                contenders.sort(key=lambda x: (-x[0], x[1]))
                token_owner[t] = contenders[0][1]

        # Re-score candidates with disambiguated expansion
        final_scored: list[tuple[float, int]] = []
        for wid in candidates:
            base_s, _ = window_scores[wid]
            # Only count expansion tokens this window "owns"
            owned_exp = sum(self.idf.get(t, 0.0) for t, owner in token_owner.items()
                           if owner == wid)
            total = base_s + 0.3 * owned_exp
            if total > 0:
                final_scored.append((total, wid))

        if not final_scored:
            return base_result[:k]

        final_scored.sort(reverse=True)
        top_score = final_scored[0][0]
        threshold = top_score * 0.5
        result = [wid for s, wid in final_scored if s >= threshold]
        return result[:k]

    @staticmethod
    def _expand_query(
        query_text: str,
        tokenizer,
        kv_gen,
        n_tokens: int = 30,
    ) -> list[int]:
        """Generate expansion tokens for a query using the model.

        Asks the model: "Keywords that would appear in a document about:
        <query>\\nSpecific words:" and collects *n_tokens* greedy tokens.
        Returns a flat list of token IDs (including case/space variants)
        suitable for adding to the TF-IDF query set.
        """
        import mlx.core as mx

        from ._sampling import sample_token

        prompt = (
            f'User query: "{query_text}"\n'
            f"List synonyms and related technical terms for this query, one per line:\n1."
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_mx = mx.array(prompt_ids)[None]

        logits, kv_store = kv_gen.prefill(prompt_mx)
        mx.eval(logits)
        seq_len = prompt_mx.shape[1]

        expansion: list[int] = []
        for _ in range(n_tokens):
            token = sample_token(logits[0, -1], 0.0)
            expansion.append(token)
            # Add case/space variants (mirrors build-time Pass 3)
            kw_text = tokenizer.decode([token]).strip()
            if kw_text and len(kw_text) >= 2:
                kw_lower = kw_text.lower()
                for variant in [kw_lower, f" {kw_lower}", kw_text, f" {kw_text}"]:
                    var_ids = tokenizer.encode(variant, add_special_tokens=False)
                    expansion.extend(var_ids)
            logits, kv_store = kv_gen.step_uncompiled(
                mx.array([[token]]),
                kv_store,
                seq_len=seq_len,
            )
            seq_len += 1

        return expansion

    def _route_tfidf(self, query_text: str, tokenizer) -> int | None:
        router = self._get_tfidf_router(tokenizer)
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)
        return router.route(query_ids)

    def _route_keyword(self, query_text: str) -> int | None:
        router = self._get_keyword_router()
        return router.route(query_text)

    def _get_tfidf_router(self, tokenizer=None) -> TFIDFRouter:
        if self._tfidf_router is None:
            stopword_ids = set()
            if tokenizer is not None:
                stopword_ids = self._build_stopword_ids(tokenizer)
            self._tfidf_router = TFIDFRouter(self.window_tokens, self.idf, stopword_ids)
        return self._tfidf_router

    @staticmethod
    def _build_stopword_ids(tokenizer) -> set[int]:
        """Build a set of token IDs for common stopwords."""
        from ..research._stopwords import FUNCTION_WORDS

        stopword_ids: set[int] = set()
        for word in FUNCTION_WORDS:
            for variant in [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                stopword_ids.update(ids)
        return stopword_ids

    def route_cosine(
        self,
        query_text: str,
        tokenizer,
        kv_gen,
        k: int = 3,
    ) -> list[int]:
        """Route via cosine similarity — no expansion needed.

        Embeds the query through prefill_to_layer, then computes cosine
        similarity against pre-computed window embeddings. <1ms routing.
        """
        router = self._get_cosine_router()
        if router.matrix is None:
            return []
        query_emb = build_query_embedding(
            kv_gen, tokenizer, query_text, self.config.crystal_layer
        )
        return router.route_window_ids(query_emb, top_k=k)

    def route_synthetic(
        self,
        query_text: str,
        tokenizer,
        kv_gen,
        k: int = 3,
    ) -> list[int]:
        """Route via cosine similarity against synthetic query embeddings.

        Skills were embedded at append time using model-generated query
        variants, placing them in "query space" for better alignment.
        """
        router = self._get_synthetic_router()
        if router.matrix is None:
            return []
        query_emb = build_query_embedding(
            kv_gen, tokenizer, query_text, self.config.crystal_layer
        )
        return router.route_window_ids(query_emb, top_k=k)

    def route_offline(
        self,
        query_text: str,
        tokenizer,
        k: int = 3,
    ) -> list[int]:
        """Route via pure software — no model invocation.

        Uses bag-of-embeddings from the saved embed_matrix.
        Tokenizer is just a lookup table. Routing is <1ms.
        """
        router = self._get_offline_router()
        if router is None:
            return []
        return router.route(query_text, tokenizer, top_k=k)

    def _get_offline_router(self) -> OfflineRouter | None:
        if self._offline_router is None:
            if self._store_path:
                self._offline_router = load_offline_router(self._store_path)
        return self._offline_router

    def _get_synthetic_router(self) -> CosineRouter:
        if self._synthetic_router is None:
            if self._store_path:
                embeddings = load_synthetic_embeddings(self._store_path)
            else:
                embeddings = {}
            self._synthetic_router = CosineRouter(embeddings)
        return self._synthetic_router

    def _get_cosine_router(self) -> CosineRouter:
        if self._cosine_router is None:
            if self._store_path:
                embeddings = load_embeddings(self._store_path)
            else:
                embeddings = {}
            self._cosine_router = CosineRouter(embeddings)
        return self._cosine_router

    def _get_keyword_router(self) -> KeywordRouter:
        if self._keyword_router is None:
            self._keyword_router = KeywordRouter(self.keywords)
        return self._keyword_router

    # ── Window access ─────────────────────────────────────────────────

    def load_boundary(self, window_id: int) -> mx.array:
        """Load boundary residual for a window (the Markov chain link).

        Returns (hidden_dim,) float32 — the cumulative state of windows 0..wid.
        Combined with token IDs, one forward pass reconstructs the full state.
        """
        if window_id in self.boundaries:
            return self.boundaries[window_id]
        if self._store_path is None:
            raise ValueError("No store path — boundaries not available")
        bnd_path = self._store_path / BOUNDARIES_DIR / f"window_{window_id:03d}.npy"
        if not bnd_path.exists():
            raise FileNotFoundError(f"Boundary not found: {bnd_path}")
        np_bnd = np.load(str(bnd_path))
        bnd = mx.array(np_bnd, dtype=mx.float32)
        mx.eval(bnd)
        return bnd

    def load_residual(self, window_id: int) -> mx.array:
        """Load L30 residual stream for a window (lazy, from disk).

        Returns (seq_len, hidden_dim) bfloat16 — the full L30 state.
        For patch_all_positions injection: replaces recipient hidden state.
        """
        # Check in-memory first (during build)
        if self.residual_streams and window_id in self.residual_streams:
            return self.residual_streams[window_id]

        # Load from disk
        if self._store_path is None:
            raise ValueError("No store path — residuals not available")
        res_path = self._store_path / RESIDUALS_DIR / f"window_{window_id:03d}.npy"
        if not res_path.exists():
            raise FileNotFoundError(f"Residual not found: {res_path}")
        np_stream = np.load(str(res_path))
        stream = mx.array(np_stream, dtype=mx.bfloat16)
        mx.eval(stream)
        return stream

    def has_residuals(self) -> bool:
        """Check if residual streams are available (on disk or in memory)."""
        if self.residual_streams:
            return True
        if self._store_path and (self._store_path / RESIDUALS_DIR).exists():
            return True
        return False

    def get_window_text(self, window_id: int, tokenizer) -> str:
        """Decode a window's tokens back to text for donor construction."""
        token_list = self.window_token_lists.get(window_id, [])
        return tokenizer.decode(token_list, skip_special_tokens=True)

    # ── Entry access ──────────────────────────────────────────────────

    def get_entries(self, window_id: int) -> list[InjectionEntry]:
        """Get injection entries for a window, sorted by window position."""
        return sorted(
            [e for e in self.entries if e.window_id == window_id],
            key=lambda e: e.position_in_window,
        )

    def get_entries_for_query(
        self,
        window_id: int,
        query_text: str,
        tokenizer=None,
        max_entries: int = 3,
    ) -> list[InjectionEntry]:
        """Select entries nearest to the routing match position.

        The routing matched a query token at some position in the window.
        The answer is NEAR that position in the text. Select stored
        entries by proximity, excluding query-redundant tokens.

        "porridge" at pos 103 → "John" at pos 98 (distance 5) wins
        over "Pitts" at pos 6 (distance 97).
        """
        all_entries = self.get_entries(window_id)
        if not all_entries:
            return []

        # Get query token IDs
        query_token_ids: set[int] = set()
        if tokenizer is not None:
            query_token_ids = set(tokenizer.encode(query_text, add_special_tokens=False))

        # Find the routing match position in the window
        match_pos: int | None = None
        wt_list = self.window_token_lists.get(window_id)
        if wt_list and query_token_ids:
            # Find the position of the highest-IDF query token in the window
            best_idf = -1.0
            for pos, tid in enumerate(wt_list):
                if tid in query_token_ids:
                    token_idf = self.idf.get(tid, 0.0)
                    if token_idf > best_idf:
                        best_idf = token_idf
                        match_pos = pos

        # Filter: exclude query-redundant tokens
        candidates = [e for e in all_entries if e.token_id not in query_token_ids]
        if not candidates:
            candidates = all_entries  # fallback if all match query

        if match_pos is not None:
            # Sort by proximity to the routing match, preferring BEFORE.
            # In natural text the answer precedes the question context:
            # "John Coyle won the porridge eating championship"
            #  ^^^^^^^^^^(98)          ^^^^^^^(108)
            def _proximity(e):
                dist = e.position_in_window - match_pos
                if dist < 0:  # before the match — likely the answer
                    return abs(dist)
                return abs(dist) + 1000  # after — penalise

            candidates.sort(key=_proximity)
        else:
            # No match position — fall back to highest IDF
            candidates.sort(key=lambda e: self.idf.get(e.token_id, 0.0), reverse=True)

        selected = candidates[:max_entries]
        # Re-sort by window position for correct injection order
        selected.sort(key=lambda e: e.position_in_window)
        return selected

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: Path | str) -> Path:
        """Save knowledge store to a directory (v10 format)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Entries
        if self.entries:
            entry_arr = _entries_to_numpy(self.entries)
            np.savez(str(path / ENTRIES_FILE), entries=entry_arr)
        else:
            np.savez(str(path / ENTRIES_FILE), entries=np.array([], dtype=_ENTRY_DTYPE))

        # Window tokens (unique sets for TF-IDF routing)
        wt_data: dict[str, np.ndarray] = {}
        for wid, tokens in self.window_tokens.items():
            wt_data[str(wid)] = np.array(sorted(tokens), dtype=np.uint32)
        np.savez(str(path / WINDOW_TOKENS_FILE), **wt_data)

        # Window token lists (ordered, for position lookup at query time)
        wtl_data: dict[str, np.ndarray] = {}
        for wid, token_list in self.window_token_lists.items():
            wtl_data[str(wid)] = np.array(token_list, dtype=np.uint32)
        np.savez(str(path / WINDOW_TOKEN_LISTS_FILE), **wtl_data)

        # IDF table
        idf_serializable = {str(k): v for k, v in self.idf.items()}
        (path / IDF_FILE).write_text(json.dumps(idf_serializable, indent=1) + "\n")

        # Keywords
        kw_serializable = {str(k): v for k, v in self.keywords.items()}
        (path / KEYWORDS_FILE).write_text(json.dumps(kw_serializable, indent=1) + "\n")

        # Boundaries (Markov chain — 10 KB each)
        if self.boundaries:
            bnd_dir = path / BOUNDARIES_DIR
            bnd_dir.mkdir(exist_ok=True)
            for wid, bnd in self.boundaries.items():
                bnd_np = np.array(bnd.tolist(), dtype=np.float32)
                np.save(str(bnd_dir / f"window_{wid:03d}.npy"), bnd_np)

        # Boundary residual
        if self.boundary_residual is not None:
            residual_np = np.array(self.boundary_residual.tolist(), dtype=np.float32)
            np.save(str(path / BOUNDARY_RESIDUAL_FILE), residual_np)

        # Residual streams (L30, bfloat16)
        has_residuals = False
        if self.residual_streams:
            res_dir = path / RESIDUALS_DIR
            res_dir.mkdir(exist_ok=True)
            for wid, stream in self.residual_streams.items():
                # stream: (seq_len, hidden_dim) — save as float16 (closest to bfloat16)
                stream_np = np.array(stream.tolist(), dtype=np.float16)
                np.save(str(res_dir / f"window_{wid:03d}.npy"), stream_np)
            has_residuals = True

        # Manifest
        manifest = {
            "version": STORE_VERSION,
            "num_entries": len(self.entries),
            "num_windows": self.num_windows,
            "num_tokens": self.num_tokens,
            "entries_per_window": self.config.entries_per_window,
            "crystal_layer": self.config.crystal_layer,
            "window_size": self.config.window_size,
            "arch_config": self.config.to_dict(),
            "has_residuals": has_residuals,
        }
        (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")

        return path

    @classmethod
    def load(cls, path: Path | str) -> KnowledgeStore:
        """Load knowledge store from a directory."""
        path = Path(path)

        # Detect v9 stores
        if (path / _V9_PASSAGES_FILE).exists() and not (path / ENTRIES_FILE).exists():
            raise ValueError(
                f"Store at {path} is v9 format (crystallised residuals). "
                "Rebuild required for v10 (injection entries). "
                "Use: lazarus knowledge build --input <doc> --output <store>"
            )

        # Manifest
        manifest = json.loads((path / MANIFEST_FILE).read_text())
        config = ArchitectureConfig.from_dict(manifest["arch_config"])

        # Entries
        npz = np.load(str(path / ENTRIES_FILE), allow_pickle=False)
        entries = _numpy_to_entries(npz["entries"]) if len(npz["entries"]) > 0 else []

        # Window tokens (unique sets)
        window_tokens: dict[int, set[int]] = {}
        if (path / WINDOW_TOKENS_FILE).exists():
            wt_npz = np.load(str(path / WINDOW_TOKENS_FILE), allow_pickle=False)
            for key in wt_npz.files:
                window_tokens[int(key)] = {int(t) for t in wt_npz[key]}

        # Window token lists (ordered, for position lookup)
        window_token_lists: dict[int, list[int]] = {}
        if (path / WINDOW_TOKEN_LISTS_FILE).exists():
            wtl_npz = np.load(str(path / WINDOW_TOKEN_LISTS_FILE), allow_pickle=False)
            for key in wtl_npz.files:
                window_token_lists[int(key)] = [int(t) for t in wtl_npz[key]]

        # IDF
        idf: dict[int, float] = {}
        if (path / IDF_FILE).exists():
            idf_raw = json.loads((path / IDF_FILE).read_text())
            idf = {int(k): float(v) for k, v in idf_raw.items()}

        # Keywords
        keywords: dict[int, list[str]] = {}
        if (path / KEYWORDS_FILE).exists():
            kw_raw = json.loads((path / KEYWORDS_FILE).read_text())
            keywords = {int(k): v for k, v in kw_raw.items()}

        # Boundary residual
        boundary_residual = None
        if (path / BOUNDARY_RESIDUAL_FILE).exists():
            residual_np = np.load(str(path / BOUNDARY_RESIDUAL_FILE))
            boundary_residual = mx.array(residual_np, dtype=mx.float32)
            if boundary_residual.ndim < 3:
                boundary_residual = boundary_residual.reshape(1, 1, -1)
            mx.eval(boundary_residual)

        store = cls(
            entries=entries,
            window_tokens=window_tokens,
            window_token_lists=window_token_lists,
            idf=idf,
            keywords=keywords,
            config=config,
            boundary_residual=boundary_residual,
            num_windows=manifest.get("num_windows", 0),
            num_tokens=manifest.get("num_tokens", 0),
        )
        store._store_path = path
        return store

    def reload_index(self) -> None:
        """Re-read the index from disk, picking up any appended skills.

        This is cheap (~1ms for a ~500KB index on SSD) and should be called
        before every query in a daemon/agentic loop to see newly appended
        skills without restarting the model.
        """
        if self._store_path is None:
            return

        path = self._store_path

        # Window tokens
        if (path / WINDOW_TOKENS_FILE).exists():
            wt_npz = np.load(str(path / WINDOW_TOKENS_FILE), allow_pickle=False)
            self.window_tokens = {int(k): {int(t) for t in wt_npz[k]} for k in wt_npz.files}

        # Window token lists
        if (path / WINDOW_TOKEN_LISTS_FILE).exists():
            wtl_npz = np.load(str(path / WINDOW_TOKEN_LISTS_FILE), allow_pickle=False)
            self.window_token_lists = {int(k): [int(t) for t in wtl_npz[k]] for k in wtl_npz.files}

        # IDF
        if (path / IDF_FILE).exists():
            idf_raw = json.loads((path / IDF_FILE).read_text())
            self.idf = {int(k): float(v) for k, v in idf_raw.items()}

        # Keywords
        if (path / KEYWORDS_FILE).exists():
            kw_raw = json.loads((path / KEYWORDS_FILE).read_text())
            self.keywords = {int(k): v for k, v in kw_raw.items()}

        # Entries
        if (path / ENTRIES_FILE).exists():
            npz = np.load(str(path / ENTRIES_FILE), allow_pickle=False)
            self.entries = _numpy_to_entries(npz["entries"]) if len(npz["entries"]) > 0 else []

        # Manifest (update num_windows/num_tokens)
        if (path / MANIFEST_FILE).exists():
            manifest = json.loads((path / MANIFEST_FILE).read_text())
            self.num_windows = manifest.get("num_windows", self.num_windows)
            self.num_tokens = manifest.get("num_tokens", self.num_tokens)

        # Invalidate cached routers so they rebuild with new data
        self._tfidf_router = None
        self._keyword_router = None
        self._cosine_router = None
        self._synthetic_router = None
        self._offline_router = None

    def log_stats(self, file=sys.stderr) -> None:
        entry_bytes = len(self.entries) * 14  # 14 bytes per entry (position_in_window is uint16)
        wt_bytes = sum(len(t) * 2 for t in self.window_tokens.values())
        idf_bytes = len(self.idf) * 12  # ~12 bytes per entry (key + float)
        kw_bytes = sum(sum(len(k) for k in kws) for kws in self.keywords.values())
        total_bytes = entry_bytes + wt_bytes + idf_bytes + kw_bytes

        print(
            f"  KnowledgeStore v{STORE_VERSION}: "
            f"{len(self.entries)} entries across {self.num_windows} windows  "
            f"entries={entry_bytes / 1024:.1f}KB  "
            f"tokens={wt_bytes / 1024:.1f}KB  "
            f"idf={idf_bytes / 1024:.1f}KB  "
            f"keywords={kw_bytes / 1024:.1f}KB  "
            f"total~{total_bytes / 1024:.1f}KB  "
            f"doc={self.num_tokens} tokens  "
            f"crystal=L{self.config.crystal_layer}  "
            f"window={self.config.window_size}",
            file=file,
        )
