"""
Sparse Index Engine — Mode 5.

Extends UnlimitedContextEngine (Mode 4) with surprise-aware keyword
extraction during prefill and prompt-based retrieval during generation.

The key insight: extract only what the model DOESN'T know. The model's
own surprise (prediction rank) at each token during prefill is the signal.
High surprise = novel = extract. Low surprise = parametric = skip.

During document processing:
  - Runs the existing window-based prefill (Mode 4 checkpoint chain)
  - After each window closes, computes per-token surprise ranks from
    the logits already computed during prefill (zero additional compute)
  - Extracts keywords only from novel/semi-parametric token positions
  - Formats as entity:fact triplets and accumulates into sparse index

During generation:
  - Renders the sparse index as a keyword prompt
  - Runs standard inference (no replay, no injection, no routing)
  - Falls back to Mode 4 replay when use_sparse=False

Usage:
    engine = SparseIndexEngine.from_pretrained("google/gemma-3-4b-it")
    engine.process_document("apollo11_transcript.txt")
    engine.save_index("apollo11.idx")

    answer = engine.generate_from_index("Where was Armstrong from?")
    answer = engine.generate(query_ids, replay_window_ids=[0, 5])  # Mode 4 fallback
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

from .sparse_index import (
    FUNCTION_WORDS,
    EntityExtractor,
    FactSpan,
    SparseEntry,
    SparseSemanticIndex,
    SurpriseClassifier,
    extract_content_words,
)
from .unlimited_engine import UnlimitedContextEngine


class SparseIndexEngine(UnlimitedContextEngine):
    """Mode 5: Sparse semantic index with optional Mode 4 fallback.

    Inherits all Mode 4 capabilities (windowed prefill, checkpoint chain,
    replay) and adds surprise-aware extraction and prompt-based generation.
    """

    def __init__(
        self,
        rs_model,
        config,
        window_size: int = 512,
        model_id: str = "",
        config_hash: str = "",
        max_keywords: int = 8,
        novel_rank_threshold: int = 50,
        context_window: int = 2,
    ):
        super().__init__(
            rs_model,
            config,
            window_size=window_size,
            model_id=model_id,
            config_hash=config_hash,
        )
        self.extractor = EntityExtractor(
            max_keywords=max_keywords,
            context_window=context_window,
            novel_rank_threshold=novel_rank_threshold,
        )
        self.classifier = SurpriseClassifier(novel_threshold=novel_rank_threshold)
        self.sparse_index = SparseSemanticIndex()
        self._tokenizer = None
        self._last_logits: mx.array | None = None  # captured from prefill

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for decode (tokens → text for extraction)."""
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "google/gemma-3-4b-it",
        window_size: int = 512,
        max_keywords: int = 8,
        **kwargs,
    ) -> SparseIndexEngine:
        """Load model and create engine in one step."""
        from ...inference import UnifiedPipeline

        pipeline = UnifiedPipeline.from_pretrained(model_id)
        engine = cls(
            pipeline.model,
            pipeline.config,
            window_size=window_size,
            model_id=model_id,
            max_keywords=max_keywords,
            **kwargs,
        )
        engine.set_tokenizer(pipeline.tokenizer)
        engine._pipeline = pipeline
        return engine

    # ------------------------------------------------------------------
    # Override: capture logits during prefill
    # ------------------------------------------------------------------

    def _extend_current_window(self, token_ids: list[int]) -> None:
        """Override: capture logits for surprise classification."""
        if not token_ids:
            return

        ids = mx.array(token_ids)[None]

        if self.kv_store is None:
            if self.current_window_id == 0:
                logits, self.kv_store, self._last_residual = self.kv_gen.prefill_with_residual(ids)
            else:
                prior_kv, _prior_abs = self.checkpoints.load(self.current_window_id - 1)
                logits, self.kv_store, self._last_residual = self.kv_gen.extend_with_residual(
                    ids, prior_kv, abs_start=self.abs_offset
                )
        else:
            abs_start = self.abs_offset + self.hot_len
            logits, self.kv_store, self._last_residual = self.kv_gen.extend_with_residual(
                ids, self.kv_store, abs_start=abs_start
            )

        mx.eval(logits)

        # Capture logits for surprise classification
        # Accumulate across chunks within the same window
        if self._last_logits is None:
            self._last_logits = logits
        else:
            self._last_logits = mx.concatenate([self._last_logits, logits], axis=1)

        self.hot_len += len(token_ids)
        self.current_window_tokens.extend(token_ids)

    def _close_window(self) -> None:
        """Override: extract keywords with surprise before closing."""
        window_id = self.current_window_id
        tokens = list(self.current_window_tokens)
        logits = self._last_logits

        # Run Mode 4 close (checkpoint + archive)
        # Save _last_logits before super() clears state
        self._last_logits = None
        super()._close_window()

        # Extract keywords using surprise-aware pipeline
        if self._tokenizer is not None and logits is not None:
            self._extract_with_surprise(window_id, tokens, logits)

    # ------------------------------------------------------------------
    # Surprise-aware extraction
    # ------------------------------------------------------------------

    def _extract_with_surprise(
        self,
        window_id: int,
        token_ids: list[int],
        logits: mx.array,
    ) -> None:
        """Extract keywords from a window using surprise ranks from logits.

        The logits are already computed during prefill — zero additional
        model computation. We just rank how surprised the model was at
        each token and only extract the novel ones.

        Steps:
            1. Compute per-token prediction rank from logits
            2. Classify each token: parametric (rank≤2), semi (3-50), novel (51+)
            3. Decode tokens to text
            4. Extract keywords only near novel token positions
            5. Format as entity:fact triplets
            6. Accumulate into index
        """
        if len(token_ids) < 2:
            self.sparse_index.add(SparseEntry(window_id=window_id, keywords=[]))
            return

        # Step 1: compute per-token surprise ranks
        # logits[0, i, :] predicts token at position i+1
        logits_f32 = logits[0].astype(mx.float32)
        mx.eval(logits_f32)

        skip = min(32, len(token_ids) - 2)  # skip boundary artifacts
        n_score = len(token_ids) - 1 - skip

        if n_score <= 0:
            self.sparse_index.add(SparseEntry(window_id=window_id, keywords=[]))
            return

        actual_ids = mx.array(token_ids[skip + 1 :])
        logits_slice = logits_f32[skip : skip + n_score]
        actual_logits = logits_slice[mx.arange(n_score), actual_ids]
        ranks = mx.sum(logits_slice > actual_logits[:, None], axis=1)
        mx.eval(ranks)

        # Build full rank array (0 for skipped positions)
        full_ranks = [0] * (skip + 1) + [int(r) for r in ranks.tolist()]
        # Pad to match token count
        while len(full_ranks) < len(token_ids):
            full_ranks.append(0)

        max_rank = max(full_ranks) if full_ranks else 0

        # Step 2: decode tokens to text
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        token_texts = [self._tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]

        # Step 3: extract keywords near novel positions
        keywords = self._extract_novel_keywords(text, token_texts, full_ranks)

        # Step 3b: extract content words (stopword-filtered, merged subwords)
        content_words = extract_content_words(token_ids, self._tokenizer)

        # Step 4: extract fact spans — top-N most surprising token positions
        n_spans = min(8, len(token_ids))
        ranked_positions = sorted(
            range(len(full_ranks)),
            key=lambda i: -full_ranks[i],
        )
        # Deduplicate: skip positions within ±5 of an already-selected one
        fact_spans = []
        selected_positions: set[int] = set()
        for pos in ranked_positions:
            if full_ranks[pos] < 3:
                break  # below semi-parametric threshold
            if any(abs(pos - sp) <= 5 for sp in selected_positions):
                continue
            fact_spans.append(FactSpan(position=pos, radius=5))
            selected_positions.add(pos)
            if len(fact_spans) >= n_spans:
                break

        # Step 5: accumulate
        entry = SparseEntry(
            window_id=window_id,
            keywords=keywords,
            content_words=content_words,
            surprise_rank=max_rank,
            fact_spans=fact_spans,
        )
        self.sparse_index.add(entry)

    def _extract_novel_keywords(
        self,
        text: str,
        token_texts: list[str],
        token_ranks: list[int],
    ) -> list[str]:
        """Extract keywords from the most surprising tokens.

        Priority: highest surprise rank first. For each surprising token,
        capture ±context words to form a fact triplet.

        This inverts the original approach: instead of finding capitalised
        words near novel positions (which grabs transcript noise like
        "CDR", "Roger"), we find the most SURPRISING tokens and build
        context around THEM.
        """
        text = " ".join(text.split())
        words = text.split()
        if not words:
            return []

        novel_threshold = self.extractor.novel_rank_threshold
        max_kw = self.extractor.max_keywords
        ctx_w = self.extractor.context_window

        # Build (token_index, rank, token_text) sorted by surprise
        ranked = []
        for i, rank in enumerate(token_ranks):
            if rank > novel_threshold and i < len(token_texts):
                tok = token_texts[i].strip()
                if len(tok) > 1:  # skip single chars
                    ranked.append((i, rank, tok))
        ranked.sort(key=lambda x: -x[1])  # highest surprise first

        if not ranked:
            return []

        keywords: list[str] = []
        seen: set[str] = set()
        used_positions: set[int] = set()

        def _add(kw: str) -> bool:
            kw = " ".join(kw.split())
            # Skip pure noise: timestamps, speaker codes, OCR artifacts
            if not kw or kw.lower() in seen:
                return False
            if len(kw) < 3:
                return False
            if len(keywords) >= max_kw:
                return False
            seen.add(kw.lower())
            keywords.append(kw)
            return True

        # For each surprising token, build a context triplet
        for tok_idx, rank, tok in ranked:
            if tok_idx in used_positions:
                continue
            if len(keywords) >= max_kw:
                break

            # Skip function words even if surprising
            if tok.lower() in FUNCTION_WORDS:
                continue
            # Skip bare punctuation/numbers-only
            clean = tok.strip(".,;:!?()[]{}\"'-_*/\\")
            if not clean or (clean.isdigit() and len(clean) < 3):
                continue

            # Find this token's approximate position in the word list
            # by matching character offset
            char_pos = 0
            for j in range(tok_idx):
                if j < len(token_texts):
                    char_pos += len(token_texts[j])

            # Find the word index closest to this character position
            word_char = 0
            word_idx = 0
            for wi, w in enumerate(words):
                if word_char + len(w) >= char_pos:
                    word_idx = wi
                    break
                word_char += len(w) + 1  # +1 for space

            # Capture ±context_window words
            start = max(0, word_idx - ctx_w)
            end = min(len(words), word_idx + ctx_w + 1)
            context_words = words[start:end]

            # Strip function words from context but keep content words
            filtered = [
                w
                for w in context_words
                if (w.lower() not in FUNCTION_WORDS and len(w) > 1) or w == words[word_idx]
            ]  # always keep the anchor

            if filtered:
                triplet = " ".join(filtered)
                _add(triplet)
                # Mark nearby positions as used to avoid overlap
                for p in range(max(0, tok_idx - 3), min(len(token_ranks), tok_idx + 4)):
                    used_positions.add(p)

        return keywords[:max_kw]

    # ------------------------------------------------------------------
    # Post-hoc extraction (for existing libraries without logits)
    # ------------------------------------------------------------------

    def extract_all_windows(self) -> None:
        """Re-extract keywords from all archived windows.

        Uses fresh forward passes to compute surprise. More expensive
        than inline extraction but works on existing libraries.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")

        self.sparse_index = SparseSemanticIndex()
        num_archived = len(self.archive)
        t0 = time.time()

        for wid in range(num_archived):
            w_tokens, _abs = self.archive.retrieve(wid)

            if len(w_tokens) < 2:
                self.sparse_index.add(SparseEntry(window_id=wid, keywords=[]))
                continue

            # Fresh forward pass for logits
            ids = mx.array(w_tokens)[None]
            logits, _kv = self.kv_gen.prefill(ids)
            mx.eval(logits)

            self._extract_with_surprise(wid, w_tokens, logits)

            if (wid + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (wid + 1) / elapsed
                remaining = (num_archived - wid - 1) / rate
                print(
                    f"\r  Sparse extraction: {wid + 1}/{num_archived} "
                    f"({elapsed:.0f}s, ~{remaining:.0f}s left)  ",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

        elapsed = time.time() - t0
        stats = self.sparse_index.stats()
        print(
            f"\r  Sparse extraction: {num_archived} windows in {elapsed:.1f}s — "
            f"{stats['non_empty']}/{num_archived} with keywords, "
            f"{stats['total_keywords']} total    ",
            file=sys.stderr,
            flush=True,
        )
        print(file=sys.stderr)

    # ------------------------------------------------------------------
    # Document processing — high-level API
    # ------------------------------------------------------------------

    def process_document(
        self,
        path: str | Path,
        max_tokens: int | None = None,
    ) -> None:
        """Process a text document: tokenize, prefill windows, extract index."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        text = Path(path).read_text(encoding="utf-8")
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)

        if max_tokens is not None:
            token_ids = token_ids[:max_tokens]

        print(
            f"Processing: {len(token_ids):,} tokens, "
            f"window_size={self.window_size}, "
            f"~{len(token_ids) // self.window_size} windows",
            file=sys.stderr,
        )

        t0 = time.time()
        self.process(token_ids)
        self.flush()
        elapsed = time.time() - t0

        stats = self.sparse_index.stats()
        print(
            f"Done: {len(self.archive)} windows in {elapsed:.1f}s — "
            f"{stats['total_keywords']} keywords extracted "
            f"({stats['non_empty']} non-empty windows)",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Generation — Mode 5 (sparse prompt) or Mode 4 fallback
    # ------------------------------------------------------------------

    def generate_from_index(
        self,
        query: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        max_keywords: int | None = None,
        chat_template: bool = True,
    ) -> str:
        """Generate answer using the sparse index as context.

        No window replay. No residual injection. Just prompt + inference.
        """
        if not self.sparse_index.entries:
            raise RuntimeError("No sparse index.")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        prompt = self.sparse_index.render_prompt(
            query,
            max_keywords=max_keywords,
            chat_template=chat_template,
        )

        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        ids = mx.array(prompt_ids)[None]

        logits, kv = self.kv_gen.prefill(ids)
        mx.eval(logits)
        seq_len = len(prompt_ids)

        stop_ids: set[int] = set()
        if self._tokenizer.eos_token_id is not None:
            stop_ids.add(self._tokenizer.eos_token_id)

        generated: list[int] = []
        for _ in range(max_new_tokens):
            last_logits = logits[0, -1]
            if temperature == 0.0:
                next_token = int(mx.argmax(last_logits).item())
            else:
                scaled = last_logits / temperature
                next_token = int(mx.random.categorical(scaled[None]).item())
            if next_token in stop_ids:
                break
            generated.append(next_token)
            logits, kv = self.kv_gen.step_uncompiled(mx.array([[next_token]]), kv, seq_len=seq_len)
            seq_len += 1

        return self._tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """Save sparse index to disk."""
        self.sparse_index.save(path)
        size = Path(path).stat().st_size
        print(
            f"Saved sparse index: {path} ({size:,} bytes, "
            f"{self.sparse_index.non_empty_count} entries)",
            file=sys.stderr,
        )

    def load_index(self, path: str | Path) -> None:
        """Load sparse index from disk."""
        self.sparse_index = SparseSemanticIndex.load(path)
        stats = self.sparse_index.stats()
        print(
            f"Loaded sparse index: {path} "
            f"({stats['num_entries']} entries, "
            f"{stats['total_keywords']} keywords)",
            file=sys.stderr,
        )
