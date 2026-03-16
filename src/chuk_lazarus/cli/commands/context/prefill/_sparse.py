"""Sparse semantic index extraction — fallback path.

This runs when `--phases sparse` is used on an existing library
(without SparseIndexEngine inline extraction). It runs a forward
pass per window to compute surprise, then uses the same
surprise-first extraction as SparseIndexEngine.

For new prefills with `--phases windows,sparse`, the inline path
in SparseIndexEngine is used instead (zero extra compute).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx


def extract_sparse(
    engine,
    tokenizer,
    output_path: Path,
    num_archived: int,
    max_keywords: int = 8,
) -> None:
    """Extract surprise-aware keyword index via separate forward passes."""
    from .....inference.context.sparse_index import (
        FUNCTION_WORDS,
        EntityExtractor,
        FactSpan,
        SparseEntry,
        SparseSemanticIndex,
    )

    extractor = EntityExtractor(max_keywords=max_keywords)
    idx = SparseSemanticIndex()
    kv_gen = engine.kv_gen
    novel_threshold = extractor.novel_rank_threshold
    ctx_w = extractor.context_window

    t0 = time.time()
    for wid in range(num_archived):
        w_tokens, _w_abs = engine.archive.retrieve(wid)

        if len(w_tokens) < 2:
            idx.add(SparseEntry(window_id=wid, keywords=[]))
            continue

        # Forward pass for logits
        ids = mx.array(w_tokens)[None]
        logits, _kv = kv_gen.prefill(ids)
        mx.eval(logits)

        # Compute per-token surprise ranks
        logits_f32 = logits[0].astype(mx.float32)
        mx.eval(logits_f32)

        skip = min(32, len(w_tokens) - 2)
        n_score = len(w_tokens) - 1 - skip

        if n_score <= 0:
            idx.add(SparseEntry(window_id=wid, keywords=[]))
            continue

        actual_ids = mx.array(w_tokens[skip + 1:])
        logits_slice = logits_f32[skip:skip + n_score]
        actual_logits = logits_slice[mx.arange(n_score), actual_ids]
        ranks = mx.sum(logits_slice > actual_logits[:, None], axis=1)
        mx.eval(ranks)

        full_ranks = [0] * (skip + 1) + [int(r) for r in ranks.tolist()]
        while len(full_ranks) < len(w_tokens):
            full_ranks.append(0)

        max_rank = max(full_ranks) if full_ranks else 0

        # Decode
        text = tokenizer.decode(w_tokens, skip_special_tokens=True)
        text = " ".join(text.split())
        words = text.split()
        token_texts = [tokenizer.decode([tid], skip_special_tokens=True)
                       for tid in w_tokens]

        # Surprise-first extraction: highest rank tokens get priority
        ranked = []
        for i, rank in enumerate(full_ranks):
            if rank > novel_threshold and i < len(token_texts):
                tok = token_texts[i].strip()
                if len(tok) > 1:
                    ranked.append((i, rank, tok))
        ranked.sort(key=lambda x: -x[1])

        if not ranked:
            idx.add(SparseEntry(window_id=wid, keywords=[], surprise_rank=max_rank))
            continue

        keywords: list[str] = []
        seen: set[str] = set()
        used_positions: set[int] = set()

        for tok_idx, rank, tok in ranked:
            if tok_idx in used_positions or len(keywords) >= max_keywords:
                break
            if tok.lower() in FUNCTION_WORDS:
                continue
            clean = tok.strip('.,;:!?()[]{}"\'-_*/\\')
            if not clean or (clean.isdigit() and len(clean) < 3):
                continue

            # Find word index from character offset
            char_pos = sum(len(token_texts[j]) for j in range(tok_idx))
            word_char = 0
            word_idx = 0
            for wi, w in enumerate(words):
                if word_char + len(w) >= char_pos:
                    word_idx = wi
                    break
                word_char += len(w) + 1

            # Capture ±context
            start = max(0, word_idx - ctx_w)
            end = min(len(words), word_idx + ctx_w + 1)
            filtered = [w for w in words[start:end]
                        if (w.lower() not in FUNCTION_WORDS and len(w) > 1)
                        or w == words[word_idx]]

            if filtered:
                triplet = " ".join(filtered)
                if triplet.lower() not in seen and len(triplet) >= 3:
                    seen.add(triplet.lower())
                    keywords.append(triplet)
                    for p in range(max(0, tok_idx - 3), min(len(full_ranks), tok_idx + 4)):
                        used_positions.add(p)

        # Extract fact spans — top-N most surprising positions, deduplicated
        n_spans = min(8, len(w_tokens))
        span_candidates = sorted(range(len(full_ranks)), key=lambda i: -full_ranks[i])
        fact_spans = []
        span_positions: set[int] = set()
        for pos in span_candidates:
            if full_ranks[pos] < 3:
                break
            if any(abs(pos - sp) <= 5 for sp in span_positions):
                continue
            fact_spans.append(FactSpan(position=pos, radius=5))
            span_positions.add(pos)
            if len(fact_spans) >= n_spans:
                break

        idx.add(SparseEntry(
            window_id=wid,
            keywords=keywords[:max_keywords],
            surprise_rank=max_rank,
            fact_spans=fact_spans,
        ))

        if (wid + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (wid + 1) / elapsed
            remaining = (num_archived - wid - 1) / rate
            novel_count = sum(1 for e in idx.entries if e.keywords)
            print(
                f"\r  Sparse index: {wid + 1}/{num_archived} windows "
                f"({novel_count} with novel content, "
                f"{elapsed:.0f}s, ~{remaining:.0f}s left)  ",
                end="", file=sys.stderr, flush=True,
            )

    elapsed = time.time() - t0
    idx.save(output_path / "sparse_index.json")

    stats = idx.stats()
    size_kb = (output_path / "sparse_index.json").stat().st_size / 1024
    parametric_count = sum(1 for e in idx.entries if not e.keywords)

    print(
        f"\r  Sparse index: {num_archived} windows in {elapsed:.1f}s — "
        f"{stats['non_empty']} novel, {parametric_count} parametric, "
        f"{stats['total_keywords']} keywords ({stats['avg_keywords']:.1f}/window), "
        f"{size_kb:.0f} KB          ",
        file=sys.stderr, flush=True,
    )
    print(file=sys.stderr)
