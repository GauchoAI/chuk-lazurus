"""Sparse two-pass generate mode — Mode 5 targeted replay.

Pass 1: Sparse index prepended to query → model answers with facts,
        citing window IDs (e.g. "W464").
Pass 2: If verbatim content needed, extract window ID from Pass 1
        response, replay that single window via Mode 4, re-query
        with full transcript text.

Usage:
    lazarus context generate --checkpoint ./lib --prompt "quote exact words" --replay sparse
"""

from __future__ import annotations

import re
import sys

from ......inference.context.sparse_index import SparseSemanticIndex

# -----------------------------------------------------------------------
# Verbatim detection
# -----------------------------------------------------------------------

VERBATIM_TRIGGERS = [
    "quote",
    "exact words",
    "exactly what",
    "word for word",
    "verbatim",
    "actual words",
    "precise wording",
    "transcript says",
    "read from the transcript",
    "what did they actually say",
    "what were the words",
    "tell me what was said",
]

DETAIL_TRIGGERS = [
    "describe",
    "detail",
    "explain",
    "tell me more",
    "what happened",
    "full story",
    "elaborate",
]


def needs_verbatim(query: str) -> bool:
    """Check if the query requests verbatim/quoted content."""
    q = query.lower()
    return any(trigger in q for trigger in VERBATIM_TRIGGERS)


def needs_detail(query: str) -> bool:
    """Check if the query requests detailed content beyond keywords."""
    q = query.lower()
    return any(trigger in q for trigger in DETAIL_TRIGGERS)


def should_auto_replay(query: str, response: str) -> bool:
    """Heuristic: should we auto-trigger Pass 2?"""
    if needs_verbatim(query):
        return True
    if needs_detail(query):
        return True
    # Response references a window but is very short
    if re.search(r"W\d+", response) and len(response.split()) < 20:
        return True
    return False


# -----------------------------------------------------------------------
# Window ID extraction
# -----------------------------------------------------------------------


def extract_window_ids(response: str) -> list[int]:
    """Extract window IDs from the model's response."""
    patterns = [
        r"W(\d+)",
        r"[Ww]indow\s*(\d+)",
        r"\(W(\d+)\)",
        r"\[W(\d+)\]",
    ]
    ids: set[int] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, response):
            ids.add(int(match.group(1)))
    return sorted(ids)


def find_windows_for_entity(
    sparse_idx: SparseSemanticIndex,
    entity_name: str,
) -> list[int]:
    """Find which windows contain entries mentioning an entity."""
    windows: list[int] = []
    entity_lower = entity_name.lower()
    for entry in sparse_idx.entries:
        for kw in entry.keywords:
            if entity_lower in kw.lower():
                windows.append(entry.window_id)
                break
    return sorted(set(windows))


def extract_entities_from_query(query: str) -> list[str]:
    """Extract capitalised entity names from the query."""
    entities: list[str] = []
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", query):
        w = m.group(1)
        if w.lower() not in {
            "what",
            "where",
            "when",
            "who",
            "how",
            "the",
            "quote",
            "find",
            "tell",
            "describe",
        }:
            entities.append(w)
    return entities


# -----------------------------------------------------------------------
# Two-pass engine
# -----------------------------------------------------------------------


def run_sparse_twopass(
    lib,
    kv_gen,
    engine,
    tokenizer,
    prompt_text: str,
    config,
    mx,
    max_keywords: int | None = None,
    max_replay_windows: int = 3,
):
    """Two-pass sparse retrieval with targeted replay.

    Pass 1: Generate from sparse index → get factual answer + window IDs.
    Pass 2: If verbatim needed, replay cited windows → generate from full text.

    Returns GenerateResult.
    """
    from ..._types import GenerateResult

    # Load sparse index
    index_path = lib.path / "sparse_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"sparse_index.json not found in {lib.path}")

    sparse_idx = SparseSemanticIndex.load(index_path)
    stats = sparse_idx.stats()

    # ------------------------------------------------------------------
    # Pass 1: Generate from sparse index
    # ------------------------------------------------------------------
    print(f"  Pass 1: sparse index ({stats['num_entries']} entries)", file=sys.stderr)

    prompt = sparse_idx.render_prompt(
        prompt_text,
        max_keywords=max_keywords or 3,
        chat_template=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"    {len(prompt_ids)} prompt tokens", file=sys.stderr)

    ids = mx.array(prompt_ids)[None]
    logits, kv = kv_gen.prefill(ids)
    mx.eval(logits)
    seq_len = len(prompt_ids)

    # Decode Pass 1
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    pass1_tokens: list[int] = []
    for _ in range(200):  # cap Pass 1 at 200 tokens
        last_logits = logits[0, -1]
        next_token = int(mx.argmax(last_logits).item())
        if next_token in stop_ids:
            break
        pass1_tokens.append(next_token)
        logits, kv = kv_gen.step_uncompiled(
            mx.array([[next_token]]),
            kv,
            seq_len=seq_len,
        )
        seq_len += 1

    pass1_response = tokenizer.decode(pass1_tokens, skip_special_tokens=True)
    print(f"    Pass 1 response: {pass1_response[:120]}...", file=sys.stderr)

    # ------------------------------------------------------------------
    # Check if Pass 2 needed
    # ------------------------------------------------------------------
    if not should_auto_replay(prompt_text, pass1_response):
        print("  Pass 2: not needed (factual answer sufficient)", file=sys.stderr)
        sys.stdout.write(pass1_response)
        sys.stdout.write("\n")
        return GenerateResult(
            response=pass1_response,
            tokens_generated=len(pass1_tokens),
            context_tokens=len(prompt_ids),
        )

    # ------------------------------------------------------------------
    # Pass 2: Targeted replay
    # ------------------------------------------------------------------
    print("  Pass 2: targeted replay (verbatim requested)", file=sys.stderr)

    # Extract window IDs from Pass 1 response
    window_ids = extract_window_ids(pass1_response)

    # Fallback: search index for entities mentioned in query
    if not window_ids:
        entities = extract_entities_from_query(prompt_text)
        for entity in entities:
            window_ids.extend(find_windows_for_entity(sparse_idx, entity))
        if window_ids:
            print(f"    Entity lookup: {entities} → windows {window_ids[:5]}", file=sys.stderr)

    if not window_ids:
        # Last fallback: use sparse BM25 routing
        from ...compass_routing._sparse import _sparse_score_windows

        scores = _sparse_score_windows(lib, prompt_text)
        window_ids = [wid for wid, _ in scores[:max_replay_windows]]
        print(f"    BM25 fallback → windows {window_ids}", file=sys.stderr)

    # Limit replay windows
    window_ids = window_ids[:max_replay_windows]
    print(f"    Replaying windows: {window_ids}", file=sys.stderr)

    # Build framed context: preamble + replay windows + postamble
    preamble_text = (
        "<start_of_turn>user\n"
        "You are answering questions based on the document transcript below. "
        "Quote exact text when possible.\n\n"
        "Here is the relevant transcript:\n\n"
    )
    preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=True)

    postamble_text = (
        f"\n\n---\nBased on the transcript above, {prompt_text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    postamble_ids = tokenizer.encode(postamble_text, add_special_tokens=False)

    # Prefill preamble
    pre_ids = mx.array(preamble_ids)[None]
    logits2, kv2 = kv_gen.prefill(pre_ids)
    mx.eval(logits2)
    seq_len2 = len(preamble_ids)

    # Replay each window using library tokens (same as standard mode)
    import time as _time

    for wid in sorted(window_ids):
        w_tokens = lib.get_window_tokens(wid)
        w_ids = mx.array(w_tokens)[None]
        t0 = _time.time()
        logits2, kv2 = kv_gen.extend(w_ids, kv2, abs_start=seq_len2)
        mx.eval(*[t for pair in kv2 for t in pair])
        elapsed_ms = (_time.time() - t0) * 1000
        seq_len2 += len(w_tokens)
        print(
            f"    Replayed W{wid} @ pos {seq_len2 - len(w_tokens)}–{seq_len2} ({elapsed_ms:.0f}ms)",
            file=sys.stderr,
        )

    # Extend with postamble
    post_ids = mx.array(postamble_ids)[None]
    logits2, kv2 = kv_gen.extend(post_ids, kv2, abs_start=seq_len2)
    seq_len2 += len(postamble_ids)

    # Decode Pass 2
    pass2_tokens: list[int] = []
    for _ in range(config.max_tokens):
        last_logits = logits2[0, -1]
        if config.temperature == 0.0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            scaled = last_logits / config.temperature
            next_token = int(mx.random.categorical(scaled[None]).item())
        if next_token in stop_ids:
            break
        pass2_tokens.append(next_token)
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()
        logits2, kv2 = kv_gen.step_uncompiled(
            mx.array([[next_token]]),
            kv2,
            seq_len=seq_len2,
        )
        seq_len2 += 1

    print()  # newline

    pass2_response = tokenizer.decode(pass2_tokens, skip_special_tokens=True)

    return GenerateResult(
        response=pass2_response,
        tokens_generated=len(pass2_tokens),
        context_tokens=seq_len2,
    )
