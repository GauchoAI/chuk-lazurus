"""Sparse index generate mode — Mode 5.

Loads the sparse keyword index from the library and uses it as the
entire context. No window replay. No checkpoint loading. Just the
index rendered as a text prompt prepended to the query.

Usage:
    lazarus context generate --checkpoint ./lib --prompt "question" --replay sparse
"""

from __future__ import annotations

import sys

from ......inference.context.sparse_index import SparseSemanticIndex


def run_sparse(
    lib,
    kv_gen,
    tokenizer,
    prompt_text: str,
    config,
    mx,
    max_keywords: int | None = None,
):
    """Build context from sparse index, return (kv, seq_len) for decode.

    Args:
        lib: CheckpointLibrary with sparse_index.json
        kv_gen: KV generator for prefill
        tokenizer: model tokenizer
        prompt_text: the user's query
        config: GenerateConfig
        mx: mlx.core module
        max_keywords: compress index (3 = triplets, None = all)

    Returns:
        (kv_store, seq_len) ready for postamble extension and decode.
    """
    # Load sparse index
    index_path = lib.path / "sparse_index.json"
    if not index_path.exists():
        print(
            f"Error: sparse_index.json not found in {lib.path}. "
            f"Run prefill with --phases sparse to generate it.",
            file=sys.stderr,
        )
        raise FileNotFoundError(f"sparse_index.json not found in {lib.path}")

    sparse_idx = SparseSemanticIndex.load(index_path)
    stats = sparse_idx.stats()
    print(
        f"  Sparse index: {stats['num_entries']} entries, {stats['total_keywords']} keywords",
        file=sys.stderr,
    )

    # Build prompt
    prompt = sparse_idx.render_prompt(
        prompt_text,
        max_keywords=max_keywords,
        chat_template=True,
    )

    # Tokenize
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(
        f"  Sparse prompt: {len(prompt_ids)} tokens (index={stats['est_tokens_full']} est + query)",
        file=sys.stderr,
    )

    # Prefill
    ids = mx.array(prompt_ids)[None]
    logits, kv = kv_gen.prefill(ids)
    mx.eval(logits)

    return kv, len(prompt_ids), logits
