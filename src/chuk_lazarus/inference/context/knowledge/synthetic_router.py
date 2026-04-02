"""Synthetic query cosine router — generate query variants, embed them, route by similarity.

At append time:
1. Feed the skill text to the model
2. Ask it to generate 10 natural language queries of increasing length
3. Embed each query via prefill_to_layer + mean pool
4. Store the mean of all query embeddings as the skill's routing vector

At query time:
1. Embed the real query via prefill_to_layer + mean pool
2. Cosine similarity against stored skill embeddings
3. Return top-k — no expansion, no TF-IDF, no stopwords

This puts skill embeddings in "query space" rather than "document space",
giving much better alignment with real user queries.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .cosine_router import CosineRouter, build_query_embedding

SYNTHETIC_EMBEDDINGS_FILE = "synthetic_embeddings.npz"

# ── Synthetic query generation ───────────────────────────────────────

SYNTH_PROMPT_TEMPLATE = """Tool description:
{skill_text}

Write 10 natural language questions a user might ask that would need this tool. Start with short simple questions (3-5 words) and progressively make them longer and more detailed. Write them as a real person would ask, not as API calls.

1."""


def generate_synthetic_queries(
    skill_text: str,
    kv_gen,
    tokenizer,
    n_queries: int = 10,
    max_tokens: int = 300,
) -> list[str]:
    """Generate synthetic user queries for a skill using the model.

    Returns a list of natural language query strings.
    """
    from ._sampling import sample_token

    # Truncate skill text to fit in context
    prompt = SYNTH_PROMPT_TEMPLATE.format(skill_text=skill_text[:800])
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    logits, kv = kv_gen.prefill(mx.array(ids)[None])
    mx.eval(logits)
    seq = len(ids)

    tokens = []
    for _ in range(max_tokens):
        t = sample_token(logits[0, -1], 0.0)
        tokens.append(t)
        logits, kv = kv_gen.step_uncompiled(mx.array([[t]]), kv, seq_len=seq)
        seq += 1

    text = "1." + tokenizer.decode(tokens, skip_special_tokens=True)

    # Parse numbered lines
    queries = []
    for line in text.split("\n"):
        line = line.strip()
        # Match lines starting with a number
        m = re.match(r"^\d+[\.\)]\s*(.+)", line)
        if m:
            q = m.group(1).strip().strip('"').strip("'")
            if len(q) >= 5:  # skip very short fragments
                queries.append(q)
            if len(queries) >= n_queries:
                break

    return queries


# ── Embedding from synthetic queries ─────────────────────────────────

def build_synthetic_embedding(
    skill_text: str,
    kv_gen,
    tokenizer,
    crystal_layer: int,
) -> mx.array:
    """Generate synthetic queries for a skill and return their mean embedding.

    Returns (hidden_dim,) float32 — normalized.
    """
    t0 = time.monotonic()
    queries = generate_synthetic_queries(skill_text, kv_gen, tokenizer)

    if not queries:
        # Fallback: embed the skill text directly
        return build_query_embedding(kv_gen, tokenizer, skill_text[:500], crystal_layer)

    # Embed each synthetic query
    embeddings = []
    for q in queries:
        emb = build_query_embedding(kv_gen, tokenizer, q, crystal_layer)
        embeddings.append(emb)

    # Mean of all query embeddings
    stacked = mx.stack(embeddings)  # (n_queries, hidden_dim)
    mean_emb = mx.mean(stacked, axis=0)  # (hidden_dim,)
    norm = mx.linalg.norm(mean_emb)
    mean_emb = mean_emb / mx.maximum(norm, mx.array(1e-8))
    mx.eval(mean_emb)

    elapsed = time.monotonic() - t0
    print(f"    Generated {len(queries)} synthetic queries, embedded in {elapsed:.1f}s",
          file=sys.stderr)

    return mean_emb


# ── Persistence ──────────────────────────────────────────────────────

def save_synthetic_embeddings(embeddings: dict[int, mx.array], store_path: Path) -> None:
    """Save synthetic query embeddings to disk."""
    store_path = Path(store_path)
    data = {}
    for wid, emb in embeddings.items():
        data[str(wid)] = np.array(emb.tolist(), dtype=np.float32)
    np.savez(str(store_path / SYNTHETIC_EMBEDDINGS_FILE), **data)


def load_synthetic_embeddings(store_path: Path) -> dict[int, mx.array]:
    """Load synthetic query embeddings from disk."""
    store_path = Path(store_path)
    emb_path = store_path / SYNTHETIC_EMBEDDINGS_FILE
    if not emb_path.exists():
        return {}
    npz = np.load(str(emb_path), allow_pickle=False)
    result = {}
    for key in npz.files:
        arr = mx.array(npz[key], dtype=mx.float32)
        norm = mx.linalg.norm(arr)
        arr = arr / mx.maximum(norm, mx.array(1e-8))
        mx.eval(arr)
        result[int(key)] = arr
    return result
