"""Offline cosine router — pure software routing, no model at query time.

Build time (offline, uses model):
  1. Generate 10 synthetic query variants per skill
  2. Embed each via bag-of-embeddings from the model's embed_matrix
  3. Save mean embedding per skill + the embed_matrix itself

Query time (pure software, no model):
  1. Tokenize query (tokenizer is a lookup table)
  2. Look up token embeddings from saved embed_matrix
  3. Mean pool + L2 normalize → query vector
  4. Cosine similarity against stored skill vectors → top-k

Zero model invocation at runtime. Routing is <1ms.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .cosine_router import CosineRouter

OFFLINE_EMBEDDINGS_FILE = "offline_embeddings.npz"
EMBED_MATRIX_FILE = "embed_matrix.npy"


# ── Bag-of-embeddings from embed_matrix ──────────────────────────────

def bag_of_embeddings(token_ids: list[int], embed_matrix: mx.array) -> mx.array:
    """Compute mean embedding from token IDs using the embed_matrix.

    Pure lookup + average — no model forward pass.
    Returns (hidden_dim,) float32, L2-normalized.
    """
    if not token_ids:
        return mx.zeros(embed_matrix.shape[1])

    # Look up embeddings for each token
    ids = mx.array(token_ids)
    embeddings = embed_matrix[ids]  # (n_tokens, hidden_dim)

    # Mean pool
    mean_emb = mx.mean(embeddings, axis=0)  # (hidden_dim,)

    # L2 normalize
    norm = mx.linalg.norm(mean_emb)
    mean_emb = mean_emb / mx.maximum(norm, mx.array(1e-8))
    mx.eval(mean_emb)
    return mean_emb


# ── Build offline embeddings (uses model for synthetic queries) ──────

def build_offline_skill_embedding(
    skill_text: str,
    kv_gen,
    tokenizer,
    embed_matrix: mx.array,
) -> mx.array:
    """Generate synthetic queries, embed via bag-of-embeddings, return mean.

    Uses the model only for synthetic query generation.
    The embedding itself is pure lookup from embed_matrix.
    """
    from .synthetic_router import generate_synthetic_queries

    t0 = time.monotonic()
    queries = generate_synthetic_queries(skill_text, kv_gen, tokenizer)

    if not queries:
        # Fallback: embed the skill text directly
        token_ids = tokenizer.encode(skill_text[:500], add_special_tokens=False)
        return bag_of_embeddings(token_ids, embed_matrix)

    # Embed each synthetic query via bag-of-embeddings (no model forward pass)
    embeddings = []
    for q in queries:
        token_ids = tokenizer.encode(q, add_special_tokens=False)
        emb = bag_of_embeddings(token_ids, embed_matrix)
        embeddings.append(emb)

    # Mean of all query embeddings
    stacked = mx.stack(embeddings)
    mean_emb = mx.mean(stacked, axis=0)
    norm = mx.linalg.norm(mean_emb)
    mean_emb = mean_emb / mx.maximum(norm, mx.array(1e-8))
    mx.eval(mean_emb)

    elapsed = time.monotonic() - t0
    print(f"    Generated {len(queries)} synthetic queries, embedded offline in {elapsed:.1f}s",
          file=sys.stderr)

    return mean_emb


# ── Pure software query routing ──────────────────────────────────────

class OfflineRouter:
    """Route queries using only tokenizer + embed_matrix. No model needed."""

    def __init__(self, skill_embeddings: dict[int, mx.array], embed_matrix: mx.array):
        self.embed_matrix = embed_matrix
        self._cosine = CosineRouter(skill_embeddings)

    def route(self, query_text: str, tokenizer, top_k: int = 3) -> list[int]:
        """Route a query — pure software, no model invocation.

        1. Tokenize (lookup table)
        2. Bag-of-embeddings from embed_matrix (matrix index)
        3. Cosine similarity (dot product)
        """
        token_ids = tokenizer.encode(query_text, add_special_tokens=False)
        query_emb = bag_of_embeddings(token_ids, self.embed_matrix)
        return self._cosine.route_window_ids(query_emb, top_k=top_k)


# ── Persistence ──────────────────────────────────────────────────────

def save_offline_data(
    skill_embeddings: dict[int, mx.array],
    embed_matrix: mx.array,
    store_path: Path,
) -> None:
    """Save offline embeddings + embed_matrix to disk."""
    store_path = Path(store_path)

    # Skill embeddings
    data = {}
    for wid, emb in skill_embeddings.items():
        data[str(wid)] = np.array(emb.tolist(), dtype=np.float32)
    np.savez(str(store_path / OFFLINE_EMBEDDINGS_FILE), **data)

    # Embed matrix (only save once — it's the same for all skills)
    matrix_path = store_path / EMBED_MATRIX_FILE
    if not matrix_path.exists():
        np.save(str(matrix_path), np.array(embed_matrix.tolist(), dtype=np.float32))
        size_mb = matrix_path.stat().st_size / (1024 * 1024)
        print(f"    Saved embed_matrix: {size_mb:.1f} MB", file=sys.stderr)


def load_offline_router(store_path: Path, tokenizer=None) -> OfflineRouter | None:
    """Load an OfflineRouter from disk. Returns None if files don't exist."""
    store_path = Path(store_path)

    emb_path = store_path / OFFLINE_EMBEDDINGS_FILE
    matrix_path = store_path / EMBED_MATRIX_FILE

    if not emb_path.exists() or not matrix_path.exists():
        return None

    # Load skill embeddings
    npz = np.load(str(emb_path), allow_pickle=False)
    skill_embeddings = {}
    for key in npz.files:
        arr = mx.array(npz[key], dtype=mx.float32)
        norm = mx.linalg.norm(arr)
        arr = arr / mx.maximum(norm, mx.array(1e-8))
        mx.eval(arr)
        skill_embeddings[int(key)] = arr

    # Load embed matrix
    embed_matrix = mx.array(np.load(str(matrix_path)), dtype=mx.float32)
    mx.eval(embed_matrix)

    return OfflineRouter(skill_embeddings, embed_matrix)
