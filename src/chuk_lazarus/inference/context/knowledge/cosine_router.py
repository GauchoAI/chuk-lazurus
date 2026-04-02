"""Cosine similarity router — semantic routing via residual stream embeddings.

Instead of TF-IDF token overlap + expensive query expansion, this router:
1. At build/append time: mean-pools each window's residual stream into a
   single embedding vector and stores it.
2. At query time: runs the query through prefill_to_layer to get its
   embedding, then computes cosine similarity against all stored embeddings.

Routing is <1ms (dot products) vs ~1.2s (LLM-based query expansion).
No stopword filtering or disambiguation needed — semantic proximity
handles vocabulary gaps natively.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

EMBEDDINGS_FILE = "embeddings.npz"


def build_window_embedding(residual_stream: mx.array) -> mx.array:
    """Mean-pool a window's residual stream into a single embedding.

    Args:
        residual_stream: (seq_len, hidden_dim) — the L30 residual stream.

    Returns:
        (hidden_dim,) float32 — normalized embedding vector.
    """
    # Mean pool across sequence length
    embedding = mx.mean(residual_stream, axis=0)  # (hidden_dim,)
    # L2 normalize for cosine similarity via dot product
    norm = mx.linalg.norm(embedding)
    embedding = embedding / mx.maximum(norm, mx.array(1e-8))
    mx.eval(embedding)
    return embedding


def build_query_embedding(
    kv_gen,
    tokenizer,
    query_text: str,
    crystal_layer: int,
) -> mx.array:
    """Run a query through the model and extract its embedding.

    Returns (hidden_dim,) float32 — normalized embedding vector.
    """
    tokens = tokenizer.encode(query_text, add_special_tokens=True)
    token_mx = mx.array(tokens)[None]

    h = kv_gen.prefill_to_layer(
        token_mx,
        target_layer=crystal_layer,
        initial_residual=None,
    )
    # h: (1, seq_len, hidden_dim)
    embedding = mx.mean(h[0], axis=0)  # (hidden_dim,)
    norm = mx.linalg.norm(embedding)
    embedding = embedding / mx.maximum(norm, mx.array(1e-8))
    mx.eval(embedding)
    return embedding


class CosineRouter:
    """Route queries by cosine similarity against stored window embeddings."""

    def __init__(self, embeddings: dict[int, mx.array]):
        """
        Args:
            embeddings: dict mapping window_id → (hidden_dim,) normalized vector.
        """
        if not embeddings:
            self.window_ids = []
            self.matrix = None
            return

        self.window_ids = sorted(embeddings.keys())
        # Stack into (num_windows, hidden_dim) matrix for batch dot product
        self.matrix = mx.stack([embeddings[wid] for wid in self.window_ids])
        mx.eval(self.matrix)

    def route(self, query_embedding: mx.array, top_k: int = 3) -> list[tuple[int, float]]:
        """Return top-k (window_id, similarity_score) pairs.

        Args:
            query_embedding: (hidden_dim,) normalized query vector.
            top_k: number of results to return.

        Returns:
            List of (window_id, cosine_similarity) sorted by score descending.
        """
        if self.matrix is None:
            return []

        # Cosine similarity = dot product (both vectors are L2-normalized)
        scores = mx.matmul(self.matrix, query_embedding)  # (num_windows,)
        mx.eval(scores)

        # Get top-k
        scores_list = scores.tolist()
        scored = [(scores_list[i], self.window_ids[i]) for i in range(len(self.window_ids))]
        scored.sort(reverse=True)

        return [(wid, score) for score, wid in scored[:top_k]]

    def route_window_ids(self, query_embedding: mx.array, top_k: int = 3) -> list[int]:
        """Return just the top-k window IDs."""
        return [wid for wid, _ in self.route(query_embedding, top_k)]


# ── Persistence ──────────────────────────────────────────────────────

def save_embeddings(embeddings: dict[int, mx.array], store_path: Path) -> None:
    """Save window embeddings to disk."""
    store_path = Path(store_path)
    data = {}
    for wid, emb in embeddings.items():
        data[str(wid)] = np.array(emb.tolist(), dtype=np.float32)
    np.savez(str(store_path / EMBEDDINGS_FILE), **data)


def load_embeddings(store_path: Path) -> dict[int, mx.array]:
    """Load window embeddings from disk."""
    store_path = Path(store_path)
    emb_path = store_path / EMBEDDINGS_FILE
    if not emb_path.exists():
        return {}
    npz = np.load(str(emb_path), allow_pickle=False)
    result = {}
    for key in npz.files:
        arr = mx.array(npz[key], dtype=mx.float32)
        # Ensure normalized
        norm = mx.linalg.norm(arr)
        arr = arr / mx.maximum(norm, mx.array(1e-8))
        mx.eval(arr)
        result[int(key)] = arr
    return result
