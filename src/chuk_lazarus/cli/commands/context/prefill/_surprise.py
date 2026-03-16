"""Surprise extraction — per-token perplexity scoring during prefill.

For each window, runs a forward pass and measures how surprised the model
is at each token given the preceding context. Stores the max surprise
per window and its position.

This finds needles — content the model considers out-of-distribution
given its context. "astronaut" in Shakespeare scores near 0% probability.
The compass can't see it (absorbed by L26 context), but surprise can.

Complementary to compass:
  Compass:   "Find content SIMILAR to my query"  → content navigation
  Surprise:  "Find content the MODEL finds weird" → anomaly detection
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def extract_surprise(
    engine,
    output_path: Path,
    num_archived: int,
    config,
) -> None:
    """Compute per-token surprise for all windows and save.

    For each window:
      1. Forward pass to get logits (1, seq_len, vocab_size)
      2. For each position, compute rank of actual next token
      3. Store max surprise (highest rank = most unexpected token) and position

    Saves surprise.npz with:
      - max_rank: (num_windows,) — rank of most surprising token per window
      - max_pos:  (num_windows,) — position of that token within the window
      - max_token: (num_windows,) — the token ID that was surprising
    """
    kv_gen = engine.kv_gen

    t0 = time.time()
    max_ranks = np.zeros(num_archived, dtype=np.int32)
    max_positions = np.zeros(num_archived, dtype=np.int32)
    max_tokens = np.zeros(num_archived, dtype=np.int32)

    for wid in range(num_archived):
        w_tokens, w_abs = engine.archive.retrieve(wid)

        if len(w_tokens) < 2:
            continue

        ids = mx.array(w_tokens)[None]  # (1, seq_len)

        # Fresh prefill per window — no KV chaining needed.
        # Surprise measures how unexpected each token is given its window context.
        logits, _kv = kv_gen.prefill(ids)

        mx.eval(logits)

        # logits[0, i, :] predicts token at position i+1
        # Compare logits[0, i] against w_tokens[i+1] for i in 0..len-2
        best_rank = 0
        best_pos = 0
        best_tok = 0

        # Rank each actual token against the model's predictions.
        # Use argmax-based ranking: count how many tokens have higher logit.
        logits_f32 = logits[0].astype(mx.float32)  # (seq_len, vocab)
        mx.eval(logits_f32)

        # Skip first 32 positions — boundary artifacts dominate there
        # (no prior context → everything looks surprising).
        skip = min(32, len(w_tokens) - 2)
        n_score = len(w_tokens) - 1 - skip  # positions to score

        if n_score <= 0:
            continue

        # For each position, count how many vocab tokens have higher logit
        # than the actual next token. rank=0 → top prediction, rank=50000 → very surprising.
        actual_ids = mx.array(w_tokens[skip + 1:])  # next tokens after skip
        logits_slice = logits_f32[skip:skip + n_score]  # (n_score, vocab)
        actual_logits = logits_slice[mx.arange(n_score), actual_ids]  # (n_score,)
        # Count tokens with higher logit per position (vectorized)
        ranks = mx.sum(logits_slice > actual_logits[:, None], axis=1)
        mx.eval(ranks)

        max_idx = int(mx.argmax(ranks).item())
        best_rank = int(ranks[max_idx].item())
        best_pos = skip + max_idx + 1  # position within the window
        best_tok = w_tokens[best_pos]

        max_ranks[wid] = best_rank
        max_positions[wid] = best_pos
        max_tokens[wid] = best_tok

        if (wid + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (wid + 1) / elapsed
            remaining = (num_archived - wid - 1) / rate
            print(
                f"\r  Surprise: {wid + 1}/{num_archived} windows "
                f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)  ",
                end="", file=sys.stderr, flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\r  Surprise: {num_archived} windows in {elapsed:.1f}s"
        f" ({num_archived / elapsed:.0f} win/s)          ",
        file=sys.stderr, flush=True,
    )
    print(file=sys.stderr)

    # Save
    np.savez(
        str(output_path / "surprise.npz"),
        max_rank=max_ranks,
        max_pos=max_positions,
        max_token=max_tokens,
    )

    # Report top-5 most surprising windows
    top_idx = np.argsort(max_ranks)[::-1][:5]
    print("  Most surprising windows:", file=sys.stderr)
    for idx in top_idx:
        print(
            f"    W{idx}: rank={max_ranks[idx]:,} at pos {max_positions[idx]} "
            f"(token_id={max_tokens[idx]})",
            file=sys.stderr,
        )

    size_kb = (output_path / "surprise.npz").stat().st_size / 1024
    print(f"  Saved: surprise.npz ({size_kb:.0f} KB)", file=sys.stderr)
