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
    from .....inference.context.kv_generator import make_kv_generator

    kv_gen = make_kv_generator(engine.model, config)

    # Warm up
    _warm = mx.array([[1, 2, 3]])
    _, _kv = kv_gen.prefill(_warm)
    mx.eval()

    t0 = time.time()
    max_ranks = np.zeros(num_archived, dtype=np.int32)
    max_positions = np.zeros(num_archived, dtype=np.int32)
    max_tokens = np.zeros(num_archived, dtype=np.int32)

    for wid in range(num_archived):
        w_tokens, w_abs = engine.archive.retrieve(wid)

        if len(w_tokens) < 2:
            continue

        ids = mx.array(w_tokens)[None]  # (1, seq_len)

        # Chain from prior window if available
        if wid > 0 and engine.checkpoints.has(wid - 1):
            prior_kv, prior_abs = engine.checkpoints.load(wid - 1)
            logits, _kv = kv_gen.extend(ids, prior_kv, abs_start=w_abs)
        else:
            logits, _kv = kv_gen.prefill(ids)

        mx.eval(logits)

        # logits[0, i, :] predicts token at position i+1
        # Compare logits[0, i] against w_tokens[i+1] for i in 0..len-2
        best_rank = 0
        best_pos = 0
        best_tok = 0

        # Get all logits as numpy for ranking
        # Use argsort descending on each position, find where actual token lands
        logits_np = np.array(logits[0].tolist(), dtype=np.float32)  # (seq_len, vocab)

        for pos in range(len(w_tokens) - 1):
            actual_token = w_tokens[pos + 1]
            # Rank: how many tokens have higher logit than the actual token?
            token_logit = logits_np[pos, actual_token]
            rank = int(np.sum(logits_np[pos] > token_logit))
            # rank=0 means it was the top prediction, rank=50000 means very surprising

            if rank > best_rank:
                best_rank = rank
                best_pos = pos + 1  # position of the surprising token
                best_tok = actual_token

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
