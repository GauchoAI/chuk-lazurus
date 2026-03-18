# Prefill Phase: kvectors

**Phase flag:** `--phases kvectors`
**Output:** `kv_route_index.npz`
**Requires:** Existing library with windows; optionally `sparse_index.json` for fact positions
**Cost:** One forward pass per window to the retrieval layer

## What it does

Extracts K vectors from the model's retrieval head (L29 KV-head-2 for Gemma 4B) at fact positions in each window. These K vectors are the model's own addressing mechanism — the same vectors that the retrieval circuit uses for Q.K matching during attention. Externalising them as a routing index lets you do the model's own fact-addressing computation in ~50ms instead of replaying full windows.

## Position selection

The `kvectors` phase selects positions using a priority cascade:

1. **Sparse index positions** (if `sparse_index.json` exists): Surprise-guided fact positions — where the model encountered novel content during prefill.
2. **Interval sampling** (fallback): 8 evenly-spaced positions per window (1.6% coverage).

This means coverage depends on what other phases have been run. If you've run `sparse` first, K-vectors are extracted at the surprise-identified fact positions. If not, you get 8 samples per window.

## Coverage limitation

Both approaches can miss facts:
- **Surprise-guided**: Only catches facts that surprised the model. Parametric knowledge (things the model already knew) gets low surprise and isn't indexed.
- **Interval**: Pure luck — 8 positions out of 512 = 1.6% coverage. Facts between samples are invisible.

For guaranteed coverage, use [`kvectors_full`](kvectors_full.md) instead.

## How routing uses it

At query time, the `kv_route` strategy:
1. Forward-passes the query to the retrieval layer
2. Extracts Q at the retrieval head (H4)
3. Computes Q.K^T against all stored K vectors
4. Top-scoring positions identify which windows contain the answer

This is the model's own attention computation, externalised. The score is exactly what Q.K^T would produce if all tokens were in the attention window.

## Usage

```bash
# Extract K-vectors (sparse positions or 8-sample fallback)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases kvectors
```

## Storage

256D bf16 per position = 512 bytes per fact.
- Sparse mode: ~1.8 MB for 3,625 facts (Apollo 11 corpus)
- Interval mode: ~0.7 MB (8 × 725 windows × 512 bytes)
- 16x smaller than compass residuals (29 MB)

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_kv_route.py`
