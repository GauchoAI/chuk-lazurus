# Prefill Phase: surprise

**Phase flag:** `--phases surprise`
**Output:** `surprise.npz`
**Requires:** Existing library with windows
**Cost:** One forward pass per window

## What it does

Computes per-token surprise (perplexity) for every window. For each token, measures how surprised the model is given the preceding context — specifically, the rank of the actual next token in the model's predicted distribution.

This finds **needles** — content the model considers out-of-distribution. "Astronaut" in a Shakespeare transcript scores near 0% probability. The compass can't see it (the residual is dominated by the surrounding context), but surprise can.

## Complementary to compass

| Signal | What it finds | Mechanism |
|--------|--------------|-----------|
| Compass | Content similar to query | Geometric distance in dark space |
| Surprise | Content the model finds unexpected | Prediction rank of actual tokens |

Compass navigates by similarity. Surprise detects anomalies. Together they cover both "find what I'm looking for" and "find what's unusual."

## What it stores

Per window:
- `max_rank`: Rank of the most surprising token (highest = most unexpected)
- `max_position`: Position of that token within the window
- `mean_rank`: Average surprise across all tokens

## How other phases use it

The `sparse` phase uses surprise scores to identify **fact positions** — tokens where the model encountered novel content. These positions seed the keyword extraction and the K-vector index.

## Usage

```bash
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases surprise
```

## Storage

Minimal — a few floats per window. ~10 KB for 725 windows.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_surprise.py`
