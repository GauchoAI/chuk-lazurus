# Prefill Phase: compass

**Phase flag:** `--phases compass`
**Output:** `compass_residuals.npz`, `compass_basis.npz`
**Requires:** Existing library with windows
**Cost:** One forward pass per window to the commitment layer (~75% depth)

## What it does

Calibrates the compass routing system by extracting residuals at the commitment layer and computing a PCA basis that separates structural patterns from content-bearing directions.

The commitment layer (~L26 for Gemma 4B's 34 layers) is where the model's internal representation crystallizes into a navigable geometry. Below this layer, representations are still forming. Above it, they're being consumed by the output head. At the commitment layer, the residual stream encodes a "compass bearing" — a geometric direction that indicates what content the model is processing.

## What it produces

**`compass_residuals.npz`**: Per-window residual vectors at the commitment layer. 8 samples per window (interval mode) or every position (full mode). These are the "map coordinates" — each window has a location in the dark space.

**`compass_basis.npz`**: The PCA basis that defines the coordinate system:
- `mean`: Corpus mean residual vector
- `basis`: PCA components (content directions)
- `structural_basis`: High-variance structural PCs (format, not content)
- `layer`: Which layer the residuals were extracted from
- `pc_start`, `pc_end`: Which PCs are content vs structural

## Auto-calibration

The compass calibration automatically discovers:

1. **Commitment layer**: ~77% of model depth (L26 for 34-layer Gemma 4B), or overridden with `--compass-layer`
2. **Structural boundary**: PCA explained variance curve — the first N PCs with steep drop-offs encode structural patterns (sentence type, formatting). These are removed before content matching.
3. **Content subspace**: The remaining ~2556 dimensions after removing structural PCs. This is the "dark space" where content lives.

## How routing uses it

At query time, the query is forward-passed to the commitment layer, projected into the same basis, and compared against stored compass residuals via cosine similarity. Structural PCs are removed first, so matching operates purely on content geometry.

## Usage

```bash
# Recalibrate compass on an existing library
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases compass

# Use a specific layer (e.g., L29 for retrieval-head alignment)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases compass \
    --compass-layer 29
```

## Storage

For Gemma 4B with 8 samples/window:
- Compass residuals: 725 windows × 8 samples × 2560D × bf16 ≈ 29 MB
- Compass basis: ~200 KB (PCA vectors + metadata)

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_compass.py`
