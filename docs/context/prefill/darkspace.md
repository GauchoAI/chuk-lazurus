# Prefill Phase: darkspace

**Phase flag:** `--phases darkspace`
**Output:** `compass_residuals.npz`, `compass_basis.npz`
**Requires:** Existing library with windows; optionally a pre-computed frame bank
**Cost:** One forward pass per window to the commitment layer

## What it does

Extracts darkspace projections for compass routing using whitened frame banks. This is an alternative to the standard `compass` phase that produces cross-corpus-compatible routing coordinates.

The key difference from `compass`: standard compass calibrates PCA from the corpus itself, so the coordinate system is corpus-specific. Darkspace uses a pre-computed or corpus-calibrated whitening transformation that produces coordinates comparable across different documents.

## Two paths

**Path A — Pre-computed frame bank**: Supply `--frame-bank ./frame_bank.npz` from a prior calibration run. Residuals are projected through this fixed bank. Coordinates are directly comparable across any corpus processed with the same bank.

**Path B — Corpus-calibrated whitening**: No frame bank supplied. The phase calibrates a whitening transformation from the corpus's own residuals, then projects through it. Produces a 64-dimensional representation that captures content variance with structural patterns removed.

## Usage

```bash
# With pre-computed frame bank
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --residual-mode darkspace \
    --frame-bank ./frame_bank.npz

# Corpus self-calibration (no external bank)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --residual-mode darkspace
```

## Frame bank calibration

Create a frame bank from diverse text:

```bash
lazarus context calibrate-frames \
    --model google/gemma-3-4b-it \
    --output ./frame_bank.npz \
    --method whitening \
    --dims 64
```

## Storage

64 dimensions × 8 samples/window × bf16 ≈ 1 KB/window. Much smaller than standard compass (2560D per sample).

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_darkspace.py`
