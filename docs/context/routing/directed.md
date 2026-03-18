# Routing Strategy: directed

**CLI flag:** `--strategy directed`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~500ms
**Best for:** Highly specific factual queries where the answer has a clear direction

## How it works

Projects all stored residuals onto the query's own direction in the dark space. The query residual at L26, minus the corpus mean, defines a single direction. Positions that deviate from the mean in the same way as the query are scored highest.

1. Load all compass residuals and compute corpus mean
2. PCA to find structural PCs; remove them (same auto-detection as compass calibration)
3. Extract query residual at L26, subtract mean, remove structural PCs
4. Normalize to get the query's direction vector
5. Cosine similarity between all stored residuals and this one direction

No PCA basis needed at routing time. No frame bank. The query IS the basis — one dimension, the query's own.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Who commanded Apollo 11?" \
    --strategy directed
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_geometric.py` (`_directed_score_windows`)
