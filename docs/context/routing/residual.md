# Routing Strategy: residual

**CLI flag:** `--strategy residual`
**Index required:** `residuals.npz`
**Speed:** ~200ms
**Best for:** Legacy compatibility; simple baseline

## How it works

Mean-centered cosine similarity of boundary residuals. The original routing strategy before compass calibration existed.

1. Extract query residual via full forward pass
2. Load all window boundary residuals
3. Subtract corpus mean from all vectors
4. Cosine similarity between query and each window

No PCA. No structural removal. No subspace projection. Raw cosine in the full residual space.

## Limitations

- No structural/content separation — format patterns dominate
- Boundary residuals only (single point per window)
- Superseded by `compass` and `geometric` which are strictly better

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Who commanded the mission?" \
    --strategy residual
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_legacy.py`
