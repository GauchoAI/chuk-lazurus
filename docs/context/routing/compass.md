# Routing Strategy: compass

**CLI flag:** `--strategy compass`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~500ms
**Best for:** Stable, reproducible routing; broad topic matching

## How it works

Cosine similarity between the query's commitment-layer residual and stored per-window compass residuals, in the calibrated subspace.

Three projection modes depending on what the library contains:

1. **Structural removal** (default): Remove high-variance structural PCs (format patterns), then match in the full remaining dark space (~2556D for Gemma 4B). This is the most accurate mode.

2. **Darkspace (frame bank)**: Stored residuals are pre-projected into a whitened frame bank space. Only the query needs projection at routing time. Cross-corpus compatible.

3. **Fixed 16D (legacy)**: Project into a fixed 16-PC content subspace. Older libraries only.

## Aggregation

Each window has multiple samples (8 in interval mode). Scores are aggregated:
- **Sparse (≤16 samples)**: Max — the best-matching position wins
- **Dense (>16 samples)**: Top-10 mean — clusters of matches win

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Tell me about the lunar module" \
    --strategy compass
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_geometric.py` (`_compass_score_windows`)
