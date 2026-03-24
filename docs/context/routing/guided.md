# Routing Strategy: guided

**CLI flag:** `--strategy guided`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~500ms
**Best for:** Queries where keyword presence should boost geometric matches

## How it works

Combines compass geometry with token overlap — both model-internal signals:

1. **Compass score**: 16D PCA subspace cosine (the model's geometric state)
2. **Token overlap**: Query token IDs present in the window (what the model read)

Combined: `score = compass_score × (1 + token_overlap_fraction)`

Windows that match geometrically AND contain query tokens rise to the top. Windows with no token overlap keep their compass score unchanged.

Token overlap filtering removes structural tokens (appearing in >50% of windows) to focus on content tokens only.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Armstrong lunar module" \
    --strategy guided
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_composite.py` (`_guided_score_windows`)
