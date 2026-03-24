# Routing Strategy: deflection

**CLI flag:** `--strategy deflection`
**Index required:** `checkpoints.npz`
**Speed:** Slow (one extend per window)
**Best for:** Research; measuring context influence

## How it works

Measures how much each window's checkpoint deflects the query's residual stream. For each window:

1. Compute bare query residual (no context)
2. Extend query against the window's boundary KV checkpoint
3. Measure L2 distance between the contextualised and bare residuals

Larger deflection = the window's context significantly changed how the model processes the query = more relevant.

## Intuition

If a window contains information relevant to the query, injecting its checkpoint will shift the model's internal state. Irrelevant windows produce small deflections. This is a direct measurement of context influence — no projections or approximations.

## Limitations

- O(N) extend operations (one per window)
- Requires checkpoint KV (not available in darkspace mode)
- Slow for large libraries

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Tell me about the EVA" \
    --strategy deflection
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_geometric.py` (`_deflection_score_windows`)
