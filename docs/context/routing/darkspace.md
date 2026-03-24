# Routing Strategy: darkspace

**CLI flag:** `--strategy darkspace`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~500ms
**Best for:** Balanced routing with complementary signals

## How it works

Dual-score routing in the 16D PCA content subspace. Two complementary readings of the same geometric map:

1. **Compass cosine**: Standard cosine in 16D — all dimensions weighted equally. Finds content that is geometrically anomalous.
2. **Directed cosine**: Query-weighted cosine in 16D — dimensions weighted by the query's own activation pattern. Finds content aligned with what the query is looking for.

Combined: `score = 0.5 × compass + 0.5 × directed`

Both are pure model geometry. Same layer. Same basis. Same subspace. Compass finds what's geometrically distinctive. Directed finds what the query targets. Together they surface windows that are both distinctive AND relevant.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Tell me about the landing" \
    --strategy darkspace
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_composite.py` (`_darkspace_score_windows`)
