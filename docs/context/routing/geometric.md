# Routing Strategy: geometric

**CLI flag:** `--strategy geometric`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~1.5s
**Best for:** General factual queries (default for Mode 7 FACTUAL)

## How it works

Fuses two geometric scoring strategies via Reciprocal Rank Fusion (RRF):

1. **Compass scoring**: Cosine similarity in the PCA-calibrated content subspace. Structural PCs are removed; matching happens in the "dark space" where content geometry lives.

2. **Contrastive scoring**: Discovers a query-specific 8D subspace at runtime. Compares the query against 30 random corpus samples to find which dimensions make the query unique, then matches windows in that frame.

Each strategy produces an independent ranking. RRF (k=5) merges them:
```
score(w) = 1/(5 + rank_compass(w)) + 1/(5 + rank_contrastive(w))
```

Windows ranked high by **either** strategy rise to the top. This makes geometric routing robust — compass catches broad topic matches, contrastive catches query-specific nuances.

## Why RRF

Compass and contrastive operate in different subspaces. Compass uses the corpus's calibrated PCA basis (fixed for all queries). Contrastive discovers a fresh basis per query. Their rankings are complementary:

- Compass is stable — same query always gets the same ranking
- Contrastive is adaptive — finds directions specific to this query
- RRF lets either one "veto" a bad ranking from the other

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy geometric
```

## Fallback

If no compass data exists in the library, falls back to BM25.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_orchestrator.py` (lines 132-168)
Scoring: `src/chuk_lazarus/cli/commands/context/compass_routing/_geometric.py`
