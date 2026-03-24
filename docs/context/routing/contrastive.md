# Routing Strategy: contrastive

**CLI flag:** `--strategy contrastive`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~1s
**Best for:** Discriminating between similar topics; finding query-specific content

## How it works

Discovers a query-specific subspace at runtime — the directions in the dark space where the query differs from typical corpus content.

1. **Extract query residual** at L26 (commitment layer)
2. **Sample contrast set**: 30 random compass residuals from the corpus — "what does typical content look like?"
3. **Cross-domain PCA**: SVD on the [query + contrast] matrix to find the principal directions of variation
4. **Fisher criterion**: Rank PCs by how well they separate the query from the contrast set. The top 8 most discriminative PCs form the query's routing frame
5. **Project and match**: All stored compass residuals are projected into this 8D frame. Cosine similarity identifies matching windows.

## Key insight

The subspace IS the address. Different queries produce different coordinate frames. A query about "baseball scores" produces a frame that separates sports content from everything else. A query about "Armstrong" produces a frame that separates Apollo 11 content.

The query defines its own routing geometry at runtime, rather than using a fixed basis for all queries.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy contrastive
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_geometric.py` (`_contrastive_score_windows`)
