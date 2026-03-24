# Routing Strategy: hybrid

**CLI flag:** `--strategy hybrid`
**Index required:** None
**Speed:** Moderate
**Best for:** Combining fast keyword pre-filtering with model-based re-ranking

## How it works

Two-stage pipeline:

1. **Stage 1 — BM25 pre-filter**: Score all windows with BM25, take the top 10 with score > 0
2. **Stage 2 — Preview re-rank**: Run the `preview` strategy (query perplexity) only on the BM25 shortlist

If BM25 finds fewer candidates than `top_k`, falls back to running preview on all windows.

This combines BM25's speed (instant keyword matching) with preview's accuracy (model-based relevance scoring), but limits the expensive preview passes to BM25-filtered candidates.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What about the EVA?" \
    --strategy hybrid
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_orchestrator.py` (lines 87-103)
