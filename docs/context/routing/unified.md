# Routing Strategy: unified

**CLI flag:** `--strategy unified`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** 2-15s (depends on classification)
**Best for:** Auto-detecting factual vs exploration queries

## How it works

Binary classification: the dark space decides whether the query is FACTUAL or EXPLORATION.

**FACTUAL** (fast path ~2s): Dual compass routing — runs both `compass` and `geometric` strategies, merges rankings via RRF (k=60), replays top-3 windows and generates.

**EXPLORATION** (thorough path ~10-15s): Delegates to the `iterative` strategy — multi-round compass navigation with generation-guided shifting. Discovers content across the document that single-shot compass never reaches.

### Classification

A query-type probe at L26 (calibrated from generic factual vs exploratory examples) projects the query residual onto a discriminative direction. Positive projection → EXPLORATION. Negative → FACTUAL.

## Relationship to Mode 7

`unified` is the predecessor to `mode7`. It handles the factual/exploration split well but doesn't distinguish engagement, tension, global, or tone queries. Mode 7 extends this to 5 query types with per-type routing. Use `mode7` for production; `unified` for the simpler binary split.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the key events?" \
    --strategy unified
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/generate/_unified.py`
