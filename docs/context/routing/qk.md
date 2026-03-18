# Routing Strategy: qk

**CLI flag:** `--strategy qk`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~500ms
**Best for:** Fast structural routing; when you want the model's own attention to decide

## How it works

Uses the model's trained W_Q and W_K weight matrices at a global attention layer near the commitment layer. The stored compass residuals are projected through W_K, the query through W_Q, and Q.K^T computes the attention scores.

This is the full dark space read through the model's own attention "glasses" — no PCA, no frame bank, no projections. The model's own learned routing mechanism operating on stored L26 residuals.

## Important limitation

QK routes by **structural patterns**, not content. Experiments showed that QK attention captures BOS sinks, entity-type patterns, and formatting structures. It does NOT reliably route by factual content.

For content-aware routing, use `geometric`, `compass`, or `kv_route` instead. QK is fast but structural.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Tell me about Armstrong" \
    --strategy qk
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_model_based.py` (`_qk_score_windows`)
