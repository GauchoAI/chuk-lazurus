# Routing Strategy: preview

**CLI flag:** `--strategy preview`
**Index required:** None (uses window tokens directly)
**Speed:** Moderate (~3-5s for 91 windows)
**Best for:** High-accuracy routing when speed isn't critical

## How it works

For each window, builds a compressed preview (first 64 + last 64 tokens), prefills it, extends the query against it, and measures the query's perplexity. Windows that make the query tokens more predictable (higher log-probability) are more relevant.

The insight: if a window's content activates the retrieval circuit for this query, the model will predict the query tokens better. The preview content provides enough signal for the model to "understand" the query context.

128 tokens per window × N windows, each as independent prefill+extend operations.

## Strengths

- High accuracy — the model directly evaluates each window's relevance
- Works without any extraction phases
- Produces calibrated scores (log-probabilities)

## Limitations

- O(N) forward passes (one per window)
- ~3-5s for 91 windows, doesn't scale well to thousands

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Who was the commander?" \
    --strategy preview
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_model_based.py` (`_preview_score_windows`)
