# Routing Strategy: attention

**CLI flag:** `--strategy attention`
**Index required:** None (uses window tokens directly)
**Speed:** Slow (~5-10s for 725 windows)
**Best for:** Ground-truth routing; research/debugging

## How it works

Lets the model's full attention mechanism decide which windows are relevant. Samples 32 token positions from each window (4 contiguous chunks of 8), concatenates all samples into a single context, and extends the query against it. The attention weights directly reveal which window regions the query attends to.

For 725 windows × 32 tokens = ~23,200 context tokens + query. One prefill + one extend. The model sees real content from every window in a single forward pass.

Attention weights are averaged across all captured heads and query positions, then summed per window to produce the final score.

## Strengths

- Uses the model's actual attention computation — no approximation
- No extraction phases needed
- Ground-truth signal for what the model considers relevant

## Limitations

- Slow — one large prefill of all sampled tokens
- Memory-intensive for large libraries
- Sampled tokens may miss key content within windows

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What about the lunar module?" \
    --strategy attention
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_model_based.py` (`_attention_score_windows`)
