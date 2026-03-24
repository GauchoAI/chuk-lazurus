# Routing Strategy: probe

**CLI flag:** `--strategy probe`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~5-10s
**Best for:** High-precision factual queries; when compass alone isn't discriminating enough

## How it works

Compass casts the net, probe ranks by grounding. No thresholds — the probe is a pure ranker.

1. **Compass → wide candidate set**: Geometric routing selects `top_k` candidates
2. **Probe scores each candidate**: For each candidate window, prefill it with the query, generate the first token, and extract the L26 residual. Project onto the grounding direction (PC1 of grounded vs ungrounded generation).
3. **Replay top-N by probe score**: The windows where the model's first token was most "grounded" are selected for final generation
4. **Generate from combined context**: Replay the best windows together and generate the full answer
5. **Mid-generation monitoring**: Every 40 tokens, probe the generation residual and log whether the model is GROUNDED, partial, or REACHING (informational only — doesn't interrupt)

### Grounding probe calibration

The grounding direction is calibrated from known-grounded (relevant window) and known-ungrounded (irrelevant window) generation examples. PC1 of this contrast captures 84.6% of the grounding variance — the model's residual cleanly separates "I have the answer" from "I'm making this up."

## Parameters

- `--top-k`: Compass candidates to probe (default: 3)
- Higher `top_k` = more probing passes but better discrimination

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the specific baseball scores?" \
    --strategy probe \
    --top-k 5
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/generate/_probe_driven.py`
Grounding calibration: `src/chuk_lazarus/cli/commands/context/generate/_grounding.py`
