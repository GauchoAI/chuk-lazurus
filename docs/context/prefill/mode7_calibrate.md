# Prefill Phase: mode7

**Phase flag:** `--phases mode7`
**Output:** `.probe_cache_v*.npz` (in checkpoint directory)
**Requires:** Existing library with windows + compass data
**Cost:** Moderate — calibrates probes from synthetic examples + library content

## What it does

Pre-calibrates the Mode 7 query classification probes so they're ready at generate time without a ~60s first-run penalty. Mode 7 uses three L26 probes to auto-classify incoming queries and dispatch them to the appropriate routing strategy:

## Probes calibrated

| Probe | Classes | Purpose |
|-------|---------|---------|
| Query-type classifier | FACTUAL, ENGAGEMENT, TENSION, GLOBAL, TONE | Routes to appropriate strategy |
| Tonal (engagement) | High/low engagement | Re-ranks BM25 candidates for engagement queries |
| Tension | High/low tension | Re-ranks candidates for tension/conflict queries |

## How calibration works

1. Generates synthetic examples for each query type (factual questions, engagement prompts, etc.)
2. Forward-passes each through the model to the commitment layer
3. Extracts L26 residuals and trains linear probes (direction vectors + thresholds)
4. Saves the probe cache to the checkpoint directory

At generate time, Mode 7 loads the cached probes instead of recalibrating.

## Usage

```bash
# Calibrate Mode 7 probes on existing library
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases mode7
```

## Storage

Small — a few hundred KB for probe direction vectors and thresholds.

## Source

Calibration: `src/chuk_lazarus/cli/commands/context/prefill/_mode7_calibrate.py`
Probes: `src/chuk_lazarus/cli/commands/context/generate/_probes.py`
