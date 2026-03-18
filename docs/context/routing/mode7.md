# Routing Strategy: mode7

**CLI flag:** `--strategy mode7`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`; optionally `sparse_index.json`, probe cache
**Speed:** Variable (depends on query type)
**Best for:** Production — one command, any query type

## How it works

Unified dark space router. One command. Any query. The engine classifies the query via a 5-class L26 probe, routes through the appropriate mechanism, selects windows, replays them, and generates.

### Query types and routing

| Query type | Detection | Routing | Windows |
|-----------|-----------|---------|---------|
| FACTUAL | Direct information request | Geometric RRF (compass + contrastive) | top-3 |
| ENGAGEMENT | "Find amusing/interesting moments" | BM25 indicator search → engagement probe re-rank | top-5 |
| TENSION | "What were the tense moments" | BM25 indicators + temporal stride → tension probe re-rank | top-5 |
| GLOBAL | "Summarize the whole document" | Temporal stride (evenly spaced) | 10 |
| TONE | "What's the overall tone" | Geometric RRF with more windows | top-7 |

### Classification pipeline

1. Forward-pass query to L26 (commitment layer)
2. Project through 5-class linear probe (calibrated during `--phases mode7`)
3. Highest activation class wins
4. Low confidence → fall back to FACTUAL (best general-purpose)

### Per-type routing

**FACTUAL**: Uses `geometric` strategy — compass + contrastive RRF. The proven best general-purpose router.

**ENGAGEMENT**: BM25 pre-filter with engagement indicator words (laughter, applause, humor markers), then re-rank candidates with the tonal probe. The probe projects L26 residuals of candidate windows onto the engagement direction.

**TENSION**: BM25 with tension indicators (emergency, warning, critical, failure) plus temporal stride candidates (tension often occurs at mission-critical timeline points). Union of candidates, re-ranked by tension probe.

**GLOBAL**: Pure temporal stride — evenly spaced windows for full document coverage. Uses a special postamble template that forces the model to distribute attention across all excerpts.

**TONE**: Geometric RRF with more windows (7 instead of 3) — tone requires broader context.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy mode7

# Pre-calibrate probes during prefill for instant startup
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases mode7
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/generate/_mode7.py`
Probes: `src/chuk_lazarus/cli/commands/context/generate/_probes.py`
