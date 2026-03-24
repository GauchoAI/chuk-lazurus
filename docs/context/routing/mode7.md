# Routing Strategy: mode7

**CLI flag:** `--strategy mode7` (default — no flag needed)
**Index required:** `compass_residuals.npz`, `compass_basis.npz`; optionally `vec_inject.npz`, probe cache
**Speed:** ~200ms (vec_inject fast path) or 2–10s (window replay)
**Best for:** Production — one command, any query type

## How it works

Unified dark space router. One command. Any query. The engine classifies the query via
a 5-class L26 probe, routes through the appropriate mechanism, and generates.

### Query types and routing

| Query type | Detection | Routing | Speed |
|-----------|-----------|---------|-------|
| FACTUAL | Direct information request | Vec inject (fast path) → geometric RRF fallback | 200ms–2s |
| ENGAGEMENT | "Find amusing/interesting moments" | Compass coarse-filter → engagement probe re-rank | 4–8s |
| TENSION | "What were the tense moments" | Compass + temporal stride → tension probe re-rank | 4–8s |
| GLOBAL | "Summarize the whole document" | Temporal stride (evenly spaced) | 8–15s |
| TONE | "What's the overall tone" | Geometric RRF with more windows (top-7) | 4–8s |

### FACTUAL routing decision tree

```
Query classified as FACTUAL
    │
    ├─ vec_inject.npz present?
    │       │
    │       ├─ YES → Q·K routing → adaptive threshold (mean × 2.0)
    │       │           │
    │       │           ├─ CONFIDENT + distinctive match
    │       │           │       → two-pass injection at L30 (~200ms)
    │       │           │
    │       │           └─ LOW CONFIDENCE or no distinctive
    │       │                   → fall through ↓
    │       │
    │       └─ NO → fall through ↓
    │
    └─ Geometric RRF (compass + contrastive), top-3 windows (~2s)
```

### Why vec_inject first

At Apollo 11 scale (3,625 facts), vec inject delivers the first token in ~200ms vs ~2s
for window replay. The adaptive threshold ensures the injection rate stays ~85% regardless
of index size (fixed 15% threshold collapses to 0% beyond N≈50 facts).

### Classification pipeline

1. Forward-pass query to L26 (commitment layer)
2. Project through 5-class linear probe (calibrated during `--phases mode7`)
3. Highest activation class wins
4. Low confidence → fall back to FACTUAL

## Usage

```bash
# No flags — Mode 7 decides everything
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Who were the crew of Apollo 11?"

# Pre-calibrate probes during prefill (avoids 45s calibration on first query)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases windows,compass,vec_inject,mode7
```

## Stderr output (verbose routing trace)

```
Query classification: factual (confidence=3286.3, 197ms)
Vec inject [CONFIDENT]: top_score=0.2352 (load=349ms, route=46.6ms)
INJECT: W472[367] score=0.2299 c=+71367.5234 tok='Nell'
Inject 2-pass: 121ms
Routing: FACTUAL → vec_inject (2574ms)
```

or on fallback:

```
Query classification: factual (confidence=4315.2, 205ms)
Vec inject below threshold — falling back to geometric.
Routing: factual → 3 windows (1820ms): [472, 224, 115]
```

## What NOT to do (ruled out by experiment)

| Approach | Why killed |
|----------|-----------|
| Fixed 15% confidence threshold | Collapses to 0% injection at N>50 facts |
| Two-layer cascade L26→L29 | Redundant signal — one layer is sufficient per query type |
| Tension probe as event detector | Domain inflation saturates score range, val accuracy below chance |
| L14 K-vectors for novel entities | L14 entity compass is parametric-only; novel facts score ~random |
| Multi-head K concatenation | 4× storage cost, minimal discrimination gain |
| Multi-layer K concatenation (L23+L29) | L23 ≈ L29 in PCA structure, redundant |
| Entity-enhanced K-vectors in hidden space | W_K projection collapses entity-discriminative directions regardless |
| Contrastive K-vectors | Same root cause — W_K can't be controlled without retraining |

## Roadmap

**Hybrid entity routing** (next): entity string match (20 bytes/fact) as pre-filter before
Q·K scoring. Entity-explicit queries ("Who was Neil Armstrong?") match by string in O(N·k),
skip Q·K entirely. ~75–83% of factual queries. Estimated injection rate: 92% at any scale.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/generate/_mode7.py`
Probes: `src/chuk_lazarus/cli/commands/context/generate/_probes.py`
Vec inject provider: `src/chuk_lazarus/inference/context/vec_inject/providers/_local_file.py`
