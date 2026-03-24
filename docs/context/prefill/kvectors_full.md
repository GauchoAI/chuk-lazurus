# Prefill Phase: kvectors_full

**Phase flag:** `--phases kvectors_full`
**Output:** `kv_route_index.npz`
**Requires:** Existing library with windows
**Cost:** One forward pass per window to the retrieval layer

## What it does

Extracts K vectors at **every position** in every window — 100% coverage, no missed facts. This is the production variant of [`kvectors`](kvectors.md) that eliminates the coverage problem.

## Why full coverage matters

K-vector routing works at the **position level** — you need the K vector at the exact token that carries the fact. Standard `kvectors` samples 8 positions per window (1.6% coverage) or uses surprise-guided positions (which miss parametric facts). Missing a position by even 8 tokens means missing the fact entirely.

Compass routing works at the **window level** — aggregate across samples, take the max. It tolerates sparse sampling. K-vectors don't.

## Coverage comparison

| Mode | Coverage | Can miss facts? | Storage (725 windows) |
|------|----------|----------------|----------------------|
| `kvectors` (sparse) | Surprise positions only | Yes — parametric facts invisible | ~1.8 MB |
| `kvectors` (interval) | 8 samples (1.6%) | Yes — facts between samples invisible | ~0.7 MB |
| `kvectors_full` | Every position (100%) | No | ~181 MB |

## When to use

- **Development / small corpora (<1000 windows)**: `kvectors` is fine. Geometric compass catches what K-vectors miss.
- **Production / large corpora**: `kvectors_full` guarantees no coverage gaps. The 181 MB cost is acceptable for sub-second factual routing on thousands of windows.
- **Speed-critical applications**: K-vector routing runs in ~50ms vs ~1.5s for geometric compass. Full coverage makes this the production fast path.

## Usage

```bash
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases kvectors_full
```

## Storage

512 positions × 256D × bf16 = 256 KB per window.
725 windows = ~181 MB.
10x larger than compass residuals (29 MB), but guaranteed complete.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_kv_route.py` (same as kvectors, with `KVectorMode.FULL`)
