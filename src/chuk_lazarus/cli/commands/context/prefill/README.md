# Prefill — Windowed Context Library Builder

Tokenizes a document and saves a windowed KV checkpoint library that the generation engine can query at inference time without re-reading the source text.

## What it produces

```
apollo11_ctx_512/
├── manifest.json              — metadata: model, window size, tokens, windows, config hash
├── tokens.bin                 — raw token IDs (uint32, little-endian)
├── windows.json               — per-window metadata: offsets, counts, text preview
├── checkpoints.npz            — boundary KV (last-position K,V at every layer per window)
├── residuals.npz              — per-window Markov residual vectors at boundaries
├── interval_residuals.npz     — 8 interior residuals per window (sub-window retrieval)
├── compass_residuals.npz      — commitment-layer residuals projected through PCA basis
├── compass_basis.npz          — PCA mean, content/structural basis, layer + PC range
├── surprise.npz               — per-token perplexity scores (anomaly/novelty detection)
├── sparse_index.json          — keyword index: per-window novel facts + BM25-ready terms
├── kv_route_index.npz         — K vectors at fact positions (model's own addressing)
└── pages.npz                  — pre-RoPE K,V pages for instant injection (optional)
```

Each file has a distinct role in the generation pipeline:

| File | Purpose |
|------|---------|
| `checkpoints.npz` | Inject a window's context without re-running the forward pass |
| `residuals.npz` | Markov state continuity across window boundaries |
| `interval_residuals.npz` | Fine-grained retrieval — find the right position within a window |
| `compass_*` | Route a query to the right windows via residual similarity |
| `surprise.npz` | Per-token novelty scores — identifies where the model encountered unexpected content |
| `sparse_index.json` | BM25 keyword routing — fast text-level window selection |
| `kv_route_index.npz` | K-vector routing — the model's own Q·K addressing externalised as a router |
| `pages.npz` | Pre-RoPE K,V for instant page injection at generate time (Mode 6) |

## Usage

```bash
# Full prefill (all phases)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --window-size 512

# Interrupted? Resume automatically
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/

# Residual modes: interval (default), full, darkspace, none
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --residual-mode darkspace \
    --frame-bank ./frame_bank.npz

# Full K-vector coverage for production routing
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases windows,compass,kvectors_full
```

## Phases

By default, prefill runs all phases. Use `--phases` to run only what you need:

```bash
# Just prefill windows — skip extraction passes entirely
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases windows

# Recalibrate compass routing on an existing library (no re-prefill)
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases compass

# Extract interval residuals only
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases interval

# Interval + compass together
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases interval,compass

# Add pages to an existing library
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --store-pages --phases pages

# Build sparse keyword index for BM25 routing
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases sparse

# K-vector routing index (sparse fact positions from surprise index)
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases kvectors

# K-vector routing with 100% position coverage (no missed facts)
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases kvectors_full

# Calibrate Mode 7 query classifier + engagement/tension probes
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases mode7

# Everything (default)
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases all
```

| Phase | What it does | Requires | Storage |
|-------|-------------|----------|---------|
| `windows` | Tokenize + forward pass per window, save checkpoints + boundary residuals | Input text | ~checkpoints + metadata |
| `interval` | Extract 8 interior residuals per window (or every position in full mode) | Existing library | ~40 KB/window |
| `compass` | Calibrate commitment-layer PCA basis for geometric routing | Existing library | ~29 MB (725 windows) |
| `darkspace` | Whitened frame bank projection for cross-corpus routing | Existing library + frame bank | Variable |
| `surprise` | Per-token perplexity scoring — identifies novel/unexpected content | Existing library | Small |
| `sparse` | Keyword extraction for BM25 routing (novel facts + terms per window) | Existing library | ~50-200 KB |
| `kvectors` | K-vector extraction at fact positions (sparse index or 8-sample fallback) | Existing library | ~1.8 MB (sparse) |
| `kvectors_full` | K-vector extraction at every position — 100% coverage, no missed facts | Existing library | ~256 KB/window (~181 MB for 725 windows) |
| `pages` | Pre-RoPE K,V page extraction for instant injection | Existing library | ~9 MB/window |
| `mode7` | Calibrate query-type classifier + engagement/tension probes | Existing library | Small |
| `all` | All of the above (default) | Input text | Everything |

When `--phases` doesn't include `windows`, the prefill loop is skipped. The engine loads the existing library from disk and runs only the requested extraction passes — useful for re-running compass calibration or adding pages without repeating the expensive forward passes.

## Routing strategies

The prefill phases produce different routing indexes, each suited to different query types and performance requirements:

| Strategy | Index used | Speed | Best for |
|----------|-----------|-------|----------|
| `geometric` | `compass_residuals.npz` | ~1.5s | General factual queries, broad topic matching |
| `contrastive` | `compass_residuals.npz` | ~1.5s | Discriminating between similar topics |
| `bm25` | `sparse_index.json` | ~10ms | Exact keyword matching, entity names |
| `kv_route` | `kv_route_index.npz` | ~50ms | Position-precise factual routing |
| `mode7` | Auto-selects based on query type | Varies | Production auto-routing |
| `temporal` | `sparse_index.json` + dates | ~10ms | Time-based queries |

Geometric compass routing works at the **window level** — it finds which windows contain relevant content. K-vector routing works at the **position level** — it finds which exact tokens carry the fact. BM25/sparse works at the **keyword level** — fast text matching without model inference.

### K-vector coverage tradeoffs

K-vectors are the model's own addressing mechanism externalised as a router. The coverage mode determines how many positions are indexed:

| Mode | Coverage | Storage (725 windows) | Tradeoff |
|------|----------|----------------------|----------|
| `kvectors` (sparse) | Surprise-guided positions | ~1.8 MB | Only indexes facts that surprised the model — parametric facts may be missed |
| `kvectors` (interval) | 8 samples/window (1.6%) | ~0.7 MB | Pure luck — facts between sample positions are invisible |
| `kvectors_full` | Every position (100%) | ~181 MB | Guaranteed coverage — no missed facts, 10x larger |

For development and small corpora (<1000 windows), `kvectors` with sparse/interval is fine — geometric compass catches what K-vectors miss. For production with large corpora where sub-second routing matters, `kvectors_full` guarantees no coverage gaps.

## Residual modes

| Mode | Interior residuals | Compass routing | Disk usage per window |
|------|-------------------|-----------------|----------------------|
| `interval` | 8 evenly-spaced samples | PCA on commitment layer | ~40 KB |
| `full` | Every position | PCA on all positions | ~5 MB (512-tok window) |
| `darkspace` | Whitened frame bank projection | Pre-computed or corpus-calibrated | Variable |
| `none` | Skipped | Skipped | Checkpoints + metadata only |

## Module layout

```
prefill/
├── __init__.py            — re-exports context_prefill_cmd
├── _cmd.py                — CLI entry point, SIGINT handling, prefill loop
├── _save.py               — save orchestrator: coordinates all file writes
├── _restore.py            — resume from partial library on disk
├── _checkpoints.py        — incremental KV checkpoint + residual zip writes
├── _compass.py            — PCA basis calibration for compass routing
├── _darkspace.py          — whitened frame bank projection + corpus calibration
├── _interval.py           — interval/full residual extraction
├── _surprise.py           — per-token perplexity scoring (novelty detection)
├── _sparse.py             — keyword extraction for BM25 sparse index
├── _kv_route.py           — K-vector extraction (sparse, interval, or full coverage)
├── _mode7_calibrate.py    — Mode 7 query classifier + probe calibration
├── _pages.py              — pre-RoPE page extraction for instant injection
├── _npz.py                — chunked mx.savez (works around 1024 kwarg limit)
└── _progress.py           — progress display helpers
```

## Key design decisions

**Incremental saves.** Periodic saves (every 5 minutes wall time) append only new windows to the existing zip files — no full rewrite. The final save also appends, then runs extraction passes.

**Memory eviction.** After each periodic save, checkpoints and residuals for already-saved windows are evicted from GPU memory. Only the most recent window is kept (needed to seed the next one). This keeps memory constant regardless of corpus size.

**Two-stage SIGINT.** First Ctrl-C finishes the current window and saves gracefully. Second Ctrl-C hard-exits. A partial library is always safe to resume.

**Lazy loading on restore.** `mx.load()` returns a lazy dict — arrays are only materialized when accessed. This caps peak memory during resume to one window at a time.

**Randomized SVD.** Compass calibration uses `sklearn.utils.extmath.randomized_svd` instead of full SVD. Since we only keep 16–64 PCs, randomized SVD caps memory and compute without meaningfully affecting basis quality.

**Selective phases.** The `--phases` flag decouples extraction passes from the prefill loop. Each phase can be run independently. This means you can iterate on compass calibration, add K-vector indexes, or calibrate Mode 7 probes without re-running the expensive window forward passes.

**K-vector coverage modes.** K-vectors work at position-level precision — missing a position means missing the fact. `kvectors_full` guarantees 100% coverage at 10x storage cost. For small corpora where geometric compass already works, the sparse default is sufficient. For production-scale routing, full coverage eliminates the coverage gap entirely.
