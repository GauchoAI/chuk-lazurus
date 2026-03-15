# Prefill — Windowed Context Library Builder

Tokenizes a document and saves a windowed KV checkpoint library that the generation engine can query at inference time without re-reading the source text.

## What it produces

```
shakespeare_ctx_512/
├── manifest.json              — metadata: model, window size, tokens, windows, config hash
├── tokens.bin                 — raw token IDs (uint32, little-endian)
├── windows.json               — per-window metadata: offsets, counts, text preview
├── checkpoints.npz            — boundary KV (last-position K,V at every layer per window)
├── residuals.npz              — per-window Markov residual vectors at boundaries
├── interval_residuals.npz     — 8 interior residuals per window (sub-window retrieval)
├── compass_residuals.npz      — commitment-layer residuals projected through PCA basis
└── compass_basis.npz          — PCA mean, content/structural basis, layer + PC range
```

Each file has a distinct role in the generation pipeline:

| File | Purpose |
|------|---------|
| `checkpoints.npz` | Inject a window's context without re-running the forward pass |
| `residuals.npz` | Markov state continuity across window boundaries; alternative injection path to boundary KV |
| `interval_residuals.npz` | Fine-grained retrieval — find the right position within a window |
| `compass_*` | Route a query to the right windows via residual similarity |

## Usage

```bash
# Full prefill (windows + interval residuals + compass calibration)
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
```

## Phases

By default, prefill runs all phases: window prefill, interval residual extraction, and compass calibration. Use `--phases` to run only what you need:

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

# Everything (default)
lazarus context prefill --model google/gemma-3-4b-it --input document.txt \
    --checkpoint ./ctx/ --phases all
```

| Phase | What it does | Requires |
|-------|-------------|----------|
| `windows` | Tokenize + forward pass per window, save checkpoints + boundary residuals | Input text |
| `interval` | Extract 8 interior residuals per window (or every position in full mode) | Existing library |
| `compass` | Calibrate commitment-layer PCA basis for routing | Existing library |
| `darkspace` | Whitened frame bank projection for cross-corpus routing | Existing library |
| `pages` | Pre-RoPE K,V page extraction for instant injection | Existing library |
| `all` | All of the above (default) | Input text |

When `--phases` doesn't include `windows`, the prefill loop is skipped. The engine loads the existing library from disk and runs only the requested extraction passes — useful for re-running compass calibration or adding pages without repeating the expensive forward passes.

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
├── __init__.py        — re-exports context_prefill_cmd
├── _cmd.py            — CLI entry point, SIGINT handling, prefill loop
├── _save.py           — save orchestrator: coordinates all file writes
├── _restore.py        — resume from partial library on disk
├── _checkpoints.py    — incremental KV checkpoint + residual zip writes
├── _compass.py        — PCA basis calibration for compass routing
├── _darkspace.py      — whitened frame bank projection + corpus calibration
├── _interval.py       — interval/full residual extraction
├── _pages.py          — pre-RoPE page extraction for instant injection
├── _npz.py            — chunked mx.savez (works around 1024 kwarg limit)
└── _progress.py       — progress display helpers
```

## Key design decisions

**Incremental saves.** Periodic saves (every 5 minutes wall time) append only new windows to the existing zip files — no full rewrite. The final save also appends, then runs extraction passes.

**Memory eviction.** After each periodic save, checkpoints and residuals for already-saved windows are evicted from GPU memory. Only the most recent window is kept (needed to seed the next one). This keeps memory constant regardless of corpus size.

**Two-stage SIGINT.** First Ctrl-C finishes the current window and saves gracefully. Second Ctrl-C hard-exits. A partial library is always safe to resume.

**Lazy loading on restore.** `mx.load()` returns a lazy dict — arrays are only materialized when accessed. This caps peak memory during resume to one window at a time.

**Randomized SVD.** Compass calibration uses `sklearn.utils.extmath.randomized_svd` instead of full SVD. Since we only keep 16–64 PCs, randomized SVD caps memory and compute without meaningfully affecting basis quality.

**Selective phases.** The `--phases` flag decouples extraction passes from the prefill loop. Each phase (windows, interval, compass, darkspace, pages) can be run independently. This means you can iterate on compass calibration or add pages to a library without re-running the expensive window forward passes.
