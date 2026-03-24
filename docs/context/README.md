# Context System — Unlimited Context via Windowed KV

The context system breaks documents into fixed-size token windows, prefills each through the model, and stores extracted signals (residuals, compass bearings, K-vectors, keywords) for fast retrieval at query time. The model never sees more than one window at a time during prefill, but at generation time, multiple windows are concatenated into the KV cache so the model can attend across them.

## Architecture

```
Document (370K tokens)
    ↓ tokenize + window
725 windows × 512 tokens
    ↓ prefill each window
Per-window artifacts:
  ├── checkpoints (boundary KV)
  ├── residuals (Markov state)
  ├── compass residuals (L26 PCA)
  ├── surprise scores (novelty)
  ├── sparse keywords (BM25)
  ├── K-vectors (fact addressing)
  └── pages (pre-RoPE K,V)
    ↓ query arrives
Routing: select relevant windows
    ↓ replay selected windows
KV cache: [preamble][W_a][W_b][W_c][postamble+query]
    ↓ autoregressive decode
Response grounded in document
```

## Two-phase operation

**Prefill** (`lazarus context prefill`): One-time cost. Process the document, extract routing indexes. Multiple extraction phases run independently — you can add compass routing to an existing library without re-running the expensive window forward passes.

**Generate** (`lazarus context generate`): Per-query. Route the query to relevant windows, replay them into the KV cache, generate a grounded response. Different routing strategies trade off speed, accuracy, and query type coverage.

## Documentation

### Prefill phases

Each phase extracts a different signal from the prefilled windows:

| Phase | Doc | What it produces |
|-------|-----|-----------------|
| windows | [windows.md](prefill/windows.md) | Core forward pass — KV checkpoints + boundary residuals |
| interval | [interval.md](prefill/interval.md) | 8 interior residuals per window for fine-grained matching |
| compass | [compass.md](prefill/compass.md) | PCA basis at commitment layer for geometric routing |
| darkspace | [darkspace.md](prefill/darkspace.md) | Whitened frame bank projections for cross-corpus routing |
| surprise | [surprise.md](prefill/surprise.md) | Per-token novelty scores (anomaly detection) |
| sparse | [sparse.md](prefill/sparse.md) | Keyword index for BM25 text-level routing |
| kvectors | [kvectors.md](prefill/kvectors.md) | K-vector routing index at fact positions |
| kvectors_full | [kvectors_full.md](prefill/kvectors_full.md) | K-vectors at every position (100% coverage) |
| vec_inject | [vec_inject.md](prefill/vec_inject.md) | 12-byte per-fact injection index (K vector + coefficient) |
| pages | [pages.md](prefill/pages.md) | Pre-RoPE K,V for instant page injection |
| mode7 | [mode7_calibrate.md](prefill/mode7_calibrate.md) | Query classifier + engagement/tension probe calibration |

### Routing strategies

Each strategy scores windows differently at query time:

| Strategy | Doc | Approach |
|----------|-----|----------|
| geometric | [geometric.md](routing/geometric.md) | Compass + contrastive RRF (default for factual) |
| compass | [compass.md](routing/compass.md) | PCA cosine at commitment layer |
| contrastive | [contrastive.md](routing/contrastive.md) | Query-specific subspace discovery |
| directed | [directed.md](routing/directed.md) | Query-defined 1D projection |
| bm25 | [bm25.md](routing/bm25.md) | Token-level keyword scoring |
| sparse | [sparse.md](routing/sparse.md) | BM25 over pre-extracted keyword index |
| kv_route | [kv_route.md](routing/kv_route.md) | L29 H4 Q.K — model's own fact addressing |
| temporal | [temporal.md](routing/temporal.md) | Evenly spaced windows for global queries |
| qk | [qk.md](routing/qk.md) | Model's own Q/K attention projections |
| attention | [attention.md](routing/attention.md) | Full attention weights over sampled windows |
| preview | [preview.md](routing/preview.md) | Query perplexity reduction per window |
| deflection | [deflection.md](routing/deflection.md) | Residual deflection from context injection |
| twopass | [twopass.md](routing/twopass.md) | Speculative generation then residual routing |
| darkspace | [darkspace.md](routing/darkspace.md) | Dual-score compass + directed in PCA space |
| guided | [guided.md](routing/guided.md) | Compass geometry x token overlap |
| hybrid | [hybrid.md](routing/hybrid.md) | BM25 pre-filter then preview re-rank |
| residual | [residual.md](routing/residual.md) | Legacy mean-centered cosine similarity |

### Meta-strategies (multi-step routing)

| Strategy | Doc | Approach |
|----------|-----|----------|
| mode7 | [mode7.md](routing/mode7.md) | Auto-classifies query type, dispatches to appropriate router |
| unified | [unified.md](routing/unified.md) | Binary factual/exploration split with auto-detection |
| iterative | [iterative.md](routing/iterative.md) | Multi-round compass navigation with generation-guided shifting |
| probe | [probe.md](routing/probe.md) | Compass candidates ranked by grounding probe |

## Quick reference

```bash
# Prefill a document (all phases)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --window-size 512

# Generate with auto-routing (Mode 7)
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy mode7

# Generate with specific strategy
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy geometric
```
