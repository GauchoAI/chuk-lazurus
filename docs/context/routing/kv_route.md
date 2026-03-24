# Routing Strategy: kv_route

**CLI flag:** `--strategy kv_route`
**Index required:** `kv_route_index.npz` (preferred) or `compass_residuals.npz` (fallback)
**Speed:** ~50ms (stored index) or ~500ms (computed from compass)
**Best for:** Position-precise factual routing; sub-second queries on large libraries

## How it works

Uses the model's own Q.K attention mechanism as an external router. The retrieval circuit at L29 H4 copies novel facts from the KV cache at 62% attention weight. The K vectors at that head are what make positions addressable. Q.K IS the model's own routing score.

### Three modes of operation

**Mode 1 — Stored K-vector index** (fastest, ~50ms): Pre-extracted K vectors from `kv_route_index.npz`. One Q.K dot product against all stored K vectors. This is the production fast path.

**Mode 2 — Computed from L29 compass residuals** (~500ms): If compass residuals were extracted at L29 (`--compass-layer 29`), project them through W_K at the retrieval head. No extra forward pass needed.

**Mode 3 — Computed from L26 compass residuals** (~500ms, approximate): If compass residuals are at L26 (default), project through L29's W_K. Approximate but still captures retrieval head geometry.

### Head mapping

For Gemma 4B (8 query heads, 4 KV heads, n_rep=2):
- Query head 4 → KV head 2 (heads 4-5 share KV head 2)
- Configurable via `--routing-layer` and `--routing-head`

## Key insight

1D per fact: knowledge IS the answer token's embedding direction. A scalar projection along the right K vector gives KL divergence of 0.000031 — essentially perfect routing in a single dimension.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy kv_route

# Custom routing head
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "What were the baseball scores?" \
    --strategy kv_route \
    --routing-layer 29 \
    --routing-head 4
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_kv_route.py`
