# Vec Inject — Vector Injection Index

**Phase**: `vec_inject`
**Output**: `vec_inject.npz`
**Experiment**: 2bd41b18

---

## What it is

Vec inject stores the minimum information needed to reproduce what L29 H4 does during a document forward pass: copy a directional component of the answer token's embedding into the residual stream.

One fact position = **12 bytes**:

| Field | Type | Meaning |
|-------|------|---------|
| `token_id` | `int32` | Answer token at this position |
| `coefficient` | `float32` | `c = dot(R_L30, embed(token_id))` |

Plus a routing K vector (256D float16 = 512 bytes) for finding the right facts. Total per fact: ~524 bytes.

---

## How it works

### Prefill (one-time)

For each window and each fact position `p`:

1. Run `prefill_to_layer(target_layer=29)` → residual `h` entering L30
2. **K vector** (routing): apply L29 K-projection to `h[p]`
3. **Coefficient** (injection): `c = dot(h[p], embed(w_tokens[p]))`
4. Store `(token_id=w_tokens[p], coefficient=c)` + K vector

Both extracted in a **single forward pass**.

### Query time (per-query)

```
query → prefill_to_layer(L29) → Q vector
      → cosine(Q, K_index)    → top-k facts      [Metal matmul]
      → scores → argsort      → top-k indices     [Metal sort]
      → gather token_ids, coefs                   [Metal gather]
      → mx.eval()             → one sync
```

### Injection (at L30 in forward pass)

```python
h = forward_to_layer(query, stop=29)          # residual before L30
h = vec_inject_all(h, result.matches, E)      # add c × (e/‖e‖²) per fact
logits = forward_from_layer(h, start=30)      # continue normally
```

`vec_inject_all` adds each match's contribution independently (linear superposition). Multiple facts are stacked, not conflated.

---

## Why it works

L29 H4 (`query_head=4`, `kv_head=2` for Gemma 4B) is a **fact-copy head**: it reads the answer token's embedding direction from the value space and writes a scaled version into the residual. The coefficient `c` captures exactly how much of that direction appears at position `p` during a normal forward pass.

Injecting `c × embed(T) / ‖embed(T)‖²` at L30 reproduces that component without replaying the entire window.

**Result**: KL divergence = 0.000031 vs full KV replay. 1D subspace beats full residual injection (99.85% vs 97.65% exact-match accuracy).

### Why 1D beats full residual

Only 0.05% of the residual energy is the answer direction. The other 99.95% is structural context that belongs to the *query*, not the fact. Injecting only the relevant component avoids contaminating the query's own structural processing.

---

## Storage

| Index | Per fact | 3625 facts (Apollo 11) |
|-------|----------|----------------------|
| KV cache (full window replay) | ~512 KB | ~181 MB |
| `vec_inject.npz` | ~524 B | ~1.86 MB |
| **Compression ratio** | — | **97:1** |

The 7.25 KB fact content represents a 7,700,000:1 compression of the full KV infrastructure.

---

## In-memory layout

After `LocalVecInjectProvider.load()`, all data is **pinned on Metal**:

```
_flat_k_mx         (n_facts, 320)   float32  L2-normalised K vectors
_flat_token_ids_mx (n_facts,)       int32    answer token IDs
_flat_coefs_mx     (n_facts,)       float32  injection coefficients
_flat_wid_mx       (n_facts,)       int32    source window
_flat_positions_mx (n_facts,)       int32    token position
```

The normalisation is computed on CPU once at load time; every subsequent retrieve is a pure Metal dispatch.

---

## Usage

### Prefill

```bash
# Add vec_inject to an existing checkpoint (no re-processing windows)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx \
    --phases vec_inject

# Or include it in a full prefill
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx \
    --phases all,vec_inject
```

Requires `sparse_index.json` for sparse mode (fact positions). Falls back to interval sampling (8 positions/window) if not available.

### Python API

```python
import asyncio
from chuk_lazarus.inference.context.vec_inject import (
    LocalVecInjectProvider,
    vec_inject_all,
)

async def query(checkpoint_dir, kv_gen, embed_matrix, query_text, query_ids):
    # Load once — all data moves to Metal
    provider = await LocalVecInjectProvider.load(checkpoint_dir, kv_gen)
    provider.log_stats()

    # Retrieve — single Metal dispatch
    result = await provider.retrieve(query_ids, query_text, top_k=5)

    print(f"Retrieved {len(result.matches)} facts in {result.retrieval_ms:.1f} ms")
    for m in result.matches:
        print(f"  W{m.window_id}[{m.position}]  score={m.score:.4f}  "
              f"tok={m.token_id}  c={m.coefficient:.4f}")

    # Inject at result.injection_layer (typically L30)
    h_at_injection = ...  # residual from forward_to_layer(query, stop=injection_layer-1)
    h_injected = vec_inject_all(h_at_injection, result.matches, embed_matrix)
    return h_injected
```

### Demo and benchmark

```bash
# Single query demo
uv run python examples/inference/vec_inject_demo.py \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx \
    --query "Who were the crew of Apollo 11?"

# With injection comparison
uv run python examples/inference/vec_inject_demo.py \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx \
    --query "Who were the crew?" \
    --inject

# Latency benchmark (50 queries)
uv run python examples/inference/vec_inject_demo.py \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx \
    --benchmark --n-queries 50
```

---

## Performance

### Typical latency breakdown (Gemma 4B, 3625 facts)

| Component | Time |
|-----------|------|
| L29 forward pass (query encoding) | ~30–50 ms |
| Matmul `(3625×320) × (320,)` on Metal | < 1 ms |
| Argsort + gather (top-5) | < 1 ms |
| `mx.eval()` sync | < 1 ms |
| **Total retrieve()** | ~30–50 ms |

The forward pass dominates. For batched or pre-encoded queries, retrieval is sub-millisecond.

### Network comparison (planned remote providers)

```
Request:  512 bytes  (Q vector at L29)
Response: 12 × top_k bytes  (token_id + coefficient per match)
```

Total round-trip for top-5: 572 bytes — smaller than a DNS response.

---

## Index format (`vec_inject.npz`)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `w{N}/k_vecs` | `(n_facts, head_dim)` | float16 | K vectors at L29 KV-head |
| `w{N}/token_ids` | `(n_facts,)` | int32 | Answer token per fact |
| `w{N}/coefs` | `(n_facts,)` | float32 | Injection coefficient c |
| `w{N}/positions` | `(n_facts,)` | int32 | Token position in window |
| `layer` | scalar | int | Retrieval layer (29) |
| `kv_head` | scalar | int | KV head index (2 for Gemma 4B) |
| `query_head` | scalar | int | Query head (4) |
| `inject_layer` | scalar | int | Injection layer (30) |

Key names are defined in `VecInjectMetaKey` (StrEnum) and `VecInjectWindowKey` — no magic strings anywhere in the codebase.

---

## Related

| Component | Description |
|-----------|-------------|
| `--phases kvectors` | Routing only (K vectors, no coefficients) |
| `kv_route_index.npz` | Legacy routing index — read by `LocalVecInjectProvider` as fallback |
| Mode 6 (`--phases pages`) | Full KV cache injection — different technique, much larger |
| `VecInjectProvider` | Protocol — implement for Redis/MCP network backends |
