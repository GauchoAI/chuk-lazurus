# Vec Inject — Vector Injection Index

**Phase**: `vec_inject`
**Output**: `vec_inject.npz`
**Experiments**: 2bd41b18 (baseline), e43eaddf (multi-fact superposition), 2178e184 (orthogonality proof)

---

## What it is

Vec inject stores the minimum information needed to reproduce what L29 H4 does during a
document forward pass: copy a directional component of the answer token's embedding into
the residual stream.

One fact position = **12 bytes**:

| Field | Type | Meaning |
|-------|------|---------|
| `token_id` | `int32` | Answer token at this position |
| `coefficient` | `float32` | `c = dot(R_L30, embed(token_id))` |

Plus a routing K vector (256D float16 = 512 bytes) and a `distinctive` flag (int32 = 4 bytes).
Total per fact: **528 bytes**.

---

## The injection formula

The L30 bare query residual is **orthogonal to all answer token embeddings**
(measured angles: 88.97°–91.76°, cosines −0.031 to +0.018).

Therefore:

```
R_patched = R_bare + (c - dot(R_bare, e)) * e
          ≈ R_bare + c * e          [since dot(R_bare, e) ≈ 0]
```

The injection is **purely additive** — writing into blank space. The coefficient `c` IS
the full magnitude of the answer signal, not a differential.

---

## How it works

### Prefill (one-time)

For each window and each fact position `p`:

1. Run `prefill_to_layer(target_layer=29)` → residual `h` entering L30
2. **K vector** (routing): apply L29 K-projection to `h[p]`
3. **Coefficient** (injection): `c = dot(h[p], embed(w_tokens[p]))`
4. **Distinctive flag**: `len(tokenizer.decode([token_id]).strip()) >= 4`
5. Store `(token_id, coefficient, k_vec, distinctive)` per fact

Both K and c extracted in a **single forward pass**.

### Donor design rule (CRITICAL)

The donor context must be constructed so the model **predicts** the answer at the last
position — the answer token appears EARLIER in the donor, absorbed through attention.

```
✅ CORRECT: "The city is Volt. Zarkov was founded in the city of"
   → last position predicts Volt (donor P=78.9%)

✗ WRONG: "Zarkov was founded in the city of Volt"
   → last position computes what comes AFTER Volt (donor P≈0)
```

### Query time (per-query)

```
query → prefill_to_layer(L29) → Q vector
      → cosine(Q, K_index)    → top-k facts      [Metal matmul]
      → adaptive threshold    → routing_confident flag
      → filter: distinctive   → safe facts only
      → inject top-1 match    → two-pass generate
```

### Adaptive threshold (replaces fixed 15%)

At small N the fixed 15% threshold works. At scale it fails:

| N facts | Max Q·K score | Fixed 15% result | Adaptive (mean × 2.0) result |
|---------|--------------|-----------------|------------------------------|
| N=12 | 20–40% | ~50% inject | ~92% inject |
| N=100 | 5–7% | ~5% inject | ~85% inject |
| N=3,625 | <1% | ~0% inject | ~85% inject |

The adaptive threshold `max(floor, mean_score × 2.0)` scales automatically with the score
distribution. The fixed floor (default 0.15) prevents false positives at tiny N.

### Two-pass generation (required)

`prefill_from_layer(start=30)` leaves L0-29 KV empty → degenerate generation after
the first token. Solution:

```
Pass 1: L0→29 → inject at L30 → L30→33  →  injection-biased first-token logits
Pass 2: full prefill(prompt)              →  proper L0-33 KV for continuation
Extend proper KV with first token → decode normally
```

Cost: ~2× prefill time (~120ms). Required for coherent generation.

---

## Inject-matched-only architecture

**Inject-all is broken.** N independent injections are mathematically non-interacting
(e_i ⊥ e_j, e_i ⊥ R_bare). But L31–L33 amplify the **largest coefficient** regardless
of entity identity — there is no address bus in the amplification layers.

Wrong injection at 0.05% of residual energy overrides 99.95% structural context at >99%
confidence. The model cannot use semantic context from the query to override the injected
signal.

```
CORRECT architecture:
  1. Q·K routing → select ONE matching fact
  2. Inject only that one fact at L30
  3. Continue L31→L33 → generate

BROKEN: inject N facts, let the model pick → wrong answer >90% of the time
```

## Token distinctiveness constraint

Facts must have **distinctive first tokens** (≥4 stripped chars, rare in vocabulary).
Non-distinctive tokens fail because the model's parametric prior for common prefixes is
too strong.

| Token | Stripped | Distinctive? | Why |
|-------|----------|-------------|-----|
| " Nell" | "Nell" | ✅ YES | 4 chars, uncommon |
| " Voltara" | "Voltara" | ✅ YES | Novel entity |
| " Cren" | "Cren" | ✅ YES | Novel entity |
| " sell" | "sell" | ✅ YES | 4 chars |
| " A" | "A" | ✗ NO | 1 char — too common |
| " Bel" | "Bel" | ✗ NO | 3 chars — borderline |
| " St" | "St" | ✗ NO | 2 chars — very common prefix |

The `distinctive` flag is set at extraction time and stored in the index. Non-distinctive
facts are not injected — they fall back to window replay.

---

## Scaling properties

| N facts | Accuracy (correct routing) | Notes |
|---------|--------------------------|-------|
| 1 | 56–99.96% | Depends on token distinctiveness |
| 3 | 59–99.97% | Flat, no degradation |
| 6 | 61–99.99% | Stable |
| 7 | 90–99.98% | Stable |
| 3,625 (Apollo 11) | ~85% inject rate | Adaptive threshold required |

**No geometric scaling knee.** The 2560D embedding space has room for hundreds of
nearly-orthogonal fact directions (mean pairwise cosine 0.062 for 7 answer tokens).
Routing precision and token distinctiveness are the limits, not geometry.

---

## What NOT to do (ruled out by experiment)

| Approach | Why killed |
|----------|-----------|
| Fixed 15% confidence threshold | Collapses to 0% injection beyond N≈50 |
| Inject-all (simultaneous N facts) | L31–L33 amplify largest coefficient, catastrophic |
| Donor ending ON the answer token | Last position computes continuation, not answer |
| L14 K-vectors for novel entities | L14 entity compass is parametric-only; novel facts near-random |
| Multi-head K concatenation | 4× storage, minimal discrimination gain |
| Multi-layer K concatenation (L23+L29) | L23 ≈ L29, redundant |
| Entity-enhanced hidden space | W_K collapses entity-discriminative directions, uncontrollable |

---

## Storage

| Index | Per fact | 3,625 facts (Apollo 11) |
|-------|----------|------------------------|
| KV cache (full window replay) | ~512 KB | ~181 MB |
| `vec_inject.npz` | ~528 B | ~1.86 MB |
| **Compression ratio** | — | **97:1** |

---

## Usage

```bash
# Full prefill including vec_inject
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx \
    --phases windows,compass,vec_inject,mode7

# Add to existing checkpoint (no re-processing windows)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx \
    --phases vec_inject

# Use explicitly (Mode 7 uses it automatically)
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx \
    --prompt "Who were the crew of Apollo 11?" \
    --replay vec_inject
```

---

## Index format (`vec_inject.npz`)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `w{N}/k_vecs` | `(n_facts, head_dim)` | float16 | K vectors at L29 KV-head |
| `w{N}/token_ids` | `(n_facts,)` | int32 | Answer token per fact |
| `w{N}/coefs` | `(n_facts,)` | float32 | Injection coefficient c |
| `w{N}/positions` | `(n_facts,)` | int32 | Token position in window |
| `w{N}/distinctive` | `(n_facts,)` | int32 | 1=distinctive, 0=common prefix |
| `layer` | scalar | int | Retrieval layer (29) |
| `kv_head` | scalar | int | KV head index (2 for Gemma 4B) |
| `query_head` | scalar | int | Query head (4) |
| `inject_layer` | scalar | int | Injection layer (30) |

Key names are defined in `VecInjectMetaKey` and `VecInjectWindowKey` — no magic strings.

---

## Roadmap

**Next: hybrid entity routing.** Store entity string (20 bytes/fact) alongside K-vector.
For entity-explicit queries ("Who was Neil Armstrong?"), match by string in O(N·k),
bypass Q·K entirely. Estimated improvement: ~92% injection rate at any scale vs ~85%
with Q·K alone. Required index change: add `w{N}/entity_strings` to `vec_inject.npz`.

---

## Related

| Component | Description |
|-----------|-------------|
| `--phases kvectors` | Routing only (K vectors, no coefficients) |
| `kv_route_index.npz` | Legacy routing index — read by `LocalVecInjectProvider` as fallback |
| Mode 6 (`--phases pages`) | Full KV cache injection — different technique, much larger |
| `VecInjectProvider` | Protocol — implement for Redis/MCP network backends |
