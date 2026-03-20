# H4 Output Routing

**Status:** Production-ready
**Accuracy:** 4/4 at N=12 same-template entity clusters
**Margins:** 2.434×–8.387× (vs 0.977×–0.984× for raw H-space cosine)

## Problem

Raw H-space cosine routing at L29 fails for same-template entity clusters at N≥8.
At N=12 (8 city-founding facts + 4 "agreed to do" facts), Zarkov (0.977×) and Nexaris
(0.984×) route to wrong facts. Root cause: the full L29 residual mixes entity signal
from H4 (copy head) with template structure from 7 other structural heads. When 8+
entities share the same template, the template signal overwhelms the entity signal.

All linear full-residual approaches fail (confirmed Item 22 on kill list):
- Variance weighting (sqrt, log, top-K): collapses to ≈1.000× — high-var dims ARE template PCs
- Entity-position routing: 0.998×/0.994× — better but doesn't cross 1.0
- Contrastive delta: Zarkov gets WORSE (0.437×) — entity delta 80% correlated across entities
- Fisher discriminant: 0.994×/0.998× — improves but never crosses 1.0

## Solution: Isolate H4's Attention Output

H4 at L29 is the copy head: it attends to entity-identity-carrying token positions and
copies their V vector into the residual. Extracting H4's contribution in isolation — before
the 7 structural heads add their template-contaminating signal — gives clean entity routing.

## Mechanism

```python
# During prefill, at layer 29 (last position)
h_pre = kv_gen.prefill_to_layer(input_ids, target_layer=28)  # output of L28 = input to L29
x = layer29.pre_attn_norm(h_pre)                            # (1, S, 2560)
q, k, v = layer29.project_qkv(x, B, S, offset=0)           # post q/k_norm, post RoPE

H4_IDX  = 4
KV_IDX  = H4_IDX // n_rep   # = 2 for Gemma 4B (n_rep=2)
dh      = 320                # head_dim for 4B

# Last query position attends over all positions (causal: no mask needed)
q_last = q[:, H4_IDX, -1:, :]       # (1, 1, 320)
k_kv   = k[:, KV_IDX,  :, :]        # (1, S, 320)
v_kv   = v[:, KV_IDX,  :, :]        # (1, S, 320)

scores = matmul(q_last, k_kv.transpose(0, 2, 1)) * attn_scale  # (1, 1, S)
attn_w = softmax(scores, axis=-1)                               # (1, 1, S)
h4_out = matmul(attn_w, v_kv)[:, 0, :]                        # (1, 320)

# Project H4's slice through O_proj → hidden space contribution
o_weight = layer29.self_attn.o_proj.weight                    # (2560, 2560) in MLX Linear
h4_contrib = h4_out @ o_weight[:, H4_IDX*dh:(H4_IDX+1)*dh].T # (1, 2560)
```

Route using: `cosine(h4_contrib_query, h4_contrib_fact)`

## Results at N=12

| Query | Correct sim | Best wrong | Ratio | vs Baseline |
|---|---|---|---|---|
| Q1 Zarkov | 0.5178 | 0.2127 | **2.434×** ✓ | was 0.977× ✗ |
| Q2 Nexaris | 0.8531 | 0.1456 | **5.859×** ✓ | was 0.984× ✗ |
| Q11 Namath | 0.9609 | 0.1146 | **8.387×** ✓ | was 1.007× ✓ |
| Q12 Marchand | 0.9572 | 0.2116 | **4.524×** ✓ | was 1.005× ✓ |

Accuracy: 4/4 (vs 2/4 baseline)

## Why H4 Attends to BOS (and Still Works)

H4 attention distribution for Q1_zarkov bare query:

| Position | Token | Weight |
|---|---|---|
| 0 | `<bos>` | 89.45% |
| 7 | ` Z` | 2.70% |
| 1 | `<start_of_turn>` | 2.70% |
| 18 | `\n` | 1.54% |
| 14 | `<end_of_turn>` | 0.88% |

BOS dominates at 89%. This is expected — H4 is partly a BOS-sink (consistent with
the L26 BOS collapse finding). However:

- **BOS V is ~constant** across all prompts. BOS (position 0) at any layer can only
  attend to itself — its residual is purely self-referential and identical across prompts.
  BOS contribution to H4's output is a constant offset that cancels in cosine similarity.

- **The 2.7% on entity token V carries the identity signal.** Entity token V at L29
  encodes that entity's identity. Even though the weight is small, the V vector is
  entity-specific and discriminative. The ratio between correct and wrong facts is
  driven by this entity V, not by the BOS component.

- **H4 output norm varies by entity:** Zarkov query ‖h4‖=0.44, Nexaris ‖h4‖=3.85.
  Smaller norm = weaker entity signal in H4, but 2.434× margin is still decisive at N=12.

## Why Variance Weighting Fails

```
Per-dim variance across 8 city facts: Mean=27,378, Max=9,485,881, Std=188,369
sqrt-weighting result: Q1=1.000×, Q2=1.000× — ratio degenerates
```

High-variance dimensions ARE the template principal components. Amplifying them by
sqrt(var) makes all cosines collapse toward 1.0000 — correct and wrong candidates become
equally similar in the high-variance subspace. Entity signal lives in the LOW-variance
tail, which weighting suppresses.

**Top-K variant (K=64..1024):** No K value resolves Zarkov. At K=64 (highest-variance dims):
Q1=0.9955×, Q2=0.9961× — still wrong. Entity discrimination is not in ANY high-variance subspace.

## Storage

- Per fact: 2560D float32 = **10,240 bytes ≈ 10 KB**
- Apollo scale (12K facts): **120 MB** total
- Alternative: 320D pre-O_proj (1,280 bytes/fact, 15.4 MB at Apollo scale) — requires
  metric alignment validation since O_proj is not an isometry

## Integration with Two-Tier Routing

```
Query arrives
    │
    ▼
K-space Q·K routing  ────────────────► inject (if score > adaptive threshold)
    │ fail (~60%)                           ~40% of queries
    ▼
H4 output routing   ─────────────────► inject (if cosine > threshold)
    │ fail (~15%)                           ~45% additional queries
    ▼
Replay fallback                             ~15% remaining
```

Expected combined injection rate: ~85% (up from 40% K-space alone)

No extra forward passes. H4 extraction adds ~O(S × head_dim) ops during existing prefill.

## Implementation Notes

- `kv_gen.prefill_to_layer(ids, target_layer=N)` runs layers 0..N (inclusive)
  and returns the residual after layer N. Use `target_layer=28` to get the input to L29.

- O_proj weight in MLX `nn.Linear` is stored as `(out_features, in_features)` =
  `(hidden_size, num_heads × head_dim)` = `(2560, 2560)`. H4 columns are `[:, 1280:1600]`.

- For Gemma 4B: `n_rep=2` (8 query heads / 4 KV heads), so H4 uses KV head index 2.

- The post-attention residual add and norms are NOT applied — we want H4's raw
  contribution to the attention output, not the full residual update.

## Experiment

Script: `examples/inference/routing_wall_breakers.py`
Results: `experiments/routing-wall-breakers/RESULTS.md`
Data: `experiments/routing-wall-breakers/m2_h4_vectors_L29.npz`
