# Routing Wall Breakers — Experiment Results
Date: 2026-03-19
Model: mlx-community/gemma-3-4b-it-bf16 (Gemma 3 4B)

## Problem
N≥8 same-template fact cluster routing failure.
Raw H-space cosine at L29: Q1 0.977× WRONG, Q2 0.984× WRONG (city template crowding)

## Results Summary

| Method | Q1 Zarkov | Q2 Nexaris | Q11 Namath | Q12 Marchand | Acc |
|---|---|---|---|---|---|
| Baseline (raw cosine) | 0.977× ✗ | 0.984× ✗ | 1.007× ✓ | 1.005× ✓ | 2/4 |
| M1b sqrt-var weighting | 1.000× ✗ | 1.000× ✗ | 1.000× ✓ | 1.000× ✓ | 2/4 |
| M1c log-var weighting | 0.989× ✗ | 0.992× ✗ | 1.003× ✓ | 1.002× ✓ | 2/4 |
| **M2 H4 output (L29)** | **2.434×✓** | **5.859×✓** | **8.387×✓** | **4.524×✓** | **4/4** |
| M3 entity-position | 0.998× ✗ | 0.994× ✗ | 1.000× ✗ | 1.003× ✓ | 1/4 |
| M4 contrastive | 0.437× ✗ | 0.939× ✗ | 1.147× ✓ | 1.491× ✓ | 2/4 |
| M5 Fisher | 0.994× ✗ | 0.998× ✗ | 1.007× ✓ | 1.006× ✓ | 2/4 |

## M2: H4 Attention Output — The Mechanism

### What it does
During the L29 forward pass, extract only H4's contribution to the attention output:
```python
h4_out = attn_weights_H4 @ V[kv_head_2]  # (320,)
h4_contrib = h4_out @ O_proj[:, 4*320:5*320].T  # (2560,)
```
Route using cosine(h4_contrib_query, h4_contrib_fact).

### Why it works
H4 is the copy head: it attends to entity-identity-carrying tokens and copies their
V vector into the residual. In the full L29 residual, 7 other heads add structural
template signal that crowds out entity signal. H4's isolated output contains
the entity signal without structural contamination.

### H4 attention distribution (Q1 Zarkov bare query)
| Position | Token | Weight |
|---|---|---|
| 0 | `<bos>` | 89.45% |
| 7 | ` Z` | 2.70% |
| 1 | `<start_of_turn>` | 2.70% |
| 18 | `\n` | 1.54% |
| 14 | `<end_of_turn>` | 0.88% |

BOS dominates at 89% — yet 2.7% on 'Z' (first token of "Zarkov") is enough
for 2.434× margin at N=12. This is because the V vector at entity positions is
entity-specific, while BOS V is approximately constant (BOS only attends to itself
through all layers → constant K/V). The entity-token V carries the identity signal.

### H4 output norms
Small norms indicate weak H4 activation for that entity/prompt:
- Zarkov query: ‖h4‖=0.44 (weakest) → smallest margin (2.434×) but still decisive
- Nexaris query: ‖h4‖=3.85 → largest margin (5.859×)

## M1 Failure Analysis: Why Variance Weighting Fails

Per-dim variance across 8 city facts:
- Mean: 27,378 | Max: 9,485,881 | Std: 188,369
- Only 1 dimension >10× mean

sqrt-variance weighting collapses ALL cosines to ≈1.0000. This is diagnostic:
**the high-variance dimensions ARE the template PCs** — they're nearly identical
for both correct and wrong candidates. Entity signal lives in LOW-variance dimensions
(< mean var), where weighting by sqrt(var) amplifies noise, not signal.

Top-K selection (64..1024 dimensions) also fails: best Q1 ratio at K=64 is 0.9955× —
still wrong. No K resolves the city queries.

Conclusion: entity discrimination cannot be recovered from the full residual
by linear reweighting. The template signal in high-variance dims overwhelms.

## M3 Failure Analysis: Entity-Position Routing

Entity position routing improves city query ratios (0.977→0.998, 0.984→0.994)
but also HURTS verb queries (1.007→1.000 for Namath). Neither city query crosses 1.0.

The entity token's residual at L29 is also template-contaminated (the entity
token has processed the full preceding context). Position helps but doesn't isolate.

## M4 Failure Analysis: Contrastive Pairs

Query delta norms: ‖delta‖/‖h_with‖ ≈ 22-28% — substantial signal.
BUT Zarkov delta gets WORSE (0.437×). Inter-entity cosine(delta_Q1, delta_Q2)=0.802 —
Q1 and Q2 deltas are 80% correlated despite different entities. The entity
perturbation in the residual is entangled with template structure in the delta.

## M5 Failure Analysis: Fisher Discriminant

Fisher weight stats: mean=0.754, max=5.50, median=0.662 — almost uniform.
No dimensions have weight >10× mean. Fisher provides mild improvement (0.977→0.994,
0.984→0.998) but cannot cross the 1.0 threshold.

## Implementation Spec: H4 Output Stage-2 Routing

### Storage
- Per fact: 2560D float32 = 10,240 bytes ≈ 10 KB
- Apollo scale (12K facts): 12K × 10KB = 120 MB
- Alternative: 320D pre-O_proj (1,280 bytes/fact, 15.4MB at 12K) — to verify

### Integration with K-space routing
- K-space handles ~40% of queries (structurally distinctive, Q·K separation sufficient)
- H4 routing handles the remaining ~60% (same-template entity-implicit queries)
- Combined expected injection rate: ~85%+ (K-space 40% + H4 60% × ~75%)

### Forward pass modification
During existing prefill, add a hook at L29:
1. After project_qkv, compute H4's attention manually (O(S) extra ops)
2. Extract h4_contrib (2560D or 320D)
3. Store alongside KV cache

No extra forward passes required.

## New Kill List Item: Confirmed Working

H4 output routing is the Stage-2 mechanism. All linear full-residual approaches
(M1 variance, M3 entity-pos, M4 contrastive, M5 Fisher) are confirmed dead ends.

Kill list addendum:
- Item 22: All linear full-residual routing for same-template N≥8 (variance weighting,
  entity-position, contrastive delta, Fisher discriminant)
- Resolution: H4 output isolation (non-linear in the sense of using one head's
  output instead of the full residual)
