## Part 12: Position-Class Pruning - How Much Can We Compress?

### 12.1 The Question

Position analysis (Part 11) showed 93% position-coded routing and massive within-class redundancy (8-11 "end" experts per layer, 6-8 "start" experts). Cross-layer ablation (Part 10) showed 30/32 experts can be removed at a single layer. Can we exploit this redundancy for compression?

The test: keep only 4 experts (1 per position class) and ablate the other 28. If facts survive, that's **87.5% expert parameter reduction**.

### 12.2 Expert Position Class Distribution

```
Layer | start | early_mid | late_mid | end
------|-------|-----------|----------|-----
L8    |     8 |         4 |        5 |  10
L12   |     8 |         2 |        3 |  12
L16   |     6 |         4 |        3 |  11
L20   |     8 |         2 |        7 |  11
```

"End" experts dominate (10-12 per layer), reflecting the model's investment in the output prediction position. At L16, the position classes are:

- **start** (6): E7, E8, E25, E20, E24, E13
- **early_mid** (4): E19, E21, E5, E17
- **late_mid** (3): E1, E22, E12
- **end** (11): E10, E26, E29, E30, E2, E14, E0, E31, E4, E11, E3

### 12.3 Expert Sets Tested

| Condition | Kept Experts | Selection Strategy |
|-----------|-------------|-------------------|
| **Diverse (1/class)** | [24, 5, 1, 2] | Best expert per position class across layers |
| **Same-position** | [2, 0, 26, 31] | Top-4 "end" specialists |
| **Arbitrary** | [28, 29, 30, 31] | Fixed indices (Part 10 baseline) |
| **Fact-specific** | [13, 14, 4, 31] | Fact's L16 top-4 routing |

### 12.4 All-Layer Results: 4 Experts Is Below the Floor

```
           Condition |         Kept | Correct | Total | Accuracy
----------------------------------------------------------------------
           diverse_4 | [24, 5, 1, 2] |       0 |     8 |      0%
          same_pos_4 | [2, 0, 26, 31] |       0 |     8 |      0%
         arbitrary_4 | [28, 29, 30, 31] |       1 |     8 |     12%
     fact_specific_4 | [13, 14, 4, 31] |       0 |     8 |      0%
```

**All conditions fail.** 4 experts at ALL 24 layers is below the minimum viable computation budget regardless of selection strategy. Position diversity doesn't rescue extreme pruning.

Output quality also degrades uniformly:

| Condition | Avg Repetition | Sample Degenerate Output |
|-----------|---------------|-------------------------|
| Diverse | 0.33 | "1. 2. 3. 4. 5. 6. 7. 8. 9. 10." |
| Same-position | 0.59 | "10.5. 10.5. 10.5. 10.5. 10.5." |
| Arbitrary | 0.55 | "a symbol that is a symbol that is a symbol" |
| Fact-specific | 0.29 | (empty output) |

### 12.5 Single-Layer Results: 87.5% Pruning Works

```
            L16_1_per_class: keep 4 experts [7, 19, 1, 10] → 8/8 facts (100%)
            L16_2_per_class: keep 8 experts → 8/8 facts (100%)
```

**At a single layer, 1 expert per position class (4 total) preserves ALL facts.** This is a 28/32 = **87.5% reduction** at that layer with zero fact loss.

Sample outputs (1 per class at L16):

```
"The capital of France is"     → "Paris. The capital of France is Paris."
"The speed of light is approx" → "299,792,458 meters per second. This is a fund..."
"The CEO of Microsoft is"      → "Satya Nadella. The CEO of Microsoft is Satya..."
```

Facts are preserved. Continuations show some repetition (avg 0.49) but are coherent and on-topic.

### 12.6 The Compression Landscape

```
Pruning Scope                                   | Fact Accuracy
------------------------------------------------|---------------
4 experts at L16 only (1/class)                 | 100%  ← WORKS
8 experts at L16 only (2/class)                 | 100%  ← WORKS
4 experts at ALL 24 layers (diverse)            |   0%  ← FAILS
4 experts at ALL 24 layers (arbitrary)          |  12%  ← FAILS
30/32 ablated at L16 only (Part 10)             | 100%  ← WORKS
28/32 ablated at all layers (Part 10)           |  12%  ← FAILS
```

The pattern: **aggressive pruning works at individual layers but not uniformly across all layers.** The model needs a minimum global computation budget that 4/32 per layer doesn't meet.

### 12.7 Practical Compression Strategy

The findings suggest a **layer-alternating pruning** approach:

```
Layer:    L0    L1    L2    L3    L4    ...   L22   L23
Experts:  4     32    4     32    4     ...   32    4
Pattern:  prune full  prune full  prune       full  prune

Expert reduction: 50% of layers × 87.5% reduction
                = 43.75% total expert parameter reduction
```

Or more aggressively:

```
Layer:    L0    L1    L2    L3    L4    L5    ...
Experts:  4     4     32    4     4     32    ...
Pattern:  prune prune full  prune prune full

Expert reduction: 67% of layers × 87.5% reduction
                = 58% total expert parameter reduction
```

Each pruned layer keeps 1 expert per position class. Full layers provide the computation budget that adjacent pruned layers lack.

### 12.8 What Didn't Work and Why

**Position diversity doesn't help at the global minimum.** The diverse set (0%) performed the same as the same-position set (0%). This is because at 4/32 experts per layer globally, the model doesn't have enough total computation to maintain coherent generation regardless of how well those 4 experts cover position space. It's like asking "should this tiny engine be a V4 or an I4?" when the real problem is that you need a bigger engine.

**Fact-specific experts are worse than arbitrary ones.** Keeping the fact's own L16 top-4 at all layers (0%) performed worse than arbitrary [28-31] (12%). This is because L16 routing preferences are optimal for L16 but suboptimal at other layers, while arbitrary experts may accidentally provide better coverage.

### 12.9 Next Step: Finding the Global Minimum

The gap to fill: between 4 experts (fails) and 28 experts (Part 10 shows this works for single-layer), what is the minimum per-layer count when applied globally?

Testing 8, 12, 16 position-diverse experts at all layers would map the compression curve and identify the practical pruning limit for uniform global pruning.

---
