## Part 1: Where Facts Live

### 1.1 Parametric Memory Probing

**Question**: At which layers can we classify fact types from hidden states?

We trained linear probes at each layer to classify prompts into four fact types: entity, numeric, temporal, and procedural.

| Layer | Depth | Category Accuracy |
|-------|-------|-------------------|
| L4    | 17%   | **100%** |
| L6    | 25%   | 67% |
| L8    | 33%   | **100%** |
| L10   | 42%   | 67% |
| L12   | 50%   | 89% |
| L13   | 54%   | **100%** |
| L16   | 67%   | 67% |
| L18   | 75%   | **100%** |
| L20   | 83%   | **100%** |

**Finding**: Fact type classification shows a bimodal pattern with peaks at L4, L8, L13, L18, L20. This suggests discrete processing stages rather than continuous refinement. The L13 peak aligns with prior findings about vocab-aligned classifiers in GPT-OSS.

### 1.2 Fact Type Clustering

We measured intra-class cosine similarity to assess how tightly each fact type clusters in hidden space.

| Fact Type | L4 | L8 | L12 | L13 |
|-----------|-----|-----|------|------|
| Temporal | **0.84** | **0.80** | 0.66 | 0.70 |
| Entity | 0.80 | 0.77 | 0.67 | 0.72 |
| Numeric | 0.67 | 0.71 | 0.66 | 0.71 |
| Procedural | 0.35 | 0.46 | 0.45 | **0.54** |

**Finding**: Procedural facts cluster much more loosely (0.35-0.54) than declarative facts (0.67-0.84). This is the first indication that procedural knowledge uses different encoding.

---
