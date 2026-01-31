## Part 14: Expert Weight Similarity

### 14.1 The Question

Parts 11-12 showed that expert routing is 93% position-coded. At L16, 11 of 32 experts are "end" specialists, 6 are "start", 4 are "early_mid", 3 are "late_mid". If experts within the same position class are functionally redundant, what kind of redundancy is it?

- **Weight duplication** (>0.9 cosine similarity): Experts are literally the same weights learned multiple times. MoE training is wasteful. Merging = pick any one.
- **Functional overlap** (0.5-0.7): Experts compute related functions. Averaging might work.
- **Diverse ensemble** (<0.5): Experts compute different things at the same position. Merging is destructive.

### 14.2 Method

Three comparison methods on L16 experts:

1. **Bias vectors** (bfloat16, unquantized): Concatenated gate_up + down biases (8,640 dimensions). Direct weight comparison.
2. **Scale vectors** (uint8 per group of 32 weights): Captures weight magnitude structure (777,600 dimensions).
3. **Functional output similarity**: Pass 50 random inputs through each expert's full SwiGLU pipeline (MXFP4 quantized_matmul), compare output vectors.

Position class assignments at L16 (from Part 12):
- **start**: E7, E8, E13, E20, E24, E25 (6 experts)
- **early_mid**: E5, E17, E19, E21 (4 experts)
- **late_mid**: E1, E12, E22 (3 experts)
- **end**: E0, E2, E4, E10, E11, E14, E26, E27, E29, E30, E31 (11 experts)
- **none** (0 activations): E3, E6, E9, E15, E16, E18, E23, E28 (8 experts)

### 14.3 Results: Bias Similarity

| Position Class | Experts | Pairs | Mean | Std | Range |
|---------------|---------|-------|------|-----|-------|
| start | 6 | 15 | 0.531 | 0.214 | [0.194, 0.804] |
| early_mid | 4 | 6 | 0.362 | 0.082 | [0.284, 0.528] |
| late_mid | 3 | 3 | 0.386 | 0.045 | [0.353, 0.450] |
| end | 11 | 55 | 0.388 | 0.147 | [0.030, 0.691] |
| none | 8 | 28 | 0.382 | 0.222 | [-0.015, 0.675] |
| **ALL CROSS** | **24** | **276** | **0.408** | **0.152** | **[-0.019, 0.804]** |

**Within-class/cross-class ratio: 0.95** (end: 0.388 / cross: 0.408)

### 14.4 Results: Functional Output Similarity

| Position Class | Experts | Pairs | Mean | Std | Range |
|---------------|---------|-------|------|-----|-------|
| start | 6 | 15 | 0.234 | 0.316 | [-0.304, 0.626] |
| early_mid | 4 | 6 | 0.187 | 0.235 | [-0.193, 0.560] |
| late_mid | 3 | 3 | -0.089 | 0.381 | [-0.384, 0.448] |
| end | 11 | 55 | 0.210 | 0.213 | [-0.501, 0.584] |
| none | 8 | 28 | 0.158 | 0.310 | [-0.502, 0.640] |
| **ALL CROSS** | **24** | **276** | **0.176** | **0.284** | -- |

**Within-class/cross-class ratio: 1.19** (end: 0.210 / cross: 0.176)

### 14.5 Diagnostic: Dequantized Full Weights

Dequantizing MXFP4 weights via `quantized_matmul` with an identity matrix yields 16.6M-dimensional vectors per expert. All pairs show cosine similarity ~0.000 (range [-0.002, 0.002]). Self-similarity = 1.000 (validated).

This is a **dimensionality artifact**: in 16M+ dimensions, even moderately correlated vectors appear orthogonal because noise dominates the angle. The bias and functional comparisons (in 8,640 and 2,880 dimensions respectively) are the meaningful metrics.

### 14.6 Interpretation

**The redundancy is a diverse ensemble, not weight duplication.**

| Metric | Within-Class Mean | Cross-Class Mean | Ratio |
|--------|-------------------|------------------|-------|
| Bias similarity | 0.388 | 0.408 | 0.95 |
| Functional similarity | 0.210 | 0.176 | 1.19 |

Three findings:

1. **Experts are NOT duplicates.** Mean functional similarity = 0.21 (well below 0.5). Experts in the same position class compute DIFFERENT transformations. They share ~20% output structure and diverge on ~80%.

2. **Position classes are barely reflected in weights.** Within-class similarity (0.39 bias, 0.21 functional) is NOT meaningfully higher than cross-class (0.41 bias, 0.18 functional). The ratio hovers around 1.0. Position-coded routing is a ROUTING property, not a WEIGHT property.

3. **Simple merging would be destructive.** With only 0.21 functional similarity, averaging two experts would destroy ~80% of their individual contributions. The "averaged expert" would write a blurred signal to the residual stream - less information than either original expert.

### 14.7 What the Redundancy Actually Is

The model has 32 experts per layer, each computing a different function. Token routing selects 4 of 32 based on position. But "position-coded routing" doesn't mean "same computation" - it means "same selection criteria."

Analogy: a restaurant has 32 chefs (experts), and tables near the entrance always get chefs 1-4 (position routing). But chefs 1-4 cook different dishes (different weights). If chef 1 calls in sick, chefs 2-4 still cover the table. The table gets slightly different food, but it still gets fed.

The ablation experiments (Parts 8-12) showed that removing any individual chef doesn't starve any table. This Part shows WHY: the remaining chefs aren't backup copies of the removed one. They're independent workers who collectively provide enough residual stream signal. The redundancy is in coverage (multiple writers at each position), not in content (same thing written multiple times).

### 14.8 Implications for Compression

| Approach | Viability | Why |
|----------|-----------|-----|
| Drop duplicates | Not applicable | No duplicates exist |
| Average within position class | Destructive | 80% of information lost |
| SVD/low-rank per expert | Possible | Each expert has internal structure to compress |
| Structured pruning (fewer experts) | Possible but limited | Part 10 shows floor at ~4/layer single-layer, collapse at ~4/layer global |
| Distillation | Most promising | Train fewer experts to approximate the ensemble |

The model's redundancy is architectural headroom, not waste. 32 experts provide a diverse enough ensemble that any 4 can handle any position. Reducing to fewer experts requires either (a) making each expert more capable (distillation) or (b) accepting degraded output diversity.

---
