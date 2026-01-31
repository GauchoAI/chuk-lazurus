## Part 16: Partial Routing Ablation

### 16.1 The Question

Part 15 showed that replacing routing at ALL 24 layers breaks everything (0/8). Parts 8-12 showed that breaking a single layer is harmless (8/8). Where is the boundary? And does it matter WHICH layers keep learned routing?

### 16.2 Conditions

Each condition keeps learned routing at designated layers and applies fixed routing (experts [0, 8, 16, 24] with equal weights) at the rest.

| Condition | Learned Layers | Pattern |
|-----------|---------------|---------|
| normal | 24/24 | All layers |
| alternating | 12/24 | Even layers: 0,2,4,...,22 |
| first_half | 12/24 | L0-L11 |
| second_half | 12/24 | L12-L23 |
| every_3rd | 9/24 | Every 3rd + L23 |
| bookends | 8/24 | L0-3 + L20-23 |
| middle_only | 8/24 | L8-L15 |
| every_4th | 7/24 | Every 4th + L23 |
| first_quarter | 6/24 | L0-L5 |
| last_quarter | 6/24 | L18-L23 |
| every_6th | 5/24 | [0, 6, 12, 18, 23] |

### 16.3 Results

| Condition | Learned | Facts | Fact% | Avg Repetition |
|-----------|---------|-------|-------|----------------|
| **normal** | **24/24** | **8/8** | **100%** | **0.29** |
| **alternating** | **12/24** | **5/8** | **62%** | **0.37** |
| every_3rd | 9/24 | 1/8 | 12% | 0.47 |
| bookends | 8/24 | 0/8 | 0% | 0.41 |
| middle_only | 8/24 | 0/8 | 0% | 0.54 |
| every_4th | 7/24 | 1/8 | 12% | 0.38 |
| first_quarter | 6/24 | 0/8 | 0% | 0.50 |
| last_quarter | 6/24 | 0/8 | 0% | 0.65 |
| every_6th | 5/24 | 0/8 | 0% | 0.63 |
| first_half | 12/24 | 1/8 | 12% | 0.37 |
| second_half | 12/24 | 1/8 | 12% | 0.68 |

### 16.4 Coverage Curve

```
 5 layers |          0/8 | every_6th
 6 layers |          0/8 | first_quarter, last_quarter
 7 layers | #        1/8 | every_4th
 8 layers |          0/8 | bookends, middle_only
 9 layers | #        1/8 | every_3rd
12 layers | #####    5/8 | alternating (every 2nd)
12 layers | #        1/8 | first_half, second_half
24 layers | ######## 8/8 | normal
```

### 16.5 The Critical Finding: Spacing Matters More Than Count

At 12 layers, three conditions were tested:
- **Alternating** (evenly spaced): **5/8 facts**
- **First half** (contiguous L0-11): **1/8 facts**
- **Second half** (contiguous L12-23): **1/8 facts**

Same number of learned layers. 5x difference in fact preservation. The variable is SPACING: evenly distributed routing "checkpoints" every 2 layers dramatically outperform any contiguous block.

### 16.6 The Routing Refresh Model

This extends the DRAM refresh analogy from Part 10:

```
The residual stream needs periodic "routing corrections."

Every 1 layer (normal):         signal stays on track → 8/8 facts
Every 2 layers (alternating):   signal drifts slightly → 5/8 facts
Every 3 layers (every_3rd):     signal drifts too far  → 1/8 facts
Every 4+ layers:                signal lost            → 0/8 facts

Contiguous blocks fail because:
  L0-11 correct, L12-23 wrong → signal diverges in second half, never recovers
  L12-23 correct, L0-11 wrong → signal never forms correctly in first half
```

The model needs a routing correction at least every 2 layers to maintain factual recall. The correction rate threshold is between every-2nd (62%) and every-3rd (12%).

### 16.7 Bookends Hypothesis Falsified

Chris hypothesized that early layers (query formation) and late layers (output formatting) might be the critical routing points, with middle layers compressible.

**Bookends (L0-3 + L20-23): 0/8 facts.** Neither the first 4 nor the last 4 layers are sufficient as "anchors." The middle 16 layers of wrong routing destroy the signal regardless of correct endpoints.

### 16.8 Implications for Compression

**The router cannot be simplified below ~50% layer coverage.** Even with optimal (evenly-spaced) placement, 12/24 learned routing layers only preserve 5/8 facts. Getting to 8/8 requires all 24 layers.

However, the alternating pattern suggests a possible architecture:

```
Full layers (learned routing + full MoE):   L0, L2, L4, ..., L22
Lite layers (simplified routing + fewer experts): L1, L3, L5, ..., L23

Parameter savings:
  - 12 full layers × 32 experts = 384 expert instances
  - 12 lite layers × 4 experts = 48 expert instances (if simplified)
  - Total: 432 vs 768 = 44% reduction

But: this only preserves 62% of facts.
     Full factual accuracy requires full routing at every layer.
```

### 16.9 The Non-Compressibility Result

After 16 experiments, the answer to "how much of this model can we compress?" is increasingly clear:

| Component | Can Remove? | Evidence |
|-----------|------------|---------|
| Individual experts at 1 layer | Yes (30/32) | Parts 8-12 |
| Individual KV heads at 1 layer | Yes (7/8) | Part 13 |
| All experts at all layers | No (collapse at 28/32) | Part 10 |
| All routing at all layers | No (0/8 for every alternative) | Part 15 |
| Routing at 50% of layers | Partial (5/8 alternating) | Part 16 |
| Routing at 33% of layers | No (1/8) | Part 16 |

The model is a tightly coupled pipeline. Each layer's routing contributes essential signal corrections. Local redundancy (any expert is expendable at any layer) coexists with global non-redundancy (the collective routing across all layers is irreplaceable).

### 16.10 Fluency Test: Normal vs Alternating

The 5/8 fact rate for alternating routing only measures the first few generated tokens. To test sustained fluency, we generated 60-token completions for 8 open-ended prompts under normal and alternating routing.

**Representative outputs:**

| Prompt | Normal | Alternating |
|--------|--------|-------------|
| "Once upon a time..." | `little girl named Lily. She lived in a small village with her parents...` | `small village of people who lived in a small village. The village was a small village. The village was...` |
| "The process of photosynthesis..." | `the absorption of light energy by chlorophyll and other pigments in the chloroplasts...` | `the absorption of light energy by the chlorophyll molecules in the plant. The absorbed light energy is converted into ch...` |
| "The best way to learn programming..." | `to practice. Start by working on small projects and gradually increase the complexity...` | `to practice. Practice is the best way to learn programming. Practice is the best way to learn programming...` |

**Repetition scores (3-gram):**

| Prompt | Normal | Alternating |
|--------|--------|-------------|
| Once upon a time | 0.00 | **0.74** |
| Learn programming | 0.00 | **0.80** |
| Most important in life | 0.06 | **0.71** |
| Photosynthesis | 0.00 | **0.04** |
| Cup of coffee | 0.04 | 0.08 |
| Cats vs dogs | 0.30 | **0.59** |
| Travel anywhere | 0.04 | **0.70** |
| History of internet | 0.00 | **0.53** |
| **Average** | **0.06** | **0.52** |

**Alternating routing produces 8.7x more repetition than normal (0.52 vs 0.06).**

The pattern: alternating outputs start coherently (first 10-15 tokens often look reasonable) but degrade into repetition loops. The first-token factual accuracy (5/8) overstates the quality because facts only measure the first few tokens. For sustained generation, alternating routing is not viable.

One exception: the photosynthesis prompt achieved near-normal quality (0.04 rep). This suggests that some topics or sentence structures happen to survive the routing disruption better than others, likely because their residual stream signals are more robust to intermediate-layer perturbations.

---
