## Part 21: Layer Spacing — How Sparse Can We Go?

### 21.1 The Question

Part 20 established L0 as critical and spacing as important. But how much spacing is too much? Every-2nd works (5/8). What about every-3rd, every-4th, every-8th?

### 21.2 Conditions

All conditions include L0. Varying the gap between learned layers:

| Condition | Layers | Gap | Learned At |
|-----------|--------|-----|-----------|
| normal | 24 | 1.0 | [0-23] |
| every_2nd_L0 | 12 | 2.0 | [0,2,4,...,22] |
| every_3rd_L0 | 9 | 2.9 | [0,3,6,9,12,15,18,21,23] |
| every_4th_L0 | 7 | 3.8 | [0,4,8,12,16,20,23] |
| every_6th_L0 | 5 | 5.8 | [0,6,12,18,23] |
| every_8th_L0 | 4 | 7.7 | [0,8,16,23] |
| L0_L23_only | 2 | 23.0 | [0,23] |
| L0_only | 1 | — | [0] |

Each tested bare and with memory bank.

### 21.3 Results

| Condition | Layers | Gap | Bare | +MB | Bare Rep | MB Rep |
|-----------|--------|-----|------|-----|---------|--------|
| normal | 24 | 1.0 | **8/8** | **8/8** | 0.360 | 0.000 |
| every_2nd_L0 | 12 | 2.0 | **5/8** | **8/8** | 0.382 | 0.164 |
| every_3rd_L0 | 9 | 2.9 | 1/8 | **6/8** | 0.504 | 0.117 |
| every_4th_L0 | 7 | 3.8 | 1/8 | **6/8** | 0.357 | 0.494 |
| every_6th_L0 | 5 | 5.8 | 0/8 | **5/8** | 0.658 | 0.545 |
| every_8th_L0 | 4 | 7.7 | 0/8 | **4/8** | 0.510 | 0.627 |
| L0_L23_only | 2 | 23.0 | 0/8 | 0/8 | 0.798 | 0.774 |
| L0_only | 1 | — | 0/8 | 0/8 | 0.732 | 0.656 |

### 21.4 The Cliff: Gap 2 → Gap 3

```
Bare facts:
  24 layers (gap 1.0) | ######## 8/8
  12 layers (gap 2.0) | #####... 5/8  ← viable
   9 layers (gap 2.9) | #....... 1/8  ← cliff
   7 layers (gap 3.8) | #....... 1/8
   5 layers (gap 5.8) | ........ 0/8
   4 layers (gap 7.7) | ........ 0/8
   2 layers (gap 23)  | ........ 0/8
   1 layer  (gap N/A) | ........ 0/8
```

**There is a sharp cliff between gap=2 and gap=3.** Every-2nd (5/8) → every-3rd (1/8). The model needs a routing correction **at most every 2 layers** to maintain factual recall. At gap=3, the residual stream drifts too far between corrections.

### 21.5 Memory Bank Shifts the Threshold

```
With memory bank:
  24 layers (gap 1.0) | ######## 8/8
  12 layers (gap 2.0) | ######## 8/8  ← full rescue
   9 layers (gap 2.9) | ######.. 6/8  ← substantial rescue
   7 layers (gap 3.8) | ######.. 6/8
   5 layers (gap 5.8) | #####... 5/8
   4 layers (gap 7.7) | ####.... 4/8
   2 layers (gap 23)  | ........ 0/8  ← MB can't rescue this
   1 layer  (gap N/A) | ........ 0/8
```

Memory bank rescue follows a gradient:
- **Gap ≤ 2**: Full rescue (8/8)
- **Gap 3-4**: Strong rescue (6/8)
- **Gap 6-8**: Partial rescue (4-5/8)
- **Gap > 8**: MB cannot rescue (0/8) — the model is too degraded to even read the memory bank

### 21.6 The Memory Bank Floor

L0_only (0/8 with MB) and L0_L23_only (0/8 with MB) reveal the **memory bank floor**: below ~4 learned layers, the model can't process the memory bank format at all. The attention mechanism that reads from `[Memory Bank]` tokens requires minimal MoE support to function. With 4+ learned layers (gap ≤ 8), the model can at least partially read external facts.

### 21.7 Efficiency Analysis

| Condition | Bare Facts | Facts/Layer |
|-----------|-----------|-------------|
| normal (24) | 8/8 | 0.33 |
| **every_2nd_L0 (12)** | **5/8** | **0.42** |
| every_3rd_L0 (9) | 1/8 | 0.11 |
| every_4th_L0 (7) | 1/8 | 0.14 |

**Every-2nd is the most efficient**: 0.42 facts per learned layer, beating even the full model (0.33). This is the sweet spot — half the routing computation, 63% of the bare factual accuracy, and 100% with memory bank.

### 21.8 Interpretation

The model's routing acts like a **digital refresh signal**:

```
Gap = 1: Full refresh every layer   → 8/8 (no drift)
Gap = 2: Refresh every other layer  → 5/8 (minor drift, recoverable)
Gap = 3: Refresh every 3rd layer    → 1/8 (critical drift, unrecoverable)
Gap ≥ 4: Sparse refresh             → 0/8 (complete drift)
```

This is strikingly similar to DRAM refresh timing: there's a hard threshold below which the signal (residual stream) degrades irreversibly. The threshold for this model is **gap = 2** (routing correction every other layer).

**Triple-spaced answer**: Every-3rd gets 1/8 bare (not viable) but 6/8 with memory bank (viable for RAG workloads). The bare cliff is at gap=2, the MB cliff is at gap~8.

---
