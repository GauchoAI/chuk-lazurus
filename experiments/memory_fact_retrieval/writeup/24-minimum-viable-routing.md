## Part 22: Minimum Viable Routing — How Few Layers Can We Learn?

### 22.1 The Question

Part 20 showed L0 is the gatekeeper (~2 facts). Part 21 showed a hard cliff at gap=3. The best 12-layer config (L0_only_extra) achieved 6/8 bare, 8/8 with MB.

But how low can we go? If L0 is critical, can L0 alone + MB achieve full accuracy? If not, what's the absolute minimum number of learned layers for 8/8 with memory bank?

### 22.2 Experiment 1: Scaling Down

All conditions include L0 (except `none`), with remaining layers evenly spaced across the 24-layer model:

| Condition | Learned Layers | Count |
|-----------|---------------|-------|
| normal | [0-23] | 24 |
| L0_plus_7 | [0, 3, 7, 11, 15, 19, 23] | 7 |
| L0_plus_5 | [0, 5, 11, 17, 23] | 5 |
| L0_plus_mid | [0, 11, 23] | 3 |
| L0_endpoints | [0, 23] | 2 |
| L0_only | [0] | 1 |
| none | [] | 0 |

### 22.3 Results: The Scaling Curve

| Condition | Learned | Bare | +MB | Bare Rep | MB Rep |
|-----------|---------|------|-----|---------|--------|
| normal | 24/24 | **8/8** | **8/8** | 0.360 | 0.000 |
| L0_plus_7 | 7/24 | 1/8 | **8/8** | 0.266 | 0.262 |
| L0_plus_5 | 5/24 | 0/8 | 4/8 | 0.316 | 0.473 |
| L0_plus_mid | 3/24 | 0/8 | 1/8 | 0.588 | 0.769 |
| L0_endpoints | 2/24 | 0/8 | 0/8 | 0.798 | 0.774 |
| L0_only | 1/24 | 0/8 | 0/8 | 0.732 | 0.656 |
| none | 0/24 | 0/8 | 0/8 | 0.843 | 0.655 |

```
Bare facts:
  24 layers | ######## 8/8
   7 layers | #....... 1/8
   5 layers | ........ 0/8
   3 layers | ........ 0/8
   2 layers | ........ 0/8
   1 layer  | ........ 0/8
   0 layers | ........ 0/8

With memory bank:
  24 layers | ######## 8/8
   7 layers | ######## 8/8  ← minimum for full MB rescue
   5 layers | ####.... 4/8  ← cliff
   3 layers | #....... 1/8
   2 layers | ........ 0/8
   1 layer  | ........ 0/8
   0 layers | ........ 0/8
```

### 22.4 L0 Alone Does Nothing

A critical revision of Part 20's "gatekeeper" finding: **L0 is necessary but not sufficient**.

```
none:    0/8 bare, 0/8 MB  (0 learned)
L0_only: 0/8 bare, 0/8 MB  (1 learned)
```

L0 alone provides zero benefit. Without downstream learned layers to propagate the routing signal, L0's correct initialization has nothing to initialize *for*. The gatekeeper needs a gate to keep.

### 22.5 The MB Cliff: Between 5 and 7 Layers

The sharp transition from 4/8 to 8/8 MB occurs between 5 and 7 learned layers. To pin the exact boundary, we tested two 6-layer configurations:

### 22.6 Experiment 2: The 6-Layer Cliff Test

| Condition | Layers | Avg Gap | Bare | +MB | Bare Rep | MB Rep |
|-----------|--------|---------|------|-----|---------|--------|
| L0_plus_6_even | [0, 5, 9, 14, 18, 23] | 4.6 | 0/8 | 2/8 | 0.751 | 0.346 |
| **L0_plus_6_tight** | **[0, 3, 7, 11, 15, 19]** | **3.4** | **2/8** | **8/8** | 0.441 | 0.467 |

Same layer count. Wildly different results.

**The tight config (gap 3-4) achieves full 8/8 MB rescue.** The even config (gap ~4.6) fails at 2/8 MB. This confirms that spacing, not count, is the binding constraint.

### 22.7 The Tight Config vs L0_plus_7

The tight 6-layer config `[0, 3, 7, 11, 15, 19]` is literally `L0_plus_7` minus L23:

| Config | Layers | Bare | +MB |
|--------|--------|------|-----|
| L0_plus_6_tight | [0, 3, 7, 11, 15, 19] | 2/8 | **8/8** |
| L0_plus_7 | [0, 3, 7, 11, 15, 19, 23] | 1/8 | **8/8** |

L23 adds nothing. The last 4 layers (20-23) don't need learned routing. The critical coverage is L0 through L19 — the first 83% of the network.

### 22.8 Complete Scaling Curve

Combining all results from Parts 20-22:

| Learned | Spacing | Bare | +MB | Status |
|---------|---------|------|-----|--------|
| 24 (100%) | gap 1 | 8/8 | 8/8 | Full model |
| 12 (50%) | gap 2 | 5-6/8 | 8/8 | Part 20 optimum (bare) |
| 7 (29%) | gap 3.4 | 1/8 | 8/8 | MB minimum (wide) |
| **6 (25%)** | **gap 3.4** | **2/8** | **8/8** | **MB minimum (tight)** |
| 6 (25%) | gap 4.6 | 0/8 | 2/8 | Fails — gaps too wide |
| 5 (21%) | gap 5.8 | 0/8 | 4/8 | MB partially fails |
| 3 (13%) | gap 11 | 0/8 | 1/8 | Broken |
| 1-2 | — | 0/8 | 0/8 | Model can't read MB |

### 22.9 The Rule

**For full MB rescue: learned layers at gap ≤ 4, covering L0 through at least L19.**

The minimum viable configuration is **6 learned layers (25%)** at `[0, 3, 7, 11, 15, 19]`:
- Bare: 2/8 facts (degraded but coherent)
- With MB: **8/8 facts (100%)**
- 75% of routers replaced with fixed experts

### 22.10 Interpretation

Three regimes emerge:

| Regime | Layers | Gap | Bare | +MB | What's happening |
|--------|--------|-----|------|-----|-----------------|
| **Full** | 12+ | ≤ 2 | 5-8/8 | 8/8 | Bare recall works; MB is insurance |
| **MB-dependent** | 6-7 | 3-4 | 1-2/8 | 8/8 | Model coherent enough to read MB |
| **Broken** | ≤ 5 | ≥ 5 | 0/8 | 0-4/8 | Model too degraded for MB to help |

The binding constraint is not the number of learned layers — it's whether the remaining learned layers maintain enough coherence for the model to process the memory bank prompt. Below 6 layers (gap > 4), the model generates repetitive nonsense and can't parse `[Memory Bank]` tokens.

**Revised architecture**: The optimal lite model needs only 6 of 24 routers (25%) with tight spacing. The remaining 18 layers use fixed expert routing. Combined with memory bank injection, this achieves 100% factual accuracy at 75% routing savings.

---
