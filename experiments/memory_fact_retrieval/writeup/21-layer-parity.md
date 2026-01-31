## Part 20: Layer Parity — Which Layers Matter?

### 20.1 The Question

Part 16 tested alternating routing with learned at even layers [0, 2, 4, ..., 22]. But we never tested the inverse. Three hypotheses:

- **H1**: Even layers are special (they do something critical that odd layers don't)
- **H2**: It's just about spacing (any alternating pattern works equally well)
- **H3**: L0 specifically is critical (first layer must be learned)

### 20.2 Conditions

| Condition | Learned Layers | Count | Has L0? |
|-----------|---------------|-------|---------|
| normal | [0-23] | 24 | Yes |
| even_learned | [0,2,4,...,22] | 12 | Yes |
| odd_learned | [1,3,5,...,23] | 12 | No |
| L0_then_odd | [0,1,3,5,...,23] | 13 | Yes |
| skip_L0 | [2,4,6,...,22] | 11 | No |
| L0_only_extra | [0,3,5,7,...,23] | 12 | Yes |

Each tested both bare (no memory bank) and with memory bank.

### 20.3 Results

| Condition | Layers | Bare Facts | MB Facts | Bare Rep | MB Rep |
|-----------|--------|-----------|---------|---------|--------|
| normal | 24/24 | **8/8** | **8/8** | 0.360 | 0.000 |
| **even_learned** | 12/24 | **5/8** | **8/8** | 0.382 | 0.164 |
| **odd_learned** | 12/24 | **3/8** | **8/8** | 0.575 | 0.013 |
| L0_then_odd | 13/24 | **5/8** | **8/8** | 0.566 | 0.026 |
| skip_L0 | 11/24 | **3/8** | **8/8** | 0.396 | 0.074 |
| **L0_only_extra** | 12/24 | **6/8** | **8/8** | 0.420 | 0.022 |

### 20.4 Even vs Odd: H1 Partially Confirmed

```
even_learned:  5/8  (rep 0.382)   ← includes L0
odd_learned:   3/8  (rep 0.575)   ← excludes L0
```

Even is better (5/8 vs 3/8), but is this because even layers are inherently special, or because the even set includes L0? Testing L0 directly:

### 20.5 L0 Is Critical: H3 Confirmed

| Condition | Has L0 | Facts | Layers |
|-----------|--------|-------|--------|
| even_learned | Yes | **5/8** | 12 |
| skip_L0 | **No** | **3/8** | 11 |
| odd_learned | **No** | **3/8** | 12 |
| L0_then_odd | Yes | **5/8** | 13 |
| L0_only_extra | Yes | **6/8** | 12 |

The pattern is stark:
- **With L0**: 5/8, 5/8, 6/8 (average 5.3/8)
- **Without L0**: 3/8, 3/8 (average 3.0/8)

Adding L0 to the odd set improves from 3/8 → 5/8 (compare odd_learned vs L0_then_odd). Removing L0 from the even set drops from 5/8 → 3/8 (compare even_learned vs skip_L0). **L0 is worth approximately 2 additional facts.**

### 20.6 Best Configuration: L0 + Spaced Odds

The best 12-layer configuration is `L0_only_extra` at **6/8 facts** — learned at [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]. This combines:
1. **L0** (critical for residual stream initialization)
2. **Regular spacing** (learned every 2 layers after L0)

This outperforms even_learned (5/8) despite having the same number of learned layers.

### 20.7 Memory Bank: Universal Rescue

**Every single condition recovers to 8/8 with memory bank**, regardless of parity:

| Condition | Bare | +MB |
|-----------|------|-----|
| even_learned | 5/8 | **8/8** |
| odd_learned | 3/8 | **8/8** |
| L0_then_odd | 5/8 | **8/8** |
| skip_L0 | 3/8 | **8/8** |
| L0_only_extra | 6/8 | **8/8** |

The memory bank mechanism is completely independent of which layers have learned routing. This further confirms that fact injection enters via attention (L0-L4), not MoE routing.

### 20.8 Interpretation

**L0 is the gatekeeper.** The first MoE layer initializes the residual stream with the correct routing pattern. Without L0's learned routing, the residual stream starts with a corrupted signal that downstream layers never fully recover from. With L0 correct, downstream layers can partially compensate for routing errors.

The full ranking of layer importance for routing:
1. **L0** (critical — worth ~2 facts)
2. **Spacing** (important — alternating > contiguous, from Part 16)
3. **Parity** (minor — even slightly better than odd, but mostly due to L0)
4. **Total count** (diminishing returns above 12 layers)

**For the compression path**: The optimal lite model uses L0 (always learned) + evenly-spaced routing at remaining layers + memory bank injection. This achieves 6/8 bare and 8/8 with MB, using only 12 of 24 routers.

---
