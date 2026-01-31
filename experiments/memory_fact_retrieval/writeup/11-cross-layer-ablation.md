## Part 10: Cross-Layer Ablation - Finding the Breaking Point

### 10.1 The Question

Part 9 proved that facts survive removal of all 4 top-k experts at any single layer. But that test only ablated one layer at a time. What happens when we ablate experts across **multiple layers simultaneously**? And how many of the 32 experts can we disable before the model loses coherence entirely?

### 10.2 Experiment Design

Three escalating conditions:

| Phase | What We Ablate | Question |
|-------|---------------|----------|
| **Layer escalation** | Fact's L16 top-4 experts across 1→5→13→24 layers | Does ablating the same experts at many layers break facts? |
| **Expert escalation** | 4→8→16→24→28→30 experts at Layer 16 only | How many experts can one layer lose? |
| **Stress test** | 28-30 experts at ALL 24 layers simultaneously | What is the minimum viable model? |

The stress test has three sub-conditions:
- **keep_high**: Ablate experts [0-27], keep [28-31] at all 24 layers (arbitrary survivors)
- **keep_fact**: Ablate 28 non-fact experts, keep fact's L16 top-4 at all layers (targeted survivors)
- **keep_2**: Ablate experts [0-29], keep only [30-31] at all 24 layers (extreme reduction)

### 10.3 Layer Escalation Results

Ablating the fact-specific top-4 experts across increasing numbers of layers:

```
         Layers |   N | Correct | Total | Rate
-------------------------------------------------------
       L16_only |   1 |       8 |     8 | 100%
        L14-L18 |   5 |       8 |     8 | 100%
         L8-L20 |  13 |       8 |     8 | 100%
     all_layers |  24 |       8 |     8 | 100%
```

**8/8 facts survive even when their specific 4 experts are ablated at ALL 24 layers simultaneously.** The model simply routes around them by selecting other experts from the remaining 28.

### 10.4 Expert Escalation Results

Ablating increasing numbers of experts at Layer 16 only:

```
        Experts |   N | Correct | Total | Rate
-------------------------------------------------------
       ablate_4 |   4 |       8 |     8 | 100%
       ablate_8 |   8 |       8 |     8 | 100%
      ablate_16 |  16 |       8 |     8 | 100%
      ablate_24 |  24 |       8 |     8 | 100%
      ablate_28 |  28 |       8 |     8 | 100%
      ablate_30 |  30 |       8 |     8 | 100%
```

**8/8 facts survive even when 30 of 32 experts are disabled at Layer 16.** With only 2 experts active, the model still generates the correct first token for every fact. A single layer needs barely any expert computation to pass factual information through.

### 10.5 Stress Test Results - The Breaking Point

Ablating 28-30 experts at ALL 24 MoE layers simultaneously:

```
                Condition | Correct | Total | Rate
-------------------------------------------------------
   28 ablated, keep [28-31] |       1 |     8 |  12%
   28 ablated, keep fact-4  |       0 |     8 |   0%
   30 ablated, keep [30-31] |       0 |     8 |   0%
```

**Facts BREAK when 28+ experts are ablated at ALL 24 layers.** The model degenerates:

| Fact | 28 ablated (keep [28-31]) | 28 ablated (keep fact-4) | 30 ablated |
|------|---------------------------|--------------------------|------------|
| Capital of France | "1.5. The capital of the bank is 1.5" | (empty) | "a number of 10" |
| Symbol for gold | "a symbol that is a symbol that is a symbol" | "the same. The symbol for gold is the same" | "(i) is the" |
| Romeo and Juliet | **"the author of the Shakespeare"** | "well-known and well-known and well-known" | "the author of the author of the author of" |
| Speed of light | "10.5. The speed of light is approximately" | "1 m/s/sm/sm/sm/sm/sm/sm" | "4. The number of the number" |
| CEO of Microsoft | "well-versed in the world" | "great example of the way" | "to be the most of the most popular" |
| Capital of Japan | "1.5. The capital of the US is 1.5" | (empty) | "a very much, a a a a a a" |
| Symbol for silver | "[Silver] [S] [S] [S] [S] [S]" | "1." | "symbol for a silver symbol, the symbol" |
| Capital of Australia | "$200$. The capital of the United Kingdom" | (empty) | "and how many people were paid" |

The model doesn't just lose facts - it loses **coherent generation entirely**. Outputs are repetitive loops, random numbers, and degenerate text. This is not fact loss; it is model collapse.

### 10.6 The Critical Boundary

```
Ablation Scope                          | Fact Survival
----------------------------------------|---------------
4 experts at 1 layer                    | 8/8 (100%)
4 experts at 24 layers                  | 8/8 (100%)
30 experts at 1 layer                   | 8/8 (100%)
28 experts at 24 layers                 | 1/8 (12%)   ← COLLAPSE
30 experts at 24 layers                 | 0/8 (0%)    ← TOTAL COLLAPSE
```

The boundary is between "many experts at one layer" (survives) and "few experts at all layers" (collapses). This tells us:

1. **Any single layer is dispensable** - its 4 experts can be fully removed and adjacent layers compensate
2. **Any single layer can operate on minimal experts** - even 2/32 is enough when other layers are intact
3. **The model needs a minimum total expert computation budget** - reducing to 4/32 at ALL layers simultaneously falls below the threshold for coherent internal representations

### 10.7 The DRAM Refresh Analogy

The residual stream is like DRAM: it holds information, but that information must be actively refreshed at each layer by expert computation. Remove the refresh circuitry at one location, and nearby cells compensate. Remove it everywhere, and the stored data decays.

```
Residual Stream:  [fact signal maintained] ──────────────────────────→
                       ↑          ↑          ↑          ↑
                    L5 MoE     L10 MoE    L16 MoE    L20 MoE
                   (refresh)  (refresh)  (refresh)  (refresh)

Remove L16 MoE:   [fact signal maintained] ──────────────────────────→
                       ↑          ↑                     ↑
                    L5 MoE     L10 MoE               L20 MoE
                   (refresh)  (refresh)              (refresh)
                   → FACT SURVIVES (adjacent layers compensate)

Remove 28/32 at ALL layers:
                   [signal degrades] ──→ [noise] ──→ [collapse]
                       ↑          ↑          ↑          ↑
                    4 experts  4 experts  4 experts  4 experts
                   (weak)     (weak)     (weak)     (weak)
                   → FACT LOST (insufficient refresh at every point)
```

### 10.8 Keeping Fact-Specific Experts Doesn't Help

An unexpected finding: keeping the fact's own L16 top-4 experts (instead of arbitrary ones) actually performed **worse** (0/8 vs 1/8). This confirms that:

- **No specific set of 4 experts "holds" a fact** - even the ones the router selects for that fact
- **Expert utility is layer-specific** - the L16 top-4 are optimal at L16 but may be suboptimal at L8 or L20
- **The model needs general computation, not fact-specific computation** - arbitrary experts [28-31] slightly outperformed "the right" experts because they provided more diverse computation across layers

### 10.9 Revised Understanding

The complete picture across Parts 9 and 10:

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Facts survive single-layer ablation | 32/32 at 4 layers (Part 9) | No single layer is a knowledge bottleneck |
| Facts survive multi-layer ablation of their specific experts | 8/8 across all 24 layers (Part 10) | No specific expert set holds knowledge |
| Facts survive 30/32 experts ablated at one layer | 8/8 at L16 (Part 10) | Minimal computation needed per layer |
| Facts break at 28/32 ablated across ALL layers | 1/8 (Part 10) | Minimum global computation budget exists |
| Model collapses, not just fact loss | Degenerate outputs (Part 10) | Threshold is below coherent generation |

**Knowledge lives in the residual stream, but the residual stream requires ongoing expert computation to maintain signal integrity.** The experts don't store facts - they maintain the computational substrate that carries facts. Like refreshing DRAM, you can skip refreshes at individual addresses, but you can't stop refreshing entirely.

---
