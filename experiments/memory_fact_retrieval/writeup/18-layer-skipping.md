## Part 17: Layer Skipping — No Signal vs Wrong Signal

### 17.1 Hypothesis

Part 16 showed that fixed routing (selecting experts [0, 8, 16, 24] with equal weights) at non-learned layers degrades output severely. But fixed routing doesn't just *fail to help* — it actively writes a **wrong signal** into the residual stream. The MoE sublayer computes `residual + MoE(layernorm(x))`, so even with wrong experts, a non-zero contribution is added.

What if we skip the MoE sublayer entirely? Instead of writing a wrong signal, the residual passes through unchanged:

- **Fixed routing**: `residual + WRONG_MoE_output` → corrupts the residual stream
- **Layer skipping**: `residual + 0` → residual passes through unmodified

If skipping works **better** than fixed routing, then the wrong MoE signal is actively harmful — silence is better than noise. If skipping works **worse**, then even wrong experts provide useful computation — any signal beats no signal.

### 17.2 Method

We patched `GptOssBlock.__call__` at the class level. At skipped layers, the block executes attention normally but bypasses the MoE sublayer entirely:

```
Normal block:    x = residual + attention(norm(x))     →  x = residual + MoE(norm(x))
Skipped block:   x = residual + attention(norm(x))     →  x = x  (identity)
```

Same 8 factual prompts and 2 coherence prompts as Part 16. 12 conditions with matching layer sets for direct comparison.

### 17.3 Results

| Condition | MoE Active | Skipped | Facts | AvgRep |
|-----------|-----------|---------|-------|--------|
| normal | 24/24 | 0/24 | **8/8** | 0.288 |
| skip_second_half | 12/24 | 12/24 | 1/8 | 0.578 |
| skip_alternating | 12/24 | 12/24 | 1/8 | 0.519 |
| skip_first_half | 12/24 | 12/24 | 0/8 | 0.791 |
| skip_every_3rd | 9/24 | 15/24 | 2/8 | 0.653 |
| skip_bookends | 8/24 | 16/24 | 0/8 | 0.584 |
| skip_middle_only | 8/24 | 16/24 | 0/8 | 0.879 |
| skip_every_4th | 7/24 | 17/24 | 1/8 | 0.724 |
| skip_first_quarter | 6/24 | 18/24 | 0/8 | 0.514 |
| skip_last_quarter | 6/24 | 18/24 | 0/8 | 0.763 |
| skip_every_6th | 5/24 | 19/24 | 0/8 | 0.801 |
| skip_all | 0/24 | 24/24 | 0/8 | 0.793 |

### 17.4 Direct Comparison: Skip vs Fixed Routing (Part 16)

| Condition | Layers Disrupted | Fixed Routing (Part 16) | Layer Skip (Part 17) | Winner |
|-----------|-----------------|------------------------|---------------------|--------|
| alternating | 12 | **5/8** | 1/8 | Fixed |
| first_half | 12 | 1/8 | 0/8 | Fixed |
| second_half | 12 | 1/8 | 1/8 | Tie |
| bookends | 16 | 0/8 | 0/8 | Tie |
| every_6th | 19 | 1/8 | 0/8 | Fixed |
| every_4th | 17 | — | 1/8 | — |
| every_3rd | 15 | — | 2/8 | — |

**Fixed routing is consistently better than or equal to layer skipping.** The most dramatic gap: alternating (5/8 vs 1/8). Even wrong experts are more useful than no experts.

### 17.5 Coverage Curve

```
 0 MoE | ........ 0/8 | skip_all
 5 MoE | ........ 0/8 | skip_every_6th
 6 MoE | ........ 0/8 | skip_first_quarter
 6 MoE | ........ 0/8 | skip_last_quarter
 7 MoE | #....... 1/8 | skip_every_4th
 8 MoE | ........ 0/8 | skip_bookends
 8 MoE | ........ 0/8 | skip_middle_only
 9 MoE | ##...... 2/8 | skip_every_3rd
12 MoE | #....... 1/8 | skip_alternating
12 MoE | ........ 0/8 | skip_first_half
12 MoE | #....... 1/8 | skip_second_half
24 MoE | ######## 8/8 | normal
```

### 17.6 Sample Outputs

**"The capital of France is"**:

| Condition | Output |
|-----------|--------|
| normal | Paris." # Test with a non-existent page... |
| skip_alternating | 7.5 million euros. The capital of France is 7.5 million euros... |
| skip_bookends | a city that is a large, but it is a small... |
| skip_all | the " ( ( () and the " () and the "... |

**"The chemical symbol for gold is"** — one of the few surviving facts under skip_alternating:

| Condition | Output | Correct? |
|-----------|--------|----------|
| normal | Au. The chemical symbol for gold is Au... | Yes |
| skip_alternating | "Au," and the chemical symbol for silver is "Ag."... | Yes |
| skip_every_3rd | "Au". The chemical symbol for silver is "Ag"... | Yes |

### 17.7 Attention-Only Model (skip_all)

Skipping ALL MoE layers produces an attention-only model. Output characteristics:
- **0/8 facts** — no factual recall
- **0.793 avg repetition** — severely degenerate
- Characteristic output: `the " ( ( () and the " () and the "...` — structural tokens only
- Resembles the fixed routing collapse from Part 15, but even less coherent

The attention-only model can produce structural tokens (parentheses, quotes) but no content. MoE experts are responsible for **all semantic content**; attention provides **syntactic scaffolding**.

### 17.8 Interpretation

**Wrong signal > No signal.** This is the key finding. Even when routing selects the wrong experts with equal weights, the resulting MoE computation still:

1. **Maintains residual stream norms** — the MoE output has the right magnitude even if wrong direction
2. **Provides some useful features** — experts share structural computation even when misrouted (Part 14 showed 0.21 functional similarity)
3. **Feeds downstream layers expected input statistics** — complete silence (zero MoE) violates what downstream layers expect far more than wrong-but-structured output

This is analogous to communication theory: a noisy channel (fixed routing) is better than a dead channel (skipping). The downstream layers can partially compensate for corrupted signal but not for absent signal.

**Implications for model compression:**
- MoE layers cannot be simply removed, even at inference time
- Any "skip" strategy is worse than even naive fixed-routing fallback
- The model's tolerance for disruption (Part 16's alternating = 5/8 with fixed routing) comes from wrong experts still providing structured computation, not from the residual stream being self-sufficient

---
