## Part 15: Routing Ablation

### 15.1 The Hypothesis

Parts 11-14 showed that experts are position-coded, functionally redundant at single layers, and have diverse (not duplicate) weights. If experts are interchangeable, maybe the router itself is overhead? Could random or fixed expert selection work just as well?

Conditions tested:
- **Normal**: Learned routing (baseline)
- **Random**: Random 4 of 32 each token, each layer
- **Fixed diverse**: Always [0, 5, 8, 22] (one per position class from Part 12)
- **Fixed arbitrary**: Always [0, 1, 2, 3]
- **Fixed popular**: Always [4, 27, 11, 31] (highest-activation from Part 12)
- **Fixed cold**: Always [7, 10, 25, 26] (lowest-activation active experts)

All non-normal conditions apply at ALL 24 layers with uniform weights (1/4 each).

### 15.2 Results

| Condition | Facts | Fact% | Avg Repetition | Degenerate |
|-----------|-------|-------|----------------|------------|
| **normal** | **8/8** | **100%** | **0.24** | **0** |
| random | 0/8 | 0% | 0.69 | 1 |
| fixed_diverse | 0/8 | 0% | 0.83 | 2 |
| fixed_arbitrary | 0/8 | 0% | 0.59 | 2 |
| fixed_popular | 0/8 | 0% | 0.50 | 4 |
| fixed_cold | 0/8 | 0% | 0.44 | 0 |

### 15.3 Sample Outputs

**"The capital of France is"**:
- Normal: `Paris." # Test with a non-existent page...` (correct, continues naturally)
- Random: `the $€€€€€€€€€€€€€€€€€€€€€€€€€€` (degenerate)
- Fixed diverse: `1.5. The price of money is 1.5. The price of money is...` (repetitive nonsense)
- Fixed popular: `a` (single token, degenerate)

**"Once upon a time there was a"**:
- Normal: `little girl named Lily. She lived in a small village...` (coherent narrative)
- Random: `time time time time time time time...` (pure repetition)
- Fixed diverse: `time of the time of the time of the time of the...` (loop)
- Fixed cold: `time, a time, a time, a time, a time, a...` (loop)

### 15.4 Analysis

**Learned routing is NOT overhead. It is essential.**

Every alternative routing scheme produces 0/8 facts and degenerate output. This is true even for:
- **Fixed popular** (the most-used experts) → 0/8, 4 degenerate outputs. The highest-activation experts are WORSE when forced on all tokens.
- **Fixed diverse** (one per position class) → 0/8, highly repetitive. Position diversity doesn't help when the WRONG expert handles each position.
- **Random** → 0/8, complete collapse. Experts are NOT interchangeable.

### 15.5 Reconciling with Prior Results

This seems to contradict Parts 8-12, which showed that ablating any expert at any single layer preserves all facts. The resolution:

| Intervention | Scope | Facts | Why |
|-------------|-------|-------|-----|
| Remove 1 expert at 1 layer | Local | 8/8 | 23 other layers provide correct routing; signal recovers |
| Remove 30/32 at 1 layer | 1 layer | 8/8 | Remaining 2 experts + 23 normal layers compensate |
| Replace router at ALL layers | Global | 0/8 | No layer has correct routing; signal never forms |
| Remove 28/32 at ALL layers | Global | 1/8 | Catastrophic collapse (Part 10) |

**Single-layer interventions are absorbed by the 23 healthy layers.** Global interventions have no healthy layers to compensate. The router's learned weights encode which expert should process each token at each layer. This per-token-per-layer selection is what builds the correct residual stream signal.

### 15.6 What the Router Actually Does

The router doesn't just "select experts." It orchestrates a 24-layer pipeline where each token gets a specific sequence of 4-expert computations at each layer. The correct sequence builds the residual stream signal that produces "Paris" from "The capital of France is."

```
Token "France" at layer 0:  Router selects E_a, E_b, E_c, E_d → writes signal S_0
Token "France" at layer 1:  Router selects E_e, E_f, E_g, E_h → writes S_1 on top of S_0
...
Token "France" at layer 23: Router selects E_w, E_x, E_y, E_z → writes S_23
Final signal: S_0 + S_1 + ... + S_23 → "Paris"
```

Replace any ONE layer's selection with random experts: the other 23 layers still contribute correct signals. S_0 + ... + S_wrong + ... + S_23 ≈ correct (Parts 8-12).

Replace ALL layers' selection: every signal is wrong. S_wrong_0 + ... + S_wrong_23 = garbage (Part 15).

### 15.7 Implications for the Compression Story

1. **Cannot delete the router.** It's the most critical component for model quality.
2. **Cannot simplify routing.** Even "smart" fixed selections (position-diverse, popular experts) fail completely.
3. **The router-expert coupling IS the model.** The weights of each expert are meaningful only in the context of which tokens the router sends to them. An expert trained to process "end-of-sequence tokens routed by the learned router" is useless when given "random tokens from any position."
4. **Expert diversity (Part 14) is NOT noise.** The 0.21 functional similarity between experts is essential - each expert is specialized for its routed inputs. The "diversity" is learned specialization, not training noise.

### 15.8 Revised Understanding

```
Previous story (Parts 8-14):
  Experts are redundant positional processors.
  Any individual expert is expendable.
  Diversity might be noise.

Revised story (Part 15):
  Experts are LOCALLY redundant (at single layers).
  The router-expert coupling is GLOBALLY essential.
  Expert diversity is learned specialization, not noise.
  The system is a distributed pipeline, not a redundant array.
```

The model isn't 32 backup copies of the same computation. It's 32 specialists, each trained to process specific inputs selected by the router. Remove one specialist at one stage: the pipeline adapts. Remove the selection mechanism entirely: the pipeline collapses.

---
