## Part 19: Memory Bank + Lite Model — The Compression Path

### 19.1 Hypothesis

Parts 15-18 showed that routing degradation destroys factual accuracy. But Part 9 showed memory banks override parametric memory at 100%. What if external fact injection **compensates** for routing degradation?

```
Full model:           8/8 facts  (routing works, parametric memory works)
Lite model:           3/8 facts  (routing broken, parametric memory fails)
Lite model + RAG:     ???        (routing broken, but facts injected externally)
```

If lite + RAG recovers to ~8/8, then: **simplified routing + external knowledge = viable compression path**.

### 19.2 Method

8 conditions combining 4 model configurations × 2 memory bank settings:

| Config | Routing | Memory Bank |
|--------|---------|-------------|
| `normal` | Full (24/24 learned) | No |
| `normal_mb` | Full (24/24 learned) | Yes |
| `fixed_alt` | Alternating (12/24 learned, 12 fixed) | No |
| `fixed_alt_mb` | Alternating (12/24 learned, 12 fixed) | Yes |
| `skip_alt` | Alternating (12/24 MoE, 12 skipped) | No |
| `skip_alt_mb` | Alternating (12/24 MoE, 12 skipped) | Yes |
| `fixed_heavy` | Every-3rd (9/24 learned, 15 fixed) | No |
| `fixed_heavy_mb` | Every-3rd (9/24 learned, 15 fixed) | Yes |

Memory bank format (proven in Part 9):
```
[Memory Bank]
- France | capital | Paris
- Gold | chemical symbol | Au
- ...all 8 facts...
[End Memory Bank]

Using the memory bank above, answer: What is the capital of France?
Answer:
```

Also tested 3 counterfactuals (Lyon, Gd, Osaka) to verify override still works under degradation.

### 19.3 Results

| Condition | Facts | Avg Rep | Description |
|-----------|-------|---------|-------------|
| normal | 8/8 100% | 0.195 | Full model, no memory bank |
| **normal_mb** | **8/8 100%** | **0.000** | Full model + memory bank |
| fixed_alt | 3/8 38% | 0.709 | Alternating fixed, no MB |
| **fixed_alt_mb** | **8/8 100%** | **0.164** | **Alternating fixed + MB** |
| skip_alt | 1/8 12% | 0.558 | Alternating skip, no MB |
| **skip_alt_mb** | **6/8 75%** | **0.171** | **Alternating skip + MB** |
| fixed_heavy | 1/8 12% | 0.688 | Every-3rd (9/24), no MB |
| **fixed_heavy_mb** | **6/8 75%** | **0.117** | **Every-3rd + MB** |

### 19.4 The Key Finding

**Memory bank injection fully rescues the alternating fixed-routing model to 100% factual accuracy.**

| Degradation | Without MB | With MB | Recovery |
|-------------|-----------|---------|----------|
| Alternating fixed (12/24) | 3/8 → | **8/8** | **FULL** |
| Alternating skip (12/24) | 1/8 → | **6/8** | Substantial |
| Every-3rd fixed (9/24) | 1/8 → | **6/8** | Substantial |

The alternating fixed-routing model with memory bank achieves **identical factual accuracy** to the full model (8/8). Repetition also drops dramatically: 0.709 → 0.164 (4.3x improvement).

### 19.5 Counterfactual Override Under Degradation

Does the memory bank still override parametric memory when routing is degraded?

| Condition | Override Rate | Examples |
|-----------|-------------|----------|
| normal_mb | **3/3 (100%)** | Lyon ✓, Gd ✓, Osaka ✓ |
| fixed_alt_mb | **3/3 (100%)** | Lyon ✓, Gd ✓, Osaka ✓ |
| skip_alt_mb | 2/3 (67%) | Lyon ✓, Gd ✓, Osaka ✗ |
| fixed_heavy_mb | 2/3 (67%) | Lyon ✗, Gd ✓, Osaka ✓ |

The alternating fixed-routing model maintains **100% counterfactual override** — same as the full model. This means the memory bank mechanism (attention-based fact injection into the residual stream) is independent of MoE routing quality.

### 19.6 Sample Outputs

**"What is the capital of France?"**

| Condition | Output |
|-----------|--------|
| normal | Paris." -> "The capital of France is Paris." |
| fixed_alt | Paris. The capital of France is Paris. The capital of France is Paris... (rep) |
| **fixed_alt_mb** | **Paris. But the user says: "What is the capital of France?" The answer is...** |
| skip_alt | 7.5. The capital of France is 7.5... (degenerate) |
| **skip_alt_mb** | **Paris. The question is: "What is the capital of France?" The answer is...** |
| fixed_heavy | the capital of the capital of the capital... (collapse) |
| **fixed_heavy_mb** | **France is the capital of France.** (partially recovered — prompt confusion) |

### 19.7 Why Memory Bank Rescues Routing Degradation

The mechanism is now clear from the full experiment series:

1. **Facts enter via attention** (L0-L4): Memory bank tokens are processed by attention heads, which embed facts into the residual stream. This happens **before** MoE routing matters.

2. **MoE routing shapes continuation**: MoE experts determine *how* the model continues after the fact (fluency, structure, elaboration). When routing is degraded, the model loses continuation quality but the fact is already in the residual stream.

3. **Memory bank constrains output space** (Part 18 finding): The structured format `[Memory Bank]...Answer:` creates a constrained output space similar to code — the model knows to output the looked-up value. This inherently reduces dependence on MoE routing.

```
Architecture of the compression path:

┌─────────────────────────────────┐
│  External Knowledge (RAG/MB)    │  ← Facts stored here
│  Injected at prompt level       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Attention layers (FULL)        │  ← Read from memory bank
│  Embed facts into residual      │     No degradation needed
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  MoE layers (SIMPLIFIED)        │  ← 50% routing simplified
│  Shape output style/structure   │     Fixed routing at odd layers
│  Facts already in residual      │     Don't need correct routing
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Output: Correct facts + okay   │  ← 8/8 facts, 0.164 rep
│  continuation quality            │     (vs 0.000 full model)
└─────────────────────────────────┘
```

### 19.8 Practical Implications

**The compression path is real.** For factual workloads:

| Metric | Full Model | Lite + RAG | Degradation |
|--------|-----------|------------|-------------|
| Facts correct | 8/8 (100%) | 8/8 (100%) | **None** |
| Counterfactual override | 3/3 (100%) | 3/3 (100%) | **None** |
| Repetition ratio | 0.000 | 0.164 | Moderate |
| Router computation | 24 layers | 12 layers | **50% saved** |

The trade-off: **perfect factual accuracy with 50% routing computation, at the cost of moderate fluency degradation** (0.164 rep vs 0.000). For Q&A, search, and knowledge retrieval tasks where the first few tokens matter most, this is viable.

For creative/conversational tasks, the lite model is not sufficient (Part 18 showed creative prompts collapse to 0.769 repetition under routing disruption).

### 19.9 What This Means for Model Architecture

This experiment validates a **hybrid architecture**:
- **Attention** handles fact retrieval (robust to MoE degradation)
- **MoE** handles output quality/continuation (degradable for factual tasks)
- **External memory** (RAG/memory bank) compensates for MoE routing errors on factual workloads

The 50% routing simplification is specific to alternating fixed routing on this model. The principle — that external knowledge injection compensates for routing degradation — likely generalizes to other MoE architectures and degradation levels.

---
