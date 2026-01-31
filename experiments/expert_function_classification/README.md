# Expert Function Classification

## Hypothesis

A significant fraction of MoE expert capacity serves as retrievable knowledge storage rather than irreducible computation. These "storage experts" can be externalized to databases without capability loss, enabling model compression.

## Method

### Phase 1: Baseline Generation

Generate outputs for all 15 test prompts without ablation. These serve as reference for all subsequent comparisons.

### Phase 2: Systematic Ablation

For each expert in layers 5-20:

1. **Quick scan**: Ablate each of the 32 experts on 2 factual prompts. Experts whose removal changes output are "causal" (~8-15 per layer).

2. **Deep test**: For causal experts, run the full 15-prompt suite (5 categories) with the expert ablated. Classify each output change as:
   - **Fact error**: Expected answer keyword missing (e.g., "Paris" no longer appears for capital of France)
   - **Structure error**: Degenerate output (repetition, truncation, incoherence)
   - **No change**: Output identical to baseline

3. **Routing disruption**: Measure JS divergence of downstream expert selection histograms before/after ablation.

### Phase 3: Recovery Testing

For experts classified as "storage": re-run ablated generation with a `[Memory Bank]` context prepended. If the correct answer is recovered, this confirms the expert's function is externalizable.

### Phase 4: Capacity Estimation

Count experts by category across all tested layers. Extrapolate to estimate what fraction of model parameters are "storage" and could be externalized.

## Expert Categories

| Category | Definition | Ablation Signature |
|----------|------------|-------------------|
| **Storage** | Retrieves facts/entities | Fact errors, structure preserved, recoverable via memory bank |
| **Computation** | Performs transformation | Structure errors (repetition, incoherence), not recoverable |
| **Routing** | Directs information flow | Low direct errors, high downstream expert disruption |
| **Redundant** | Backed up by other experts | No measurable output change |

## Classification Logic

```
no_change_rate > 0.9          → REDUNDANT
routing_disruption > 0.15     → ROUTING  (if low direct errors)
fact_error_rate > struct_rate  → STORAGE  (if fact_error_rate > 0.3)
struct_rate > fact_error_rate  → COMPUTATION (if struct_rate > 0.3)
```

## Validation Criteria

1. **L16E4 classifies as storage** (validates prior memory_fact_retrieval findings)
2. **Storage experts show >70% recovery rate** with memory bank injection
3. **Computation experts show <30% recovery rate**
4. **At least 10% of experts are classifiable as storage**

## Model

- **GPT-OSS 20B** (`openai/gpt-oss-20b`)
- 24 layers, all MoE
- 32 experts per layer, top-4 routing
- ~21B total params, ~3.6B active per token

## Run

```bash
# Full experiment (layers 5-20, ~1-2 hours)
python experiments/expert_function_classification/experiment.py

# Single layer (~5-10 minutes)
python experiments/expert_function_classification/experiment.py --layer 16
```

## Connection to Prior Work

This experiment builds on findings from the memory_fact_retrieval experiment:
- **L16E4** was identified as the "fact lookup" expert handling 25% of declarative routing
- **Memory bank format** `[Memory Bank]...[End Memory Bank]` achieved 100% fact override
- **Declarative vs procedural split** showed fundamentally different expert routing patterns

This experiment tests whether that architectural distinction extends to a general storage/computation classification across all experts.
