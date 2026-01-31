# Model Comparison — Runs 19-25

**Part of**: [RESULTS.md](RESULTS.md) | **Prev**: [Schema Evolution (Runs 9-18)](RESULTS_SCHEMA_EVOLUTION.md)

Model scaling and comparison experiments across TinyLlama, SmolLM2, and Llama-3.2 variants.

See also: [TINYLLAMA_1.1B.md](models/TINYLLAMA_1.1B.md) | [SMOLLM2_1.7B.md](models/SMOLLM2_1.7B.md) | [LLAMA32_1B.md](models/LLAMA32_1B.md)

---

## Run 19: 100-Sample GSM-8K Evaluation (Critical Finding)

**Date**: 2026-01-26
**Config**: 3000 examples, 1 epoch SFT, 20 RL iterations, max_tokens=750

### The Experiment

Run 18's 90% accuracy on the 10-sample GSM-8K probe suggested we were close to solving GSM-8K. Run 19 scaled up training data and RL iterations to push further, then evaluated on a larger 100-sample subset.

### Results

| Metric | Value |
|--------|-------|
| Training examples | 3000 |
| SFT accuracy | **100%** |
| GSM-8K 10-sample | 7/10 (70%) |
| GSM-8K 100-sample | **~0-2%** |
| 100-sample parse rate | **100%** |

### The Critical Discovery

**100% parse rate with ~0% accuracy reveals a fundamental gap: the model learns trace FORMAT, not REASONING.**

The model produces perfectly valid YAML traces that parse without errors. The traces have correct structure, correct expert routing, correct trace steps. But the computed answers are wrong because the model:

1. **Pattern matches instead of understanding** — Memorized trace templates, not problem semantics
2. **Extracts wrong values** — Confuses which numbers map to which variables
3. **Truncates multi-step problems** — Stops after 2-3 steps when 6+ are needed
4. **Misroutes expert types** — Routes percentage problems to arithmetic, etc.

### Failure Categories (100-sample analysis)

| Category | % of Failures | Example |
|----------|---------------|---------|
| Multi-entity confusion | ~30% | "Bob has twice as many as Alice" → wrong entity wiring |
| Value extraction error | ~25% | Extracts wrong numbers from problem text |
| Multi-step truncation | ~20% | 6-step problems get 2-3 step traces |
| Rate/time confusion | ~15% | Confuses rate, time, and quantity roles |
| Decimal/fraction handling | ~10% | 0.5 becomes 5, "half" not recognized |

### Bug Fixes Applied

During Run 19 preparation, several generator bugs were discovered and fixed:

| Bug | Impact | Fix |
|-----|--------|-----|
| `type: choice` not handled in vocab | `mult_word` always 0 | Added choice handling in `_sample_vocab()` |
| `_generate_variables` only checked `options` | `multiplier` from `values` list failed | Check both `options` and `values` |
| No auto-pluralization | "coinss", "sandwichs" | Added `_pluralize()` method |
| `mult_word`/`multiplier` independent | Word didn't match number | Auto-generate from `multiplier` |
| ratio_split div-by-zero | `"ratio + 1"` not evaluated | Changed to explicit compute step |
| Template var issues | `${person1}` not resolved | Fixed to `${name1}` in template_vars |

### New Schemas Added (10 total)

Created pattern files for schemas that had templates in schema JSON but no pattern file:

- `twice_relationship.json` — "X has twice as many as Y"
- `multi_item_cost.json` — Multi-item shopping problems
- `nested_groups.json` — Groups within groups
- `growth_doubled.json` — Value doubles/triples then adds
- `two_period_sum.json` — Sum across two time periods
- `recover_then_multiply.json` — Recover original, then multiply
- `average_three.json` — Average of three values
- `ratio_split.json` — Split by ratio (division)
- `remaining_capacity.json` — Capacity minus used
- `time_from_distance.json` — Distance/speed calculations

**Total schemas**: 59 (all verified passing)

### Key Insight: The 10-Sample Trap

The 10-sample probe was **not representative** of GSM-8K difficulty distribution:

| 10-Sample Results | 100-Sample Results |
|-------------------|-------------------|
| 90% accuracy | ~2% accuracy |
| Mostly simple patterns | Full difficulty range |
| High template overlap with training | Low template overlap |

The 10 problems happened to match our training templates. The remaining 1309 problems use different phrasings, more complex reasoning chains, and problem structures we haven't seen.

### Implications

1. **Format mastery ≠ Reasoning** — Producing valid structured output is a necessary but insufficient condition for math reasoning

2. **Small probe sets are dangerous** — 10 samples gave a false sense of progress; 100 samples exposed the reality

3. **Linguistic diversity is critical** — GSM-8K has thousands of unique phrasings; our templates cover a small fraction

4. **Model capacity may be limiting** — TinyLlama 1.1B may not have sufficient capacity to generalize from ~50 schemas to ~1300 problem variants

### Next Experiment: SmolLM2-1.7B

Given the capacity hypothesis, the next experiment uses a larger model:

**Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct
**Parameters**: 1.7B (55% larger than TinyLlama)
**Architecture**: Optimized for instruction-following

Command:
```bash
python experiments/csp_cot_gsm8k/train_gsm8k_yaml.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --n-train 3000 \
    --sft-epochs 1 \
    --rl-iters 20 \
    --max-tokens 750
```

**Hypothesis**: A larger model with better instruction-following capability may generalize from training patterns to novel GSM-8K phrasings more effectively.

---

## Run 20: SmolLM2-1.7B (Larger Model Experiment)

**Date**: 2026-01-26
**Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct (1.7B parameters)
**Config**: 3000 examples, 1 epoch SFT, 20 RL iterations, max_tokens=750

### Hypothesis

TinyLlama 1.1B may lack capacity to generalize from training patterns to novel GSM-8K phrasings. A larger model (55% more parameters) with instruction-tuning might bridge this gap.

### Results

| Metric | TinyLlama 1.1B (Run 18) | SmolLM2-1.7B (Run 20) |
|--------|------------------------|----------------------|
| SFT accuracy | 100% | **65%** |
| SFT parse rate | 100% | 95% |
| Final training | 96% | **78%** |
| GSM-8K 10-sample | 90% | **30%** |
| Composition | 100% | **22%** |

### By Expert (Final Eval, 50 samples)

| Expert | SmolLM2-1.7B | TinyLlama (Run 18) |
|--------|--------------|-------------------|
| arithmetic | 81% (13/16) | 93% |
| comparison | 100% (7/7) | 89% |
| composition | **22% (2/9)** | 100% |
| entity_track | 92% (11/12) | 100% |
| percentage | 100% (3/3) | 100% |
| rate_equation | 100% (3/3) | 100% |

### GSM-8K 10-Sample Results

| Problem | Status | Notes |
|---------|--------|-------|
| Janet's ducks | ✓ | Correct |
| Robe fiber | ✗ | wrong_answer:6 (expected 3) |
| House flipping | ✗ | invalid_trace: PercentIncreaseStep in arithmetic |
| Problems 4-10 | — | Not shown in output |
| **Total** | **3/10 (30%)** | |

### Analysis: Why Larger Model Performs Worse

**1. Chat Template Mismatch**

SmolLM2-Instruct uses a different chat template than TinyLlama. The model may be fighting against its instruction-tuning when learning our YAML format:

```
SmolLM2 baseline output: <|entity_track|><|rate_equation|><|percentage|>...
```

The model initially tries to output special tokens rather than YAML — suggesting the instruction-tuning is interfering.

**2. Composition Catastrophic Failure (22%)**

The most striking regression is composition accuracy: 100% → 22%. Multi-expert traces require:
- Longer generation (250-350 tokens)
- Precise YAML list formatting
- Cross-expert variable wiring (`source: prev.result`)

SmolLM2 appears to struggle with the list-of-dicts YAML format for composition.

**3. Expert Boundary Confusion**

```
House flipping: expert=arithmetic, but trace includes PercentIncreaseStep
```

The model routes to the wrong expert (arithmetic instead of composition) and then emits percentage operations — same cross-expert contamination seen in early TinyLlama runs.

**4. Possible Causes**

| Hypothesis | Evidence |
|------------|----------|
| Chat template incompatibility | Baseline outputs special tokens, not YAML |
| Over-regularization from instruction-tuning | Model resists learning new format |
| Different tokenization | YAML syntax may tokenize differently |
| Hyperparameter mismatch | Learning rate may need adjustment for larger model |

### Key Insight: Bigger ≠ Better for Format Learning

The instruction-tuning that makes SmolLM2 good at chat may actively hurt its ability to learn a new structured output format. The model's priors fight against the YAML trace format.

**TinyLlama's advantage**: As a base model (not instruction-tuned), TinyLlama has weaker priors and more easily adopts the YAML format.

### GSM-8K 30-Sample Evaluation (Final)

| Metric | Value |
|--------|-------|
| **Correct** | **2/30 (7%)** |
| Parsed | 25/30 (83%) |
| Valid traces | 25/30 (83%) |
| Wrong answer | 16 |
| Invalid trace | 5 |

### Failure Pattern: "Almost Right But Missing Final Step"

| Problem | Model Did | Missing Step | Expected |
|---------|-----------|--------------|----------|
| Pizza tip | 15 × 0.2 = 3 | + 15 | **18** |
| Typing average | 47+52+57 = 156 | / 3 | **52** |
| Milk calories | 8 × 2 = 16 | × 3 | **48** |
| Gum packs | 4 × 15 × 30 = 1800 | Should be (4×30)/15 | **8** |

### Failure Pattern: Wrong Operation Selection

| Problem | Model Did | Should Do |
|---------|-----------|-----------|
| Bridge weight | 5000 + 3755 = 8755 | 5000 - 3755 = 1245 |
| Salary total | 5000 × 5000 | 5000 + 10000 + 30000 |

### Failure Pattern: Variable Overwriting

```yaml
# Beanstalk problem - reinitializes computed variable!
- {op: compute, compute_op: mul, args: [start, factor], var: step1}  # step1 = 6
- {op: init, var: step1, value: 3}  # OVERWRITES step1 = 3 !!!
```

### Failure Pattern: Expert Boundary Violation

```yaml
# Swimming problem - percentage op in arithmetic expert
expert: arithmetic
trace:
- {op: percent_of, base: distance, rate: 60, var: step1}  # INVALID!
```

### New Failure Patterns (30-Sample)

**Repeated sub-traces**: For "10 times a month", model emits same trace 10 times instead of multiplying:
```yaml
- expert: arithmetic
  trace: [{rate×time}]
- expert: arithmetic
  trace: [{rate×time}]  # REPEATED 10x!
```

**Invented operations**: Model creates non-existent ops:
```yaml
{op: add, args: [...]}  # Should be: {op: compute, compute_op: add}
```

**String as value**: Variable names as strings instead of references:
```yaml
{op: init, var: chinese, value: step1}  # 'step1' is a STRING!
```

See [SMOLLM2_1.7B.md](models/SMOLLM2_1.7B.md) for detailed analysis.

---

## Run 21: Llama-3.2-1B Base (Format vs Reasoning)

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-1B (base, no instruction-tuning)
**Config**: 3000 examples, 1 epoch SFT, 20 RL iterations

### Results

| Metric | Llama-3.2-1B (base) | SmolLM2-1.7B (instruct) |
|--------|---------------------|------------------------|
| SFT accuracy | **95%** | 65% |
| Parse rate (training) | **100%** | 95% |
| GSM-8K 30-sample | **7%** | **7%** |
| GSM-8K parse rate | **93%** | 83% |
| Valid traces | **93%** | 83% |
| Wrong answers | 23 | 16 |
| Invalid traces | 2 | 5 |

### Key Finding: Format ≠ Reasoning

Both models achieve **identical 7% GSM-8K accuracy** despite very different format learning:

| Aspect | Llama-3.2-1B | SmolLM2-1.7B |
|--------|--------------|--------------|
| Learns format | ✅ Excellent | ❌ Poor |
| Clean traces | ✅ 93% valid | ⚠️ 83% valid |
| GSM-8K accuracy | 7% | 7% |

**Conclusion**: The bottleneck is **reasoning**, not format. Both models can produce valid YAML, but neither can reason about novel problems.

### Base Model Advantage Confirmed

Base models learn structured output formats better than instruction-tuned models:

| Model Type | SFT Accuracy | Parse Rate |
|------------|--------------|------------|
| Base (Llama-3.2-1B) | **95%** | **100%** |
| Instruct (SmolLM2-1.7B) | 65% | 95% |

See [LLAMA32_1B.md](models/LLAMA32_1B.md) for detailed analysis.

---

## Run 22: Llama-3.2-1B-Instruct (BEST RESULT!)

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-1B-Instruct

### Results

| Metric | Llama Instruct | Llama Base | SmolLM2 Instruct |
|--------|----------------|------------|------------------|
| SFT accuracy | **95%** | 95% | 65% |
| Parse rate | **100%** | 93% | 83% |
| Valid traces | **100%** | 93% | 83% |
| GSM-8K | **17% (5/30)** | 7% (2/30) | 7% (2/30) |

### Key Finding: Not All Instruction-Tuning Is Equal

| Model | Instruction-Tuning Effect |
|-------|---------------------------|
| Llama-3.2-1B-Instruct | **Helps** — format preserved, reasoning improved |
| SmolLM2-1.7B-Instruct | **Hurts** — format degraded, reasoning unchanged |

Llama's instruction-tuning is compatible with learning new structured output formats. SmolLM2's is not.

### GSM-8K 30-Sample Breakdown

```
Correct: 5/30 (17%)
Parsed: 30/30 (100%)
Valid traces: 30/30 (100%)
Wrong answer: 22
Invalid trace: 0
```

**This is 2.5x better than any other model tested!**

### Checkpoints

| Checkpoint | Accuracy |
|------------|----------|
| `llama32_1b_instruct_run1_sft` | 95% (SFT) |
| `llama32_1b_instruct_run1` | 95% (final) |

---

## Run 23: Llama-3.2-3B-Instruct (Largest Model)

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-3B-Instruct (3.2B parameters)
**Config**: 3000 examples, 1 epoch SFT, 20 RL iterations, 6 unfrozen layers

### Results

| Metric | 3B-Instruct | 1B-Instruct | Delta |
|--------|-------------|-------------|-------|
| SFT accuracy | **100%** | 95% | +5% |
| Parse rate | 29/30 (97%) | 30/30 (100%) | -3% |
| Valid traces | 29/30 (97%) | 30/30 (100%) | -3% |
| GSM-8K | **27% (8/30)** | 17% (5/30) | **+10%** |

### Key Finding: Sublinear Scaling

**3x parameters → only 1.6x performance**

The 3B model shows improvement over 1B (27% vs 17%), but the scaling is sublinear. This suggests:

1. **Capacity helps somewhat** — More parameters = better pattern matching
2. **Diminishing returns** — 3x params doesn't mean 3x accuracy
3. **Same failure patterns** — Bridge, average, multi-entity problems still fail

### Same Semantic Errors

| Problem | 1B Answer | 3B Answer | Expected |
|---------|-----------|-----------|----------|
| Bridge boxes | 1.33 | 583 | 83 |
| Typing average | 156 | 114 | 52 |
| Multi-entity (robots) | 20 | 20 | 70 |

Both models make the **same conceptual errors** — proving the bottleneck is semantic understanding, not capacity.

### New Bugs in 3B

The 3B model introduced new structural issues not seen in 1B:

```yaml
# Expressions in init (invalid!)
{op: init, var: total, value: 4 * 18}

# Undefined variables
{op: compute, compute_op: mul, args: [count2, factor2], var: step2}
# count2 never initialized!
```

### Checkpoints

| Checkpoint | Accuracy |
|------------|----------|
| `llama32_3b_instruct_run1_sft` | 100% (SFT) |
| `llama32_3b_instruct_run1` | 100% (final) |

---

## Run 24: Layer Unfreezing Experiment (8 layers)

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-1B-Instruct
**Config**: 3000 examples, 1 epoch SFT, 20 RL iterations, **8 unfrozen layers**

### Hypothesis

Unfreezing more layers (50% vs 37% of model) would improve semantic understanding by training the middle "task classification" layers.

### Results

| Metric | 6 layers | 8 layers | Delta |
|--------|----------|----------|-------|
| Layers unfrozen | 6/16 (37%) | 8/16 (50%) | +13% |
| SFT accuracy | 95% | 95% | = |
| GSM-8K | 17% (5/30) | **17% (5/30)** | **= (no change!)** |

### Key Finding: More Layers ≠ Better

**Hypothesis rejected.** Unfreezing 2 additional layers provided zero improvement on GSM-8K.

| Config | Layers | GSM-8K |
|--------|--------|--------|
| 1B + 6 layers | 37% | 17% |
| 1B + 8 layers | 50% | 17% |
| 3B + 6 layers | 21% | 27% |

The 3B model's improvement came from **raw capacity**, not layer depth. Unfreezing more layers on a smaller model doesn't bridge the gap.

### Checkpoints

| Checkpoint | Accuracy |
|------------|----------|
| `llama32_1b_instruct_8layers_run1_sft` | 95% (SFT) |
| `llama32_1b_instruct_8layers_run1` | 95% (final) |

---

## Run 25: Full Fine-Tune (Catastrophic Forgetting)

**Date**: 2026-01-26
**Model**: meta-llama/Llama-3.2-1B-Instruct
**Config**: 3000 examples, 1 epoch SFT, 20 RL iterations, **ALL 16 layers unfrozen**

### Hypothesis

Full fine-tuning (100% of layers) would maximize the model's ability to learn new patterns.

### Results

| Metric | 6 layers | 16 layers (full) | Delta |
|--------|----------|------------------|-------|
| Layers unfrozen | 37% | **100%** | +63% |
| SFT accuracy | 95% | **100%** | +5% |
| GSM-8K | 17% (5/30) | **7% (2/30)** | **-10%!** |

### Key Finding: CATASTROPHIC FORGETTING

**Full fine-tune made the model WORSE!**

The model achieved 100% on training data but collapsed to 7% on GSM-8K — the same as the base model before any Llama-specific tuning benefits.

### What Went Wrong

The full fine-tune **overwrote** the model's base capabilities:

```yaml
# Run 22 (partial fine-tune): Correctly divides by 2 for "half"
{op: init, var: half, value: 2}
{op: compute, compute_op: div, args: [total, half], var: result}

# Run 25 (full fine-tune): Nonsensical
{op: init, var: half, value: 3}  # Why 3?!
{op: compute, compute_op: div, args: [step1, half], var: result}
```

The model memorized our narrow training templates so aggressively that it **unlearned** basic math concepts.

### Failure Examples

| Problem | Partial (6 layers) | Full (16 layers) |
|---------|-------------------|------------------|
| Diaper changes | ✓ Correct | 3.33 (wrong) |
| Bridge boxes | 1.33 (wrong) | 1.33 (same) |
| Pet legs | 35 (wrong) | 110 (worse!) |
| Typing average | 156 (wrong) | **invalid trace** |

### The Lesson

```
Partial fine-tune: Preserves base knowledge, adds format
Full fine-tune:    Overwrites everything with training data
```

With only 3,000 narrow examples, full fine-tune = disaster.

### Checkpoints

| Checkpoint | Accuracy |
|------------|----------|
| `llama32_1b_instruct_full_run1_sft` | 100% (SFT) |
| `llama32_1b_instruct_full_run1` | 100% (final) |
