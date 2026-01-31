# Memory & Fact Retrieval Experiments

Investigates how LLMs store, retrieve, and process factual knowledge.

## Key Questions

1. **Parametric Memory**: Where are facts stored in the model weights? Which layers encode different fact types?
2. **Context vs Memory**: How does the model distinguish facts from weights vs in-context information?
3. **Fact Manipulation**: Can we steer/suppress/inject facts via activation manipulation?
4. **Retrieval Dynamics**: What attention patterns emerge during fact lookup?

## Hypotheses

Based on findings from `probe_classifier` and `ir_emission`:

| Hypothesis | Basis |
|------------|-------|
| Facts are linearly separable at intermediate layers (L8-L12) | Task info is linearly separable at 90%+ accuracy |
| Entity facts vs numeric facts have different storage patterns | MoE experts specialize for different patterns |
| Context facts override parametric facts via attention | Attention provides 89-98% of routing signal |
| Fact retrieval can be steered like CoT generation | Format gate steering achieves 100% flip rate |

## Analyses

### 1. Parametric Memory Probing (`parametric_probing`)

**Question**: At which layers are different fact types encoded?

```
Method:
1. Create fact completion prompts: "The capital of France is"
2. Extract hidden states at each layer
3. Train linear probes to predict fact category AND fact value
4. Measure layer-wise accuracy curves

Expected:
- L4: Category detection (entity/numeric/temporal)
- L8: Relation encoding (capital_of, born_in, etc.)
- L12+: Value retrieval (Paris, 1945, etc.)
```

### 2. Context vs Memory Discrimination (`context_memory`)

**Question**: How does the model prioritize conflicting information?

```
Method:
1. Create conflict scenarios:
   - Parametric: "Paris is the capital of France"
   - Context: "The capital of France was changed to Lyon"
2. Probe which source dominates at each layer
3. Identify the "override layer" where context takes precedence

Expected:
- Early layers: Parametric memory activates
- Middle layers: Context representation builds
- Late layers: Context overrides (if strong enough)
```

### 3. Fact Manipulation via Steering (`fact_steering`)

**Question**: Can we inject or suppress facts via activation steering?

```
Method:
1. Collect "fact-present" vs "fact-absent" activations
2. Extract steering directions via PCA/difference
3. Test:
   - Suppression: Block known facts from being retrieved
   - Injection: Make model retrieve incorrect facts
   - Enhancement: Increase confidence in correct facts

Expected:
- Facts are suppressible with strength 3-5 at L8-L12
- Injection requires stronger steering (may cause incoherence)
- Enhancement is easier than suppression
```

### 4. Retrieval Dynamics (`retrieval_attention`)

**Question**: What attention patterns characterize fact lookup?

```
Method:
1. Compare attention patterns for:
   - Known facts (confident retrieval)
   - Unknown facts (uncertainty/hedging)
   - Hallucinated facts (false confidence)
2. Identify "retrieval heads" that specialize in fact lookup
3. Track attention flow: query → context → fact tokens

Expected:
- Retrieval heads emerge at L4-L8
- Known facts show sharp attention spikes
- Hallucinations show diffuse attention patterns
```

### 5. Fact Type Specialization (`fact_types`)

**Question**: Do different fact types have different neural signatures?

```
Fact Types:
- Entity: "Paris is the capital of France"
- Numeric: "Water boils at 100 degrees Celsius"
- Temporal: "World War II ended in 1945"
- Procedural: "To make coffee, grind beans first"

Method:
1. Collect hidden states for each fact type
2. Train multi-class classifier on fact type
3. Analyze which layers specialize for which types
4. Check if MoE models route different facts to different experts
```

### 6. Retrieval Confidence Calibration (`confidence`)

**Question**: Does internal confidence correlate with factual accuracy?

```
Method:
1. Collect hidden states before fact generation
2. Train probe to predict correctness
3. Compare probe confidence vs output probability
4. Identify "uncertainty markers" in hidden states

Expected:
- Internal states contain calibration signal
- Hallucinations have detectable uncertainty markers
- Could enable fact verification before generation
```

## Fact Categories

### Entity Facts
```python
ENTITY_FACTS = [
    ("The capital of France is", "Paris"),
    ("The CEO of Apple is", "Tim Cook"),
    ("The author of 1984 is", "George Orwell"),
    ("The chemical symbol for gold is", "Au"),
    ("The largest planet is", "Jupiter"),
]
```

### Numeric Facts
```python
NUMERIC_FACTS = [
    ("The speed of light is approximately", "299,792,458 meters per second"),
    ("Water freezes at", "0 degrees Celsius"),
    ("The human body has", "206 bones"),
    ("A year has", "365 days"),
    ("Pi equals approximately", "3.14159"),
]
```

### Temporal Facts
```python
TEMPORAL_FACTS = [
    ("World War II ended in", "1945"),
    ("The Declaration of Independence was signed in", "1776"),
    ("The first moon landing was in", "1969"),
    ("The Berlin Wall fell in", "1989"),
    ("Shakespeare was born in", "1564"),
]
```

### Procedural Facts
```python
PROCEDURAL_FACTS = [
    ("To boil water, heat it to", "100 degrees Celsius"),
    ("The first step in CPR is to", "check responsiveness"),
    ("To convert Celsius to Fahrenheit, multiply by 9/5 and add", "32"),
    ("The order of operations in math is", "PEMDAS"),
]
```

## Run

```bash
# All analyses
lazarus experiment run memory_fact_retrieval

# Specific analysis
lazarus experiment run memory_fact_retrieval --analysis parametric_probing
lazarus experiment run memory_fact_retrieval --analysis context_memory
lazarus experiment run memory_fact_retrieval --analysis fact_steering
```

## Expected Insights

1. **Layer-wise Fact Localization**: Facts emerge at specific layers based on type
2. **Context Override Mechanism**: Identifies how/where context overrides parametric memory
3. **Steering Feasibility**: Determines if facts can be manipulated post-hoc
4. **Retrieval Heads**: Identifies attention heads specializing in fact lookup
5. **Hallucination Detection**: Internal markers for unreliable outputs

## Connections to Existing Work

| This Experiment | Related Work |
|-----------------|--------------|
| Parametric probing | `probe_classifier` - same methodology |
| Context override | `ir_attention_routing` - attention retrieval |
| Fact steering | `format_gate` - steering vectors at L8 |
| Retrieval dynamics | `moe_attention_routing` - attention analysis |
| Fact types | `moe_expert_patterns` - token type specialization |

## Architecture Implications

If facts are linearly separable and steerable:
- **Fact Verification**: Pre-generation verification of fact confidence
- **Grounded Generation**: Inject facts from external knowledge base
- **Selective Forgetting**: Suppress outdated or incorrect facts
- **Fact Circuits**: Identify and analyze fact-specific circuits
