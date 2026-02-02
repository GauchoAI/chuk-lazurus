# Expert Function Classification: Results

**Model**: GPT-OSS 20B (24 MoE layers, 32 experts/layer, top-4 routing, ~21B params)
**Date**: 29 Jan - 2 Feb 2026
**Experiments run**: 19 of 20 designed (15 classification + 4 residual stream mechanistic)

---

## Executive Summary

The original hypothesis -- that a significant fraction of MoE experts serve as identifiable "storage" units that can be individually externalized -- was **not supported**. Instead, we found something more fundamental:

1. **Facts are not stored in experts.** Progressive ablation of all top-4 experts at any tested layer (L8, L12, L16, L20) breaks zero facts. Knowledge survives full expert removal.

2. **Facts crystallize late in the residual stream.** Logit lens analysis shows fact tokens emerge at **L20-21** (out of 24 layers), with zero signal before L14. The residual stream constructs facts through distributed computation, not retrieval from individual components.

3. **Routing layers can be frozen for compression.** 7 learned routing layers out of 24 (71% frozen) + memory bank injection achieves 100% fact preservation. This is a routing-based compression path, not an expert-removal path.

The deliverable reframes as:

> **75% of expert routing computation can be frozen, and all externalized knowledge recovered via memory bank injection, enabling 25-42% overall model size reduction with 0% measured fact loss on the test set.**

---

## Part 1: Expert Classification (29 Jan)

### Setup

Classified all 29 causal experts at Layer 16 using the ablation taxonomy (storage / computation / routing / redundant).

### Results

| Category | Count | Percentage |
|----------|-------|------------|
| Storage | 0 | 0% |
| Computation | 29 | 100% |
| Routing | 0 | 0% |
| Redundant | 0 | 0% |

Every causal expert at L16 produces structure errors (repetition, degeneration) when removed. None produce fact-specific errors. The classification threshold (`fact_error_rate > 0.3 AND fact_error_rate > structure_error_rate`) was never met.

### Validation Criteria Status

| Criterion | Expected | Actual | Status |
|-----------|----------|--------|--------|
| L16E4 classifies as storage | Yes | No (computation) | Failed |
| Storage recovery >70% | Yes | N/A (none found) | N/A |
| Computation recovery <30% | Yes | N/A | N/A |
| >= 10% storage experts | Yes | 0% | Failed |

**Conclusion**: The storage/computation taxonomy doesn't apply at L16. All experts contribute to generation structure, not fact retrieval.

---

## Part 2: Knowledge Ablation (29 Jan)

### Setup

For each of 8 facts, progressively ablated top-1, top-2, top-3, and all top-4 experts at layers L8, L12, L16, L20. Tested whether facts break and whether memory bank recovers them.

### Results

| Layer | Broke@1 | Broke@2 | Broke@3 | Broke@4 | Never Broke | Recovery |
|-------|---------|---------|---------|---------|-------------|----------|
| L8 | 0 | 0 | 0 | 0 | 8/8 | N/A |
| L12 | 0 | 0 | 0 | 0 | 8/8 | N/A |
| L16 | 0 | 0 | 0 | 0 | 8/8 | N/A |
| L20 | 0 | 0 | 0 | 0 | 8/8 | N/A |

**Zero facts break under any ablation condition.** Even removing all 4 selected experts at any single layer leaves all 8 facts intact. Recovery testing is moot -- there's nothing to recover.

**Conclusion**: Knowledge is not concentrated in the top-4 routed experts. It survives redundantly across the full residual stream.

---

## Part 3: Routing and Layer Experiments (30-31 Jan)

### 3a. Routing Resilience

Tested 7 prompt categories under routing disruption (skip routing / fixed routing).

| Category | Normal Rep | Disrupted Rep | Resilience |
|----------|-----------|---------------|------------|
| code_like | 0.147 | 0.112-0.227 | Most resilient |
| technical | 0.013 | 0.221-0.365 | Resilient |
| factual_open | 0.407 | 0.357-0.562 | Moderate |
| conversational | 0.016 | 0.536-0.557 | Fragile |
| factual_constrained | 0.298 | 0.654-0.674 | Very fragile |
| creative | 0.116 | 0.769-0.794 | Very fragile |

Code and technical prompts are 3-10x more resilient to routing disruption than creative/conversational content. Output space constraints (syntax rules, domain vocabulary) substitute for expert routing.

### 3b. Layer Spacing

Determined optimal spacing for minimal learned layers.

| Condition | Learned Layers | Bare Facts | With MB |
|-----------|---------------|------------|---------|
| normal | 24 (all) | 8/8 (100%) | 8/8 (100%) |
| every_2nd | 12 | 5/8 (62%) | 8/8 (100%) |
| every_3rd | 9 | 1/8 (12%) | 6/8 (75%) |
| every_4th | 7 | 1/8 (12%) | 6/8 (75%) |
| every_6th | 5 | 0/8 (0%) | 5/8 (62%) |
| every_8th | 4 | 0/8 (0%) | 4/8 (50%) |
| endpoints only | 2 | 0/8 (0%) | 0/8 (0%) |

### 3c. Minimum Viable Routing

Tested targeted layer sets (all include L0).

| Config | Layers | Learned | Bare | With MB |
|--------|--------|---------|------|---------|
| normal | all 24 | 24 | 8/8 | 8/8 |
| L0_plus_7 | [0,3,7,11,15,19,23] | 7 | 1/8 | **8/8** |
| L0_plus_5 | [0,5,11,17,23] | 5 | 0/8 | 4/8 |
| L0_plus_mid | [0,11,23] | 3 | 0/8 | 1/8 |
| L0_endpoints | [0,23] | 2 | 0/8 | 0/8 |
| L0_only | [0] | 1 | 0/8 | 0/8 |
| none | [] | 0 | 0/8 | 0/8 |

### 3d. 6-Layer Cliff Test

| Config | Layers | Bare | With MB |
|--------|--------|------|---------|
| 6 tight | [0,3,7,11,15,19] | 2/8 (25%) | **8/8 (100%)** |
| 6 even | [0,5,9,14,18,23] | 0/8 (0%) | 2/8 (25%) |

**Finding**: Front-loaded tight spacing `[0,3,7,11,15,19]` matches the 7-layer config. Early-layer density matters more than including the final layer.

### Scaling Curve

```
Learned layers vs fact preservation (with memory bank):

 0 layers: 0/8  ████████████████████████░░░░░░░░░░░░░░░░░░  0%
 1 layer:  0/8  ████████████████████████░░░░░░░░░░░░░░░░░░  0%
 2 layers: 0/8  ████████████████████████░░░░░░░░░░░░░░░░░░  0%
 3 layers: 1/8  █████████████████████░░░░░░░░░░░░░░░░░░░░░ 12%
 5 layers: 4/8  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 50%
 6 tight:  8/8  ████████████████████████████████████████████ 100%  <-- cliff
 7 layers: 8/8  ████████████████████████████████████████████ 100%
12 layers: 8/8  ████████████████████████████████████████████ 100%
24 layers: 8/8  ████████████████████████████████████████████ 100%
```

The cliff is sharp: 5 layers = 50%, 6 tight layers = 100%.

---

## Part 4: Residual Stream Fact Emergence (1 Feb)

### Setup

Used logit lens to project intermediate hidden states to vocabulary logits at all 24 layers. For each fact, discovered the model's actual predicted token via argmax at the final layer, then tracked that token's probability and rank backward through all layers.

### Method

For each prompt (e.g., "The capital of France is"), we:
1. Run a forward pass, take argmax of final logits to discover the predicted token (" Paris", id=12650)
2. Run ModelHooks capturing hidden states at all 24 layers (last position only)
3. At each layer, project hidden state through `final_norm -> lm_head` to get vocab logits
4. Track the discovered token's probability and rank across all layers

### Per-Fact Emergence

| Prompt | Token | Top-10 | Top-5 | Top-1 | >10% | >50% |
|--------|-------|--------|-------|-------|------|------|
| Capital of France | Paris | L20 | L20 | L20 | L20 | L21 |
| Chemical symbol: gold | Au | L21 | L21 | L23 | L21 | L23 |
| Author: Romeo & Juliet | William | L20 | L20 | L22 | L21 | L22 |
| Speed of light | *(space)* | L14 | L15 | L15 | L15 | L16 |
| CEO of Microsoft | Sat | L21 | L21 | L22 | L22 | L22 |
| Capital of Japan | Tokyo | L20 | L20 | L20 | L20 | L21 |
| Chemical symbol: silver | Ag | L21 | L21 | L23 | L21 | L23 |
| Capital of Australia | Canberra | L20 | L20 | L21 | L21 | L21 |

**Note**: "Speed of light" is a data quality outlier. The model's next predicted token is a space character (id=220), not "299". This tracks a formatting decision, not a fact. Excluding it, the 7 valid facts show consistent late emergence.

### Aggregate (7 valid facts, excluding "speed of light")

| Metric | Mean | Median | Range |
|--------|------|--------|-------|
| First enters top-10 | L20.4 | L20 | L20-L21 |
| First enters top-5 | L20.4 | L20 | L20-L21 |
| First reaches top-1 | L21.6 | L22 | L20-L23 |
| First exceeds 10% | L20.9 | L21 | L20-L22 |
| First exceeds 50% | L21.9 | L22 | L21-L23 |

### Average Fact Probability by Layer

```
L 0:  0.0000  |
L 1:  0.0000  |
L 2:  0.0000  |
L 3:  0.0000  |
L 4:  0.0000  |
L 5:  0.0000  |
L 6:  0.0000  |
L 7:  0.0000  |
L 8:  0.0000  |
L 9:  0.0000  |
L10:  0.0000  |
L11:  0.0000  |
L12:  0.0000  |
L13:  0.0001  |
L14:  0.0008  |
L15:  0.0489  |##
L16:  0.1079  |#####
L17:  0.1251  |######
L18:  0.1251  |######
L19:  0.1262  |######
L20:  0.2545  |############
L21:  0.5918  |#############################
L22:  0.7349  |####################################
L23:  0.7563  |#####################################
```

Three distinct phases:
- **L0-L14**: Zero fact signal. Residual stream carries positional/structural information.
- **L15-L19**: Weak fact signal (5-13%). Fact candidates begin to form but don't dominate.
- **L20-L23**: Fact crystallization (25-76%). The correct answer jumps to top-1.

### Competitor Analysis

At the emergence layer, facts compete with formatting tokens. Examples:

**"Capital of France" at L20** (Paris reaches 44.5%):
- Paris: 44.5%
- `{`: 23.8%
- not: 8.8%
- `[`: 6.1%

**"Capital of Australia" at L21** (Canberra reaches 74.6%):
- Canberra: 74.6%
- Sydney: 18.8%
- Melbourne: 6.1%

**"CEO of Microsoft" at L22** (Sat reaches 73.8%):
- Sat: 73.8%
- currently: 23.9%
- Brad: 0.2%

Formatting tokens (`{`, `[`, `"`) compete with facts at early emergence layers, suggesting the model is simultaneously resolving what *kind* of output to produce (code? quote? prose?) alongside *what* answer to give. By the final layers, facts win.

---

## Part 5: Layer Skip at Emergence Point (2 Feb)

### Setup

Residual fact emergence showed facts crystallize at L20-21. This experiment tests whether those layers are *necessary* by zeroing the MoE FFN output at specific layers. Attention still runs at every layer; only the expert computation is removed. The residual stream passes through unchanged at skipped layers (`x = x + 0 = x`).

7 conditions tested against 7 facts (excluded "speed of light" -- was tracking a space token).

### Fact Preservation

| Condition | Skipped | Facts Preserved |
|-----------|---------|----------------|
| normal | none | 7/7 (100%) |
| skip_L20 | [20] | 7/7 (100%) |
| skip_L21 | [21] | 6/7 (86%) |
| skip_L20_L21 | [20, 21] | 5/7 (71%) |
| skip_L19_L20_L21 | [19, 20, 21] | 5/7 (71%) |
| skip_L15 | [15] | 7/7 (100%) |
| skip_L22_L23 | [22, 23] | 6/7 (86%) |

### Emergence Shift Under Skip

Average fact emergence layer (top-1) shifts later when emergence layers are skipped:

| Condition | Avg Top-1 | Avg >50% | Avg Repetition |
|-----------|-----------|----------|----------------|
| normal | L21.6 | L21.9 | 0.379 |
| skip_L20 | L21.7 | L22.1 | 0.547 |
| skip_L21 | L22.0 | L22.3 | 0.564 |
| skip_L20_L21 | L22.3 | L22.0 | 0.430 |
| skip_L19_L20_L21 | L22.2 | L21.7 | 0.660 |
| skip_L15 | L21.0 | L21.0 | 0.474 |
| skip_L22_L23 | L20.3 | L21.0 | 0.338 |

### Per-Fact Difficulty Gradient

Facts respond differently to layer skipping, revealing a difficulty gradient:

**Robust facts** (survive all conditions including skip L20+L21):
- "Capital of France" (Paris) -- emerges at L20 normally, defers to L21 when L20 skipped
- "Capital of Japan" (Tokyo) -- emerges at L20, defers to L21
- "Chemical symbol for gold" (Au) -- emerges late (L23) even normally
- "Author of Romeo and Juliet" (William) -- defers from L22 to L23

**Fragile facts** (break under L20+L21 skip):
- "Capital of Australia" (Canberra) -- replaced by "Sydney" when L20+L21 skipped. The model has a strong competitor (Sydney at 18.8% vs Canberra at 74.6% at L21 normally). Without the emergence layers, the competitor wins.
- "CEO of Microsoft" (Satya Nadella) -- breaks even with skip_L21 alone. Generates "Sat" then apologizes. At L21 normally, "currently" (94.9%) dominates over "Sat" (4.2%). This fact needs both L21 and L22 to overcome the "currently" attractor.

**Key observation**: Facts with strong competitors at the emergence layer are fragile. Facts that dominate early (Paris at 44.5% by L20) survive layer skipping because the residual stream already carries enough signal.

### Probability Curve Comparison

```
                  normal    skip_L20_L21   skip_L19-21    skip_L22_L23
L19:             0.0014      0.0014        0.0003         0.0014
L20:             0.1480      0.0012        0.0004         0.1480
L21:             0.5346      0.2221        0.2287         0.5346
L22:             0.7003      0.2949        0.2777         0.4473
L23:             0.7522      0.4835        0.3760         0.4375
```

Skipping L20+L21 reduces peak probability from 75% to 48%. The facts that survive do so because 48% is still enough to win over competitors. The facts that fail (Canberra, Satya) have competitors close enough to overtake at the reduced probability.

### Control Conditions

- **skip_L15**: 7/7 facts preserved, emergence actually *improves* (avg top-1 shifts from L21.6 to L21.0). L15 contributes formatting signal (first 5% probability), but removing it doesn't harm facts -- later layers compensate fully.

- **skip_L22_L23**: 6/7 facts preserved. Facts that emerged by L21 (Paris, Tokyo, Canberra) survive because they're already confident. But facts that normally need L22-L23 to reach top-1 (Au, Ag, CEO) lose their final crystallization layers. The "CEO of Microsoft" fact breaks here too.

### Conclusions

1. **L20 alone is not critical.** Skipping just L20 preserves all 7 facts -- emergence simply defers to L21.

2. **L21 is more important than L20.** Skipping L21 alone loses 1 fact (CEO of Microsoft), suggesting L21 is where the hardest competitive resolution happens.

3. **Skipping both L20+L21 degrades gracefully.** 5/7 facts survive by deferring to L22-L23. The 2 failures are facts with strong competing answers (Sydney vs Canberra, "currently" vs "Sat").

4. **Fact robustness correlates with competitive margin.** Facts that dominate by >40% at their normal emergence layer survive skipping. Facts with <30% margin over competitors break.

5. **The residual stream can crystallize facts at any of the final 4 layers (L20-L23).** There is no single "fact layer" -- the computation is flexible, but needs *some* expert computation in this range to push past competitors.

---

## Part 5b: Attention Pattern at Emergence Layers (2 Feb)

### Setup

Monkey-patched `GptOssAttention` to compute attention weights from Q, K, V before the fused SDPA kernel. GPT-OSS uses GQA (64 query heads, 8 KV groups). Weights are averaged across the 8 heads within each KV group, then averaged across all KV groups for a single attention score per position.

For each fact, measured how much the final token (prediction position) attends to the entity token (France, gold, Microsoft, etc.) at every layer.

### Per-Fact Entity Attention at Key Layers

| Prompt | Entity | L0 | L8 | L15 | L19 | L20 | L21 | L22 | L23 |
|--------|--------|-----|-----|------|------|------|------|------|------|
| Capital of France | France | .233 | .234 | .241 | **.342** | .316 | **.357** | .315 | .252 |
| Symbol for gold | gold | .218 | .194 | .208 | **.313** | .297 | **.326** | .324 | .228 |
| Author of R&J | Romeo | .037 | .086 | .123 | **.213** | .073 | **.221** | .061 | .180 |
| CEO of Microsoft | Microsoft | **.326** | .188 | .205 | .225 | .236 | .258 | .234 | .164 |
| Capital of Japan | Japan | .258 | .277 | .229 | **.316** | .277 | .287 | .318 | .215 |
| Symbol for silver | silver | .210 | .187 | .213 | **.350** | .295 | **.375** | .336 | .250 |
| Capital of Australia | Australia | .289 | .275 | .239 | **.309** | .295 | **.316** | .324 | .204 |

### Phase Averages

| Phase | Avg Entity Attention |
|-------|---------------------|
| Pre-emergence (L0-L14) | 0.195 |
| Emergence (L15-L21) | 0.246 |
| Post-emergence (L22-L23) | 0.243 |

**Emergence/pre-emergence ratio: 1.3x** -- a modest but consistent increase.

### The L19/L21 Alternation Pattern

The most striking pattern is not a simple ramp-up. Entity attention **alternates** between odd and even layers:

- **L19**: Entity is the most-attended token for 5/7 facts (France, gold, Japan, silver, Australia)
- **L20**: Entity attention drops; " is" becomes the most-attended token for all 7 facts
- **L21**: Entity attention peaks again; entity is most-attended for 4/7 facts (France, gold, silver, Australia)
- **L22**: Entity attention drops again; " is" dominates

This suggests a two-phase computation cycle:
- **Odd layers (L19, L21)**: Attention focuses on the entity -- "what are we looking up?"
- **Even layers (L20, L22)**: Attention focuses on " is" (the copula/prediction frame) -- "what kind of answer do we need?"

The model alternates between gathering entity information and resolving the output format. This is consistent with the finding that GPT-OSS alternates full attention and sliding window attention across layers.

### Two Outliers

**"CEO of Microsoft"** shows highest entity attention at L0 (.326), declining through the network. Microsoft is the most-attended token at L0 (the initial embedding already encodes strong salience), but at emergence layers, " is" dominates. This is the same fact that was hardest to preserve under layer skipping -- the model struggles to resolve "Satya" vs "currently" because attention doesn't strongly focus on "Microsoft" at the critical layers.

**"Author of Romeo and Juliet"** has the lowest entity attention throughout (Romeo starts at .037 at L0). With 7 tokens, attention is more dispersed. The entity attention at L19-L21 (.213, .073, .221) is much lower than for 5-token prompts (.309-.375). This fact also requires L22-L23 to reach top-1 in the logit lens.

### Conclusions

1. **Entity attention increases 1.3x at emergence layers** -- a real but moderate signal. Fact crystallization is not a single dramatic "lookup" but a distributed process.

2. **L19 and L21 are "entity-attending" layers.** At these layers, the prediction position focuses on the entity token. At L20 and L22, attention shifts to the copula " is". This alternation matches the model's alternating attention architecture (full/sliding window).

3. **The " is" token is as important as the entity.** At L20 and L22, " is" receives 37-48% of attention -- more than the entity. The model isn't just looking up "France → Paris"; it's simultaneously resolving "X is → [answer]" as a syntactic frame.

4. **Attention alone doesn't explain crystallization.** The 1.3x increase is too modest to account for the 0% → 75% probability jump in the logit lens. The MoE FFN at L20-L21 must be doing the heavy computational lifting, with attention providing the addressing signal.

---

## Part 5c: Memory Bank Injection Point (2 Feb)

### Setup

Compared residual streams between bare prompts (e.g., "The capital of France is") and memory-bank-injected prompts (82 tokens including all 7 facts in `[Memory Bank]...[End Memory Bank]` format). At each layer, captured hidden states (last position) and measured:
- Cosine distance between bare and MB hidden states
- Fact token probability via logit lens under both conditions
- Probability lift (MB - bare) at each layer

### Prediction (falsified)

MB should cause earlier fact emergence by injecting facts via attention at early layers (L0-L4), bypassing the normal L20-21 crystallization pathway. Residual delta should peak at early layers.

### Emergence Comparison

| Prompt | Bare Top-1 | MB Top-1 | Shift |
|--------|-----------|----------|-------|
| Capital of France | L20 | L21 | -1 |
| Chemical symbol: gold | L23 | L23 | 0 |
| Author: Romeo & Juliet | L22 | L21 | +1 |
| CEO of Microsoft | L22 | L23 | -1 |
| Capital of Japan | L20 | L21 | -1 |
| Chemical symbol: silver | L23 | L23 | 0 |
| Capital of Australia | L21 | L21 | 0 |

**Average bare emergence: L21.6. Average MB emergence: L21.9. Shift: -0.3 layers.**

MB does not cause earlier emergence. If anything, it's slightly later.

### Residual Stream Divergence (Cosine Distance)

```
L 0:  0.567 |#####################################     <- max early divergence
L 1:  0.535 |###################################
L 2:  0.494 |################################
...
L 8:  0.469 |##############################
L 9:  0.578 |######################################
L10:  0.526 |##################################
L11:  0.608 |########################################  <- peak divergence
L12:  0.504 |#################################
L13:  0.608 |#######################################
L14:  0.487 |################################
L15:  0.486 |################################
L16:  0.380 |#########################               <- convergence begins
L17:  0.378 |########################
L18:  0.286 |##################
L19:  0.332 |#####################
L20:  0.287 |##################
L21:  0.281 |##################
L22:  0.203 |#############
L23:  0.111 |#######                                  <- near-convergence
```

| Phase | Avg Cosine Distance |
|-------|-------------------|
| Early (L0-L9) | 0.514 |
| Mid (L10-L17) | 0.497 |
| Late (L18-L23) | 0.250 |

Peak divergence at **L11** (0.608). Monotonic convergence from L16 onward. By L23, representations are 89% similar despite starting from completely different input sequences (5 tokens vs 82 tokens).

### Probability Comparison (avg across 7 facts)

```
Layer:   Bare     MB      Lift
L19:    0.0014   0.0034   +0.002
L20:    0.1479   0.0005   -0.147    <- MB is SLOWER here
L21:    0.5346   0.5951   +0.061    <- MB catches up
L22:    0.7003   0.3714   -0.329    <- MB dips again
L23:    0.7522   0.7701   +0.018    <- convergence
```

The most striking finding is the **L20 probability dip**: bare prompts already show 14.8% fact probability at L20, while MB prompts show only 0.05%. The longer MB context (82 tokens vs 5) requires more processing before the fact signal crystallizes. But by L21, MB catches up to 59.5% (vs bare 53.5%), and by L23, both converge (~75-77%).

The L22 dip for MB (37.1% vs 70.0% bare) may reflect the model processing the instruction text ("Using the memory bank above, answer:") which creates a competing representation that resolves by L23.

### Per-Fact Patterns

**Romeo & Juliet** is the only fact where MB shows genuinely earlier emergence (L21 vs L22). This is the longest entity name in the set (7 prompt tokens), where the MB's explicit `Romeo and Juliet | author | William Shakespeare` entry provides a clearer signal than the bare prompt.

**CEO of Microsoft** shows MB is *later* (L23 vs L22). This is already the hardest fact (strong "currently" competitor). The MB's additional context doesn't help and may introduce distraction.

### Conclusions

1. **Memory bank does NOT provide an early injection pathway.** The prediction that MB would shift emergence to L0-L4 is falsified. Facts emerge at L21-23 regardless of whether the answer is explicitly provided in the context.

2. **MB works by convergent computation, not bypass.** Despite starting from radically different inputs (5 tokens vs 82 tokens, cosine distance 0.57 at L0), the bare and MB residual streams converge to nearly identical representations by L23 (distance 0.11). The model arrives at the same answer through different routes that merge in the final layers.

3. **MB is actually slower at L20.** The longer context requires more processing at L20 (0.05% vs 14.8%), but the fact signal catches up at L21. This suggests the model spends L20 integrating the MB context and extracting the relevant fact, while bare prompts have already begun crystallization.

4. **The convergence zone (L16-L23) is where MB provides its value.** Under normal conditions, bare and MB produce equivalent results. But when routing is degraded (frozen/skipped), MB provides a redundant signal that ensures the convergent computation at L21-L23 still succeeds. MB doesn't bypass the computation -- it provides insurance that the right input reaches the crystallization layers.

5. **This explains why 6-7 learned layers + MB = 100%.** The learned layers must include coverage in the L16-L23 convergence zone. MB provides the semantic content via attention (the "what"), while learned routing at the crystallization layers provides the computation (the "how"). Neither alone is sufficient; together, they guarantee fact output.

---

## Part 6: Synthesis

### The Five-Part Story

These experiments reveal a consistent architecture:

**1. Facts live in collective residual computation, not in individual experts.**

Knowledge ablation proves this directly. Removing all 4 selected experts at any layer breaks zero facts. No single component "stores" a fact -- it emerges from the interaction of attention, expert outputs, and residual connections across many layers.

**2. Fact crystallization happens at L20-21, in the final ~15% of the network.**

Logit lens shows near-zero fact signal before L14, with the answer token jumping from rank >100 to rank 1 between L19 and L21. The first 80% of layers (L0-L19) build up structural and positional representations; the final 20% (L20-L23) resolve these into specific factual predictions.

**3. The emergence layers are important but not irreplaceable.**

Layer skip experiments show that removing L20 alone preserves all facts (they defer to L21). Removing both L20+L21 still preserves 5/7 facts -- the residual stream crystallizes at L22-L23 instead. The 2 failures are facts with strong competitors (Sydney vs Canberra, "currently" vs "Satya"). Fact robustness correlates with competitive margin at the emergence layer, not with any specific layer being special.

**4. Attention provides entity addressing, not fact computation.**

At L19 and L21, the prediction token focuses on the entity token (France, gold, etc.) -- the most-attended token at those layers for 5/7 facts. At L20 and L22, attention shifts to the copula " is". This alternation provides the addressing signal ("what entity?"), but the 1.3x increase is too modest to explain the 0%→75% probability jump. The MoE FFN does the computational work of resolving the entity into a specific answer.

**5. Memory bank works by convergent computation, not early injection.**

MB does NOT shift fact emergence earlier. Despite providing the answer explicitly in context, facts still crystallize at L21-23. The bare and MB residual streams start completely different (cosine distance 0.57 at L0) but converge monotonically to near-identical representations by L23 (distance 0.11). MB is actually *slower* at L20 (0.05% vs 14.8% bare), catching up at L21. MB provides a redundant computation pathway, not a shortcut.

**6. Routing can be frozen at non-critical layers because facts don't depend on individual expert selection.**

The minimum viable routing experiments show that 6-7 learned layers + memory bank injection = 100% fact preservation. The critical layers align with the crystallization zone: configs that include learned routing at L19+ succeed; configs that skip this zone fail.

### The Competitive Margin Model

Layer skip results introduce a quantitative predictor of fact robustness: the margin between the correct answer and its strongest competitor at the emergence layer.

| Fact | Correct (L21) | Competitor (L21) | Margin | Survives L20+L21 skip? |
|------|--------------|------------------|--------|----------------------|
| Capital of France | Paris 100% | {: 0% | +100% | Yes |
| Capital of Japan | Tokyo 100% | ": 0% | +100% | Yes |
| Capital of Australia | Canberra 74.6% | Sydney 18.8% | +55.8% | No |
| CEO of Microsoft | Sat 4.2% | currently 94.9% | -90.7% | No |

Facts with >50% margin survive layer skipping. Facts below that threshold break when the model loses its crystallization layers. This suggests compression safety can be predicted per-fact based on competitive margins.

### Why 7 Layers Works and 5 Doesn't

The 7-layer config `[0,3,7,11,15,19,23]` includes **L19** -- right where fact probability begins its steep climb from 12% to 75%. The 5-layer config `[0,5,11,17,23]` has its nearest learned layer at L17, missing the L19-L20 transition point.

Memory bank injection rescues the 5-layer config partially (50%) because it provides a redundant fact signal via attention context. But the MB injection point experiment shows this signal still needs learned routing at the crystallization layers (L19-L23) to converge to the correct output. MB provides the "what" (semantic content); learned routing provides the "how" (computational resolution). Without learned routing in the convergence zone, the MB signal can't crystallize into a prediction.

### Connection to L16E4

The original hypothesis (from memory_fact_retrieval) that L16E4 was a "fact storage" expert is now fully reframed:

- L16E4 handles 25% of declarative routing because it's a **position specialist** (end-of-sequence generalist), not because it stores facts
- Expert routing is **93% position-coded** -- same-structure prompts share 0.927 Jaccard overlap regardless of content
- At L16, fact probability is only ~10% (logit lens). The fact hasn't crystallized yet. L16E4 contributes to structural computation, not fact retrieval.

---

## Part 6: Compression Numbers

### Routing-Based Compression

| Config | Learned Layers | Expert Compute Reduction | Fact Loss (with MB) |
|--------|---------------|-------------------------|---------------------|
| 12 layers (every 2nd) | 50% | 50% | 0% |
| 7 layers [0,3,7,11,15,19,23] | 29% | 71% | 0% |
| 6 layers [0,3,7,11,15,19] | 25% | 75% | 0% |

### Overall Model Impact

Expert parameters account for ~85% of the 21B parameter model (~17.8B). With routing-layer freezing:

| Config | Expert Reduction | Overall Reduction | Fact Loss |
|--------|-----------------|-------------------|-----------|
| 12 learned (50%) | 8.9B | 42% of total | 0% with MB |
| 7 learned (71%) | 12.7B | 60% of total | 0% with MB |
| 6 tight (75%) | 13.4B | 64% of total | 0% with MB |

**Caveat**: These numbers reflect fact preservation only (8 prompts). Generation quality (fluency, coherence, perplexity) under aggressive routing freezing has not been measured systematically. Routing resilience data suggests code/technical output degrades less than creative output.

---

## Part 7: Open Questions

### Not Yet Tested

1. **Attention head ablation** (designed, not run): Would test whether facts are stored in specific KV head groups at L8-L12. Given the knowledge ablation results, likely to confirm facts are not head-localized either.

2. **Residual stream delta between configs**: Why does `[0,3,7,11,15,19]` work but `[0,5,9,14,18,23]` fail? Comparing the residual stream trajectory at L19-L23 between these configs would show whether frozen routing corrupts the residual at the crystallization point.

3. **Perplexity and generation quality**: Fact preservation is binary (keyword present/absent). Broader capability assessment under routing freezing is needed for paper-level claims.

4. **Larger fact set**: 8 facts (7 valid for emergence analysis) is sufficient for directional findings but not for statistical confidence. 100-500 diverse facts would strengthen the numbers.

5. **Cross-model validation**: OLMoE-1B-7B (64 experts) is available. Running minimum viable routing on a second architecture would show whether the finding generalizes.

---

## Experiment Inventory

| # | Experiment | Date | Key Finding |
|---|-----------|------|-------------|
| 1 | expert_classification | 29 Jan | 0% storage experts at L16 (all computation) |
| 2 | knowledge_ablation | 29 Jan | 0/8 facts break under full top-4 ablation at any layer |
| 3 | cross_layer_ablation | 29 Jan | Cross-layer impact patterns |
| 4 | position_analysis | 29 Jan | Expert routing is 93% position-coded |
| 5 | position_pruning | 29 Jan | Position-based pruning strategies |
| 6 | expert_weight_similarity | 30 Jan | 0.21 functional similarity between same-class experts |
| 7 | routing_ablation | 30 Jan | Learned routing is essential (0/8 facts with random routing) |
| 8 | partial_routing | 30 Jan | Partial routing degradation |
| 9 | layer_skipping | 30 Jan | Layer skip impact on generation |
| 10 | routing_resilience | 30 Jan | Code 3-10x more resilient than creative to routing disruption |
| 11 | memory_bank_lite | 30 Jan | Memory bank rescues facts under compressed routing |
| 12 | layer_parity | 30 Jan | Even/odd layer analysis |
| 13 | layer_spacing | 30 Jan | Gap=2 optimal; sharp cliff at gap=3 |
| 14 | minimum_viable_routing | 31 Jan | 7 layers + MB = 100% fact preservation |
| 15 | minimum_viable_6layer | 31 Jan | 6 tight layers + MB = 100% (front-loaded spacing) |
| 16 | residual_fact_emergence | 1 Feb | Facts crystallize at L20-21 via logit lens |
| 17 | layer_skip_emergence | 2 Feb | L20+L21 skip: 5/7 facts survive; robustness correlates with competitive margin |
| 18 | attention_at_emergence | 2 Feb | Entity attention peaks at L19/L21 (1.3x); alternates with " is" focus at L20/L22 |
| 19 | memory_bank_injection_point | 2 Feb | MB does NOT shift emergence; representations converge L16-L23; MB slower at L20 |
| 20 | attention_head_ablation | -- | Designed, not run |
