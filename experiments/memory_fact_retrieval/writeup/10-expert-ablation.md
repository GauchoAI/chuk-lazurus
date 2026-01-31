## Part 9: Expert Ablation - Where Does Knowledge Actually Live?

### 9.1 The Question

The MoE routing analysis (Part 2) found that L16E4 handles 25% of declarative fact routing. The natural follow-up: if you remove L16E4, does the model lose facts? More broadly, what fraction of expert capacity is "storage" (externalizable to a database) vs "computation" (irreducible)?

### 9.2 Experiment Design

We systematically ablated each expert in Layer 16 by masking it from the router (setting its logits to -1e9 so no tokens route to it). For each of the 32 experts, we:

1. **Quick scan**: Test 2 factual prompts. Does output change?
2. **Deep test**: If causal, test all 15 prompts across 5 categories (factual, procedural, reasoning, linguistic, generation)
3. **Classify errors**: fact error (expected keyword lost), structure error (degenerate output), or no change

### 9.3 Results: Nothing Breaks

```
Layer 16: 29/32 experts are causal (change output on at least 1 prompt)
          29/29 classified as REDUNDANT
          0/29 fact errors across all experts and all prompts
          0/29 structure errors (after fixing false positive detection)
```

**No single expert ablation causes fact loss.** Here's why:

#### Facts survive every ablation

"The capital of France is" changed output in 29/29 ablations. But in every single case, the first token generated was still "Paris":

```
Baseline:  Paris." # Test with a non-existent article...
L16E0:     Paris." # Test with a non-existent page...
L16E4:     Paris." Sure! Here's a simple example...
L16E20:    Paris." # Test with a non-existent page...
```

The **fact** ("Paris") is always correct. What changes is the **continuation** after the fact - different code snippets, different conversational responses. The expert ablation affects generation style, not factual recall.

#### Output sensitivity varies by prompt type

| Prompt | Changed by N/29 experts |
|--------|------------------------|
| Capital of France | 29/29 (100%) |
| CEO of Microsoft | 24/29 (83%) |
| Math operations order | 24/29 (83%) |
| Once upon a time | 20/29 (69%) |
| All cats are mammals | 9/29 (31%) |
| Pattern 2,4,8,16 | 3/29 (10%) |
| Speed of light | 0/29 (0%) |
| Opposite of hot | 0/29 (0%) |

Open-ended prompts ("capital of France is [then what?]") are highly sensitive to expert routing. Constrained prompts ("opposite of hot is [cold]") are not. The fact itself is always preserved.

#### L16E4 stands out only in routing disruption

| Expert | Routing Disruption (JS div) | No-Change Rate |
|--------|----------------------------|----------------|
| **L16E4** | **0.108** | 73% |
| L16E11 | 0.035 | 87% |
| L16E13 | 0.035 | 73% |
| L16E24 | 0.035 | 73% |
| L16E0 | 0.000 | 87% |
| All others | 0.000 | 73-87% |

L16E4 has 3x the routing disruption of any other expert. When you remove it, downstream experts shift their selection patterns. This confirms its role as a routing hub for factual queries, even though its removal doesn't break facts.

### 9.4 Why Single-Expert Ablation Doesn't Work

GPT-OSS uses **top-4 routing** - every token goes to 4 of 32 experts. When you ablate 1, the remaining 3 compensate. The model was trained with load balancing, so no single expert is a bottleneck.

This means the original hypothesis - "some experts are storage, and ablating them loses facts" - is wrong at the single-expert level. Knowledge in a top-k MoE is:

1. **Not concentrated in one expert** - it's distributed across the top-k set
2. **Not even limited to the MoE layer** - the residual stream carries factual information independently (we showed in Part 1 that facts are linearly classifiable from hidden states at L4, long before L16)
3. **Resilient by design** - load-balanced training ensures no single point of failure

### 9.5 What This Means for Externalization

The finding is more nuanced than "experts are storage" or "experts are computation":

| Model | What We Expected | What We Found |
|-------|------------------|---------------|
| **Single expert = fact store** | Remove expert → lose fact | Remove expert → fact preserved, continuation changes |
| **Top-4 = redundant storage** | Remove 1 of 4 → lose fact | Remove 1 of 4 → nothing breaks |
| **Experts = style/routing** | N/A | Experts control *how* the model continues, not *what fact* it retrieves |

The actual architecture is: **facts live in the residual stream** (shown by L4 probing), and **experts shape the generation style** (shown by ablation changing continuations but not facts). L16E4 is not a "fact lookup" expert - it's a "fact routing" expert that directs how factual knowledge flows to later layers.

### 9.6 Implications for the Virtual Expert Architecture

This doesn't invalidate the memory bank approach from Part 8 - that still works at 100%. It changes *why* it works:

- **Old theory**: Memory bank replaces a "storage expert" (L16E4) that holds facts
- **New theory**: Memory bank provides facts via the **residual stream** (context at L4), bypassing expert routing entirely. The model's attention mechanism reads the memory bank and places the fact into the residual stream before MoE routing even happens.

This is actually *better* for externalization:
- You don't need to identify or ablate specific experts
- The memory bank works at the prompt level, not the expert level
- It's architecture-agnostic - works regardless of how many experts are active or which ones are selected

### 9.7 Revised Architecture

```
                    Query Input
                        │
                        ▼
┌────────────────────────────────────────────┐
│  [Memory Bank] injected at prompt level     │
│  Facts enter the residual stream at L0-L4   │
│  via attention to memory bank tokens        │
└────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────┐
│  MoE routing (L5-L20)                       │
│  Experts shape style/continuation           │
│  Facts already in residual stream           │
│  Expert selection affects HOW, not WHAT     │
└────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────┐
│  Generation: fact from residual stream      │
│  + style from expert routing                │
└────────────────────────────────────────────┘
```

### 9.8 Progressive Ablation: The Definitive Test

Single-expert ablation doesn't find knowledge loss because top-4 routing provides redundancy. The `knowledge_ablation.py` script tests progressive ablation: for each fact, identify the 4 experts selected at the last token position, then remove 1, then 2, then 3, then all 4. At what point does the model lose the fact?

#### Setup

- **8 facts**: capitals (France, Japan, Australia), chemical symbols (gold, silver), speed of light, CEO of Microsoft, author of Romeo and Juliet
- **4 layers**: L8, L12, L16, L20
- **32 total fact-layer combinations** (8 facts x 4 layers)
- For each combination: identify top-4 experts, progressively ablate 1→2→3→4, test if expected keyword survives

#### Results

```
Layer | Broke@1 | Broke@2 | Broke@3 | Broke@4 | Never | Avg Break | Recovery
------|---------|---------|---------|---------|-------|-----------|----------
L8   |       0 |       0 |       0 |       0 |     8 |       inf | N/A
L12  |       0 |       0 |       0 |       0 |     8 |       inf | N/A
L16  |       0 |       0 |       0 |       0 |     8 |       inf | N/A
L20  |       0 |       0 |       0 |       0 |     8 |       inf | N/A
```

**0 out of 32 fact-layer combinations broke under full top-4 ablation.**

Every fact survived removal of all 4 experts selected for it at every layer tested.

#### Per-Fact Detail (Layer 16)

| Fact | Top-4 Experts | Ablate 1 | Ablate 2 | Ablate 3 | Ablate 4 |
|------|---------------|----------|----------|----------|----------|
| Capital of France | [13,14,4,31] | Paris | Paris | Paris | Paris |
| Chemical symbol for gold | [31,2,4,27] | Au | Au | Au | Au |
| Author of Romeo and Juliet | [2,4,31,27] | Shakespeare | Shakespeare | Shakespeare | Shakespeare |
| Speed of light | [2,14,4,27] | 299M | 299M | 299M | 299M |
| CEO of Microsoft | [30,2,31,27] | Nadella | Nadella | Nadella | Nadella |
| Capital of Japan | [27,14,4,31] | Tokyo | Tokyo | Tokyo | Tokyo |
| Chemical symbol for silver | [30,2,4,27] | Ag | Ag | Ag | Ag |
| Capital of Australia | [14,27,31,4] | Canberra | Canberra | Canberra | Canberra |

The fact token is always the first token generated, regardless of how many experts are ablated. What changes is the continuation *after* the fact.

#### What Changes Under Full Ablation

The continuations change even when facts don't:

```
Capital of France (L16, all 4 ablated):
  Baseline: Paris." # Test with a non-existent article...
  Ablated:  Paris. The capital of France is Paris. The capital of France is Paris...

Capital of France (L20, all 4 ablated):
  Baseline: Paris." # Test with a non-existent article...
  Ablated:  Paris." Sure! Here's a simple example of a Python program...
```

With all top-4 experts removed, the model falls into repetitive patterns or shifts to different continuation styles, but the factual first token never changes.

#### Cross-Layer Expert Routing

Different layers route the same fact through completely different experts:

| Fact | L8 Experts | L12 Experts | L16 Experts | L20 Experts |
|------|------------|-------------|-------------|-------------|
| Capital of France | [4,17,0,13] | [15,5,21,14] | [13,14,4,31] | [29,12,1,20] |
| Chemical symbol for gold | [4,19,13,0] | [23,21,15,14] | [31,2,4,27] | [0,12,20,9] |
| CEO of Microsoft | [23,13,19,0] | [15,17,23,14] | [30,2,31,27] | [26,11,20,1] |

No expert appears consistently across all layers for the same fact. L16E4 appears in 7/8 facts at Layer 16 but is absent at other layers. Knowledge cannot be localized to any specific expert or set of experts across the full network.

#### Memory Bank Recovery Under Full Ablation

Even with all 4 experts ablated, the memory bank works:

| Layer | Recovery Success |
|-------|-----------------|
| L8 | 8/8 (100%) |
| L12 | 7/8 (88%) |
| L16 | 7/8 (88%) |
| L20 | 8/8 (100%) |

The one failure (speed of light at L12 and L16) is a detection artifact: the model generated "3.0 x 10^8 m/s" (scientifically correct) but the expected keyword was "299" (the numeric prefix). The fact was recovered; the format changed.

### 9.9 Conclusion (Refined in Part 10)

The progressive ablation experiment settles the single-layer question:

**No single layer's experts hold factual knowledge. Facts survive even full top-4 ablation at any layer.**

Evidence:
1. Removing all 4 top-k experts at any single layer preserves 100% of facts (32/32)
2. Different layers route the same fact through completely different expert sets
3. The residual stream carries factual representations from L4 onward (Part 1)
4. Memory bank injection works even under full expert ablation (30/32 = 94%)

However, Part 10 reveals a critical nuance: while no *specific* experts hold facts, the *collective computation* of experts across all layers is required to maintain the residual stream's factual representations. Ablating 28/32 experts at ALL 24 layers causes model collapse.

**Memory banks work because they enter facts via the residual stream** through attention at early layers. This operates independently of expert routing and doesn't require manipulating specific experts.
