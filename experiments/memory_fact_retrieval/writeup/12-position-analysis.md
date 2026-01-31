## Part 11: Position Analysis - What Experts Actually Specialize In

### 11.1 The Question

We know experts don't store facts (Part 9), and they maintain the residual stream through position-independent computation (Part 10). But L16E4 has 3x the routing disruption of other experts. Why? What makes one expert more impactful than another?

Hypothesis: experts specialize by **token position** within the sequence, not by semantic content. If "The capital of France is" and "The capital of Japan is" route to the same experts at the same positions, expert routing is position-coded.

### 11.2 Experiment Design

- **17 prompts** across 5 groups:
  - Same structure, different content: "The capital of France/Japan/Australia/Germany is"
  - Same structure, different content: "The chemical symbol for gold/silver/iron is"
  - Mixed factual: author, CEO, speed of light
  - **Different structure, same content**: "France's capital city is", "What is the capital of France?", "Paris is the capital of"
  - Different types: creative, linguistic, reasoning
- **4 layers**: L8, L12, L16, L20
- For each prompt: capture expert routing at every token position
- Measure cross-prompt overlap at matched structural positions

### 11.3 Position Selectivity Results

Many experts fire ONLY at specific positions in the sequence:

```
Layer | Avg Selectivity | High Sel (>0.1) | Selectivity=1.0 Experts
------|-----------------|-----------------|-------------------------------------------
L8    | 0.400           | 24/32 (75%)     | E11 (late_mid), E29 (early_mid)
L12   | 0.534           | 17/32 (53%)     | E2 (end), E5 (end), E6 (end)
L16   | 0.515           | 20/32 (63%)     | E1 (late_mid), E7 (start), E8 (start)
L20   | 0.394           | 23/32 (72%)     | E7 (end), E14 (start), E22 (end)
```

At Layer 16:
- **E7 and E8**: selectivity=1.0, ONLY fire at sequence start positions
- **E1**: selectivity=1.0, ONLY fires at late-middle positions
- **L16E4** (the "fact routing" expert): selectivity=**0.02**, fires at "end" positions - nearly the LEAST selective expert

L16E4 is not a fact specialist. It's an **end-of-sequence generalist**. Its 25% share of declarative routing (Part 2) occurs because declarative fact prompts predict the answer at the last token, which is an "end" position where L16E4 naturally activates.

### 11.4 Cross-Prompt Consistency - The Key Result

For prompt pairs with the same structure but different content:

```
Same structure, different content:  0.927 Jaccard overlap
Different structure, same content:  0.416 Jaccard overlap
```

**92.7% of expert routing is shared** between "The capital of France is" and "The capital of Japan is" at matched positions. Only **41.6%** is shared between "The capital of France is" and "France's capital city is" (same fact, different phrasing).

At the last token (where the fact is predicted):

```
Last-token overlap (same structure):  0.800
Last-token overlap (diff structure):  0.321
```

Even at the prediction position, **80%** of expert selection is determined by structure, not content. The model routes "Paris" and "Tokyo" through the same experts because their prompts have the same shape.

### 11.5 What This Means

```
Expert routing decision tree:

  Token arrives at MoE layer
      │
      ▼
  "What position is this token in the sequence?"
      │
      ├── Start (0-25%):  → E7, E8 (position specialists)
      ├── Early-mid:      → E29, etc.
      ├── Late-mid:       → E1, etc.
      └── End (75-100%):  → E4, E11, etc. (end generalists)
      │
      ▼
  "What is this token's content?"
      │
      └── Fine-grained selection within position-appropriate pool
```

The routing hierarchy is: **position first, content second.**

### 11.6 Reframing L16E4

| Original Interpretation (Part 2) | Revised Interpretation (Part 11) |
|----------------------------------|----------------------------------|
| L16E4 is a "fact lookup" expert | L16E4 is an end-of-sequence generalist |
| It handles 25% of declarative routing | It fires at end positions where facts happen to be predicted |
| Procedural facts bypass it | Procedural prompts may have different positional patterns |
| Its 0.108 routing disruption = fact hub | Its disruption = positional disruption at sequence end |

L16E4 appeared to specialize in facts because our factual prompts all predicted answers at the last token, and L16E4 is an end-position expert. The apparent content specialization was an artifact of position-content correlation in our test set.

### 11.7 Ablation: Position Selectivity Doesn't Predict Impact

| Expert Category | Facts Preserved |
|----------------|-----------------|
| High selectivity (position specialists) | 9/9 (100%) |
| Low selectivity (generalists) | 9/9 (100%) |

Neither position-specialists nor generalists are critical for factual recall when ablated individually. This is consistent with Parts 9-10: facts are in the residual stream, and any single expert is dispensable.

### 11.8 Position-Coding Across Layers

The position preference changes across layers:

| Layer | Selectivity=1.0 Experts | Preferred Regions |
|-------|------------------------|-------------------|
| L8 | E11, E29 | late_mid, early_mid |
| L12 | E2, E5, E6 | end, end, end |
| L16 | E1, E7, E8 | late_mid, start, start |
| L20 | E7, E14, E22 | end, start, end |

The same expert index (e.g., E7) can be a "start" specialist at L16 but an "end" specialist at L20. Position-coding is layer-specific, not a fixed property of the expert index.

### 11.9 Implications for Model Architecture

1. **MoE routing is primarily positional**: 92.7% of routing is explained by sequence position, leaving only ~7% for content-based specialization. This aligns with the prior trigram finding that expert selection tracks positional context.

2. **Expert routing is closer to a CNN than to content-addressed memory**: Like convolutional filters that apply position-dependent transformations, MoE experts apply position-dependent computation to the residual stream.

3. **The "expert as lookup table" hypothesis is wrong**: Experts don't store facts, don't specialize by content, and aren't addressed by semantic similarity. They provide position-specific computational transformations that maintain the residual stream.

4. **This explains the DRAM refresh finding (Part 10)**: Since experts provide position-specific computation, removing most experts at all layers means no position gets adequate computation. The residual stream degrades uniformly, not because specific facts are lost, but because the positional processing pipeline breaks.

## Part 12: Position-Class Pruning - How Much Can We Compress?
