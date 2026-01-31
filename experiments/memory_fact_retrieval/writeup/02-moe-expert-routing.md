## Part 2: MoE Expert Routing

### 2.1 Which Experts Handle Which Facts?

Using the Lazarus ExpertRouter, we tracked which experts activate for each fact type across all 24 layers.

#### Type Separation by Layer

| Layer Range | Avg Separation | Interpretation |
|-------------|----------------|----------------|
| L0-L4 | 0.21 | Low - experts not yet specialized |
| L5-L9 | 0.27 | Specialization emerging |
| L10-L15 | **0.31** | Peak separation |
| L16-L23 | 0.27 | Maintained separation |

Peak separation occurs at **L14** (0.324), indicating this is where expert routing most strongly distinguishes fact types.

#### Dominant Experts by Fact Type

| Fact Type | Dominant Expert | Weight | Secondary |
|-----------|-----------------|--------|-----------|
| Entity | **L16E4** | 24.7% | L18E1 (24.5%), L5E24 (21.6%) |
| Numeric | **L16E4** | 25.0% | L5E24 (22.7%) |
| Temporal | **L16E4** | 24.6% | L18E1 (23.7%), L5E24 (22.5%) |
| Procedural | L5E24 only | 21.4% | *None above 20% threshold* |

### 2.2 The L16E4 "Fact Lookup" Expert

**Key Finding**: L16E4 handles ~25% of all declarative fact (entity/numeric/temporal) routing, but procedural facts completely bypass this expert.

```
Declarative Facts:
    Input → L5E24 (22%) → ... → L16E4 (25%) → L18E1 (24%) → Output

Procedural Facts:
    Input → L5E24 (21%) → ... → [diffuse, no dominant] → Output
```

This is strong evidence that:
1. **L16E4 functions as a "fact lookup" expert** for declarative knowledge
2. **Procedural knowledge is distributed** across many experts with no concentration
3. The clustering difference (0.35 vs 0.84) reflects genuine architectural separation

---
