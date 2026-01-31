## Part 3: Context vs Parametric Memory

### 3.1 Where Does Context Override Begin?

We measured hidden state divergence between prompts with and without conflicting context.

| Layer | Context Weight | Interpretation |
|-------|----------------|----------------|
| L4 | **91.4%** | Context already dominant |
| L8 | 378% | Strong amplification |
| L13 | 1398% | Peak at vocab classifier |
| L20 | 6575% | Maximum divergence |

**Finding**: Context begins overriding parametric memory at **L4**, with the signal amplifying through later layers.

### 3.2 Override Success by Fact Type

We tested conflict scenarios where context contradicts parametric knowledge.

#### Declarative Facts

| Query | Parametric | With Context | Override? |
|-------|------------|--------------|-----------|
| "Capital of France?" | Paris | Paris | No |
| "Water boils at? (here)" | 100°C | **85°C** | **Yes** |
| "Speed of light?" | 299,792,458 | 299,792,458 | No |

#### Procedural Facts

| Query | Parametric | With Context | Override? |
|-------|------------|--------------|-----------|
| "Convert 20°C (this system)" | 68°F (standard) | 68°F | No |
| "2+3*4 (this notation)" | 14 (PEMDAS) | 14 | No |
| "Tie bowline (this method)" | Traditional | Traditional | No |

### 3.3 The Override Asymmetry

| Fact Type | Override Success |
|-----------|------------------|
| Declarative | **1/3 (33%)** |
| Procedural | **0/3 (0%)** |

**Critical Finding**: Procedural facts completely resist context override.

This makes architectural sense:
- Declarative facts route through concentrated expert (L16E4) → single intervention point
- Procedural facts are diffusely encoded → no single point to override

**Implication**: External fact injection via context will work for declarative knowledge but requires a different strategy for procedural knowledge.

---
