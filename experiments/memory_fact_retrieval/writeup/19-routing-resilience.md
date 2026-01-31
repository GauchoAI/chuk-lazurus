## Part 18: Routing Resilience — Why Photosynthesis Survived

### 18.1 The Anomaly

In Part 16, the photosynthesis prompt survived alternating routing almost perfectly (0.04 repetition vs 0.52 average). Why? Four hypotheses:

- **H1**: Technical/structured topics are more resilient than creative/open ones
- **H2**: Prompts that strongly constrain the output space survive better
- **H3**: Domain-specific vocabulary anchors generation against drift
- **H4**: Prompt length provides more context to stabilize generation

### 18.2 Method

35 prompts across 7 categories, each tested under 3 conditions:
- **Normal**: Full model (baseline)
- **Skip**: Alternating layer skipping (MoE at even layers, skip at odd — Part 17 style)
- **Fixed**: Alternating fixed routing (learned at even layers, fixed at odd — Part 16 style)

60 tokens generated per prompt (longer than prior experiments to measure sustained fluency).

Categories: `technical` (5), `factual_constrained` (5), `factual_open` (5), `creative` (5), `conversational` (5), `code_like` (5), `structured_list` (5).

### 18.3 Results by Category

| Category | Normal Rep | Skip Rep | Fixed Rep | Skip/Normal | Fixed/Normal |
|----------|-----------|---------|----------|-------------|-------------|
| **code_like** | 0.147 | **0.227** | **0.112** | 1.5x | 0.8x |
| **technical** | 0.013 | **0.365** | 0.221 | 29.2x | 17.7x |
| conversational | 0.016 | 0.536 | 0.557 | 33.5x | 34.8x |
| factual_open | 0.407 | 0.562 | 0.357 | 1.4x | 0.9x |
| structured_list | 0.299 | 0.567 | 0.331 | 1.9x | 1.1x |
| factual_constrained | 0.298 | 0.654 | 0.674 | 2.2x | 2.3x |
| **creative** | 0.116 | **0.769** | **0.794** | 6.6x | 6.8x |

### 18.4 Resilience Ranking

```
1. code_like:            #########............................... 0.227
2. technical:            ##############.......................... 0.365
3. conversational:       #####################................... 0.536
4. factual_open:         ######################.................. 0.562
5. structured_list:      ######################.................. 0.567
6. factual_constrained:  ##########################.............. 0.654
7. creative:             ##############################.......... 0.769
```

**Code is 3.4x more resilient than creative writing** (0.227 vs 0.769).

### 18.5 Most Resilient Individual Prompts

| Rep | Category | Prompt | Output (first 60 chars) |
|-----|----------|--------|------------------------|
| 0.000 | conversational | "Cats and dogs are different because" | they have different shapes. This sentence is a bit confusin |
| 0.000 | code_like | "class Node: def __init__(self, value):" | self.value = value self. |
| 0.000 | code_like | "for i in range(10):" | print("i", i) print("j", j) print("k", |
| 0.048 | technical | "The TCP/IP protocol stack consists of" | the following components: 1. TCP/IP Protocol: This is t |
| 0.111 | code_like | "def fibonacci(n):" | if n <= 0: return 0 if n == 1: |
| 0.161 | technical | "Mitochondria are often called..." | they produce energy for the cell. They are also called the |

### 18.6 Least Resilient Individual Prompts

| Rep | Category | Prompt | Output (first 60 chars) |
|-----|----------|--------|------------------------|
| 0.914 | creative | "The old lighthouse keeper had one secret:" | he had a secret that he had a secret that he had a secret th |
| 0.944 | structured_list | "Steps to make a PB sandwich: 1." | 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. |
| 0.828 | creative | "Once upon a time there was a" | small, but very beautiful, little, little, little, little |
| 0.826 | code_like | "import numpy as np; x = np.array..." | print(x); print('x'); print('x'); print('x') |
| 0.808 | conversational | "My favorite thing about summer is" | that it is a good time for the summer. It is a good time for |

### 18.7 Vocab Diversity Under Disruption

| Category | Normal | Skip | Retention |
|----------|--------|------|-----------|
| **code_like** | 0.710 | 0.604 | **85%** |
| factual_open | 0.497 | 0.337 | 68% |
| technical | 0.670 | 0.436 | 65% |
| structured_list | 0.545 | 0.279 | 51% |
| factual_constrained | 0.524 | 0.260 | 50% |
| conversational | 0.719 | 0.344 | 48% |
| **creative** | 0.630 | 0.171 | **27%** |

Code retains 85% of its vocabulary diversity; creative writing collapses to 27%. The output space constraint of programming syntax protects against repetition.

### 18.8 Interpretation

**H1 confirmed: Technical prompts are more resilient, but code is even better.**

The resilience ranking reveals a clear pattern: **output space constraint is the dominant factor**. The more the prompt constrains what can come next, the less the model depends on correct MoE routing:

1. **Code** (most constrained): Syntax rules, keyword sequences, and indentation patterns are enforced by attention patterns alone. Code has a small, predictable vocabulary. Even with disrupted MoE, the model produces syntactically valid Python.

2. **Technical** (moderately constrained): Domain-specific terminology (TCP/IP, nucleophilic, chloroplast) limits the plausible vocabulary. Photosynthesis survived because "chlorophyll," "light energy," "glucose" are nearly deterministic given the prompt.

3. **Creative** (least constrained): Any word could follow. With no syntactic or domain constraints, the model must rely entirely on MoE computation to choose among millions of plausible continuations. When MoE is disrupted, it falls into the lowest-energy attractor: repetition.

**Why this matters for compression:**
- **Code generation models** may be compressible — syntax constraints substitute for MoE computation
- **Creative/conversational models** are not compressible via routing simplification
- The "compressibility" of a model depends on the **task distribution**, not just the architecture
- A model serving structured queries (SQL, API calls, templates) could tolerate routing degradation that would destroy a chatbot

**The photosynthesis exception explained:** It's not special about photosynthesis — it's special about **domain-constrained technical content**. TCP/IP (0.048) and mitochondria (0.161) are equally resilient. The common factor: prompts where the next word is strongly predicted by domain vocabulary, not by general-purpose reasoning.

---
