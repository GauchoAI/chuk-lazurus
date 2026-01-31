## Part 8: Memory Bank Proof of Concept

### 8.1 The Idea

A language model has parametric memory - facts baked into its weights during training. But these facts can be wrong, outdated, or hallucinated. What if we give the model an external memory bank that it treats as authoritative?

The concept is simple: prepend a structured `[Memory Bank]` block to the prompt. The model reads from it like a person checking their notes instead of going from memory.

```
[Memory Bank]
- France | capital | Lyon
- Japan | capital | Osaka
- Gold | symbol | Gd
[End Memory Bank]

Using the memory bank above, answer: What is the capital of France?
Answer:
```

### 8.2 What We Tested

Three progressively harder tests:

| Test | Question |
|------|----------|
| **Correct lookup** | Does the model read from the memory bank? |
| **Counterfactual override** | Does the memory bank beat parametric memory? |
| **Multi-fact selection** | Can the model pick the right fact from several entries? |

### 8.3 Results

```
SUMMARY
══════════════════════════════════════════
  Correct fact lookup:     4/4
  Counterfactual override: 3/3
  Multi-fact selection:    3/3
  ────────────────────────────────────────
  Total:                   10/10 (100%)
══════════════════════════════════════════
```

### 8.4 Correct Fact Lookup (4/4)

The model reads single-entry memory banks and returns the correct value.

| Query | Memory Bank Entry | Model Output |
|-------|-------------------|--------------|
| Capital of France? | France \| capital \| Paris | **Paris** |
| Capital of Australia? | Australia \| capital \| Canberra | **Canberra** |
| Symbol for gold? | Gold \| symbol \| Au | **Au** |
| CEO of Microsoft? | Microsoft \| ceo \| Satya Nadella | **Satya Nadella** |

### 8.5 Counterfactual Override (3/3)

The memory bank overrides what the model "knows" from training. This is the critical test.

| Query | Parametric (from weights) | Memory Bank | Model Output |
|-------|---------------------------|-------------|--------------|
| Capital of France? | Paris | France \| capital \| **Lyon** | **Lyon** |
| Capital of Japan? | Tokyo | Japan \| capital \| **Osaka** | **Osaka** |
| Symbol for gold? | Au | Gold \| symbol \| **Gd** | **Gd** |

The model confidently generates "Au" for gold's chemical symbol without the memory bank. But when the memory bank says "Gd", the very first token it generates is "Gd". The memory bank takes precedence over parametric memory.

### 8.6 Multi-Fact Selection (3/3)

A memory bank with five entries, mixing correct and counterfactual data:

```
[Memory Bank]
- France | capital | Lyon          ← counterfactual
- Japan | capital | Osaka          ← counterfactual
- Germany | capital | Berlin       ← correct
- Gold | symbol | Gd               ← counterfactual
- Silver | symbol | Ag             ← correct
[End Memory Bank]
```

| Query | Expected (from bank) | Model Output |
|-------|---------------------|--------------|
| Capital of France? | Lyon | **Lyon** |
| Capital of Germany? | Berlin | **Berlin** |
| Symbol for silver? | Ag | **Ag** |

The model correctly selects the right entry from the bank for each query. It returns "Lyon" for France (counterfactual) and "Berlin" for Germany (correct) - it doesn't confuse entries or default to parametric memory.

### 8.7 Why This Works

The `[Memory Bank]` format succeeds where narrative context failed:

| Approach | Format | Override Rate |
|----------|--------|---------------|
| Narrative | "Due to recent changes, the capital has been moved to Lyon" | 33% |
| Memory Bank | `[Memory Bank]\n- France \| capital \| Lyon\n[End Memory Bank]` | **100%** |

Three properties make the memory bank effective:

1. **Structured delimiters** - `[Memory Bank]` / `[End Memory Bank]` create a clear boundary between external facts and the query
2. **Tabular format** - `entity | relation | value` is unambiguous, no room for the model to reinterpret
3. **Explicit instruction** - "Using the memory bank above" tells the model where to look

This is not prompt engineering tricks. It's an architectural pattern: give the model a structured external memory and an instruction to prefer it over its weights.

### 8.8 Implications

| Property | Parametric Memory (weights) | Memory Bank (external) |
|----------|---------------------------|----------------------|
| Accuracy | Model-dependent | **Deterministic** |
| Updatable | Requires retraining | **Edit the bank** |
| Verifiable | No citation | **Source attached** |
| Hallucination risk | Yes | **None (for covered facts)** |
| Latency | Zero | Bank lookup time |
| Coverage | Unbounded | Bank-dependent |

The memory bank gives you a clean separation: the model handles language and reasoning, the bank handles facts. Each does what it's good at.

---
