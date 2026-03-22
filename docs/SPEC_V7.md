# SPEC: Knowledge Store — Final Architecture v7

**Author:** Chris Hay
**Date:** 2026-03-21
**Status:** Implementation Spec (all experiments complete)
**Repository:** chuk-mlx (chuk-lazarus)

---

## 1. What the Experiments Proved

Seven rounds of experiments. Each answered a question. Together
they define the architecture.

| Experiment | Finding |
|------------|---------|
| Markov property (Video 1) | Residual IS the complete state. KL=0.0 on swap. |
| Two-circuit discovery | L26 FFN = parametric. L29 H4 = in-context. Zero shared heads. |
| 12-byte injection | 100% P(target) for novel entities. Direct residual write at L30. |
| Token frequency sweep | H4 rare, H5 common, H3 ultra-common. Three copy services. |
| Synthetic KV injection | V-projection lossy in isolation. 38% attention, 0.06% P(target). |
| Attention-routed injection | K-space routes correctly. 4.5x discrimination. 100% P(target). |
| Apollo diagnosis | Interval sampling captures punctuation. Chat template required. |
| K-norm sampling | 12.4x better content capture. 87.5% content tokens. |
| Parametric override | Model confabulates well-known topics even with correct context. |

### The Three-Problem Stack

| Problem | Status | Fix |
|---------|--------|-----|
| Sampling captures punctuation | **SOLVED** | K-norm sampling (top-N by norm) |
| RoPE positional bias | **SOLVED** | Store format equalises positions |
| Parametric override on known topics | **BY DESIGN** | Not a bug — two circuits serve different populations |

### What Injection Can and Cannot Do

**CAN:** Add facts the model doesn't know parametrically. Novel
entities (Voltara), specific transcript details (exact timestamps,
radio callsigns, technical readouts, informal crew conversations),
document-specific content that isn't in training data.

**CANNOT:** Override facts the model already "knows." The parametric
circuit (L26 FFN) and the copy circuit (L29 H4) are separate.
Injection writes into the copy circuit. It doesn't suppress the
parametric circuit. For well-known topics (Apollo 11 + baseball),
the model's parametric association wins.

**This is correct behaviour.** The two circuits exist because the
model needs both: fast recall of trained knowledge AND the ability
to read new facts from context. Injection replaces the context-
reading mechanism. It doesn't replace the knowledge mechanism.

---

## 2. The Architecture

```
Query (chat template) + final_residual (document context)
    |
    v
L0 -> L28: normal forward pass
    |      parametric circuit builds through these layers
    |      residual navigates toward the answer
    |
    v
L29: attention-routed selection
    |  Q from query . K from stored entries
    |  softmax selects the best content entry
    |  4/8 heads route correctly (equalized positions)
    |
    v
L30: 12-byte injection
    |  h += coefficient x (embed(token_id) / ||embed||^2)
    |  direct residual write — bypasses V-projection
    |
    v
L31 -> L33: normal forward pass
    |        injection + parametric merge in the residual
    |
    v
Generate
```

Three knowledge paths operate simultaneously:

| Path | Source | Mechanism | Serves |
|------|--------|-----------|--------|
| Parametric | L26 FFN weights | Always active, free | Known facts (Paris, Armstrong) |
| Document context | Final residual (10 KB) | Biases generation toward document | Topic, tone, relevance |
| In-context facts | KV index + injection | L29 Q.K routing + L30 12-byte write | Novel facts only |

The parametric circuit and the copy circuit share zero heads.
They don't compete — they serve different populations of facts.
For novel entities: injection fills the vacuum. For known entities:
parametric handles it. The final residual biases both toward the
document.

---

## 3. K-Norm Sampling

### Why

Interval sampling (every Nth position) captures punctuation and
whitespace. Content tokens (nouns, entities, key phrases) have
K-vector norms ~5.2x higher than punctuation at L29. The K-norm
IS the content signal.

### How

```python
def sample_by_k_norm(
    backend, h, offset, chunk_length, n_samples, config,
):
    layer = backend.get_layer(config.copy_layer)
    primary = config.copy_heads[0]

    h_window = h[:, offset:offset + chunk_length, :]
    x = layer.pre_attn_norm(h_window)
    _q, k, _v = layer.project_qkv(x, 1, chunk_length, offset=0)
    k_head = k[:, primary.kv_head, :, :]

    norms = sqrt(sum(k_head[0] ** 2, axis=-1))
    eval(norms)

    valid = chunk_length - 1
    indices = argsort(norms[:valid], descending=True)
    selected = sorted(indices[:n_samples].tolist())

    return selected
```

### Validation

| Metric | Interval | K-norm | Improvement |
|--------|----------|--------|-------------|
| Content tokens | 68% | 87.5% | 1.3x |
| Mean H4 attention | 0.000251 | 0.003120 | 12.4x |
| Content/punct ratio | — | 5.2x per position | — |

---

## 4. Chat Template Requirement

The chat template activates the retrieval circuit at L29. Without
it, Zarkov drops from rank #1 to rank #925 among 1,473 entries.

All queries must be chat-template wrapped.

---

## 5. Store Format

### Per entry: 518 bytes

| Field | Size | Purpose |
|-------|-----:|---------|
| K-vector | 256 x 2 = 512 B | Routing (Q.K attention) |
| token_id | 2 B | Injection (answer token) |
| coefficient | 4 B | Injection (magnitude) |

### Directory layout

```
knowledge_store/
+-- manifest.json
+-- kv_index.npz         # K-vectors + injection data
+-- final_residual.npy   # document Markov state
+-- tokens.bin           # token archive (replay fallback)
+-- windows.json         # window boundaries
```

### Storage budget (512-token windows, 32 samples/window, K-norm)

| Document | Windows | Entries | Store | Total |
|----------|---------|---------|-------|-------|
| 100K tokens | 195 | 6,240 | 3.2 MB | ~4.6 MB |
| 370K tokens | 724 | 23,168 | 11.7 MB | ~13.2 MB |
| 1M tokens | 1,953 | 62,496 | 31.6 MB | ~33 MB |
| 10M tokens | 19,531 | 625,000 | 316 MB | ~318 MB |

vs. full KV cache for 370K tokens: 56 GB. **Compression: 4,200x.**

---

## 6. Cost Model

### Prefill (512-token windows, K-norm sampling)

| Step | Per window | 724 windows (Apollo 11) |
|------|-----------|------------------------|
| Forward L0 -> L29 | ~0.8s | ~10 min |
| K-norm computation | ~0.05s | ~36s |
| Entry extraction | ~0.1s | ~72s |
| Chain residual | <1ms | <1s |
| **Total** | **~1.0s** | **~12 min** |

### Query

| Step | Time |
|------|------|
| Load store | ~30ms |
| L0 -> L28 prefill | ~80ms |
| L29 Q.K routing | ~10ms |
| L29 normal forward | ~3ms |
| L30 injection | <1ms |
| L31 -> L33 | ~12ms |
| Generate (50 tokens) | ~500ms |
| **Total** | **~635ms** |

---

## 7. The Progression

```
v1  Full KV cache                      56 GB      brute force
v2  12-byte injection + external routing 15.5 MB   routing bottleneck
v3  Synthetic KV (V-delivery)          1.4 MB     FAILED — V lossy
v4  Attention-routed injection         2.2 MB     mechanism validated
v5  Apollo diagnosis                   —          sampling + style gap
v6  K-norm sampling + chat template    13.2 MB    content-aware store
v7  Three-problem resolution           13.2 MB    parametric override understood
```

The final architecture isn't a compromise. It's the correct
decomposition of two circuits that serve different populations.
The parametric circuit handles what the model knows. The injection
circuit handles what it doesn't. The final residual carries the
document context that biases both.

**13 MB. Three knowledge paths. No context window.**

---

## 8. CLI

```bash
# Build
lazarus knowledge build \
    --model google/gemma-3-4b-it \
    --input docs/apollo11_clean.txt \
    --output ./apollo11_store/ \
    --window-size 512 \
    --samples-per-window 32

# Query (single)
lazarus knowledge query \
    --model google/gemma-3-4b-it \
    --store ./apollo11_store/ \
    --prompt "What were the O2 flow readings before EVA prep?"

# Chat (multi-turn, store grows)
lazarus knowledge chat \
    --model google/gemma-3-4b-it \
    --store ./apollo11_store/
```

---

## 9. What's Next

### For Video 2 (record now)
- Build 512-window K-norm store for Apollo 11
- Select demo queries targeting novel transcript facts
- Run the demos: nav map, map swap, four passes, injection, Apollo queries
- The story: three knowledge paths, 13 MB, no context window

### For Video 3 (after Video 2 ships)
- 10M token scale demo (multiple documents)
- "There Is No Context Limit"

### For Video 4 (research)
- Multi-head routing consensus
- Parametric override investigation (steering vectors?)
- The full copy circuit map: "Attention Is a Service"
