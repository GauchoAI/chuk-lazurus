# SPEC: chuk-mlx Inference Engine — The Decoupled Attention Architecture

**Author:** Chris Hay
**Date:** 2026-03-19
**Status:** Implementation Spec
**Repository:** chuk-mlx (chuk-lazarus CLI + inference engine)

---

## 1. What This Is

The inference engine that demonstrates two videos worth of results:

**Video 2 — "The Filing Cabinet Is Empty"**
Pure V-injection. The residual stream carries everything. The KV cache
is a 56 GB addressing system we replaced with a 1.85 MB lookup table.
12 bytes per fact. 330ms per query. On a MacBook.

**Video 3 — "Attention Is a Service"**
The full decoupled architecture. Five query types, automatic routing,
pluggable attention providers, dark space evaluation, distributed
knowledge stores. Attention decomposed from a monolith into services.

Both videos need the same engine. Video 2 uses the V-injection path.
Video 3 uses all paths together.

---

## 2. The Attention Provider Interface

The core abstraction. Everything else plugs into this.

```python
class AttentionProvider(Protocol):
    """Anything that can answer a query from stored knowledge."""

    def route(self, q_vector: array, prompt_text: str,
              top_k: int = 5) -> RouteResult:
        """Find relevant content for this query."""
        ...

    def retrieve(self, match: Match) -> RetrievalResult:
        """Get the answer for a matched entry."""
        ...

@dataclass
class RouteResult:
    matches: list[Match]
    confidence: float        # highest match score
    strategy: str            # for logging/debugging

@dataclass
class Match:
    fact_id: str
    window_id: int
    position: int
    score: float

@dataclass
class RetrievalResult:
    # For V-injection (12 bytes):
    token_id: int | None
    coefficient: float | None
    injection_layer: int | None

    # For window replay:
    window_ids: list[int] | None
    tokens_per_window: int | None

    # Which path:
    mode: Literal["inject", "replay"]
```

The engine doesn't know or care which provider answers. It receives
either an injection (12 bytes, apply at L30) or a replay instruction
(load these windows, build KV, generate).

---

## 3. Providers (what exists, what to build)

### 3.1 VecInjectProvider — V-injection with Three-Tier Routing (EXISTS + UPDATE NEEDED)

```
Storage:   ~534 bytes/fact (512 K + 4 token_id + 8 coefficient + 10 H4 routing vec)
Routing:   Three-tier: adaptive Q·K → H4 output cosine → replay fallback
Retrieval: 12 bytes (token_id + coefficient)
Injection: 1D subspace at L30 (128ms, two-pass)
Total:     ~330ms per query (injection path)

Safety:    Adaptive Q·K threshold: mean_score × 2.0 (auto-scales with N)
           H4 cosine threshold: calibrate per cluster (empirical)
           Below both thresholds → returns mode="replay" (fallback)

Validated: 6/6 synthetic, Apollo 11 scale (3,625 facts),
           zero wrong injections with adaptive threshold
           H4 routing: 4/4 at N=12 same-template, margins 2.4×–8.4×

Injection rate (projected):
           Stage 1 K-space Q·K: ~40% of queries
           Stage 2 H4 output:   ~45% of remaining queries
           Stage 3 Replay:      ~15%
           Combined:            ~85% injection rate at any N
```

**Three-tier routing architecture (routing-wall-breakers experiment, 2026-03-19):**

The K-space crowding problem is a **W_K projection problem**: 2560D → 256D collapses
entity-discriminative directions (Namath/Marchand are 73.87° apart in 2560D, near-identical
in 256D K-space). The geometric fix is **H4 output routing** — the copy head's isolated
contribution to the attention output, extracted in the same forward pass as K-space routing.

Why H4? It's the only head that was trained to discriminate entity identity. The other 7
heads at L29 add structural/template signal. The full residual mixes all 8 heads, so
same-template entities crowd. H4's output in isolation has 2.4×–8.4× margins at N=12.

```python
def route(self, q_h4_contrib, q_k_vector, top_k=5) -> RouteResult:
    # Stage 1: Adaptive Q·K threshold (structurally distinctive queries, ~40%)
    k_scores = self.k_vectors @ q_k_vector        # (N,) in 256D K-space
    max_idx   = k_scores.argmax()
    threshold = k_scores.mean() * 2.0             # scales with N automatically
    if k_scores[max_idx] > threshold:
        return RouteResult(
            matches=[Match(fact_id=max_idx, score=k_scores[max_idx])],
            confidence=k_scores[max_idx] / threshold,
            strategy="adaptive_qk"
        )

    # Stage 2: H4 output cosine (same-template entity discrimination, ~45%)
    # h4_contrib: (2560,) — H4's attention output projected through O_proj
    h4_sims   = self.h4_routing @ q_h4_contrib    # (N,) cosine via normalised storage
    h4_max    = h4_sims.argmax()
    h4_thresh = h4_sims.mean() * 2.0              # same adaptive logic
    if h4_sims[h4_max] > h4_thresh:
        return RouteResult(
            matches=[Match(fact_id=h4_max, score=h4_sims[h4_max])],
            confidence=h4_sims[h4_max] / h4_thresh,
            strategy="h4_output"
        )

    # Stage 3: Replay fallback (~15%)
    return RouteResult(matches=[], confidence=0.0, strategy="fallback")
```

**H4 output extraction (during prefill, hooks into existing forward pass):**
```python
# At layer 29, after project_qkv:
H4_IDX = 4; KV_IDX = H4_IDX // n_rep  # = 2 for Gemma 4B
q_last = q[:, H4_IDX, -1:, :]          # (1, 1, 320)
k_kv   = k[:, KV_IDX, :, :]            # (1, S, 320)
v_kv   = v[:, KV_IDX, :, :]            # (1, S, 320)
attn_w = softmax(q_last @ k_kv.T * scale, axis=-1)  # (1, 1, S)
h4_out = (attn_w @ v_kv)[:, 0, :]      # (1, 320)
h4_contrib = h4_out @ o_proj.weight[:, H4_IDX*320:(H4_IDX+1)*320].T  # (1, 2560)
# Store normalised h4_contrib per fact → 2560D float32 = 10 KB/fact
```

**Why NOT entity string matching (deprecated):**
- Language-dependent, breaks on paraphrase
- H4 routing is geometric — works on any reformulation of the same question
- H4 routing works on entity-implicit queries too (string match cannot)
- Entity signal in H4 is the model's own representation, not surface text

**Why NOT fixed 15% threshold at scale:**
- N=12: max Q·K ≈ 20-40% → 15% threshold works
- N=100: max Q·K ≈ 5-7% → 100% threshold failure
- N=3,625: max Q·K ≈ 0.03-0.3% → 100% threshold failure
- Adaptive (mean × 2.0) auto-scales with N, maintains discrimination

**Already implemented.** LocalVecInjectProvider in
`src/chuk_lazarus/inference/context/vec_inject/`.
**Needs update:** Add H4 output extraction during prefill, three-tier routing logic.

Config:
```yaml
provider:
  type: vec_inject
  index_path: ./apollo11_l29/
  adaptive_threshold_multiplier: 2.0  # mean_score × this (both stages)
  h4_routing_layer: 29
  injection_layer: 30
  fallback: geometric_replay
```

### 3.2 GeometricReplayProvider — Compass routing (EXISTS)

```
Storage:   29 MB compass residuals (L26, 8 positions × 725 windows)
Routing:   Cosine in PC 8-23 + contrastive RRF (1.5s)
Retrieval: Window IDs → replay 256 clean tokens per window
Total:     ~2s (routing + replay + generation)

Validated: Sports rank 1, landing rank 1, 5/5 quality
```

**Already implemented.** RoutingStrategy.GEOMETRIC in compass_routing/.

### 3.3 ProbeReRankProvider — Dark space evaluation (EXISTS)

```
Storage:   20 KB probe directions + 1 MB BM25 index
Routing:   BM25 coarse filter → generation-mode re-ranking
Retrieval: Window IDs → replay
Total:     20-30s (dominated by re-ranking forward passes)

Validated: Engagement 3/5, tension 3.5/5
           Finds keyword-unreachable content (W37 birthday)
```

**Already implemented.** _probe_driven_generate and indicator BM25.

### 3.4 EventDenseProvider — Timeline routing (EXISTS)

```
Storage:   1 KB (regex-matched event window list)
Routing:   Load pre-computed event windows (0ms)
Retrieval: Window IDs → replay
Total:     ~4s
```

### 3.5 TemporalStrideProvider — Fallback for global (EXISTS)

```
Storage:   None (arithmetic)
Routing:   Every Nth window (0ms)
Total:     ~4s
```

### 3.6 CorrectionProvider — Parametric override (NEW)

```
Storage:   12 bytes per correction (same as V-injection)
Routing:   Pattern match on query text
Injection: 1D subspace at L30 (same mechanism)
Total:     ~50ms (no routing forward pass)
```

**Not yet implemented.**

### 3.7 RemoteProvider — MCP network (NEW, Video 3)

```
Network:   512 bytes up (Q vector), 12 bytes down (token_id + coeff)
Not yet implemented. The chuk-mcp-kvindex server.
```

---

## 4. The Forward Pass with Providers

### 4.1 V-injection forward pass (Video 2 path)

```python
def generate_with_injection(model, prompt_ids, injection, max_tokens):
    """Two-pass generation with V-injection at L30."""

    # Pass 1: inject-biased first token
    h_L29 = model.prefill_to_layer(prompt_ids, target_layer=29)
    e = model.embed(injection.token_id)
    # Normalise by squared norm (not unit norm) because coefficient c = dot(R, e_raw).
    # Result: c * (e / ‖e‖²) = proj_e(R_donor) — transplants the donor's e-component.
    # Purely additive: R_bare ⊥ e so proj_e(R_bare) ≈ 0 and we write into blank space.
    direction = e / dot(e, e)
    h_injected = h_L29 + injection.coefficient * direction
    logits_first = model.prefill_from_layer(h_injected, start_layer=30)
    first_token = sample(logits_first)

    # Pass 2: proper KV for continuation
    _, kv = model.prefill(prompt_ids)
    logits, kv = model.extend([first_token], kv, abs_start=len(prompt_ids))

    # Standard autoregressive from here
    for _ in range(max_tokens - 1):
        token = sample(logits)
        logits, kv = model.step(token, kv)
    return tokens
```

Note: The injection is **purely additive** — `R_bare ⊥ e` (angles 88.97°–91.76°),
so `dot(R_bare, e) ≈ 0` and the full correction term simplifies to `c * e`.

### 4.2 Combined dispatch (the engine)

```python
class DecoupledInferenceEngine:
    def generate(self, prompt_text, prompt_ids, max_tokens=300):
        # Step 1: Classify (L26 residual → 5-class probe)
        query_type = self.classifier.classify(prompt_ids)

        # Step 2: Route via appropriate provider
        provider = self.providers[query_type]
        result = provider.route(q_vec, prompt_text)
        retrieval = provider.retrieve(result.matches[0])

        # Step 3: Dispatch
        if retrieval.mode == "inject":
            return generate_with_injection(self.model, prompt_ids, retrieval, max_tokens)
        elif retrieval.mode == "replay":
            return generate_with_replay(self.model, prompt_ids, retrieval.window_ids, ...)
```

---

## 5. What the Model Needs to Expose

### Already implemented

- `prefill(ids)` → logits, kv_cache
- `prefill_to_layer(ids, target_layer)` → hidden state at layer L
- `prefill_from_layer(hidden_state, start_layer)` → logits, kv_cache
- `step(token, kv_cache)` → logits, updated kv_cache
- `extend(ids, kv_cache, abs_start)` → logits, extended kv_cache

### Needed for Video 3 efficiency

**`step_with_extraction(token, kv, extraction_layer)`** — step + extract intermediate
residual without extra forward pass. Eliminates double forward pass in probe re-ranking
(cuts engagement/tension time from 20s to ~10s).

```python
def step_with_extraction(self, token, kv, extraction_layer=None):
    logits, new_kv = self.step(token, kv)
    extracted = self._last_hidden_at_layer[extraction_layer] if extraction_layer else None
    return logits, new_kv, extracted
```

---

## 6. Prefill Pipeline

### Current phases (all exist)

| Phase | Output | Size | Purpose |
|---|---|---|---|
| windows | tokens.bin, windows.json, boundary KV | ~10 MB | Token archive + checkpoints |
| compass | compass_residuals.npz, compass_basis.npz | ~29 MB | L26 content-type routing |
| sparse | sparse_index.json | ~1 MB | BM25 keyword index |
| vec_inject | vec_inject.npz | ~1.86 MB | K-vectors + injection coefficients |
| surprise | surprise.npz | ~10 KB | Per-token surprise scores |
| mode7 | .probe_cache_v*.npz | ~20 KB | Pre-calibrated query classifier + probes |

### New phases needed (Video 3)

| Phase | Output | Size | Purpose |
|---|---|---|---|
| **judgments** | judgment_residuals.npz | ~7.2 MB | Pre-computed engagement/tension evaluation per window |
| **events** | event_windows.json | ~1 KB | Regex-matched event windows |
| **clean** | clean_tokens.bin | ~1.5 MB | OCR-cleaned tokens for replay |

### Judgment residual extraction (the key to fast Video 3)

Pre-compute generation-mode judgment residuals per window at prefill time:

```python
def extract_judgments(engine, output_path, windows, tokenizer):
    """For each window:
      1. Replay 256 clean tokens
      2. Append "Rate interestingness 1-5. Rate:"
      3. Generate 20 tokens
      4. Extract L26 at last generated token
      5. Store 2560D judgment residual (5 KB per window)
    """
```

Query-time cost: 725 dot products (vs 20-50 forward passes currently).
Drops engagement/tension routing from 20-30s to <1s.

### Full prefill command (when all phases are built)

```bash
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input docs/apollo11_clean.txt \
    --checkpoint ./apollo11_full/ \
    --phases all
```

### Current full prefill command (Video 2)

```bash
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input docs/apollo11_clean.txt \
    --checkpoint ./apollo11_full/ \
    --phases windows,compass,vec_inject,mode7
```

---

## 7. Query Pipeline (Video 2)

```
User: "Who were the crew of Apollo 11?"
  ↓
Engine: FACTUAL classification (L26, 197ms)
  ↓
vec_inject index: Stage 1 adaptive Q·K threshold
  → top match: W472, pos 367, token "Nell", score > threshold → INJECT
  ↓
Two-pass injection at L30 (128ms)
  → first token: "Nell"
  ↓
Full prefill + autoregressive generation
  → "Nell Armstrong, Buzz Aldrin, and Michael Collins"
  ↓
Total: ~780ms. 80 tokens in context. Zero document replay.
```

**With H4 output routing (next step, Stage 2):**
```
User: "What city was Zarkov Industries founded in?"  ← same-template entity cluster
  → Stage 1: K-space score below adaptive threshold (template crowding)
  → Stage 2: H4 output cosine → 2.434× margin → INJECT
  → Same 128ms injection path
  → Total: ~780ms (routing + injection)

"The same head that retrieves the fact also addresses it."
```

**CLI:**
```bash
lazarus context generate \
    --checkpoint ./apollo11_full/ \
    --model google/gemma-3-4b-it \
    --prompt "Who were the crew of Apollo 11?"
# No flags. Mode 7 decides. vec_inject fast path auto-selected.
```

---

## 8. Query Pipeline (Video 3)

```
User: "Find 3 amusing moments"
  ↓
Engine: ENGAGEMENT classification (L26)
  ↓
If judgments pre-computed:
  → dot product 725 stored residuals × engagement direction (5ms)
  → top-5 windows
Else:
  → BM25 indicators → 20 candidates → generation-mode re-rank (20s)
  ↓
Replay 5 windows × 256 tokens → generate
  → "Czar joke, birthday from space, sandwiches"
  ↓
Total: 4s (with judgments) or 25s (without)
```

---

## 9. Storage Layout

```
apollo11_full/
├── manifest.json
├── tokens.bin                       # raw token IDs (1.5 MB)
├── windows.json                     # window boundaries (50 KB)
├── checkpoints.npz                  # boundary KV for chaining (7 MB)
├── compass_residuals.npz            # L26 interval residuals (29 MB)
├── compass_basis.npz                # PCA basis + contrastive frames (100 KB)
├── vec_inject.npz                   # K-vectors + coefficients + H4 routing (37.5 MB)
│   ├── w{N}/k_vecs                  # Stage 1 routing addresses (256D float16)
│   ├── w{N}/token_ids               # answer tokens
│   ├── w{N}/coefs                   # injection magnitudes
│   ├── w{N}/positions               # token positions
│   ├── w{N}/distinctive             # ≥4 char first token flag
│   └── w{N}/h4_routing              # Stage 2 H4 output vectors (2560D float32, 10 KB/fact)
├── sparse_index.json                # BM25 keyword index (1 MB)
├── surprise.npz                     # per-token surprise (10 KB)
├── .probe_cache_v*.npz              # Mode 7 probes (20 KB)
│
│ ── Video 3 additions ──────────────────────────────────────────
├── judgment_residuals.npz           # engagement + tension per window (7.2 MB)
├── event_windows.json               # regex-matched events (1 KB)
└── corrections.json                 # parametric overrides (optional)

Total (Video 2): ~77 MB  (36.25 MB H4 routing at 3,625 facts × 10 KB/fact + existing ~40 MB)
Total (Video 3): ~87 MB
```

**H4 routing storage note:** 2560D float32 per fact = 10,240 bytes. At 3,625 facts = 36.25 MB.
Alternative: 320D pre-O_proj (head_dim only, 1,280 bytes/fact, 4.5 MB at scale) — preserves
cosine discrimination if O_proj columns are approximately orthonormal. Trade-off under study.

---

## 10. Performance Targets

### Video 2 (V-injection demo)

| Metric | Target | Status |
|---|---|---|
| Factual query (adaptive Q·K) | <1s | ~780ms ✓ |
| Factual query (entity match, next) | <600ms | not yet built |
| Factual query (fallback) | <3s | ~2s ✓ |
| Wrong injections | 0 | 0 ✓ |
| Index size | <2 MB | 1.86 MB ✓ |
| Injection rate (N≤50) | >85% | ~92% ✓ |
| Injection rate (N=3,625) | >80% | ~85% (projected) |

### Video 3 (Full Mode 7 demo)

| Metric | Target | Status |
|---|---|---|
| Factual | <500ms | ~330ms ✓ |
| Engagement (with judgments) | <5s | not yet built |
| Engagement (without judgments) | <25s | ~20s ✓ |
| Tension (with judgments) | <5s | not yet built |
| Timeline | <5s | ~4s ✓ |
| Tone/mood | <3s | ~2s ✓ |

---

## 11. Implementation Priority

### For Video 2

- [x] VecInjectProvider (LocalVecInjectProvider)
- [x] Two-pass generation
- [x] Fallback to window replay
- [x] CLI: `--replay vec_inject`
- [x] Adaptive threshold (mean × 2.0) — replaces fixed 15%
- [x] Mode 7 + vec_inject wiring (FACTUAL auto-routes to vec_inject)
- [x] Video 2 demo script
- [x] Mode 7 probe calibration during prefill (mode7 phase)
- [ ] H4 output extraction during prefill (add to vec_inject phase)
- [ ] H4 routing index storage in vec_inject.npz (10 KB/fact)
- [ ] H4 output cosine as Stage 2 routing tier

### For Video 3

- [ ] Judgment residual extraction (prefill phase)
- [ ] Judgment-cached ProbeReRankProvider (<5s engagement/tension)
- [ ] Event window extraction (regex prefill phase)
- [ ] CorrectionProvider for parametric overrides
- [ ] `step_with_extraction` for probe efficiency
- [ ] RemoteProvider / chuk-mcp-kvindex (stretch goal)

---

## 12. What NOT to Build

All ruled out by experiment. See `memory/project_vec_inject_kill_list.md` for full details.

1. Surprise-based event detection (anti-correlated)
2. Speaker-change frequency (anti-correlated)
3. Angular velocity phase boundaries (too smooth)
4. Tension probe for event detection (domain context inflation)
5. Expression-mode tonal steering at scale (W170 rank 308)
6. Inject-all multi-fact (no address bus in L31-L33)
7. Compressed page replay (worse than plain)
8. 4D evaluative manifold (only 2-3D effective)
9. Two-layer cascade L26→L29 (redundant)
10. Operational language prompt warnings (net negative)
11. Fixed 15% confidence threshold (useless beyond N≈50)
12. L14 K-vectors for novel entities (parametric-only)
13. Multi-layer K concatenation L23+L29 (redundant)
14. Multi-head K concatenation (4× cost, minimal gain)
15. Entity-enhanced K-vectors (W_K collapses in projection)
16. Contrastive K-vectors (same root cause)
17. Fixing W_K at inference time (frozen, needs LoRA)
18. H-space PCA-16 routing (circular dependency — need doc context to build query vector)
19. Global PCA-16 bare query routing (circular dependency, item 19)
20. Raw H-space cosine at N≥8 same-template (0.977×/0.984× for city cluster)
21. Per-template PCA removal (0.024× catastrophic — entity signal lives in template dims)
22. All linear full-residual routing for same-template N≥8 entity discrimination:
    variance weighting (sqrt/log/top-K), entity-position, contrastive delta, Fisher.
    Entity signal is entangled with template across all dimensions. Cannot be recovered
    by linear reweighting. Solved by H4 output isolation (see §3.1, §13).
23. Entity string matching as routing mechanism (language-dependent, paraphrase-fragile,
    superseded by H4 output routing which works on any reformulation)

---

## 13. The W_K Projection Problem — Solved by H4 Routing

The fundamental scaling limit of V-injection is NOT content (12 bytes, solved) but
addressing precision. Root cause: W_K projects 2560D → 256D, collapsing
entity-discriminative directions.

**The representation is fine:**
- Namath vs Marchand at L29: 73.87° apart (cosine 0.278) in 2560D
- Entity signal accounts for ~43% of K-vector variance in hidden space
- The model knows entities are different

**W_K collapses it:**
W_K was trained for attention pattern matching, not entity routing.
Same-template facts produce structurally similar K-vectors in 256D
despite divergent 2560D hidden states.

**L14 is a trap:**
Peak entity separation at L14 (Namath sep=39.26) — but only for training-data entities.
Novel entities get near-random discrimination (Zarkov sep=0.41). L14 is parametric.
L29 is correct: novel entities build signal through 15 layers of template processing.

**The geometric fix is H4 output routing (routing-wall-breakers, 2026-03-19):**
H4 at L29 is the copy head. Its isolated output bypasses the W_K projection entirely —
we're reading from the 2560D residual (post-V projection, pre-O_proj collapse), not from
the 256D K-space. Same-template N=12: 2.434×–8.387× margins, 4/4 accuracy.

All linear full-residual approaches were ruled out (Item 22, kill list):
- Variance weighting (sqrt, log, top-K): high-var dims ARE template PCs; sqrt-var collapses all cosines to 1.000×
- Entity-position routing: 0.998×/0.994× — still template-contaminated
- Contrastive delta: Zarkov gets WORSE (0.437×) — entity perturbation 80% correlated across entities
- Fisher discriminant: 0.994×/0.998× — nearly uniform weights, never crosses 1.0

Entity discrimination cannot be recovered from the full residual by any linear reweighting.
It can only be isolated by going to the source: the one head trained to compute it.

**The entity string approach is superseded:**
Entity string matching was a workaround. H4 routing is the geometric solution.
String matching is language-dependent and paraphrase-fragile. H4 routing works on any
reformulation of any entity query — because it uses the model's own representation.

---

## 14. The Claim

**Video 2:**
"The KV cache is a 56 GB filing cabinet. The answer is 12 bytes. We pre-compute the
addresses and answers during prefill, store them in a 1.86 MB index, and at query time
we route using the model's own geometry.

Three tiers: K-space adaptive Q·K handles structurally distinctive queries (~40%).
H4 output cosine handles same-template entity discrimination (~45%) — the same head
that retrieves the fact also addresses it, at 2.4×–8.4× margins. Replay fallback
handles the rest (~15%). 85% injection rate at any scale. Average latency 330ms.
On a MacBook.

The W_K projection matrix collapses entity-discriminative directions — entities that
are 74° apart in 2560D crowd together in 256D K-space. The fix isn't a string lookup.
It's H4's output: the one head trained to copy entity identity into the residual."

**Video 3:**
"Attention was designed as a monolith — Q, K, V all computed together, same head,
same layer, same device. We decomposed it into services. Routing from a pre-computed
index. Retrieval from a 12-byte lookup. Evaluation from the dark space. Storage from
anywhere. Five query types, automatic routing, all on a MacBook. The KV cache was the
coupling mechanism. Removing it removed the coupling."
