# SPEC: Prefill — What Exists, What's Missing, What to Build

**Author:** Chris Hay
**Date:** 2026-03-20
**Status:** Implementation Spec (aligned with codebase)
**Repository:** chuk-mlx (chuk-lazarus)

---

## 1. Current State of the Code

The prefill pipeline exists and runs. It processes documents into
checkpoint libraries with multiple extraction phases. But the
current architecture grew organically from the original unlimited-
context engine. It needs to be refactored into the clean knowledge
store architecture.

### What exists (working)

- `context_prefill_cmd` — CLI entry point, window-by-window processing
- `UnlimitedContextEngine.process()` — window processing with KV archival
- `SparseIndexEngine` — extends unlimited engine with inline sparse extraction
- `save_library()` — orchestrator that runs all extraction phases
- `extract_vec_inject_index()` — K-vectors, coefficients, H4 outputs
- `extract_kv_route_index()` — K-vectors only (lighter routing index)
- `calibrate_compass()` — L26 PCA basis for content routing
- `extract_surprise()` — per-token perplexity scoring
- `extract_sparse()` — surprise-aware keyword extraction
- `extract_darkspace()` — whitened frame bank projections
- `extract_interval_residuals()` — 8 evenly-spaced residuals per window
- `extract_pages()` — pre-RoPE KV pages for instant injection
- `calibrate_mode7_probes()` — query classifier + engagement/tension
- Resume support via `restore_engine()`
- Export mode (skip KV checkpoints, portable index)
- Incremental saves every 5 minutes

### What's missing

1. **No residual chaining across windows during extraction.**
   Every extraction pass calls `kv_gen.prefill_to_layer(w_ids, target_layer=L)`
   independently per window. No `initial_residual`. Each window
   is processed in isolation — the extraction doesn't see cumulative
   document context.

2. **Coefficient computed at wrong layer.**
   `extract_vec_inject_index()` computes `dot(h[pos], embed(token))`
   where h is at the retrieval layer (L29). The spec says the
   coefficient should be at the injection layer (L30). The current
   code uses `h` from `prefill_to_layer(target_layer=retrieval_layer)`,
   not `continue_from_layer(retrieval_layer+1, injection_layer)`.

3. **No final residual saved.**
   The extraction passes don't produce a document-level Markov state.
   There's no `final_residual` in the output. Query-time has no
   document context to start from.

4. **No generation continuation.**
   After generating a response, nothing gets extracted. The store
   doesn't grow. The residual doesn't update.

5. **Not model-agnostic in extraction.**
   `ArchitectureConfig.from_model_config()` exists and resolves
   retrieval_layer/query_head/kv_head from the model config. But
   the extraction code has some Gemma-specific assumptions in the
   H4 output extraction (direct access to `o_proj.weight` slicing).

---

## 2. Architecture Config (Exists)

```python
# From arch_config.py — already implemented
@dataclass
class ArchitectureConfig:
    retrieval_layer: int
    query_head: int
    injection_layer: int   # retrieval_layer + 1
    kv_head: int = -1
    head_dim: int = 0
    hidden_dim: int = 0
    k_dim: int = 0
    threshold_multiplier: float = 2.0

    @classmethod
    def from_model_config(cls, config) -> "ArchitectureConfig":
        """Resolve from model config. Raises if not calibrated."""
        ...

    def with_geometry(self, *, kv_head, head_dim, hidden_dim):
        """Return a copy with geometry fields populated from the model."""
        ...

    def to_dict(self) -> dict: ...
```

The config is embedded in the manifest at save time.

---

## 3. The Refactored Prefill Pipeline

### 3.1 Phase 1: Window Processing (exists, needs residual chaining)

The current `UnlimitedContextEngine.process(chunk)` handles:
- Tokenisation
- Forward pass
- KV archival (boundary KV at last position)
- Window metadata

**Change needed:** After each window's forward pass, save the
boundary residual. Thread it into the next window's forward pass.

### 3.2 `prefill_to_layer` with `initial_residual` (needs implementation)

The current `kv_gen.prefill_to_layer(ids, target_layer)` doesn't
accept an `initial_residual` parameter. This is the core change.

**The boundary residual is a position.** Prepended at position 0.
Attention sees it alongside the window's tokens. Causal mask
allows all positions to attend to it. This is the standard
transformer mechanism — no special injection logic.

### 3.3 Phase 2: Vec-Inject Extraction (exists, needs fixes)

**Issue 1: No residual chaining.** Each window is processed
independently.

**Fix:** Chain during the main prefill loop and extract inline.
One forward pass per window, not two.

**Issue 2: Coefficient at wrong layer.** Currently uses h at L29.
However the current demos work with L29 coefficient (100% in
`02_the_injection.py`). **Decision:** Keep L29 coefficient for now.

### 3.4 Phase 2b: H4 Output Extraction (exists, model-specific)

The current H4 extraction accesses `o_proj.weight` directly.
For model-agnostic: add `project_head_output()` to the layer
protocol. Future work.

### 3.5 Phase 3: Final Residual (missing)

After all windows are processed, save the boundary residual as
the document's Markov state.

**Storage:** hidden_dim × 4 bytes. Gemma 4B: 10 KB. Gemma 1B: 5 KB.

---

## 4. Inline Extraction (the clean path)

Instead of separate extraction passes (each re-prefilling every
window), extract everything during the main prefill loop.

One forward pass per window (to retrieval_layer). Everything
extracted from that single pass:
- K-vectors (from K projection at retrieval_layer)
- Coefficients (from dot(h, embed) at retrieval_layer)
- H4 output vectors (from head-isolated attention at retrieval_layer)
- Boundary residual (last position)

---

## 5. Generation Continuation

After generating a response, extract K vectors + coefficients
from the generated tokens and append to the provider. Update
the residual. The store grows with the conversation.

This is implemented in `_extract_and_append()` in the vec_inject
generate mode.

---

## 6. Implementation Order

### Phase 1: Residual Chaining (unlocks final residual)

1. Add `initial_residual` parameter to `prefill_to_layer`
2. Thread `boundary_residual` through the prefill loop
3. Save `final_residual.npy` in `save_library()`
4. Load at query time in the generate command

### Phase 2: Inline Extraction (replaces separate passes)

1. Move vec_inject extraction into the window processing loop
2. Extract K-vectors, coefficients, H4 from the chained forward pass
3. Remove the separate `extract_vec_inject_index()` pass

### Phase 3: Generation Continuation

1. After generation: extract entries, append to provider
2. Update residual
3. Save updated store

### Phase 4: Model-Agnostic H4 Extraction

1. Add `project_head_output()` to TransformerLayer protocol
2. Implement for Gemma, Llama, Qwen backends
3. Remove direct `o_proj.weight` access

---

## 7. What Doesn't Change

- The vec_inject.npz format (same keys, same dtypes)
- The sparse_index.json format
- The CLI interface
- The export mode flag
- The resume mechanism
- The 12-byte injection primitive (`vec_inject()`)
- The two-pass generate architecture

---

## 8. Cost Impact

| Step | Current | After refactoring |
|------|---------|-------------------|
| Prefill (per window) | ~12s forward + ~6s extraction | ~12s forward + ~3s inline extraction |
| Total prefill (Apollo 11) | ~14 min | ~10 min (one fewer pass) |
| Peak memory | 143 MB | 143 MB + 5 KB residual |
| Output size | ~15 MB | ~15 MB + 10 KB final residual |
| Query time | ~800ms | ~800ms |
| Generation overhead | 0 | ~350ms per turn (extraction) |
| Store growth | 0 | ~55 KB per turn |
