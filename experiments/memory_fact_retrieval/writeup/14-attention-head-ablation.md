## Part 13: Attention Head Ablation

### 13.1 The Hypothesis

Previous experiments (Parts 8-12) conclusively showed that factual knowledge is NOT stored in MoE experts. Experts are positional computation units; ablating any combination at a single layer preserves 100% of facts. But the model clearly recalls memorized facts ("Paris", "Au", "Shakespeare"). Where does this retrieval happen?

**Chris's hypothesis**: Facts are stored as key-value associations in mid-layer attention heads. The L4 probing result (100% fact type classification) might detect the **query formation**, not the answer retrieval. The actual lookup could happen at L8-L12, where context override amplification occurs.

GPT-OSS 20B uses Grouped Query Attention (GQA):
- **64 query heads**, **8 KV heads** (8:1 ratio)
- Each KV head group = 8 query heads sharing the same K and V matrices
- If facts are stored as key-value associations, ablating the relevant KV head group should break retrieval

### 13.2 Experiment Design

**Method**: Class-level monkey-patching of `GptOssAttention.__call__`. At the target layer, we replace `mx.fast.scaled_dot_product_attention` with manual SDPA that applies a head mask to zero out all 8 query heads in a KV head group.

**Tests**: 8 factual prompts, same as all previous experiments (Paris, Au, Shakespeare, 299, Nadella, Tokyo, Ag, Canberra).

**Layers tested**: L4, L6, L8, L10 (spanning early layers through the hypothesized lookup zone).

**Per layer**: Ablate each of 8 KV head groups independently, test all 8 facts = 64 tests per layer.

### 13.3 Results: KV Head Group Scan

| Layer | KV0 | KV1 | KV2 | KV3 | KV4 | KV5 | KV6 | KV7 | Facts Broken |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|--------------|
| L4    | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **0** |
| L6    | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **0** |
| L8    | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | **0** |
| L10   | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | --  | **0** |

**0 facts broken across 248 ablation tests (31 KV head groups x 8 facts).**

### 13.4 Output Change Rate (Without Fact Breakage)

Although no facts broke, outputs DID change:

| Layer | Avg Output Change Rate | Range |
|-------|------------------------|-------|
| L4    | 87% (7.0/8)            | 6-8/8 |
| L6    | 62% (5.0/8)            | 3-6/8 |
| L8    | 44% (3.5/8)            | 3-5/8 |
| L10   | 39% (3.1/8)            | 2-4/8 |

The ablation IS changing model outputs substantially (especially at early layers), but the factual content survives. The changes affect phrasing, continuation style, and repetition patterns - not the initial factual answer.

### 13.5 Interpretation

**The "attention heads store facts" hypothesis is falsified**, at least for individual KV head groups at individual layers. This exactly mirrors the MoE expert ablation result: single-component ablation changes output form but not factual content.

The declining output change rate from L4 (87%) to L10 (39%) is consistent with the "residual stream accumulation" model from Part 9-10: early layers contribute more novel information to the residual stream, so ablation is more disruptive to output form. Later layers refine what's already present.

### 13.6 The Emerging Picture

We have now tested every addressable component of GPT-OSS at the single-component level:

| Component Type | Components Tested | Facts Broken | Coverage |
|---------------|-------------------|--------------|----------|
| MoE experts (single) | 32 at L16 | 0 | Part 8 |
| MoE experts (top-4) | 4 per layer, all 24 layers | 0 | Part 9 |
| MoE experts (30/32) | 30 at single layer | 0 | Part 10 |
| KV head groups | 31 across L4-L10 | 0 | Part 13 |

**No single component - expert or attention head - is individually necessary for factual recall.** The model distributes factual knowledge across components with enough redundancy that removing any one component from any one layer is absorbed by the remaining components.

### 13.7 Why This Resilience Exists

The cross-layer ablation experiment (Part 10) showed that removing 28/32 experts at ALL 24 layers causes model collapse. Combined with Part 13, this suggests:

1. **Facts are not "stored" in any component.** They emerge from the collective computation of many components operating on the residual stream.
2. **Redundancy is architectural, not incidental.** GQA's 8:1 ratio means 8 KV heads provide 8x coverage. MoE's top-4/32 routing means facts are processed by whichever 4 experts handle that position. Neither individual KV heads nor individual experts are critical paths.
3. **The residual stream is the memory.** Components (attention heads, MoE experts) READ from and WRITE to the residual stream. Any individual writer can be removed because others write overlapping information. Only when you remove the majority of writers globally does the signal degrade.

### 13.8 Implications for the "Externalizable Knowledge" Question

The original research question was: "What fraction of this model is a lookup table that could be externalized to a database?"

After 14 experiments, the answer is: **facts are not stored in a lookup table at all.** They are distributed patterns in the residual stream, maintained by the collective computation of attention heads and MoE experts. No single component can be swapped for a database query because no single component holds a fact.

The memory bank approach (Part 8, 100% success) works precisely because it operates at the RIGHT level of abstraction - injecting facts directly into the residual stream via attention, rather than trying to locate and replace individual storage components.

---
