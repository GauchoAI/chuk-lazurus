## Appendix: Experiment Configuration

### Model
- **ID**: `openai/gpt-oss-20b`
- **Layers**: 24
- **Hidden dim**: 2880
- **Experts**: 32 per MoE layer

### Probe Layers
- Parametric probing: L4, L6, L8, L10, L12, L13, L16, L18, L20
- Context override: L4, L8, L12, L13, L16, L20
- MoE routing: All 24 layers

### Fact Dataset
- Entity: 15 facts (capitals, symbols, authors)
- Numeric: 10 facts (physical constants, quantities)
- Temporal: 10 facts (historical dates)
- Procedural: 8 facts (conversions, methods)

### Analyses Run
1. `parametric_probing` - Linear probe accuracy by layer
2. `context_memory` - Override detection and success rates
3. `fact_steering` - Simulated suppression/injection
4. `retrieval_attention` - Attention pattern analysis
5. `fact_types` - Clustering and cross-type confusion
6. `confidence` - Calibration curves and ECE
7. `moe_fact_routing` - Expert activation by fact type
8. `expert_function_classification` - Systematic expert ablation (29/32 experts at L16)
9. `memory_bank_proof` - Memory bank override testing (10/10)
10. `knowledge_ablation` - Progressive top-k ablation (8 facts x 4 layers, 0/32 broke)
11. `cross_layer_ablation` - Cross-layer stress test (104 tests, collapse at 28/32 experts x 24 layers)
12. `position_analysis` - Sequence-position expert coding (0.927 same-structure Jaccard)
13. `position_pruning` - Position-class pruning (87.5% at single layer, 0% at all layers)
14. `attention_head_ablation` - KV head group ablation (31 groups x 8 facts = 248 tests, 0 broken)
15. `expert_weight_similarity` - Bias, scale, and functional output similarity (0.21 mean functional, NOT duplicates)
16. `routing_ablation` - Random/fixed routing vs learned (normal: 8/8, ALL alternatives: 0/8)
17. `partial_routing` - Partial routing coverage (alternating 12/24: 5/8, contiguous 12/24: 1/8)
18. `layer_skipping` - MoE skip vs fixed routing (wrong signal > no signal; alternating skip: 1/8 vs fixed: 5/8)
19. `routing_resilience` - Why photosynthesis survived (code 0.227 rep vs creative 0.769; output constraint = resilience)
20. `memory_bank_lite` - Memory bank + degraded routing (fixed_alt+MB: 8/8 facts, skip_alt+MB: 6/8, counterfactual 100%)
21. `layer_parity` - Even vs odd layers + L0 importance (L0 worth ~2 facts; best 12-layer: L0+spaced odds = 6/8; all configs 8/8 with MB)
22. `layer_spacing` - How sparse can routing be (cliff at gap=3: 5/8→1/8; MB rescue gradient: 8/8 at gap=2, 4/8 at gap=8, 0/8 at gap>8)
