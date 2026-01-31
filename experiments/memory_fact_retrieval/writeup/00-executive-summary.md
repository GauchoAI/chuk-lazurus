# Memory & Fact Retrieval in GPT-OSS: A Mechanistic Study

**Model**: GPT-OSS 20B (24 layers, 2880 hidden dim, 32 experts)
**Date**: 2026-01-29
**Author**: Chris Hay

---

## Executive Summary

This study investigates how large language models store, retrieve, and process factual knowledge, with particular focus on the architectural distinction between declarative and procedural knowledge. Using linear probing, MoE expert tracking, context override experiments, and systematic expert ablation on GPT-OSS 20B, we find:

1. **Fact types are perfectly classifiable** at multiple layers (L4, L8, L13) with 100% accuracy
2. **L16E4 is an end-of-sequence generalist**, not a fact expert - its 25% declarative routing share is a position artifact
3. **Knowledge lives in the residual stream, maintained by collective computation** - ablating all top-4 experts at any layer preserves facts (32/32); ablating any KV head group at any layer preserves facts (248/248); only ablating 28/32 experts at ALL 24 layers causes collapse
4. **No single component stores facts** - 0/248 facts broken by KV head group ablation (L4-L10), 0/32 by single expert ablation (L16), 0/32 by full top-4 ablation at any layer. Facts emerge from collective computation, not individual storage
5. **Expert routing is 93% position-coded** - same-structure prompts share 0.927 Jaccard overlap regardless of content; experts are positional computation units, not content-addressed memories
6. **Learned routing is essential** - replacing the router with random, fixed, or frequency-based selection at all layers breaks ALL facts (0/8) and produces degenerate output. The router-expert coupling is the model's critical mechanism
7. **Expert diversity is learned specialization, not noise** - 0.21 functional similarity between same-class experts; position classes not reflected in weights. Each expert is trained for its routed inputs
8. **Memory bank injection works at 100%** because it enters the residual stream via attention, bypassing positional expert routing entirely
9. **Procedural facts use fundamentally different circuits** - diffuse routing, context override fails (0%)
10. **Wrong signal > No signal** - Skipping MoE entirely (identity pass-through) is worse than fixed routing with wrong experts. Alternating skip: 1/8 facts vs alternating fixed routing: 5/8. Even wrong experts maintain residual stream statistics that downstream layers depend on
11. **Routing resilience is task-dependent** - Code (0.227 rep) is 3.4x more resilient than creative writing (0.769 rep) under routing disruption. Output space constraint substitutes for MoE computation: syntax rules, domain vocabulary, and template structure reduce dependence on correct expert selection
12. **Memory bank fully rescues the lite model** - Alternating fixed routing + memory bank injection = 8/8 facts (100%), identical to the full model. External knowledge compensates for 50% routing simplification on factual workloads. Counterfactual override also preserved at 100%. This validates a hybrid architecture: attention handles fact retrieval, MoE handles output quality, external memory compensates for MoE degradation
13. **L0 is the gatekeeper** - Removing L0 from learned routing drops facts from 5/8 → 3/8. Adding L0 to any alternating set adds ~2 facts. Best 12-layer config: L0 + spaced odds = 6/8 bare, 8/8 with MB. Even vs odd parity is mostly an L0 effect, not a fundamental layer property
14. **Hard cliff at gap=3** - Routing correction every 2 layers: 5/8 facts. Every 3 layers: 1/8 (cliff). Every 4+: 0/8. Memory bank shifts the threshold: 8/8 at gap=2, 6/8 at gap=3-4, 4-5/8 at gap=6-8, 0/8 at gap>8. Below ~4 learned layers, the model can't even read the memory bank

---
