# Memory & Fact Retrieval in GPT-OSS: A Mechanistic Study

**Model**: GPT-OSS 20B (24 layers, 2880 hidden dim, 32 experts)
**Date**: 2026-01-29
**Author**: Chris Hay

---

## Table of Contents

0. [Executive Summary](00-executive-summary.md)

### Core Analysis (Parts 1-7)

1. [Where Facts Live](01-where-facts-live.md) - Linear probing, fact type clustering
2. [MoE Expert Routing](02-moe-expert-routing.md) - Expert activation by fact type, L16E4 analysis
3. [Context vs Parametric Memory](03-context-vs-parametric-memory.md) - Override experiments, declarative/procedural asymmetry
4. [Confidence & Calibration](04-confidence-calibration.md) - Calibration curves, hallucination detection
5. [Architectural Implications](05-architectural-implications.md) - Virtual expert architecture, design decisions
6. [Connections to Prior Work](06-prior-work.md)
7. [Limitations & Future Work](07-limitations-future-work.md)

### [Conclusion](08-conclusion.md)

### Deep-Dive Experiments (Parts 8-21)

8. [Memory Bank Proof of Concept](09-memory-bank-poc.md) - Structured external memory, 100% override
9. [Expert Ablation](10-expert-ablation.md) - Single-expert and progressive top-k ablation
10. [Cross-Layer Ablation](11-cross-layer-ablation.md) - Finding the breaking point, DRAM refresh analogy
11. [Position Analysis](12-position-analysis.md) - 93% position-coded routing, L16E4 reframed
12. [Position-Class Pruning](13-position-class-pruning.md) - Compression limits, 87.5% single-layer reduction
13. [Attention Head Ablation](14-attention-head-ablation.md) - KV head group scan, 0/248 facts broken
14. [Expert Weight Similarity](15-expert-weight-similarity.md) - Diverse ensemble, not duplicates
15. [Routing Ablation](16-routing-ablation.md) - Learned routing is essential, 0/8 for all alternatives
16. [Partial Routing Ablation](17-partial-routing-ablation.md) - Spacing matters more than count
17. [Layer Skipping](18-layer-skipping.md) - Wrong signal > no signal
18. [Routing Resilience](19-routing-resilience.md) - Why photosynthesis survived, task-dependent compressibility
19. [Memory Bank + Lite Model](20-memory-bank-lite-model.md) - The compression path, hybrid architecture
20. [Layer Parity](21-layer-parity.md) - L0 is the gatekeeper, even vs odd
21. [Layer Spacing](22-layer-spacing.md) - Hard cliff at gap=3, MB rescue gradient

### [Appendix: Experiment Configuration](23-appendix.md)
