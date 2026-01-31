## Conclusion

This study reveals three layers of how GPT-OSS processes knowledge:

### Where knowledge lives

**Facts are in the residual stream, maintained by collective computation.** Linear probes classify fact types at L4 with 100% accuracy. Removing ALL 4 top-k experts at any layer preserves 100% of facts (32/32). Even ablating 30 of 32 experts at a single layer preserves 100%. Ablating any of 8 KV head groups at layers L4-L10 preserves 100% of facts (248/248 tests). But ablating 28/32 experts at ALL 24 layers simultaneously causes model collapse (1/8 survival). The residual stream carries facts, but requires ongoing computation from both attention heads and MoE experts to maintain signal integrity - like DRAM that needs refresh cycles.

### What components do

**Components are locally redundant but globally orchestrated.** Expert routing is 93% determined by token position, and no single component (expert or attention head) is individually necessary for facts. But replacing the learned router with ANY alternative (random, fixed, frequency-based) at all layers breaks ALL facts (0/8) and produces degenerate output. The router-expert coupling - which expert processes which token at which layer - is the model's critical mechanism. Weight similarity confirms experts are genuinely diverse (0.21 functional similarity), not duplicates. Each expert is specialized for the inputs the router sends it. Critically, even wrong experts are better than no experts: skipping MoE entirely (1/8 facts with alternating skip) is worse than fixed routing (5/8 with alternating). MoE layers cannot be removed; they maintain residual stream statistics that downstream layers depend on.

### Why memory banks work

**Memory banks enter facts via the residual stream at L0-L4, bypassing component-level routing entirely.** The `[Memory Bank]` format achieves 100% override (including counterfactuals like "France | capital | Lyon") because attention to the memory bank tokens places facts directly into the residual stream. Memory banks continue working even under full top-4 expert ablation (94% success rate across all layers). This is architecture-agnostic and doesn't require identifying or manipulating specific components.

### The declarative/procedural split

**Declarative knowledge** (facts about the world):
- Encoded in the residual stream from early layers
- Not localized to any single expert or attention head (0 facts broken in 248+ ablation tests)
- Overridable via memory bank injection (100%)
- **Externalizable via memory banks, not via component replacement**

**Procedural knowledge** (how to do things):
- Diffusely encoded across many experts
- No dominant routing expert
- Resists context override entirely (0%)
- **Not externalizable via current methods**

### Practical implication

For production systems requiring factual accuracy: **inject facts via memory banks at the prompt level**. This works because facts live in the residual stream, not in any individual component (expert or attention head). No model surgery needed - the architecture already supports external fact injection via attention. Attempts to externalize facts by identifying and replacing specific "storage" components will fail because no such components exist.

### The compression path

Memory bank injection + simplified routing achieves **100% factual accuracy with 50% routing computation**. Alternating fixed routing (learned at even layers, fixed at odd) loses all facts without memory bank (3/8) but recovers fully with it (8/8). Counterfactual override also preserved at 100%. The trade-off: moderate fluency degradation (0.164 repetition ratio vs 0.000 for full model). For factual Q&A, search, and knowledge retrieval: viable. For creative/conversational generation: not viable (0.769 repetition). Model compressibility is task-dependent.

---

---
