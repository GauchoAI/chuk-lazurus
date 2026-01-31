## Part 5: Architectural Implications

### 5.1 The Declarative/Procedural Split

| Property | Declarative | Procedural |
|----------|-------------|------------|
| Clustering | Tight (0.67-0.84) | Loose (0.35-0.54) |
| Dominant Expert | L16E4 (25%) | None (diffuse) |
| Context Override | 33% success | 0% success |
| External Lookup | Viable | Not viable |

### 5.2 Virtual Expert Architecture

The findings enable a clean architecture for externalizing declarative fact retrieval:

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Input                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  L13 Probe: is_declarative? (100% accuracy)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
          is_declarative              is_procedural
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  EXTERNAL KB LOOKUP       │   │  PARAMETRIC GENERATION    │
│  - Extract entity         │   │  - Diffuse expert routing │
│  - Query knowledge base   │   │  - No intervention point  │
│  - Deterministic result   │   │  - Prompt-level only      │
└───────────────────────────┘   └───────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Context Injection at L4                                    │
│  "VERIFIED FACT from [source]: [fact]. Therefore..."        │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Generation (L16E4 bypassed for external facts)             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Detect at L13 | 100% classification accuracy |
| Inject at L4 | Where context override begins |
| Strong markers | "VERIFIED FACT" enables override for confident parametric facts |
| Procedural at prompt level | Context injection doesn't work |

### 5.4 Expected Benefits

| Metric | Parametric | With External KB |
|--------|------------|------------------|
| Factual accuracy | ~85% | **99%+** |
| Hallucination rate | 5-15% | **<1%** |
| Updatability | Requires retraining | **Edit database** |
| Verifiability | None | **Full citations** |
| Latency | 0ms | ~50-100ms |

---
