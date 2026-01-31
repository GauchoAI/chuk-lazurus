# Fact Lookup Virtual Expert Design

## Overview

Based on the memory/fact retrieval experiment findings, this document proposes a virtual expert architecture for externalizing declarative fact lookup.

## Key Findings That Enable This

| Finding | Implication |
|---------|-------------|
| 100% fact type classification at L13 | Reliable detection of declarative queries |
| L16E4 handles 25% of declarative routing | Clear intervention point |
| Context override works at L4 for declarative | Injection point identified |
| Procedural facts resist override | Different strategy needed |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Input                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  L4-L8: Early Processing                                        │
│  - Fact type classifier probe (100% accuracy)                   │
│  - Extract: is_declarative, entity, relation                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              is_declarative           is_procedural
                    │                       │
                    ▼                       ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│  EXTERNAL: Knowledge Base    │   │  INTERNAL: Parametric Memory │
│  - Entity lookup             │   │  - Diffuse routing           │
│  - Deterministic retrieval   │   │  - No single expert          │
│  - Citation attached         │   │  - Cannot reliably override  │
└─────────────────────────────┘   └─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Context Injection at L4                                        │
│  Format: "According to [source]: [fact]. Therefore, ..."        │
│  Strong markers ensure override for confident parametric facts  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  L16+: Generation (L16E4 bypassed for external facts)           │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation: chuk-mcp-facts

### MCP Server Interface

```python
# tools/fact_lookup.py
@mcp_tool
async def fact_lookup(
    entity: str,
    relation: str,
    fact_type: Literal["entity", "numeric", "temporal"]
) -> FactResult:
    """
    Look up a declarative fact from the knowledge base.

    Args:
        entity: The subject entity (e.g., "France")
        relation: The relation being queried (e.g., "capital_of")
        fact_type: Classification from L13 probe

    Returns:
        FactResult with value, source, confidence
    """
    result = await kb.query(entity, relation)
    return FactResult(
        value=result.value,
        source=result.source,
        confidence=result.confidence,
        injection_format=f"According to {result.source}: {result.statement}. "
    )
```

### Detection at L13

```python
class FactTypeDetector:
    """Probe at L13 to detect declarative vs procedural."""

    def __init__(self, model):
        self.probe = load_probe("L13_fact_classifier")  # Trained probe

    async def detect(self, hidden_state: mx.array) -> FactClassification:
        logits = self.probe(hidden_state)
        probs = mx.softmax(logits)

        return FactClassification(
            fact_type=["entity", "numeric", "temporal", "procedural"][mx.argmax(probs)],
            confidence=float(mx.max(probs)),
            is_declarative=mx.argmax(probs) < 3  # Not procedural
        )
```

### Entity Extraction

```python
class EntityExtractor:
    """Extract query entity from prompt."""

    PATTERNS = {
        "capital_of": r"capital of (\w+)",
        "located_in": r"where is (\w+)",
        "born_in": r"when was (\w+) born",
        "symbol_of": r"symbol (?:for|of) (\w+)",
    }

    async def extract(self, prompt: str, fact_type: str) -> QueryEntity:
        # Use regex + LLM fallback for complex queries
        for relation, pattern in self.PATTERNS.items():
            if match := re.search(pattern, prompt, re.I):
                return QueryEntity(
                    entity=match.group(1),
                    relation=relation
                )

        # Fallback: use model to extract
        return await self._llm_extract(prompt)
```

### Context Injection

```python
class ContextInjector:
    """Inject facts as context at L4."""

    # Use contextual qualifiers that mirror successful override patterns
    # "VERIFIED" is ungrounded - "in this context" works better empirically
    INJECTION_TEMPLATES = {
        "entity": "In this context, {entity}'s {relation} is {value} (source: {source}). ",
        "numeric": "For this query, {statement} (source: {source}). ",
        "temporal": "As of {timestamp}, {statement} (source: {source}). ",
    }

    def format_injection(self, fact: FactResult, timestamp: str = None) -> str:
        template = self.INJECTION_TEMPLATES[fact.fact_type]
        return template.format(
            source=fact.source,
            entity=fact.entity,
            relation=fact.relation,
            value=fact.value,
            statement=fact.statement,
            timestamp=timestamp or "current records"
        )
```

**Note**: The "in this context" framing mirrors the contextual qualifiers ("here", "in this region") that succeeded in override experiments. Authority markers like "VERIFIED" are just tokens to the model - they don't carry grounded meaning.

## Integration with Rogue-1

The fact_lookup expert fits the YAML trace pattern:

```yaml
# Example trace for "What is the capital of France?"
expert: fact_lookup
classification:
  fact_type: entity
  confidence: 0.98
  is_declarative: true
extraction:
  entity: France
  relation: capital_of
lookup:
  source: wikidata
  value: Paris
  confidence: 1.0
injection:
  format: "VERIFIED FACT from Wikidata: France capital_of Paris. "
  position: L4
result: Paris
```

## Knowledge Base Options

| Option | Pros | Cons |
|--------|------|------|
| **Wikidata** | Comprehensive, structured | Query complexity |
| **SQLite local** | Fast, offline | Limited coverage |
| **Vector DB** | Semantic search | Embedding overhead |
| **Hybrid** | Best of both | Complexity |

### Recommended: Wikidata + Local Cache

```python
class HybridKB:
    def __init__(self):
        self.cache = SQLiteCache()  # Fast local lookup
        self.wikidata = WikidataClient()  # Fallback for cache miss

    async def query(self, entity: str, relation: str) -> Optional[Fact]:
        # Try cache first
        if cached := await self.cache.get(entity, relation):
            return cached

        # Fallback to Wikidata
        if result := await self.wikidata.query(entity, relation):
            await self.cache.set(entity, relation, result)
            return result

        return None  # Unknown fact, let model generate
```

## Handling Procedural Queries

Since procedural context override fails, use prompt-level injection instead:

```python
async def handle_procedural(query: str, context: str) -> str:
    """For procedural queries, prepend instructions rather than inject at L4."""
    return f"""Follow these specific instructions for this query:

{context}

Now, following the above instructions exactly:
{query}"""
```

## Metrics

| Metric | Target | Baseline (parametric) |
|--------|--------|----------------------|
| Factual accuracy | 99%+ | ~85% (varies by fact type) |
| Hallucination rate | <1% | ~5-15% |
| Latency | <100ms | 0ms (but unreliable) |
| Coverage | KB-dependent | Unbounded (but wrong) |

## Implementation Path

### V1: Simple Tool (No Hidden State Access)

Start with a simpler version that validates KB lookup works before building the full L13 probe.

```python
# v1: Prompt-based classification (no model internals)
class FactLookupToolV1:
    """Simple fact lookup as a tool, not virtual expert."""

    # Keyword patterns for entity fact detection
    ENTITY_PATTERNS = [
        r"what is the capital of",
        r"who is the (?:president|ceo|author) of",
        r"what is the symbol for",
        r"when was .+ born",
        r"where is .+ located",
    ]

    async def should_lookup(self, query: str) -> bool:
        """Prompt-level classification (no hidden states)."""
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in self.ENTITY_PATTERNS)

    async def lookup(self, query: str) -> Optional[str]:
        if not await self.should_lookup(query):
            return None

        entity = self._extract_entity(query)
        relation = self._extract_relation(query)

        if fact := await self.kb.query(entity, relation):
            return f"In this context, {fact.statement} (source: {fact.source}). "

        return None  # Let model handle
```

**Add to chuk-tool-processor**:
```yaml
tools:
  - name: fact_lookup
    description: "Look up factual information about entities"
    trigger: "entity fact queries (capitals, symbols, dates)"
```

### V2: Hidden State Probe

After validating V1 works, add L13 probe for more accurate detection.

```python
class FactLookupToolV2(FactLookupToolV1):
    """V2: Add L13 probe for better classification."""

    def __init__(self, model):
        super().__init__()
        self.probe = self._load_probe("L13_fact_classifier")

    async def should_lookup(self, query: str) -> bool:
        # Try prompt patterns first (fast)
        if super().should_lookup(query):
            return True

        # Fall back to L13 probe (requires forward pass)
        hidden = await self._get_hidden_state(query, layer=13)
        classification = self.probe(hidden)
        return classification.is_declarative and classification.confidence > 0.9

    async def _get_hidden_state(self, query: str, layer: int) -> mx.array:
        tokens = self.tokenizer(query, return_tensors="np")
        output = self.model(mx.array(tokens["input_ids"]), output_hidden_states=True)
        return output.hidden_states[layer][0, -1, :]
```

### V3: Full Virtual Expert

After V2 validates, integrate as true virtual expert with L4 injection.

```python
class FactLookupVirtualExpert:
    """V3: Full virtual expert with hidden state injection."""

    async def process(self, query: str, model_state: ModelState) -> VirtualExpertResult:
        # Detect at L13
        classification = await self.classify(model_state.hidden_states[13])

        if not classification.is_declarative:
            return VirtualExpertResult(handled=False)

        # Extract and lookup
        entity, relation = await self.extract(query)
        fact = await self.kb.query(entity, relation)

        if not fact:
            return VirtualExpertResult(handled=False)

        # Inject at L4
        injection = self.format_injection(fact)
        return VirtualExpertResult(
            handled=True,
            injection=injection,
            injection_layer=4,
            bypass_experts=[("L16", 4)],  # Skip fact lookup expert
        )
```

### Migration Path

| Phase | Scope | Validation Metric |
|-------|-------|-------------------|
| **V1** | Entity facts only, prompt patterns | Override success rate > 50% |
| **V2** | Add L13 probe | Classification accuracy > 95% |
| **V3** | Full virtual expert | Factual accuracy > 99% |

### Immediate Next Steps

1. **V1 Implementation**
   - Add `fact_lookup` tool to chuk-tool-processor
   - Connect to Wikidata SPARQL for entity queries
   - Measure override success in production

2. **KB Setup**
   - Start with capitals, chemical symbols, historical dates
   - ~1000 high-confidence facts
   - SQLite cache + Wikidata fallback

3. **Validation**
   - A/B test: parametric vs KB-augmented
   - Track hallucination rate by fact type
   - Measure latency impact

## Open Questions (with Proposed Solutions)

### 1. Partial Matches

**Problem**: Query asks about "French Republic" but KB has "France".

**Solution**: Fuzzy entity matching with confidence penalty.

```python
class EntityMatcher:
    def match(self, query_entity: str) -> MatchResult:
        # Exact match
        if exact := self.kb.get(query_entity):
            return MatchResult(entity=query_entity, confidence=1.0, fact=exact)

        # Fuzzy match with aliases
        if fuzzy := self.kb.fuzzy_search(query_entity, threshold=0.85):
            return MatchResult(
                entity=fuzzy.canonical,
                confidence=0.8,  # Penalty for fuzzy
                fact=fuzzy.fact
            )

        return None  # Let parametric handle
```

**Decision threshold**: Only inject if confidence > 0.7. Below that, let the model generate and accept hallucination risk.

### 2. Multi-hop Facts

**Problem**: "What is the capital of the country that has the Eiffel Tower?"

**Solution**: Integrate with chuk-ai-planner for decomposition.

```python
class MultiHopResolver:
    """Decompose multi-hop queries into execution graph."""

    async def resolve(self, query: str) -> list[FactResult]:
        # Detect if multi-hop (nested entity references)
        if not self._is_multi_hop(query):
            return [await self.single_lookup(query)]

        # Plan decomposition
        plan = await self.planner.decompose(query)
        # Example plan:
        # [
        #   {"hop": 1, "query": "Eiffel Tower located_in ?", "binds": "country"},
        #   {"hop": 2, "query": "{country} capital_of ?", "binds": "capital"}
        # ]

        results = []
        bindings = {}
        for step in plan:
            resolved_query = step["query"].format(**bindings)
            result = await self.single_lookup(resolved_query)
            bindings[step["binds"]] = result.value
            results.append(result)

        return results

    def _is_multi_hop(self, query: str) -> bool:
        # Heuristics: nested relative clauses, "of the X that", etc.
        patterns = [
            r"of the \w+ that",
            r"where .+ is located",
            r"whose .+ is",
        ]
        return any(re.search(p, query, re.I) for p in patterns)
```

### 3. Temporal Facts

**Problem**: "Who is the Prime Minister of the UK?" changes over time.

**Solution**: Version KB entries with validity periods.

```python
@dataclass
class TemporalFact:
    entity: str
    relation: str
    value: str
    valid_from: datetime
    valid_to: datetime | None  # None = current

class TemporalKB:
    async def query(
        self, entity: str, relation: str, as_of: datetime = None
    ) -> TemporalFact:
        as_of = as_of or datetime.now()

        fact = await self.db.query(
            entity=entity,
            relation=relation,
            valid_from__lte=as_of,
            valid_to__gte=as_of  # or NULL for current
        )

        return fact

# Injection includes timestamp
"As of January 2026, the Prime Minister of the UK is Keir Starmer (source: gov.uk). "
```

### 4. Procedural Knowledge (Experimental)

**Observation**: L5E24 shows 21% activation for procedural - it's shared with declarative as a "knowledge activation" expert.

**Hypothesis**: Gentle amplification of L5E24 during procedural queries might improve context integration without full steering.

```python
class ProceduralAmplifier:
    """Experimental: amplify L5E24 for procedural queries."""

    def __init__(self, model, strength: float = 0.5):
        self.strength = strength  # Lower than steering (0.5 vs 5.0)
        self.target_layer = 5
        self.target_expert = 24

    async def generate_with_amplification(
        self, prompt: str, context: str
    ) -> str:
        # Inject context at prompt level (known to work)
        full_prompt = f"{context}\n\n{prompt}"

        # Amplify L5E24 routing weight
        # This is speculative - needs validation
        with self.model.amplify_expert(
            layer=self.target_layer,
            expert=self.target_expert,
            factor=1.0 + self.strength
        ):
            return await self.model.generate(full_prompt)
```

**Status**: Speculative. Needs experimentation to validate if L5E24 amplification improves procedural context integration.
