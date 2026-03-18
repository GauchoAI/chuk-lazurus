# Routing Strategy: sparse

**CLI flag:** `--strategy sparse`
**Index required:** `sparse_index.json`
**Speed:** ~10ms
**Best for:** Semantic keyword matching from pre-extracted entities

## How it works

BM25 scoring over the pre-extracted keyword index, not raw window text. The `sparse` prefill phase extracts entity names, novel terms, and content words at surprise-identified fact positions. This strategy queries that index.

Advantages over raw BM25:
- **Semantically richer**: Keywords are entity+context triplets (e.g., "Neil Armstrong commander"), not raw word frequencies
- **Faster**: No tokenizer decode needed — the index is pre-computed JSON
- **Surprise-aware**: Only indexes content the model found novel, filtering out parametric knowledge noise

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Who was the commander?" \
    --strategy sparse
```

## Fallback

If no `sparse_index.json` exists, falls back to raw BM25.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_sparse.py`
