# Prefill Phase: sparse

**Phase flag:** `--phases sparse`
**Output:** `sparse_index.json`
**Requires:** Existing library with windows
**Cost:** One forward pass per window (or zero if SparseIndexEngine was used during prefill)

## What it does

Builds a keyword index for each window — entity names, novel terms, and content words extracted via surprise-guided analysis. This enables BM25 text-level routing without decoding full window tokens at query time.

## Two extraction paths

**Inline (zero cost):** When prefill uses `SparseIndexEngine` (the sparse-aware engine), keywords are extracted during the window forward pass itself. No extra compute needed — the index is built as a side effect of prefill.

**Separate pass (fallback):** When running `--phases sparse` on an existing library that wasn't built with inline extraction, a separate forward pass per window computes surprise scores and extracts keywords.

## What it stores

Per window in `sparse_index.json`:
- `window_id`: Window identifier
- `keywords`: Entity/concept triplets (e.g., "Neil Armstrong commander", "Eagle lunar module")
- `content_words`: High-information content words
- `fact_spans`: Positions of novel facts with surrounding context
- `is_parametric`: Whether the window contains only knowledge the model already knew

## Surprise-guided extraction

The extraction is driven by surprise: positions where the model was most surprised (high prediction rank) are identified as novel content. Keywords are extracted from the context around these surprise peaks. This means the index naturally focuses on what's **new** to the model, not what it already knows parametrically.

## Usage

```bash
# Build sparse index on existing library
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases sparse

# Or include during full prefill (uses inline extraction)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases all
```

## Storage

JSON text — typically 50-200 KB for 725 windows. Contains human-readable keyword triplets.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_sparse.py`
Inline engine: `src/chuk_lazarus/inference/context/sparse_engine.py`
