# Prefill Phase: pages

**Phase flag:** `--phases pages` (also requires `--store-pages`)
**Output:** `pages.npz`
**Requires:** Existing library with windows
**Cost:** One forward pass per window

## What it does

Extracts pre-RoPE K,V tensors at sampled positions within each window. These are K,V values before rotary position embedding is applied — position-independent representations that can be injected at any sequence position at generation time.

This enables **instant page injection** (Mode 6): instead of replaying a full window through the model, inject the pre-computed K,V directly into the attention layers. The model attends to the injected pages as if it had processed the tokens, but without the forward pass cost.

## Pre-RoPE vs post-RoPE

Standard KV cache stores post-RoPE values — K,V with position information baked in. These can only be used at the sequence positions where they were computed.

Pre-RoPE pages store K,V before position encoding. At injection time, RoPE is applied with the target position. This means the same page can be injected at any position in the generation sequence.

## Usage

```bash
# Extract pages on existing library
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --store-pages \
    --phases pages
```

## Storage

8 pages × 34 layers × 2 (K,V) × 4 KV heads × 320 head_dim × bf16 per window.
~1 GB for 725 windows. Heavy — only use when instant injection speed is needed.

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_pages.py`
