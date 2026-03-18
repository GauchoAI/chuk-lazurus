# Prefill Phase: windows

**Phase flag:** `--phases windows`
**Output:** `checkpoints.npz`, `residuals.npz`, `tokens.bin`, `windows.json`, `manifest.json`
**Requires:** Input text file
**Cost:** One full forward pass per window

## What it does

The `windows` phase is the core prefill loop. It tokenizes the input document, splits it into fixed-size windows, and runs a full forward pass through each window sequentially. Each window builds on the KV cache state left by the previous window, maintaining Markov continuity across the document.

For each window, it stores:

- **Boundary KV checkpoint** (`checkpoints.npz`): The K,V tensors at the last position of each layer. These are the "seam" between windows — injecting this checkpoint lets the model continue as if it had just read that window.
- **Boundary residual** (`residuals.npz`): The residual stream vector at the last position. This is the Markov state — the complete forward state compressed to a single vector.
- **Token IDs** (`tokens.bin`): Raw uint32 token IDs for the entire document.
- **Window metadata** (`windows.json`): Per-window offsets, token counts, and text previews.

## How it works

```
Window 0: tokens[0:512]     → forward pass → checkpoint_0, residual_0
Window 1: tokens[512:1024]  → forward pass (seeded from checkpoint_0) → checkpoint_1, residual_1
Window 2: tokens[1024:1536] → forward pass (seeded from checkpoint_1) → checkpoint_2, residual_2
...
```

Each window's forward pass is seeded with the previous window's boundary KV, so the model maintains context continuity. The Gemma sliding window attention (pattern=6) means non-global layers only attend to the current window, while global layers (every 6th) attend to all accumulated KV.

## Incremental saves

The prefill loop saves periodically (every 5 minutes wall time) by appending only new windows to the existing zip files. This makes it safe to interrupt — Ctrl-C saves the current state, and `--resume` picks up where it left off.

## Usage

```bash
# Prefill only — no extraction passes
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases windows

# Resume after interruption (automatic)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/
```

## Storage

Storage per window depends on model size. For Gemma 4B (34 layers, hidden=2560, 4 KV heads, head_dim=320):
- Checkpoint: 34 layers × 2 (K,V) × 4 heads × 1 position × 320D × bf16 = ~87 KB/window
- Residual: 2560D × bf16 = 5 KB/window
- Total: ~92 KB/window, ~65 MB for 725 windows

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_cmd.py`
Serialization: `src/chuk_lazarus/cli/commands/context/prefill/_checkpoints.py`
