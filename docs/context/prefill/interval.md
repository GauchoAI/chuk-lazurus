# Prefill Phase: interval

**Phase flag:** `--phases interval`
**Output:** `interval_residuals.npz`
**Requires:** Existing library with windows
**Cost:** One forward pass per window (re-reads from archive)

## What it does

Extracts interior residual vectors at sampled positions within each window. While the `windows` phase only stores the boundary residual (last position), the `interval` phase captures the residual stream at 8 evenly-spaced positions throughout each window.

These interior residuals enable fine-grained matching — not just "which window?" but "where within the window?"

## Modes

Controlled by `--residual-mode`:

| Mode | Samples per window | Storage per window | Use case |
|------|-------------------|-------------------|----------|
| `interval` | 8 evenly spaced | ~40 KB | Default — good coverage with low storage |
| `full` | Every position (512) | ~5 MB | Maximum precision, 125x more storage |

## How it works

For each window, run a forward pass and extract the residual stream vector at positions 0, 73, 146, 219, 292, 365, 438, 511 (for a 512-token window). These 8 vectors capture the model's internal state at representative points throughout the window.

At routing time, strategies like `twopass` use these to match query residuals against interior positions, finding the most relevant region within a window.

## Usage

```bash
# Extract interval residuals on an existing library
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --phases interval

# Full residuals (every position)
lazarus context prefill \
    --model google/gemma-3-4b-it \
    --input document.txt \
    --checkpoint ./ctx/ \
    --residual-mode full \
    --phases interval
```

## Storage

For Gemma 4B (hidden=2560) at 512-token windows:
- Interval mode: 8 × 2560D × bf16 = ~40 KB/window, ~29 MB for 725 windows
- Full mode: 512 × 2560D × bf16 = ~2.5 MB/window, ~1.8 GB for 725 windows

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/prefill/_interval.py`
