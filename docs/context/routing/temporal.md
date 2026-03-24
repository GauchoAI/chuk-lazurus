# Routing Strategy: temporal

**CLI flag:** `--strategy temporal`
**Index required:** None
**Speed:** Instant
**Best for:** Global/timeline queries; "summarize the whole document"; chronological coverage

## How it works

Selects evenly-spaced windows across the document. No scoring, no model inference — pure structural sampling that guarantees coverage of the full document timeline.

For `top_k=10` across 725 windows, selects windows at positions 0, 72, 144, 216, 288, 360, 432, 504, 576, 648 — one every ~72 windows.

Scores are uniform (1.0 descending to 0.0) so the ordering is preserved for downstream compatibility, but the selection is purely positional.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Summarize the key events from beginning to end" \
    --strategy temporal \
    --top-k 10
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_temporal.py`
