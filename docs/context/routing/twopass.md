# Routing Strategy: twopass

**CLI flag:** `--strategy twopass`
**Index required:** `residuals.npz` or `interval_residuals.npz`
**Speed:** Moderate (~1-2s)
**Best for:** Research; observing how the model's compass shifts during generation

## How it works

Two-pass speculative routing — the model's hallucination IS the signal.

**Pass 1**: Generate N tokens without any context (speculative generation). At each step, capture the residual and compare it against stored checkpoint/interval residuals. The routing table at every step shows where the compass points as the model transitions from format tokens to content tokens.

**Pass 2**: Route using the residual at the final speculative step. Replay the top-k windows and regenerate with full context.

## What it reveals

The step-by-step routing table shows the model's internal navigation:
- Steps 1-3: Format tokens ("The", ",") — compass wanders
- Steps 4-8: Content tokens ("Armstrong", "lunar") — compass locks onto the relevant window
- Final step: Use this residual for routing

This is observable evidence that the model's residual stream encodes content-specific directions even during unsupported generation.

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Who commanded Apollo 11?" \
    --strategy twopass \
    --speculative-tokens 10
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/compass_routing/_twopass.py`
