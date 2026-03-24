# Routing Strategy: iterative

**CLI flag:** `--strategy iterative`
**Index required:** `compass_residuals.npz`, `compass_basis.npz`
**Speed:** ~10-15s (multiple rounds)
**Best for:** Exploration queries; answers spanning multiple document regions

## How it works

Multi-round compass navigation with generation-guided shifting. Each round:

1. **Route**: Compass geometric routing (excluding already-visited windows)
2. **Replay**: Load the best window into context
3. **Explore**: Generate 50 tokens with a note-taking prompt (not a full answer)
4. **Shift**: Extract the L26 generation residual and use it to steer the compass for the next round

After all rounds, replay the last 3 discovered windows together and generate the final answer.

### Note-taking prompt

Exploration rounds use a special prompt: "DO NOT answer yet. Write 1-2 sentences of reading notes: what specific content is in this excerpt and how relevant is it to the task?"

This puts the model in judgment/assessment mode, which produces residuals that encode WHAT the model found noteworthy — steering the compass toward content the model deems relevant, not answer structure.

### Generation-guided compass shift

The key insight: after the model reads a window and generates notes, its L26 residual has shifted to reflect what it learned. This shifted residual becomes the query vector for the next compass call. The compass naturally moves toward related but unvisited content.

## Parameters

- `--max-rounds`: Number of exploration rounds (default: 3)
- `--top-k`: Windows per compass call (default: 1)

## Usage

```bash
lazarus context generate \
    --model google/gemma-3-4b-it \
    --checkpoint ./ctx/ \
    --prompt "Find the most amusing moments in the transcript" \
    --strategy iterative \
    --max-rounds 5
```

## Source

Implementation: `src/chuk_lazarus/cli/commands/context/generate/_iterative.py`
