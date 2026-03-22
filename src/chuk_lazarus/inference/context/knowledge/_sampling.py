"""Shared sampling utility for context generation functions."""

from __future__ import annotations

import mlx.core as mx


def sample_token(logits: mx.array, temperature: float) -> int:
    """Sample a single token from logits.

    Parameters
    ----------
    logits      : (vocab_size,) unnormalised log-probabilities.
    temperature : 0.0 = greedy argmax, >0.0 = categorical sampling.
    """
    if temperature == 0.0:
        return int(mx.argmax(logits).item())
    scaled = logits / temperature
    return int(mx.random.categorical(scaled[None]).item())
