"""
Gemma Residual Stream architecture.

Markov inference: the residual tensor at each layer IS the complete forward state.
No KV cache is ever stored between forward passes.
"""

from .model import (
    GemmaResidualStreamForCausalLM,
    ResidualStreamOutput,
    PartialResidualOutput,
)

__all__ = [
    "GemmaResidualStreamForCausalLM",
    "ResidualStreamOutput",
    "PartialResidualOutput",
]
