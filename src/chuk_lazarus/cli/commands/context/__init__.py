"""Context management CLI commands.

Commands:
    context_prefill_cmd  — prefill a document and save a KV checkpoint
    context_generate_cmd — generate from a saved KV checkpoint
"""

from ._types import GenerateConfig, GenerateResult, PrefillConfig, PrefillResult
from .generate import context_generate_cmd
from .prefill import context_prefill_cmd

__all__ = [
    "GenerateConfig",
    "GenerateResult",
    "PrefillConfig",
    "PrefillResult",
    "context_generate_cmd",
    "context_prefill_cmd",
]
