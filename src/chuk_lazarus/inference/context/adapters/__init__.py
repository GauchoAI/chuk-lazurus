"""Model adapters implementing TransformerLayerProtocol / ModelBackboneProtocol."""

from .gemma_adapter import GemmaBackboneAdapter, GemmaLayerAdapter
from .llama_adapter import LlamaBackboneAdapter, LlamaLayerAdapter

__all__ = [
    "GemmaBackboneAdapter",
    "GemmaLayerAdapter",
    "LlamaBackboneAdapter",
    "LlamaLayerAdapter",
]
