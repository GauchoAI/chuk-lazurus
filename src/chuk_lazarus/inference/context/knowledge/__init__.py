"""Knowledge store — the production path for document understanding.

Build once, query fast. Three entry points:

    streaming_prefill()              — read document, extract injection entries
    KnowledgeStore.route()           — TF-IDF / keyword → window index
    generate_with_injection()        — query + entries → answer

Usage
-----
    from chuk_lazarus.inference.context.knowledge import (
        KnowledgeStore,
        streaming_prefill,
        generate_with_injection,
        ArchitectureConfig,
    )
"""

from .build import streaming_prefill
from .config import ArchitectureConfig, ArchitectureNotCalibrated
from .inject import generate_with_injection, inject_1d
from .route import (
    KeywordRouter,
    SparseKeywordIndex,
    TFIDFRouter,
    _extract_keywords_from_text,
    extract_window_keywords,
)
from .store import InjectionEntry, KnowledgeStore

__all__ = [
    "ArchitectureConfig",
    "ArchitectureNotCalibrated",
    "InjectionEntry",
    "KeywordRouter",
    "KnowledgeStore",
    "SparseKeywordIndex",
    "TFIDFRouter",
    "extract_window_keywords",
    "generate_with_injection",
    "inject_1d",
    "streaming_prefill",
]
