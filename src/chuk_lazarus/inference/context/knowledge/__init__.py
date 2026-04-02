"""Knowledge store — the production path for document understanding."""

from .append import append_skill, build_base_state, load_base_state, load_index, save_index
from .build import streaming_prefill
from .config import ArchitectureConfig, ArchitectureNotCalibrated
from .inject import (
    extract_donor_residual,
    generate_with_boundary,
    generate_with_injection,
    generate_with_markov_injection,
    generate_with_persistent_injection,
    inject_1d,
)
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
    "_extract_keywords_from_text",
    "extract_donor_residual",
    "extract_window_keywords",
    "generate_with_boundary",
    "generate_with_injection",
    "generate_with_markov_injection",
    "generate_with_persistent_injection",
    "inject_1d",
    "streaming_prefill",
]
