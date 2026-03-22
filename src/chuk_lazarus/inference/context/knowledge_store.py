"""Shim — canonical locations: knowledge/store.py, knowledge/route.py, knowledge/build.py"""
from .knowledge.build import streaming_prefill  # noqa: F401
from .knowledge.route import (  # noqa: F401
    SparseKeywordIndex,
    _extract_keywords_from_text,
    extract_window_keywords,
)
from .knowledge.store import InjectionEntry, KnowledgeStore  # noqa: F401
