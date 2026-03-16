"""
Context management for stateful inference.

Five generation strategies:

  KVDirectGenerator        — model-agnostic KV-direct step generator (Mode 2)
  CompiledRSGenerator      — compiled residual-stream generator (Mode 1b, Gemma-only)
  BoundedKVEngine (Mode 3) — three-tier bounded memory (HOT/WARM/COLD, Gemma-only)
  UnlimitedContextEngine   — checkpoint-chained window replay (Mode 4)
  KV Injection (Mode 6)    — prefix caching with pre-saved full KV per window

Protocols and adapters for any transformer architecture:
  ModelBackboneProtocol    — interface that any backbone adapter must satisfy
  TransformerLayerProtocol — per-layer interface
  GemmaBackboneAdapter     — wraps GemmaResidualStreamForCausalLM
  LlamaBackboneAdapter     — wraps LlamaForCausalLM / Mistral

Factory:
  make_kv_generator(model) — auto-detects family, returns KVDirectGenerator

Checkpoint library format for pre-filled knowledge bases:
  CheckpointLibrary, LibraryManifest, WindowMeta, etc.

Usage
-----
    from chuk_lazarus.inference.context import (
        KVDirectGenerator,
        make_kv_generator,
        GemmaBackboneAdapter,
        LlamaBackboneAdapter,
        UnlimitedContextEngine,
        CheckpointLibrary,
        LibrarySource,
    )
"""

from .adapters import (
    GemmaBackboneAdapter,
    GemmaLayerAdapter,
    LlamaBackboneAdapter,
    LlamaLayerAdapter,
)
from .bounded_engine import (
    BoundedKVEngine,
    Checkpoint,
    ConversationState,
    GenerationMode,
    MemoryReport,
    PathLabel,
    TurnStats,
)
from .checkpoint_library import (
    CheckpointLibrary,
    LibraryFile,
    LibraryFormatVersion,
    LibraryManifest,
    WindowMeta,
)
from .kv_checkpoint import (
    CheckpointMeta,
    ContextCheckpointFile,
    ContextCheckpointStatus,
    KVCheckpoint,
)
from .kv_generator import KVDirectGenerator, make_kv_generator
from .protocols import ModelBackboneProtocol, TransformerLayerProtocol
from .rs_generator import CompiledRSGenerator
from .unlimited_engine import (
    CheckpointStore,
    EngineStats,
    KVGeneratorProtocol,
    KVStore,
    LibrarySource,
    ResidualStore,
    TokenArchive,
    UnlimitedContextEngine,
)
from .sparse_index import (
    EntityExtractor,
    FactSpan,
    SparseEntry,
    SparseSemanticIndex,
    SurpriseClassifier,
)
from .sparse_engine import SparseIndexEngine

__all__ = [
    # Protocols
    "ModelBackboneProtocol",
    "TransformerLayerProtocol",
    # Adapters
    "GemmaBackboneAdapter",
    "GemmaLayerAdapter",
    "LlamaBackboneAdapter",
    "LlamaLayerAdapter",
    # Generators
    "KVDirectGenerator",
    "make_kv_generator",
    "CompiledRSGenerator",
    # Mode 3 — bounded engine (Gemma-only)
    "GenerationMode",
    "PathLabel",
    "MemoryReport",
    "TurnStats",
    "Checkpoint",
    "ConversationState",
    "BoundedKVEngine",
    # Mode 4 — unlimited context engine
    "KVStore",
    "KVGeneratorProtocol",
    "LibrarySource",
    "EngineStats",
    "CheckpointStore",
    "ResidualStore",
    "TokenArchive",
    "UnlimitedContextEngine",
    # Checkpoint library format
    "LibraryFile",
    "LibraryFormatVersion",
    "WindowMeta",
    "LibraryManifest",
    "CheckpointLibrary",
    # KV checkpoint
    "CheckpointMeta",
    "ContextCheckpointFile",
    "ContextCheckpointStatus",
    "KVCheckpoint",
    # Mode 5 — sparse semantic index
    "EntityExtractor",
    "SparseEntry",
    "SparseSemanticIndex",
    "SurpriseClassifier",
    "SparseIndexEngine",
]
