"""
Context management for stateful inference.

Four generation strategies:

  KVDirectGenerator        — raw KV-direct step generator (Mode 2)
  CompiledRSGenerator      — compiled residual-stream generator (Mode 1b)
  BoundedKVEngine (Mode 3) — three-tier bounded memory (HOT/WARM/COLD)
  UnlimitedContextEngine   — checkpoint-chained window replay (Mode 4)

And the checkpoint library format for pre-filled knowledge bases:
  CheckpointLibrary, LibraryManifest, WindowMeta, etc.

Usage
-----
    from chuk_lazarus.inference.context import (
        KVDirectGenerator,
        UnlimitedContextEngine,
        CheckpointLibrary,
        LibrarySource,
    )
"""

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
from .kv_generator import KVDirectGenerator
from .rs_generator import CompiledRSGenerator
from .unlimited_engine import (
    CheckpointStore,
    EngineStats,
    KVGeneratorProtocol,
    KVStore,
    LibrarySource,
    TokenArchive,
    UnlimitedContextEngine,
)

__all__ = [
    # Generators
    "KVDirectGenerator",
    "CompiledRSGenerator",
    # Mode 3 — bounded engine
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
    "TokenArchive",
    "UnlimitedContextEngine",
    # Checkpoint library format
    "LibraryFile",
    "LibraryFormatVersion",
    "WindowMeta",
    "LibraryManifest",
    "CheckpointLibrary",
]
