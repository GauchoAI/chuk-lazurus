"""
Inference and text generation utilities.

Provides a high-level API for loading and running inference
with any supported model family.

Recommended: Use UnifiedPipeline which auto-detects model family:

    from chuk_lazarus.inference import UnifiedPipeline

    # One-liner for any supported model - auto-detects family!
    pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Chat
    response = pipeline.chat("What is the capital of France?")
    print(response.text)

    # Generation with custom settings
    response = pipeline.generate(
        "Write a poem about AI",
        max_new_tokens=200,
        temperature=0.9,
    )
"""

# Chat utilities
from .chat import (
    ASSISTANT_SUFFIX,
    ChatHistory,
    ChatMessage,
    FallbackTemplate,
    Role,
    format_chat_prompt,
    format_history,
)

# Generation utilities
from .generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    generate,
    generate_stream,
    get_stop_tokens,
)

# Loader utilities
from .loader import (
    DownloadConfig,
    DownloadResult,
    DType,
    HFLoader,
    LoadedWeights,
    StandardWeightConverter,
    WeightConverter,
)

# Unified pipeline (recommended)
from .unified import (
    IntrospectionResult,
    UnifiedPipeline,
    UnifiedPipelineConfig,
    UnifiedPipelineState,
)

# Context management — stateful inference engines and checkpoint libraries
from .context import (
    # Generators
    KVDirectGenerator,
    CompiledRSGenerator,
    # Mode 3: bounded engine
    GenerationMode,
    PathLabel,
    MemoryReport,
    TurnStats,
    Checkpoint,
    ConversationState,
    BoundedKVEngine,
    # Mode 4: unlimited context engine
    KVStore,
    KVGeneratorProtocol,
    LibrarySource,
    EngineStats,
    CheckpointStore,
    TokenArchive,
    UnlimitedContextEngine,
    # Checkpoint library format
    LibraryFile,
    LibraryFormatVersion,
    WindowMeta,
    LibraryManifest,
    CheckpointLibrary,
)

# Virtual expert system for MoE and dense models
from .virtual_expert import (
    MathExpertPlugin,
    SafeMathEvaluator,
    VirtualDenseRouter,
    VirtualDenseWrapper,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertRegistry,
    VirtualExpertResult,
    VirtualMoEWrapper,
    VirtualRouter,
    create_virtual_dense_wrapper,
    create_virtual_expert_wrapper,
    get_default_registry,
)

__all__ = [
    # Loader
    "DownloadConfig",
    "DownloadResult",
    "DType",
    "HFLoader",
    "LoadedWeights",
    "StandardWeightConverter",
    "WeightConverter",
    # Chat
    "ASSISTANT_SUFFIX",
    "ChatHistory",
    "ChatMessage",
    "FallbackTemplate",
    "Role",
    "format_chat_prompt",
    "format_history",
    # Generation
    "GenerationConfig",
    "GenerationResult",
    "GenerationStats",
    "generate",
    "generate_stream",
    "get_stop_tokens",
    # Unified Pipeline (recommended)
    "UnifiedPipeline",
    "UnifiedPipelineConfig",
    "UnifiedPipelineState",
    "IntrospectionResult",
    # Context — generators
    "KVDirectGenerator",
    "CompiledRSGenerator",
    # Context — bounded engine (Mode 3)
    "GenerationMode",
    "PathLabel",
    "MemoryReport",
    "TurnStats",
    "Checkpoint",
    "ConversationState",
    "BoundedKVEngine",
    # Context — unlimited engine (Mode 4)
    "KVStore",
    "KVGeneratorProtocol",
    "LibrarySource",
    "EngineStats",
    "CheckpointStore",
    "TokenArchive",
    "UnlimitedContextEngine",
    # Context — checkpoint library
    "LibraryFile",
    "LibraryFormatVersion",
    "WindowMeta",
    "LibraryManifest",
    "CheckpointLibrary",
    # Virtual Expert System (MoE)
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "VirtualMoEWrapper",
    "VirtualRouter",
    "create_virtual_expert_wrapper",
    # Virtual Expert System (Dense)
    "VirtualDenseWrapper",
    "VirtualDenseRouter",
    "create_virtual_dense_wrapper",
    # Built-in plugins
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "get_default_registry",
]
