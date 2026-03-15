"""Memory and memory-inject introspect parsers."""

import asyncio

from ....commands.introspect import (
    introspect_memory,
    introspect_memory_inject,
)


def register_memory_parsers(introspect_subparsers):
    """Register memory and memory-inject subcommands."""
    # Memory command - extract memory organization structure
    memory_parser = introspect_subparsers.add_parser(
        "memory",
        help="Extract memory organization structure for facts",
        description="""Extract how facts are organized in model memory by analyzing
neighborhood activation patterns.

For each query, captures what other facts co-activate, revealing:
- Memory organization (row vs column, clusters)
- Asymmetry (does A->B activate same as B->A?)
- Attractor nodes (frequently co-activated facts)
- Difficulty patterns (which facts are hardest to retrieve)

Built-in fact types:
- multiplication: Single-digit times tables (2-9)
- addition: Single-digit addition
- capitals: Country capitals
- elements: Periodic table elements

Custom facts via CSV/JSON file.

Examples:
    # Extract times table memory structure
    lazarus introspect memory -m model --facts multiplication --layer 20

    # Extract capital city memory
    lazarus introspect memory -m model --facts capitals --layer 15

    # Custom facts from file
    lazarus introspect memory -m model --facts @my_facts.json --layer 20
        """,
    )
    memory_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    memory_parser.add_argument(
        "--facts",
        "-f",
        required=True,
        help="Fact type: 'multiplication', 'addition', 'capitals', 'elements', or @file.json",
    )
    memory_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to analyze (default: ~80%% of model depth)",
    )
    memory_parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top predictions to capture per query (default: 30)",
    )
    memory_parser.add_argument(
        "--output",
        "-o",
        help="Save detailed results to JSON file",
    )
    memory_parser.add_argument(
        "--save-plot",
        help="Save visualization to file (e.g., memory_structure.png)",
    )
    memory_parser.add_argument(
        "--classify",
        action="store_true",
        help="Show memorization classification (memorized/partial/weak/not memorized)",
    )
    memory_parser.set_defaults(func=lambda args: asyncio.run(introspect_memory(args)))

    # Memory-inject command - external memory injection
    memory_inject_parser = introspect_subparsers.add_parser(
        "memory-inject",
        help="External memory injection for fact retrieval",
        description="""External memory injection: Inject correct answers from an external store.

This provides circuit-guided memory externalization by:
1. Building a store of (query, value) vector pairs from known facts
2. Matching input queries to stored entries by similarity
3. Injecting retrieved values into the residual stream

Use cases:
- Override incorrect model answers
- Rescue out-of-distribution query formats
- Add new facts without fine-tuning

Examples:
    # Test on multiplication with standard query
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --query "7*8="

    # Rescue non-standard format (force injection even if below threshold)
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --query "seven times eight equals" --force

    # Multiple queries
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --queries "7*8=|6*7=|9*9="

    # Evaluate on all facts
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --evaluate
        """,
    )
    memory_inject_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    memory_inject_parser.add_argument(
        "--facts",
        "-f",
        required=True,
        help="Fact type: 'multiplication', 'addition', or @file.json",
    )
    memory_inject_parser.add_argument(
        "--query",
        "-q",
        help="Single query to test",
    )
    memory_inject_parser.add_argument(
        "--queries",
        help="Multiple queries separated by | (e.g., '7*8=|6*7=')",
    )
    memory_inject_parser.add_argument(
        "--query-layer",
        type=int,
        help="Layer for query matching (default: ~92%% of model depth)",
    )
    memory_inject_parser.add_argument(
        "--inject-layer",
        type=int,
        help="Layer to inject values (default: ~88%% of model depth)",
    )
    memory_inject_parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend factor: 0=no injection, 1=full replacement (default: 1.0)",
    )
    memory_inject_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity to use injection (default: 0.7)",
    )
    memory_inject_parser.add_argument(
        "--force",
        action="store_true",
        help="Force injection even if similarity is below threshold",
    )
    memory_inject_parser.add_argument(
        "--save-store",
        help="Save memory store to file (e.g., memory.npz)",
    )
    memory_inject_parser.add_argument(
        "--load-store",
        help="Load memory store from file",
    )
    memory_inject_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate baseline vs injected accuracy on all facts",
    )
    memory_inject_parser.set_defaults(func=lambda args: asyncio.run(introspect_memory_inject(args)))
