"""Embedding, early-layers, commutativity introspect parsers."""

from ...commands.introspect import (
    introspect_commutativity,
    introspect_early_layers,
    introspect_embedding,
)


def register_embedding_parsers(introspect_subparsers):
    """Register embedding, early-layers, and commutativity subcommands."""
    # Embedding command - analyze what's encoded at embedding level
    embedding_parser = introspect_subparsers.add_parser(
        "embedding",
        help="Analyze what information is encoded at embedding level vs after layers",
        description="""Test the RLVF backprop hypothesis: does task information exist in raw embeddings?

Tests:
1. Task type detection (arithmetic vs language) from embeddings
2. Operation type detection (mult vs add) from embeddings
3. Answer correlation with embeddings vs after layers

If task type is 100% detectable from embeddings, this suggests RLVF gradients
backpropagate all the way to the embedding layer.

Examples:
    # Test embedding analysis
    lazarus introspect embedding -m model

    # Test with specific operation
    lazarus introspect embedding -m model --operation mult

    # Analyze specific layers
    lazarus introspect embedding -m model --layers 0,1,2,4
        """,
    )
    embedding_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    embedding_parser.add_argument(
        "--operation",
        choices=["mult", "add", "all", "*", "+"],
        help="Operation type to test (default: all)",
    )
    embedding_parser.add_argument(
        "--layers",
        help="Layers to compare against embeddings (comma-separated, default: 0,1,2)",
    )
    embedding_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    embedding_parser.set_defaults(func=introspect_embedding)

    # Commutativity command - test if representations respect A*B = B*A
    commutativity_parser = introspect_subparsers.add_parser(
        "commutativity",
        help="Test if internal representations respect commutativity (A*B = B*A)",
        description="""Test commutativity in internal representations.

For multiplication, A*B and B*A should produce the same answer. This test checks
whether the internal representations for commutative pairs are similar, which
would indicate a lookup table structure rather than an algorithm.

High commutativity similarity (>0.99) suggests the model memorizes individual facts
rather than computing them algorithmically.

Examples:
    # Test all commutative pairs (2-9)
    lazarus introspect commutativity -m model

    # Test specific pairs
    lazarus introspect commutativity -m model \\
        --pairs "2*3,3*2|7*8,8*7|4*5,5*4"

    # Analyze at specific layer
    lazarus introspect commutativity -m model --layer 20
        """,
    )
    commutativity_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    commutativity_parser.add_argument(
        "--pairs",
        help="Explicit commutative pairs to test (e.g., '2*3,3*2|7*8,8*7')",
    )
    commutativity_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to analyze (default: ~60%% of model depth)",
    )
    commutativity_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    commutativity_parser.set_defaults(func=introspect_commutativity)

    # Early layers command - analyze what information is encoded in early layers
    early_layers_parser = introspect_subparsers.add_parser(
        "early-layers",
        help="Analyze what information is encoded in early layers",
        description="""Analyze early layer information encoding using linear probes.

This command reveals how information is organized in early transformer layers:
- Cross-expression similarity at the '=' position
- Linear probe extraction of operation type, operands, and answer
- The "orthogonal subspaces paradox": high similarity but separable information

Key insight: Even when cosine similarity is high (0.997), information can be
linearly extracted because it's encoded in orthogonal directions.

Examples:
    # Basic analysis with default settings
    lazarus introspect early-layers -m model

    # Analyze specific layers
    lazarus introspect early-layers -m model --layers 0,1,2,4,8

    # Include position-wise analysis
    lazarus introspect early-layers -m model --analyze-positions

    # Test specific operations
    lazarus introspect early-layers -m model --operations "*,+,-"
        """,
    )
    early_layers_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    early_layers_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated, default: 0,1,2,4,8,12)",
    )
    early_layers_parser.add_argument(
        "--operations",
        help="Operations to test (comma-separated, default: *,+)",
    )
    early_layers_parser.add_argument(
        "--digits",
        help="Digit range for operands (e.g., 2-8, default: 2-8)",
    )
    early_layers_parser.add_argument(
        "--analyze-positions",
        action="store_true",
        help="Include position-wise analysis (slower but more detailed)",
    )
    early_layers_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    early_layers_parser.set_defaults(func=introspect_early_layers)
