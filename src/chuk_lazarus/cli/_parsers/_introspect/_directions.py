"""Directions and operand-directions introspect parsers."""

from ....commands.introspect import (
    introspect_directions,
    introspect_operand_directions,
)


def register_directions_parsers(introspect_subparsers):
    """Register directions and operand-directions subcommands."""
    # Directions command - compare multiple direction vectors for orthogonality
    directions_parser = introspect_subparsers.add_parser(
        "directions",
        help="Compare direction vectors for orthogonality",
        description="""Compare multiple saved direction vectors to check if they are orthogonal.

This confirms whether extracted dimensions (e.g., difficulty, operation, format)
represent truly independent features in activation space.

Example:
    lazarus introspect directions \\
        difficulty.npz uncertainty.npz format.npz operation.npz
        """,
    )
    directions_parser.add_argument(
        "files",
        nargs="+",
        help="Direction files to compare (.npz format from 'introspect probe --save-direction')",
    )
    directions_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Cosine similarity threshold for 'orthogonal' (default: 0.1)",
    )
    directions_parser.add_argument(
        "--output",
        "-o",
        help="Save similarity matrix to JSON file",
    )
    directions_parser.set_defaults(func=introspect_directions)

    # Operand-directions command - analyze how operands are encoded
    operand_directions_parser = introspect_subparsers.add_parser(
        "operand-directions",
        help="Analyze how operands A and B are encoded in activation space",
        description="""Extract operand directions (A_d and B_d) to analyze encoding structure.

This is useful for understanding if a model uses:
- Compositional encoding (like GPT-OSS): A and B in separate orthogonal subspaces
- Holistic encoding (like Gemma): entire expression encoded together

Examples:
    # Analyze multiplication operand encoding
    lazarus introspect operand-directions -m model \\
        --digits 2,3,4,5,6,7,8,9 --operation "*" --layers 8,16,20,24

    # Save directions for later analysis
    lazarus introspect operand-directions -m model \\
        --output operand_dirs.npz
        """,
    )
    operand_directions_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    operand_directions_parser.add_argument(
        "--digits",
        help="Digits to use (comma-separated, default: 2,3,4,5,6,7,8,9)",
    )
    operand_directions_parser.add_argument(
        "--operation",
        default="*",
        help="Operation to test (default: '*')",
    )
    operand_directions_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated, default: auto key layers)",
    )
    operand_directions_parser.add_argument(
        "--output",
        "-o",
        help="Save results to file (.json or .npz)",
    )
    operand_directions_parser.set_defaults(func=introspect_operand_directions)
