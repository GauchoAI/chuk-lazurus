"""Patch introspect parser."""

from ...commands.introspect import introspect_patch


def register_patch_parsers(introspect_subparsers):
    """Register patch subcommand."""
    patch_parser = introspect_subparsers.add_parser(
        "patch",
        help="Perform activation patching between source and target prompts",
        description="""Activation patching: transfer activations from source to target prompt.

This is a causal intervention technique that tests whether activations from
one prompt can transfer computation to another prompt.

For example, patching activations from "7*8=" into "7+8=" at the right layer
should cause the model to output "56" instead of "15".

Examples:
    # Patch multiplication into addition
    lazarus introspect patch -m model \\
        --source "7*8=" --target "7+8="

    # Patch at specific layer
    lazarus introspect patch -m model \\
        --source "7*8=" --target "7+8=" --layer 20

    # Patch with partial blend
    lazarus introspect patch -m model \\
        --source "7*8=" --target "7+8=" --blend 0.5
        """,
    )
    patch_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    patch_parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Source prompt to patch FROM",
    )
    patch_parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target prompt to patch INTO",
    )
    patch_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Single layer to patch at",
    )
    patch_parser.add_argument(
        "--layers",
        help="Multiple layers to sweep (comma-separated, default: all key layers)",
    )
    patch_parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend factor: 0=no change, 1=full replacement (default: 1.0)",
    )
    patch_parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=10,
        help="Max tokens to generate (default: 10)",
    )
    patch_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    patch_parser.set_defaults(func=introspect_patch)
