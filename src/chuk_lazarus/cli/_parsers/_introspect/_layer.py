"""Layer and format-sensitivity introspect parsers."""

from ....commands.introspect import (
    introspect_format_sensitivity,
    introspect_layer,
)


def register_layer_parsers(introspect_subparsers):
    """Register layer and format-sensitivity subcommands."""
    # Layer analysis command - representation similarity and clustering
    layer_parser = introspect_subparsers.add_parser(
        "layer",
        help="Analyze what specific layers do with representation similarity",
        description="Analyze representations at specific layers to understand what layers 'see'.",
    )
    layer_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    layer_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to analyze (pipe-separated or @file.txt). Example: 'prompt1|prompt2|prompt3'",
    )
    layer_parser.add_argument(
        "--labels",
        "-l",
        help="Labels for prompts (comma-separated, same order as prompts). Example: 'working,broken,working,broken'",
    )
    layer_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated, e.g., '2,4,6,8'). Default: auto (key layers)",
    )
    layer_parser.add_argument(
        "--attention",
        "-a",
        action="store_true",
        help="Also analyze attention patterns",
    )
    layer_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    layer_parser.set_defaults(func=introspect_layer)

    # Format sensitivity command - quick test for trailing space effects
    format_parser = introspect_subparsers.add_parser(
        "format-sensitivity",
        help="Quick format sensitivity check (trailing space vs no space)",
        description="Automatically test prompts with and without trailing space.",
    )
    format_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    format_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Base prompts (pipe-separated or @file.txt). Trailing space will be added/removed automatically.",
    )
    format_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated). Default: auto",
    )
    format_parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Only show summary (skip detailed output)",
    )
    format_parser.set_defaults(func=introspect_format_sensitivity)
