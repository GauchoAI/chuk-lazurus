"""Arithmetic and uncertainty introspect parsers."""

import asyncio

from ....commands.introspect import (
    introspect_arithmetic,
    introspect_uncertainty,
)


def register_arithmetic_parsers(introspect_subparsers):
    """Register arithmetic and uncertainty subcommands."""
    # Arithmetic command - systematic arithmetic circuit study
    arithmetic_parser = introspect_subparsers.add_parser(
        "arithmetic",
        help="Run systematic arithmetic study to find emergence layers",
        description="""Test arithmetic problems of varying difficulty and track when
answers first emerge as top predictions across layers.

This reveals where computation happens in the model:
- Easy problems may emerge early
- Hard problems may require more layers
- Multiplication may use different circuits than addition
        """,
    )
    arithmetic_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    arithmetic_parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick mode (subset of tests)",
    )
    arithmetic_parser.add_argument(
        "--easy-only",
        action="store_true",
        help="Only run easy problems (1-digit)",
    )
    arithmetic_parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only run hard problems (3-digit)",
    )
    arithmetic_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template",
    )
    arithmetic_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    arithmetic_parser.set_defaults(func=introspect_arithmetic)

    # Uncertainty command - detect model confidence before generation
    uncertainty_parser = introspect_subparsers.add_parser(
        "uncertainty",
        help="Detect model uncertainty using hidden state geometry",
        description="""Predict model confidence before generation by analyzing
hidden state geometry at a key layer.

Uses distance to "compute center" vs "refusal center" to predict
whether the model will produce a confident answer or show uncertainty.

Key insight: Working prompts cluster in one region of hidden space,
broken/uncertain prompts cluster in another. The distance ratio
predicts output behavior.
        """,
    )
    uncertainty_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    uncertainty_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to test (pipe-separated or @file.txt)",
    )
    uncertainty_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Detection layer (default: ~70%% of model depth)",
    )
    uncertainty_parser.add_argument(
        "--working",
        "-w",
        help="Comma-separated working examples for calibration (default: arithmetic with trailing space)",
    )
    uncertainty_parser.add_argument(
        "--broken",
        "-b",
        help="Comma-separated broken examples for calibration (default: arithmetic without trailing space)",
    )
    uncertainty_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    uncertainty_parser.set_defaults(func=lambda args: asyncio.run(introspect_uncertainty(args)))
