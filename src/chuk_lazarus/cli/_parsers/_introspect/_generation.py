"""Generate and metacognitive introspect parsers."""

import asyncio

from ...commands.introspect import (
    introspect_generate,
    introspect_metacognitive,
)


def register_generation_parsers(introspect_subparsers):
    """Register generate and metacognitive subcommands."""
    # Generate command - multi-token generation to test next-token lock hypothesis
    generate_parser = introspect_subparsers.add_parser(
        "generate",
        help="Generate multiple tokens to test next-token lock hypothesis",
        description="Test whether format issues cause simple next-token lock or complex gates.",
    )
    generate_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    generate_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to test (pipe-separated or @file.txt)",
    )
    generate_parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=10,
        help="Maximum tokens to generate (default: 10)",
    )
    generate_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Temperature (0=greedy, default: 0)",
    )
    generate_parser.add_argument(
        "--compare-format",
        "-c",
        action="store_true",
        help="Auto-create with/without trailing space variants and compare",
    )
    generate_parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show individual generated tokens",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    generate_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template (for non-chat models or direct testing)",
    )
    generate_parser.set_defaults(func=lambda args: asyncio.run(introspect_generate(args)))

    # Metacognitive command - detect strategy switch
    metacog_parser = introspect_subparsers.add_parser(
        "metacognitive",
        help="Detect metacognitive strategy switch (direct vs chain-of-thought)",
        description="""Probe the model's decision layer to detect strategy selection.

At approximately 70% depth, the model's token prediction reveals its strategy:
- DIRECT: Predicts a digit → will output answer immediately
- CoT: Predicts space/word → will use chain-of-thought reasoning

This is the "metacognitive switch" - the model deciding HOW to solve, not WHAT the answer is.
        """,
    )
    metacog_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    metacog_parser.add_argument(
        "--problems",
        "-p",
        default="5 + 5 =",
        help="Arithmetic problems (pipe-separated or @file.txt). Default: '5 + 5 ='",
    )
    metacog_parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Auto-generate random arithmetic problems",
    )
    metacog_parser.add_argument(
        "--num-problems",
        "-n",
        type=int,
        default=20,
        help="Number of problems to generate (with --generate, default: 20)",
    )
    metacog_parser.add_argument(
        "--decision-layer",
        "-l",
        type=int,
        help="Layer to probe for strategy (default: ~70%% of model depth)",
    )
    metacog_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for problem generation (default: 42)",
    )
    metacog_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template",
    )
    metacog_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    metacog_parser.set_defaults(func=lambda args: asyncio.run(introspect_metacognitive(args)))
