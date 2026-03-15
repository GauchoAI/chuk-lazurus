"""Classifier and logit-lens introspect parsers."""

import asyncio

from ...commands.introspect import (
    introspect_classifier,
    introspect_logit_lens,
)


def register_classifier_parsers(introspect_subparsers):
    """Register classifier and logit-lens subcommands."""
    # Multi-class classifier probe
    classifier_parser = introspect_subparsers.add_parser(
        "classifier",
        help="Train multi-class linear probe for operation classification",
        description="""Train logistic regression probes at each layer to find where
the model distinguishes between multiple operation types (e.g., multiply, add, subtract, divide).

Example:
  lazarus introspect classifier -m meta-llama/Llama-3.2-1B \\
    --classes "multiply:7*8=|12*5=" \\
    --classes "add:23+45=|17+38=" \\
    --classes "subtract:50-23=|89-34=" \\
    --classes "divide:48/6=|81/9=" \\
    --test "11*12=|11+12=|15-6=|12/4="
        """,
    )
    classifier_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    classifier_parser.add_argument(
        "--classes",
        "-c",
        action="append",
        required=True,
        help="Class definition in format 'label:prompt1|prompt2|...' (can specify multiple)",
    )
    classifier_parser.add_argument(
        "--test",
        "-t",
        help="Test prompts to classify (pipe-separated or @file.txt)",
    )
    classifier_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    classifier_parser.set_defaults(func=lambda args: asyncio.run(introspect_classifier(args)))

    # Logit lens analysis
    logit_lens_parser = introspect_subparsers.add_parser(
        "logit-lens",
        help="Apply logit lens to check vocabulary-mappable classifiers",
        description="""Project hidden states at specified layer through the unembedding
matrix to see which vocabulary tokens emerge. Useful for checking if classifiers
project to specific tokens (e.g., 'multiply', 'add').

Example:
  lazarus introspect logit-lens -m meta-llama/Llama-3.2-1B \\
    --prompts "7*8=|23+45=|50-23=|48/6=" \\
    --layer 8 \\
    --targets "multiply|add|subtract|divide"
        """,
    )
    logit_lens_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    logit_lens_parser.add_argument(
        "--adapter",
        "-a",
        help="Path to LoRA adapter directory (for analyzing fine-tuned models)",
    )
    logit_lens_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to analyze (pipe-separated or @file.txt)",
    )
    logit_lens_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to analyze (default: 55%% depth)",
    )
    logit_lens_parser.add_argument(
        "--targets",
        "-t",
        action="append",
        help="Target tokens to track probability (can specify multiple)",
    )
    logit_lens_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    logit_lens_parser.set_defaults(func=lambda args: asyncio.run(introspect_logit_lens(args)))
