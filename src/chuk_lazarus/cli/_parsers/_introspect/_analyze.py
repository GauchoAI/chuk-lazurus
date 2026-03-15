"""Analyze, compare, hooks introspect parsers."""

import asyncio

from ....commands.introspect import (
    introspect_analyze,
    introspect_compare,
    introspect_hooks,
)


def register_analyze_parsers(introspect_subparsers):
    """Register analyze, compare, and hooks subcommands."""
    # Analyze command - main logit lens analysis
    analyze_parser = introspect_subparsers.add_parser(
        "analyze",
        help="Run logit lens analysis on a prompt",
        description="Analyze how model predictions evolve across layers using logit lens.",
    )
    analyze_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    analyze_parser.add_argument(
        "--adapter",
        "-a",
        help="Path to LoRA adapter weights (for analyzing fine-tuned models)",
    )
    analyze_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to analyze (required unless --prefix is used)",
    )
    analyze_parser.add_argument(
        "--layer-strategy",
        choices=["all", "evenly_spaced", "first_last", "custom"],
        default="evenly_spaced",
        help="Layer selection strategy (default: evenly_spaced)",
    )
    analyze_parser.add_argument(
        "--layer-step",
        "-s",
        type=int,
        default=4,
        help="Step size for evenly_spaced strategy (default: 4)",
    )
    analyze_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    analyze_parser.add_argument(
        "--track",
        "-t",
        action="append",
        help="Token to track evolution (can specify multiple, e.g., --track Paris --track ' Paris')",
    )
    analyze_parser.add_argument(
        "--embedding-scale",
        type=float,
        help="Embedding scale factor (e.g., 33.94 for Gemma with hidden_size=1152)",
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    analyze_parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Capture all layers (overrides --layer-strategy)",
    )
    analyze_parser.add_argument(
        "--layers",
        "-l",
        help="Specific layers to analyze (comma-separated, e.g., '13,14,15,16,17'). Overrides --layer-strategy",
    )
    analyze_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template (for non-chat models or direct testing)",
    )
    analyze_parser.add_argument(
        "--find-answer",
        action="store_true",
        help="Generate tokens first to find where the answer starts, then analyze at that position (default for chat mode)",
    )
    analyze_parser.add_argument(
        "--no-find-answer",
        action="store_true",
        help="Disable answer position detection (analyze immediate next token)",
    )
    analyze_parser.add_argument(
        "--expected",
        help="Expected answer token(s) to find in generated output (used with --find-answer)",
    )
    analyze_parser.add_argument(
        "--gen-tokens",
        "-n",
        type=int,
        default=30,
        help="Number of tokens to generate when using --find-answer (default: 30)",
    )
    analyze_parser.add_argument(
        "--prefix",
        help="Analyze at a specific prefix (bypasses --prompt, --raw, --find-answer). Useful for testing specific positions in generated output.",
    )
    analyze_parser.add_argument(
        "--steer",
        help="Apply steering during analysis. Either 'direction.npz:coefficient' or just 'direction.npz' (use --strength for coefficient)",
    )
    analyze_parser.add_argument(
        "--steer-neuron",
        type=int,
        help="Single neuron index to steer (alternative to --steer with a direction file)",
    )
    analyze_parser.add_argument(
        "--steer-layer",
        type=int,
        help="Layer to apply neuron steering (required with --steer-neuron)",
    )
    analyze_parser.add_argument(
        "--strength",
        type=float,
        help="Steering strength/coefficient when using --steer or --steer-neuron (default: 1.0)",
    )
    analyze_parser.add_argument(
        "--inject-layer",
        type=int,
        help="Layer at which to inject a token embedding (use with --inject-token)",
    )
    analyze_parser.add_argument(
        "--inject-token",
        help="Token to inject at --inject-layer (e.g., '2491' to force that answer)",
    )
    analyze_parser.add_argument(
        "--inject-blend",
        type=float,
        default=1.0,
        help="Blend factor for injection: 0=original, 1=full replacement (default: 1.0)",
    )
    analyze_parser.add_argument(
        "--compute-override",
        choices=["arithmetic", "none"],
        default="none",
        help="Override model computation with Python at layer boundary. "
        "'arithmetic' detects A*B=, A+B=, etc and injects correct answer at --compute-layer",
    )
    analyze_parser.add_argument(
        "--compute-layer",
        type=int,
        help="Layer at which to inject computed answer (default: 80%% of model depth)",
    )
    analyze_parser.set_defaults(func=lambda args: asyncio.run(introspect_analyze(args)))

    # Compare command - compare two models
    compare_introspect_parser = introspect_subparsers.add_parser(
        "compare",
        help="Compare two models' predictions using logit lens",
        description="Compare how predictions evolve in two different models.",
    )
    compare_introspect_parser.add_argument(
        "--model1",
        "-m1",
        required=True,
        help="First model name or HuggingFace ID",
    )
    compare_introspect_parser.add_argument(
        "--model2",
        "-m2",
        required=True,
        help="Second model name or HuggingFace ID",
    )
    compare_introspect_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Prompt to analyze",
    )
    compare_introspect_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    compare_introspect_parser.add_argument(
        "--track",
        help="Tokens to track evolution (comma-separated)",
    )
    compare_introspect_parser.set_defaults(func=lambda args: asyncio.run(introspect_compare(args)))

    # Hooks command - low-level hook demonstration
    hooks_parser = introspect_subparsers.add_parser(
        "hooks",
        help="Low-level hook demonstration",
        description="Demonstrate low-level hook API for capturing intermediate states.",
    )
    hooks_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    hooks_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Prompt to analyze",
    )
    hooks_parser.add_argument(
        "--layers",
        help="Layers to capture (comma-separated, e.g., '0,4,8,12')",
    )
    hooks_parser.add_argument(
        "--capture-attention",
        action="store_true",
        help="Also capture attention weights",
    )
    hooks_parser.add_argument(
        "--last-only",
        action="store_true",
        help="Only capture last sequence position (more memory efficient)",
    )
    hooks_parser.add_argument(
        "--no-logit-lens",
        action="store_true",
        help="Skip logit lens analysis",
    )
    hooks_parser.set_defaults(func=lambda args: asyncio.run(introspect_hooks(args)))
