"""Ablation, weight-diff, activation-diff introspect parsers."""

from ...commands.introspect import (
    introspect_ablate,
    introspect_activation_diff,
    introspect_weight_diff,
)


def register_ablation_parsers(introspect_subparsers):
    """Register ablate, weight-diff, and activation-diff subcommands."""
    # Ablation command - run ablation studies
    ablation_parser = introspect_subparsers.add_parser(
        "ablate",
        help="Run ablation studies to identify causal circuits",
        description="Ablate model components to identify which layers/heads are causal for specific behaviors.",
    )
    ablation_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    ablation_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to test (required unless using --prompts)",
    )
    ablation_parser.add_argument(
        "--criterion",
        "-c",
        help="Criterion to check (e.g., 'function_call', 'sorry', 'positive', or expected text)",
    )
    ablation_parser.add_argument(
        "--component",
        choices=["mlp", "attention", "both"],
        default="mlp",
        help="Component to ablate (default: mlp)",
    )
    ablation_parser.add_argument(
        "--layers",
        help="Layers to test (comma-separated, e.g., '5,8,10,11,12'). Default: all",
    )
    ablation_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    ablation_parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum tokens to generate (default: 60)",
    )
    ablation_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show actual generated outputs (original and ablated)",
    )
    ablation_parser.add_argument(
        "--multi",
        action="store_true",
        help="Ablate all specified layers together (default: sweep each layer separately)",
    )
    ablation_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template",
    )
    ablation_parser.add_argument(
        "--prompts",
        help="Multiple prompts to test (pipe-separated, e.g., '10*10=|45*45=|47*47=')",
    )
    ablation_parser.set_defaults(func=introspect_ablate)

    # Weight divergence command - compare weights between two models
    weight_div_parser = introspect_subparsers.add_parser(
        "weight-diff",
        help="Compare weight divergence between two models",
        description="Compute per-layer, per-component weight differences between base and fine-tuned models.",
    )
    weight_div_parser.add_argument(
        "--base",
        "-b",
        required=True,
        help="Base model (HuggingFace ID or path)",
    )
    weight_div_parser.add_argument(
        "--finetuned",
        "-f",
        required=True,
        help="Fine-tuned model (HuggingFace ID or path)",
    )
    weight_div_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    weight_div_parser.set_defaults(func=introspect_weight_diff)

    # Activation divergence command - compare activations on same prompt
    activation_div_parser = introspect_subparsers.add_parser(
        "activation-diff",
        help="Compare activation divergence between two models",
        description="Run same prompts through two models and compare hidden state representations.",
    )
    activation_div_parser.add_argument(
        "--base",
        "-b",
        required=True,
        help="Base model (HuggingFace ID or path)",
    )
    activation_div_parser.add_argument(
        "--finetuned",
        "-f",
        required=True,
        help="Fine-tuned model (HuggingFace ID or path)",
    )
    activation_div_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to test (comma-separated or @file.txt)",
    )
    activation_div_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    activation_div_parser.set_defaults(func=introspect_activation_diff)
