"""Virtual-expert and moe-expert introspect parsers."""

import argparse
import asyncio

from ...commands.introspect import (
    introspect_moe_expert,
    introspect_virtual_expert,
)


def register_moe_parsers(introspect_subparsers):
    """Register virtual-expert and moe-expert subcommands."""
    # Virtual Expert command - add virtual experts to models
    virtual_expert_parser = introspect_subparsers.add_parser(
        "virtual-expert",
        help="Add virtual expert (tool) capabilities to models",
        description="""Virtual Expert System - route to external tools via MoE routing.

For MoE models (like GPT-OSS), intercepts actual router decisions.
For dense models (like LLaMA), creates virtual routing in activation space.

CoT Rewriting:
  By default, assumes the model is CoT-trained and can generate VirtualExpertAction
  format directly. For non-CoT-trained models, use --use-few-shot-rewriter to enable
  FewShotCoTRewriter which uses in-context learning to normalize queries to action format.

Actions:
  analyze   - Analyze which experts activate for different prompt categories (MoE only)
  solve     - Solve a single problem with virtual expert
  benchmark - Run benchmark comparing model vs virtual expert
  compare   - Compare model-only vs virtual expert on a prompt
  interactive - Interactive REPL mode

Examples:
    # Solve with CoT-trained model (no rewriter needed)
    lazarus introspect virtual-expert solve -m my-cot-model -p "127 * 89 = "

    # Solve with non-CoT-trained model (use few-shot rewriter)
    lazarus introspect virtual-expert solve -m mlx-community/SmolLM-135M-fp16 \\
        -p "What is 256 times 4?" --use-few-shot-rewriter

    # Run benchmark with few-shot rewriter
    lazarus introspect virtual-expert benchmark -m model --use-few-shot-rewriter

    # Compare approaches
    lazarus introspect virtual-expert compare -m model -p "127 * 89 = "

    # Interactive mode
    lazarus introspect virtual-expert interactive -m model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    virtual_expert_parser.add_argument(
        "action",
        nargs="?",
        choices=["analyze", "solve", "benchmark", "compare", "interactive"],
        default="solve",
        help="Action to perform (default: solve)",
    )
    virtual_expert_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    virtual_expert_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to solve/compare (required for solve/compare)",
    )
    virtual_expert_parser.add_argument(
        "--problems",
        help="Problems for benchmark (pipe-separated or @file.txt)",
    )
    virtual_expert_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    virtual_expert_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed routing decisions (layer-by-layer trace)",
    )
    virtual_expert_parser.add_argument(
        "--use-few-shot-rewriter",
        action="store_true",
        dest="use_few_shot_rewriter",
        help="Use FewShotCoTRewriter to normalize queries (for non-CoT-trained models)",
    )
    virtual_expert_parser.set_defaults(
        func=lambda args: asyncio.run(introspect_virtual_expert(args))
    )

    # MoE Expert command - direct expert manipulation
    moe_expert_parser = introspect_subparsers.add_parser(
        "moe-expert",
        help="Direct manipulation of MoE expert routing",
        description="""MoE Expert Explorer - Analyze how MoE models route tokens to experts.

Actions:
  explore       - Interactive REPL for real-time expert analysis (default)
  analyze       - Identify expert specializations across all categories
  chat          - Force all routing through a single expert
  compare       - Compare outputs from multiple specific experts
  ablate        - Remove an expert from routing (see what breaks)
  weights       - Show router weights for a prompt
  trace         - Trace expert assignments across layers
  heatmap       - Generate routing heatmap visualization
  full-taxonomy  - Semantic trigram pattern analysis across categories
  domain-test    - Demonstrate that domain experts don't exist
  token-routing  - Demonstrate that single token routing is context-dependent
  context-test   - Test context independence of routing
  context-window   - Test how much context the router uses (trigram vs attention)
  attention-routing - Analyze how attention patterns drive expert routing
  attention-pattern - Show attention weights for a specific position

MoE Type Detection & Compression:
  moe-type-analyze   - Detect pseudo vs native MoE (is it compressible?)
  moe-type-compare   - Compare MoE types between two models
  moe-overlay-compute  - Compute overlay representation (base + low-rank deltas)
  moe-overlay-verify   - Verify reconstruction accuracy (<1% error)
  moe-overlay-estimate - Estimate storage savings for full model
  moe-overlay-compress - Compress model to overlay format (saves to disk)

Quick Start:
    # Interactive explorer (recommended starting point)
    lazarus introspect moe-expert explore -m openai/gpt-oss-20b

Examples:
    # Prove domain experts don't exist (7+ experts handle ALL domains)
    lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b

    # Show same token routes to different experts based on context
    lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b

    # Full semantic trigram taxonomy analysis
    lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b

    # Generate routing heatmap visualization
    lazarus introspect moe-expert heatmap -m openai/gpt-oss-20b -p "def fibonacci(n):"

    # Chat with Expert 6 (force all tokens through it)
    lazarus introspect moe-expert chat -m openai/gpt-oss-20b --expert 6 -p "127 * 89 = "

    # Kill an expert and see what breaks
    lazarus introspect moe-expert ablate -m openai/gpt-oss-20b --expert 6 -p "127 * 89 = " --benchmark

MoE Compression Examples:
    # Detect if model is compressible (pseudo-MoE vs native-MoE)
    lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

    # Compare two models
    lazarus introspect moe-expert moe-type-compare -m openai/gpt-oss-20b -c allenai/OLMoE-1B-7B-0924

    # Full compression pipeline
    lazarus introspect moe-expert moe-overlay-compute -m openai/gpt-oss-20b
    lazarus introspect moe-expert moe-overlay-verify -m openai/gpt-oss-20b
    lazarus introspect moe-expert moe-overlay-estimate -m openai/gpt-oss-20b

    # Actually compress model to disk (36GB -> ~7GB)
    lazarus introspect moe-expert moe-overlay-compress -m openai/gpt-oss-20b
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    moe_expert_parser.add_argument(
        "action",
        nargs="?",
        choices=[
            # Interactive
            "explore",
            # Core analysis
            "analyze",
            "chat",
            "compare",
            "ablate",
            # Routing visualization
            "weights",
            "trace",
            "heatmap",
            # Semantic trigram methodology
            "full-taxonomy",
            "domain-test",
            "token-routing",
            "context-test",
            "context-window",
            "attention-routing",
            "attention-pattern",
            # MoE type detection
            "moe-type-analyze",
            "moe-type-compare",
            # MoE compression
            "moe-overlay-compute",
            "moe-overlay-verify",
            "moe-overlay-estimate",
            "moe-overlay-compress",
            # Expert dynamics analysis
            "cold-experts",
            "generation-dynamics",
            "expert-circuits",
            "expert-interference",
            "expert-merging",
            "attention-prediction",
            "task-prediction",
            "routing-manipulation",
            "context-attention-routing",
        ],
        default="explore",
        help="Action to perform (default: explore)",
    )
    moe_expert_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID (must be MoE model)",
    )
    moe_expert_parser.add_argument(
        "--expert",
        "-e",
        type=int,
        help="Expert index for chat/ablate (0-based)",
    )
    moe_expert_parser.add_argument(
        "--experts",
        help="Expert indices for compare (comma-separated, e.g., '6,7,20')",
    )
    moe_expert_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to test",
    )
    moe_expert_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run ablation benchmark on multiple problems",
    )
    moe_expert_parser.add_argument(
        "--layer",
        type=int,
        help="Target MoE layer for analysis (default: middle layer)",
    )
    moe_expert_parser.add_argument(
        "--layers",
        help="Layers to analyze for trace (comma-separated or 'all')",
    )
    moe_expert_parser.add_argument(
        "--examples",
        type=int,
        default=4,
        help="Number of example prompts to show per pattern (default: 4)",
    )
    moe_expert_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    moe_expert_parser.add_argument(
        "--token",
        "-t",
        help="Target token for context-test (e.g., 'the', 'def', '127')",
    )
    moe_expert_parser.add_argument(
        "--contexts",
        help="Comma-separated contexts to test (e.g., 'the cat,the dog,under the bridge')",
    )
    # Arguments for heatmap action
    moe_expert_parser.add_argument(
        "--prompts",
        nargs="+",
        help="Multiple prompts for heatmap (e.g., --prompts 'Hello' 'World')",
    )
    moe_expert_parser.add_argument(
        "--show-weights",
        action="store_true",
        help="For heatmap: show raw weight values in addition to expert indices",
    )
    # Arguments for full-taxonomy action
    moe_expert_parser.add_argument(
        "--categories",
        help="Comma-separated categories for full-taxonomy (e.g., 'arithmetic,code,analogy')",
    )
    moe_expert_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output (e.g., expert specializations for full-taxonomy)",
    )
    # Arguments for moe-type-compare action
    moe_expert_parser.add_argument(
        "--compare-model",
        "-c",
        help="Second model for moe-type-compare action",
    )
    # Arguments for moe-type-analyze action
    moe_expert_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show expert orthogonality heatmap visualization (for moe-type-analyze)",
    )
    # Arguments for moe-overlay-* actions (compression)
    moe_expert_parser.add_argument(
        "--quality",
        "-q",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Compression quality preset: fast (~12x), balanced (~8x, default), quality (~5x)",
    )
    moe_expert_parser.add_argument(
        "--gate-rank",
        type=int,
        help="Override gate projection rank (advanced)",
    )
    moe_expert_parser.add_argument(
        "--up-rank",
        type=int,
        help="Override up projection rank (advanced)",
    )
    moe_expert_parser.add_argument(
        "--down-rank",
        type=int,
        help="Override down projection rank (advanced)",
    )
    moe_expert_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Start fresh instead of resuming from checkpoint (for moe-overlay-compress)",
    )
    moe_expert_parser.set_defaults(func=introspect_moe_expert)
