"""Data command parsers."""

from ..commands.data import (
    data_batch_generate,
    data_batching_analyze,
    data_batching_histogram,
    data_batching_suggest,
    data_batchplan_build,
    data_batchplan_info,
    data_batchplan_shard,
    data_batchplan_verify,
    data_lengths_build,
    data_lengths_stats,
)


def register_data_parsers(subparsers):
    """Register the data subcommand and all sub-subcommands."""
    data_parser = subparsers.add_parser("data", help="Data processing utilities")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data commands")

    # === Lengths subcommands ===
    lengths_parser = data_subparsers.add_parser("lengths", help="Length cache utilities")
    lengths_subparsers = lengths_parser.add_subparsers(
        dest="lengths_command", help="Lengths commands"
    )

    # Build length cache
    lengths_build_parser = lengths_subparsers.add_parser(
        "build", help="Build length cache from dataset"
    )
    lengths_build_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset file (JSONL or JSON)"
    )
    lengths_build_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    lengths_build_parser.add_argument(
        "--output", "-o", required=True, help="Output cache file path"
    )
    lengths_build_parser.set_defaults(func=data_lengths_build)

    # Length cache stats
    lengths_stats_parser = lengths_subparsers.add_parser(
        "stats", help="Show length cache statistics"
    )
    lengths_stats_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    lengths_stats_parser.set_defaults(func=data_lengths_stats)

    # === BatchPlan subcommands ===
    batchplan_parser = data_subparsers.add_parser("batchplan", help="Batch plan utilities")
    batchplan_subparsers = batchplan_parser.add_subparsers(
        dest="batchplan_command", help="BatchPlan commands"
    )

    # Build batch plan
    batchplan_build_parser = batchplan_subparsers.add_parser(
        "build", help="Build batch plan from length cache"
    )
    batchplan_build_parser.add_argument("--lengths", "-l", required=True, help="Length cache file")
    batchplan_build_parser.add_argument(
        "--epochs", "-e", type=int, default=1, help="Number of epochs (default: 1)"
    )
    batchplan_build_parser.add_argument(
        "--token-budget",
        "-b",
        type=int,
        default=4096,
        help="Token budget per batch (default: 4096)",
    )
    batchplan_build_parser.add_argument(
        "--bucket-edges",
        default="128,256,512",
        help="Bucket edges (comma-separated, default: 128,256,512)",
    )
    batchplan_build_parser.add_argument(
        "--overflow-max",
        type=int,
        default=2048,
        help="Max length for overflow bucket (default: 2048)",
    )
    batchplan_build_parser.add_argument(
        "--predictable",
        "-p",
        action="store_true",
        help="Use predictable mode (deterministic batching)",
    )
    batchplan_build_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for predictable mode (default: 42)",
    )
    batchplan_build_parser.add_argument("--dataset-hash", help="Dataset hash for fingerprinting")
    batchplan_build_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for batch plan"
    )
    batchplan_build_parser.set_defaults(func=data_batchplan_build)

    # Batch plan info
    batchplan_info_parser = batchplan_subparsers.add_parser(
        "info", help="Show batch plan information"
    )
    batchplan_info_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batchplan_info_parser.add_argument(
        "--show-batches",
        "-n",
        type=int,
        default=0,
        help="Number of sample batches to show",
    )
    batchplan_info_parser.add_argument(
        "--rank",
        "-r",
        type=int,
        default=None,
        help="Worker rank for sharded view (0-indexed)",
    )
    batchplan_info_parser.add_argument(
        "--world-size", "-w", type=int, default=None, help="Total number of workers"
    )
    batchplan_info_parser.set_defaults(func=data_batchplan_info)

    # Batch plan verify
    batchplan_verify_parser = batchplan_subparsers.add_parser(
        "verify", help="Verify batch plan reproducibility"
    )
    batchplan_verify_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batchplan_verify_parser.add_argument("--lengths", "-l", required=True, help="Length cache file")
    batchplan_verify_parser.set_defaults(func=data_batchplan_verify)

    # Batch plan shard
    batchplan_shard_parser = batchplan_subparsers.add_parser(
        "shard", help="Create sharded batch plans for distributed training"
    )
    batchplan_shard_parser.add_argument(
        "--plan", "-p", required=True, help="Source batch plan directory"
    )
    batchplan_shard_parser.add_argument(
        "--world-size",
        "-w",
        type=int,
        required=True,
        help="Number of distributed workers",
    )
    batchplan_shard_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for sharded plans"
    )
    batchplan_shard_parser.set_defaults(func=data_batchplan_shard)

    # === Batching analysis subcommands ===
    batching_parser = data_subparsers.add_parser("batching", help="Batching analysis utilities")
    batching_subparsers = batching_parser.add_subparsers(
        dest="batching_command", help="Batching commands"
    )

    # Analyze batching efficiency
    batching_analyze_parser = batching_subparsers.add_parser(
        "analyze", help="Analyze batching efficiency"
    )
    batching_analyze_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_analyze_parser.add_argument(
        "--bucket-edges",
        default="128,256,512",
        help="Bucket edges to analyze (comma-separated, default: 128,256,512)",
    )
    batching_analyze_parser.add_argument(
        "--overflow-max",
        type=int,
        default=2048,
        help="Max length for overflow bucket (default: 2048)",
    )
    batching_analyze_parser.add_argument("--output", "-o", help="Save JSON report to file")
    batching_analyze_parser.set_defaults(func=data_batching_analyze)

    # Length histogram
    batching_histogram_parser = batching_subparsers.add_parser(
        "histogram", help="Display length histogram"
    )
    batching_histogram_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_histogram_parser.add_argument(
        "--bins", type=int, default=15, help="Number of histogram bins (default: 15)"
    )
    batching_histogram_parser.add_argument(
        "--width", type=int, default=50, help="Chart width (default: 50)"
    )
    batching_histogram_parser.set_defaults(func=data_batching_histogram)

    # Suggest bucket edges
    batching_suggest_parser = batching_subparsers.add_parser(
        "suggest", help="Suggest optimal bucket edges"
    )
    batching_suggest_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_suggest_parser.add_argument(
        "--num-buckets",
        "-n",
        type=int,
        default=4,
        help="Number of buckets (default: 4)",
    )
    batching_suggest_parser.add_argument(
        "--goal",
        "-g",
        choices=["waste", "balance", "memory"],
        default="waste",
        help="Optimization goal: waste (minimize padding), balance (even bucket sizes), memory (power-of-2 edges)",
    )
    batching_suggest_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    batching_suggest_parser.set_defaults(func=data_batching_suggest)

    # === Batch generation subcommands ===
    batch_parser = data_subparsers.add_parser("batch", help="Batch file generation")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", help="Batch commands")

    # Generate NPZ batch files
    batch_generate_parser = batch_subparsers.add_parser(
        "generate", help="Generate NPZ batch files from BatchPlan"
    )
    batch_generate_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batch_generate_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset file (JSONL or JSON)"
    )
    batch_generate_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    batch_generate_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for NPZ files"
    )
    batch_generate_parser.set_defaults(func=data_batch_generate)
