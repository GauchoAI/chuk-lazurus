"""Bench command parser."""

from ..commands.gym import bench_pipeline


def register_bench_parser(subparsers):
    """Register the bench subcommand."""
    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark the batching pipeline",
        description="Run comprehensive benchmarks on tokenization, batching, packing, and efficiency.",
    )
    bench_parser.add_argument(
        "-d",
        "--dataset",
        help="JSONL dataset file (optional - uses synthetic data if not provided)",
    )
    bench_parser.add_argument(
        "-t",
        "--tokenizer",
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    bench_parser.add_argument(
        "--bucket-edges",
        default="128,256,512,1024",
        help="Bucket edge lengths (comma-separated, default: 128,256,512,1024)",
    )
    bench_parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
        help="Token budget per microbatch (default: 4096)",
    )
    bench_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    bench_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process from dataset",
    )
    bench_parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (when no dataset, default: 1000)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    bench_parser.set_defaults(func=bench_pipeline)
