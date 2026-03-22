"""
Main CLI entry point for chuk-lazarus.

Usage:
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data/train.jsonl
    lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences.jsonl
    lazarus train grpo --model ./checkpoints/sft/final --reward-script ./reward.py
    lazarus generate --type math --output ./data/lazarus_math
    lazarus infer --model ./checkpoints/dpo/final --prompt "Calculate 2+2"

Data Commands:
    lazarus data lengths build -d train.jsonl -t gpt2 -o lengths.jsonl
    lazarus data lengths stats -c lengths.jsonl
    lazarus data batchplan build -l lengths.jsonl -e 3 -b 4096 -o batch_plan/
    lazarus data batchplan info -p batch_plan/
    lazarus data batchplan info -p batch_plan/ --rank 0 --world-size 4  # sharded view
    lazarus data batchplan verify -p batch_plan/ -l lengths.jsonl
    lazarus data batchplan shard -p batch_plan/ -w 4 -o shards/  # distributed sharding

Batching Analysis Commands:
    lazarus data batching analyze -c lengths.jsonl --bucket-edges 128,256,512
    lazarus data batching histogram -c lengths.jsonl --bins 20
    lazarus data batching suggest -c lengths.jsonl --goal waste --num-buckets 4
    lazarus data batch generate -p batch_plan/ -d train.jsonl -t gpt2 -o batches/

Tokenizer Commands:
    lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
    lazarus tokenizer decode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ids "1,2,3"
    lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"
    lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test" --verbose
    lazarus tokenizer doctor -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t model --save fingerprint.json
    lazarus tokenizer fingerprint -t model --verify fingerprint.json --strict
    lazarus tokenizer benchmark -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer benchmark -t model --samples 5000 --compare
    lazarus tokenizer analyze coverage -t model --file corpus.txt
    lazarus tokenizer analyze entropy -t model --file corpus.txt
    lazarus tokenizer analyze fit-score -t model --file corpus.txt
    lazarus tokenizer analyze efficiency -t model --file corpus.txt
    lazarus tokenizer analyze vocab-suggest -t model --file corpus.txt
    lazarus tokenizer curriculum length-buckets -t model --file corpus.txt
    lazarus tokenizer curriculum reasoning-density -t model --file corpus.txt
    lazarus tokenizer training throughput -t model --file corpus.txt
    lazarus tokenizer training pack -t model --file corpus.txt --max-length 512
    lazarus tokenizer regression run -t model --tests tests.yaml
    lazarus tokenizer research soft-tokens -n 10 -d 768 --prefix task
    lazarus tokenizer research analyze-embeddings -f embeddings.json --cluster
    lazarus tokenizer research morph -f embeddings.json -s 0 -t 1 --method spherical
    lazarus tokenizer instrument histogram -t model --file corpus.txt
    lazarus tokenizer instrument oov -t model --file corpus.txt
    lazarus tokenizer instrument waste -t model --file corpus.txt --max-length 512
    lazarus tokenizer instrument vocab-diff -t1 model1 -t2 model2 --file corpus.txt

Gym Commands (Online Learning):
    lazarus gym run -t gpt2 --mock --num-episodes 10  # Test with mock stream
    lazarus gym run -t gpt2 --host localhost --port 8023  # Connect to puzzle arcade
    lazarus gym run -t gpt2 --mock --output buffer.json  # Save samples to buffer
    lazarus gym info  # Display gym stream configuration

Benchmark Commands:
    lazarus bench  # Run benchmark with synthetic data
    lazarus bench -d train.jsonl -t gpt2  # Benchmark with real dataset
    lazarus bench --bucket-edges 128,256,512 --token-budget 4096  # Custom config

Introspection Commands (Model Analysis):
    lazarus introspect analyze -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p "The capital of France is"
    lazarus introspect analyze -m model -p "Hello" --track "world,there" --layer-strategy all
    lazarus introspect compare -m1 model1 -m2 model2 -p "The answer is" --track "42"
    lazarus introspect hooks -m model -p "Test" --layers 0,4,8 --capture-attention

Ablation Study Commands (Causal Circuit Discovery):
    lazarus introspect ablate -m model -p "prompt" -c function_call --component mlp
    lazarus introspect ablate -m model -p "prompt" -c sorry --layers 5,8,10,11,12
    lazarus introspect weight-diff -b base_model -f finetuned_model -o diff.json
    lazarus introspect activation-diff -b base -f finetuned -p "prompt1,prompt2"

Activation Steering Commands:
    lazarus introspect steer -m model --extract --positive "good prompt" --negative "bad prompt" -o direction.npz
    lazarus introspect steer -m model -d direction.npz -p "test prompt" -c 1.0
    lazarus introspect steer -m model -d direction.npz -p "test prompt" --compare "-2,-1,0,1,2"

Circuit Analysis Commands:
    lazarus introspect arithmetic -m model --quick  # Test arithmetic emergence across layers
    lazarus introspect uncertainty -m model -p "100 - 37 = |100 - 37 ="  # Predict confidence

Training with Batching:
    lazarus train sft --model model --data train.jsonl --batchplan batch_plan/
    lazarus train sft --model model --data train.jsonl --bucket-edges 128,256,512 --token-budget 4096
    lazarus train sft --model model --data train.jsonl --pack --pack-max-len 2048
    lazarus train sft --model model --data train.jsonl --online --gym-host localhost --gym-port 8023
"""

import argparse
import logging
import sys

from ._parsers import (
    register_bench_parser,
    register_context_parsers,
    register_data_parsers,
    register_experiment_parsers,
    register_gym_parsers,
    register_infer_parser,
    register_introspect_parsers,
    register_knowledge_parsers,
    register_tokenizer_parsers,
    register_train_parsers,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def app():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="chuk-lazarus: MLX-based LLM training framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train SFT
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl

    # Train DPO
    lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

    # Generate training data
    lazarus generate --type math --output ./data/lazarus

    # Run inference
    lazarus infer --model ./checkpoints/dpo/final --prompt "What is 2+2?"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register all parser groups
    register_train_parsers(subparsers)
    register_infer_parser(subparsers)
    register_context_parsers(subparsers)
    register_tokenizer_parsers(subparsers)
    register_data_parsers(subparsers)
    register_gym_parsers(subparsers)
    register_experiment_parsers(subparsers)
    register_bench_parser(subparsers)
    register_introspect_parsers(subparsers)
    register_knowledge_parsers(subparsers)

    # Serve subcommand — optional (requires chuk-lazarus[server])
    try:
        from chuk_lazarus.server.cli import add_serve_parser

        add_serve_parser(subparsers)
    except ImportError:
        pass  # server deps not installed; serve command silently unavailable

    return parser


def main():
    """Main entry point."""
    parser = app()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    elif args.command == "train" and args.train_type is None:
        parser.parse_args(["train", "--help"])
    elif args.command == "tokenizer" and args.tok_command is None:
        parser.parse_args(["tokenizer", "--help"])
    elif args.command == "gym" and getattr(args, "gym_command", None) is None:
        parser.parse_args(["gym", "--help"])
    elif args.command == "introspect" and getattr(args, "introspect_command", None) is None:
        parser.parse_args(["introspect", "--help"])
    elif args.command == "context" and getattr(args, "ctx_command", None) is None:
        parser.parse_args(["context", "--help"])
    elif args.command == "experiment" and getattr(args, "exp_command", None) is None:
        parser.parse_args(["experiment", "--help"])
    elif args.command == "serve":
        if not hasattr(args, "func"):
            print(
                "ERROR: Server dependencies not installed.\n"
                "Install with: uv add 'chuk-lazarus[server]'",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
