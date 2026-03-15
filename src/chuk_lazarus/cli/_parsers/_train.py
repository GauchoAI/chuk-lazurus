"""Train command parsers."""

import asyncio

from ...commands.train import generate_data_cmd, train_dpo_cmd, train_grpo_cmd, train_sft_cmd


def register_train_parsers(subparsers):
    """Register train and generate subcommands."""
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="train_type", help="Training type")

    # SFT training
    sft_parser = train_subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser.add_argument("--model", required=True, help="Model name or path")
    sft_parser.add_argument("--data", required=True, help="Training data path (JSONL)")
    sft_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    sft_parser.add_argument("--output", default="./checkpoints/sft", help="Output directory")
    sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_parser.add_argument("--max-steps", type=int, help="Max training steps (overrides epochs)")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    sft_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    sft_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    sft_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    sft_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    sft_parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (default: q_proj,v_proj). "
        "Options: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    sft_parser.add_argument(
        "--freeze-layers",
        help="Layers to freeze (e.g., '0-12' or '0,1,2,3'). Frozen layers are not trained.",
    )
    sft_parser.add_argument(
        "--config",
        help="YAML config file (overrides other arguments)",
    )
    sft_parser.add_argument("--mask-prompt", action="store_true", help="Mask prompt in loss")
    sft_parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    # Batching options
    sft_parser.add_argument("--batchplan", help="Use pre-computed batch plan directory")
    sft_parser.add_argument(
        "--bucket-edges",
        help="Bucket edges for length-based batching (e.g., 128,256,512)",
    )
    sft_parser.add_argument(
        "--token-budget",
        type=int,
        help="Token budget for dynamic batching (replaces --batch-size)",
    )
    sft_parser.add_argument("--pack", action="store_true", help="Enable sequence packing")
    sft_parser.add_argument("--pack-max-len", type=int, help="Max length for packed sequences")
    sft_parser.add_argument(
        "--pack-mode",
        choices=["first_fit", "best_fit", "greedy"],
        default="first_fit",
        help="Packing algorithm",
    )
    # Online training options
    sft_parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online training with gym stream",
    )
    sft_parser.add_argument(
        "--gym-host",
        default="localhost",
        help="Gym server host for online training",
    )
    sft_parser.add_argument(
        "--gym-port",
        type=int,
        default=8023,
        help="Gym server port for online training",
    )
    sft_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size for online training",
    )
    sft_parser.set_defaults(func=lambda args: asyncio.run(train_sft_cmd(args)))

    # DPO training
    dpo_parser = train_subparsers.add_parser("dpo", help="Direct Preference Optimization")
    dpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    dpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    dpo_parser.add_argument("--data", required=True, help="Preference data path (JSONL)")
    dpo_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    dpo_parser.add_argument("--output", default="./checkpoints/dpo", help="Output directory")
    dpo_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    dpo_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    dpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    dpo_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    dpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    dpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    dpo_parser.set_defaults(func=lambda args: asyncio.run(train_dpo_cmd(args)))

    # GRPO training
    grpo_parser = train_subparsers.add_parser(
        "grpo", help="Group Relative Policy Optimization (RL with verifiable rewards)"
    )
    grpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    grpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    grpo_parser.add_argument("--output", default="./checkpoints/grpo", help="Output directory")
    grpo_parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    grpo_parser.add_argument(
        "--prompts-per-iteration", type=int, default=16, help="Prompts per iteration"
    )
    grpo_parser.add_argument("--group-size", type=int, default=4, help="Responses per prompt")
    grpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    grpo_parser.add_argument("--kl-coef", type=float, default=0.1, help="KL penalty coefficient")
    grpo_parser.add_argument(
        "--max-response-length", type=int, default=256, help="Max response tokens"
    )
    grpo_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    grpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    grpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    grpo_parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (default: q_proj,v_proj)",
    )
    grpo_parser.add_argument(
        "--freeze-layers",
        help="Layers to freeze (e.g., '0-12' or '0,1,2,3')",
    )
    grpo_parser.add_argument(
        "--reward-script",
        required=True,
        help="Python script defining reward_fn(prompt, response) -> float and get_prompts() -> list[str]",
    )
    grpo_parser.add_argument(
        "--config",
        help="YAML config file (overrides other arguments)",
    )
    grpo_parser.set_defaults(func=lambda args: asyncio.run(train_grpo_cmd(args)))

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--type", required=True, choices=["math"], help="Data type")
    gen_parser.add_argument("--output", default="./data/generated", help="Output directory")
    gen_parser.add_argument("--sft-samples", type=int, default=10000, help="SFT samples")
    gen_parser.add_argument("--dpo-samples", type=int, default=5000, help="DPO samples")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.set_defaults(func=lambda args: asyncio.run(generate_data_cmd(args)))
