"""Infer command parser."""

import asyncio

from ..commands.infer import run_inference


def register_infer_parser(subparsers):
    """Register the infer subcommand."""
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", required=True, help="Model name or path")
    infer_parser.add_argument("--adapter", help="LoRA adapter path")
    infer_parser.add_argument("--prompt", help="Single prompt")
    infer_parser.add_argument("--prompt-file", help="File with prompts (one per line)")
    infer_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    infer_parser.add_argument(
        "--chat", action="store_true", help="Use chat template (for chat models)"
    )
    infer_parser.add_argument("--system", help="System prompt (only used with --chat)")
    infer_parser.add_argument(
        "--engine",
        choices=["standard", "kv_direct"],
        default="standard",
        help="Inference engine (default: standard)",
    )
    infer_parser.set_defaults(func=lambda args: asyncio.run(run_inference(args)))
