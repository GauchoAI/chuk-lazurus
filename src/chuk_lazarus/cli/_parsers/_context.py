"""Context command parsers."""

import asyncio

from ..commands.context import context_generate_cmd, context_prefill_cmd
from ..commands.context.calibrate_frames import context_calibrate_frames_cmd


def register_context_parsers(subparsers):
    """Register the context subcommand and its sub-subcommands."""
    context_parser = subparsers.add_parser(
        "context", help="KV context checkpoint tools (prefill / generate)"
    )
    ctx_subparsers = context_parser.add_subparsers(dest="ctx_command", help="Context commands")

    # context prefill
    ctx_prefill = ctx_subparsers.add_parser(
        "prefill", help="Prefill a document into a windowed checkpoint library"
    )
    ctx_prefill.add_argument("--model", "-m", required=True, help="Model ID or local path")
    ctx_prefill.add_argument("--input", "-i", required=True, help="Input text file to prefill")
    ctx_prefill.add_argument("--checkpoint", "-c", required=True, help="Output library directory")
    ctx_prefill.add_argument(
        "--window-size",
        type=int,
        default=None,
        dest="window_size",
        help="Tokens per window (default: 8192)",
    )
    ctx_prefill.add_argument(
        "--max-tokens",
        type=int,
        dest="max_tokens",
        help="Truncate input to at most N tokens",
    )
    ctx_prefill.add_argument(
        "--name",
        help="Human-readable library name (defaults to input filename stem)",
    )
    ctx_prefill.add_argument(
        "--no-resume",
        action="store_true",
        dest="no_resume",
        help="Ignore existing partial library and start fresh",
    )
    ctx_prefill.add_argument(
        "--residual-mode",
        choices=["interval", "full", "none", "darkspace"],
        default="interval",
        dest="residual_mode",
        help="Residual extraction: interval (8 samples), full (every position), darkspace (frame bank), none (skip)",
    )
    ctx_prefill.add_argument(
        "--frame-bank",
        dest="frame_bank",
        help="Path to frame_bank.npz (required for --residual-mode darkspace)",
    )
    ctx_prefill.add_argument(
        "--store-pages",
        action="store_true",
        dest="store_pages",
        help="Store pre-RoPE K,V pages for instant page injection at generate time",
    )
    ctx_prefill.add_argument(
        "--store-kv-full",
        action="store_true",
        dest="store_kv_full",
        help="Save full KV cache per window for Mode 6 KV injection (~9MB/window for 4B)",
    )
    ctx_prefill.add_argument(
        "--phases",
        default="all",
        help=(
            "Comma-separated phases to run: windows, interval, compass, darkspace, pages, surprise, sparse, kvectors, kvectors_full, mode7, all. "
            "E.g. --phases windows to prefill only, --phases kvectors to extract K-vector routing index "
            "(sparse/interval sampling), --phases kvectors_full for 100%% position coverage (~256KB/window), "
            "--phases mode7 to calibrate Mode 7 probes. Default: all"
        ),
    )
    ctx_prefill.add_argument(
        "--compass-layer",
        type=int,
        default=None,
        dest="compass_layer",
        help="Explicit layer for compass extraction (default: auto ~77%% depth, e.g. 29 for retrieval-layer routing)",
    )
    ctx_prefill.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "export"],
        help=(
            "Prefill mode: standard (default, writes KV checkpoints for fast replay) or "
            "export (streaming, no KV checkpoints — portable ~33 MB index for demo/distribution)"
        ),
    )
    ctx_prefill.set_defaults(func=lambda args: asyncio.run(context_prefill_cmd(args)))

    # context generate
    ctx_generate = ctx_subparsers.add_parser(
        "generate", help="Generate text (optionally from a checkpoint library)"
    )
    ctx_generate.add_argument("--model", "-m", required=True, help="Model ID or local path")
    ctx_generate.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Library directory to load (optional — omit for plain generation)",
    )
    ctx_generate.add_argument("--prompt", "-p", help="Prompt text")
    ctx_generate.add_argument(
        "--prompt-file", dest="prompt_file", help="File containing the prompt"
    )
    ctx_generate.add_argument(
        "--max-tokens", type=int, default=200, dest="max_tokens", help="Max tokens to generate"
    )
    ctx_generate.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ctx_generate.add_argument(
        "--replay",
        nargs="*",
        default=None,
        help='Window IDs to replay: "auto" (default, compass routing), "all", "last", "kv" (Mode 6 prefix caching), "sparse" (Mode 5), "vec_inject" (1D subspace injection at L30), or specific IDs (e.g. --replay 0 1 45)',
    )
    ctx_generate.add_argument(
        "--find",
        default=None,
        help="Auto-find and replay the window containing this term",
    )
    ctx_generate.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Number of windows to select (default: 3)",
    )
    ctx_generate.add_argument(
        "--strategy",
        default=None,
        choices=[
            "mode7",
            "unified",
            "bm25",
            "compass",
            "qk",
            "kv_route",
            "geometric",
            "contrastive",
            "darkspace",
            "guided",
            "directed",
            "twopass",
            "attention",
            "deflection",
            "preview",
            "hybrid",
            "iterative",
            "probe",
            "residual",
            "sparse",
            "temporal",
        ],
        help="Routing strategy: mode7 (default, auto-classifies query), unified (legacy), geometric, iterative, probe, bm25, temporal, residual (legacy)",
    )
    ctx_generate.add_argument(
        "--speculative-tokens",
        type=int,
        default=50,
        dest="speculative_tokens",
        help="Tokens to generate in Pass 1 of twopass strategy (default: 50)",
    )
    ctx_generate.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        dest="max_rounds",
        help="Max navigation rounds for iterative strategy (default: 3)",
    )
    ctx_generate.add_argument(
        "--system-prompt",
        default=None,
        dest="system_prompt",
        help="System prompt prepended to the chat template",
    )
    ctx_generate.add_argument(
        "--no-chat-template",
        action="store_true",
        dest="no_chat_template",
        help="Send prompt as raw text without chat template wrapping",
    )
    ctx_generate.add_argument(
        "--max-keywords",
        type=int,
        default=3,
        dest="max_keywords",
        help="Max keywords per window in sparse mode (default: 3 for triplet compression)",
    )
    ctx_generate.add_argument(
        "--no-framing",
        action="store_true",
        dest="no_framing",
        help="KV mode: raw [span tokens][query] prefill, no preamble/postamble",
    )
    ctx_generate.add_argument(
        "--span-radius",
        type=int,
        default=None,
        dest="span_radius",
        help="Override fact span radius (default: 5, try 15 for sentence-boundary spans)",
    )
    ctx_generate.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "fact", "broad"],
        help="KV mode: auto (BM25 score selects), fact (Mode 6), broad (Mode 7)",
    )
    ctx_generate.add_argument(
        "--token-budget",
        type=int,
        default=None,
        dest="token_budget",
        help="Max tokens for broad context mode (default: 3000)",
    )
    ctx_generate.add_argument(
        "--broad-windows",
        type=int,
        default=None,
        dest="broad_windows",
        help="Number of windows for broad mode (default: 20)",
    )
    ctx_generate.add_argument(
        "--broad-strategy",
        default=None,
        dest="broad_strategy",
        choices=["contrastive", "compass", "geometric"],
        help="Geometric strategy for broad mode (default: contrastive)",
    )
    ctx_generate.add_argument(
        "--routing-layer",
        type=int,
        default=29,
        dest="routing_layer",
        help="Layer for kv_route strategy (default: 29, the retrieval layer)",
    )
    ctx_generate.add_argument(
        "--routing-head",
        type=int,
        default=4,
        dest="routing_head",
        help="Query head for kv_route strategy (default: 4, the fact-copying head)",
    )
    ctx_generate.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        dest="confidence_threshold",
        help="Min Q·K score to trust vec_inject routing (default: 0.15; below = fallback to replay)",
    )
    ctx_generate.set_defaults(func=lambda args: asyncio.run(context_generate_cmd(args)))

    # context calibrate-frames
    ctx_calibrate = ctx_subparsers.add_parser(
        "calibrate-frames", help="Discover dark space coordinate frames for a model"
    )
    ctx_calibrate.add_argument("--model", "-m", required=True, help="Model ID or local path")
    ctx_calibrate.add_argument(
        "--output", "-o", required=True, help="Output path for frame_bank.npz"
    )
    ctx_calibrate.add_argument(
        "--method",
        choices=["whitening", "category"],
        default="whitening",
        help="Discovery method: whitening (model-driven, default) or category (human-defined)",
    )
    ctx_calibrate.add_argument(
        "--dims",
        type=int,
        default=64,
        dest="dims_per_frame",
        help="Dimensions in frame bank (default: 64)",
    )
    ctx_calibrate.add_argument(
        "--layer-frac",
        type=float,
        default=0.77,
        dest="layer_frac",
        help="Commitment layer as fraction of model depth (default: 0.77)",
    )
    ctx_calibrate.set_defaults(func=lambda args: asyncio.run(context_calibrate_frames_cmd(args)))
