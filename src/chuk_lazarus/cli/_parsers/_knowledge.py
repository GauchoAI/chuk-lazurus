"""Knowledge store command parsers."""

import asyncio


def register_knowledge_parsers(subparsers):
    """Register the knowledge subcommand and its sub-subcommands."""
    knowledge_parser = subparsers.add_parser(
        "knowledge",
        help="Knowledge store — build / query / chat",
    )
    kn_subparsers = knowledge_parser.add_subparsers(dest="kn_command", help="Knowledge commands")

    # knowledge build
    kn_build = kn_subparsers.add_parser(
        "build",
        help="Build knowledge store from a document",
    )
    kn_build.add_argument("--model", "-m", required=True, help="Model ID or local path")
    kn_build.add_argument("--input", "-i", required=True, help="Input text file")
    kn_build.add_argument("--output", "-o", required=True, help="Output knowledge store directory")
    kn_build.add_argument(
        "--window-size",
        type=int,
        default=512,
        dest="window_size",
        help="Tokens per window (default: 512)",
    )
    kn_build.add_argument(
        "--entries-per-window",
        type=int,
        default=8,
        dest="entries_per_window",
        help="Injection entries per window (default: 8)",
    )
    kn_build.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        help="Truncate input to at most N tokens",
    )
    kn_build.set_defaults(func=lambda args: asyncio.run(_run_build(args)))

    # knowledge query
    kn_query = kn_subparsers.add_parser(
        "query",
        help="Query a knowledge store with persistent injection",
    )
    kn_query.add_argument("--model", "-m", required=True, help="Model ID or local path")
    kn_query.add_argument("--store", "-s", required=True, help="Knowledge store directory")
    kn_query.add_argument("--prompt", "-p", required=True, help="Query text")
    kn_query.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        dest="max_tokens",
        help="Max tokens to generate (default: 80)",
    )
    kn_query.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    kn_query.set_defaults(func=lambda args: asyncio.run(_run_query(args)))

    # knowledge chat
    kn_chat = kn_subparsers.add_parser(
        "chat",
        help="Interactive chat with a knowledge store",
    )
    kn_chat.add_argument("--model", "-m", required=True, help="Model ID or local path")
    kn_chat.add_argument("--store", "-s", required=True, help="Knowledge store directory")
    kn_chat.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        dest="max_tokens",
        help="Max tokens to generate per turn (default: 80)",
    )
    kn_chat.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    kn_chat.set_defaults(func=lambda args: asyncio.run(_run_chat(args)))


async def _run_build(args):
    from ..commands.knowledge import knowledge_build_cmd

    await knowledge_build_cmd(args)


async def _run_query(args):
    from ..commands.knowledge import knowledge_query_cmd

    await knowledge_query_cmd(args)


async def _run_chat(args):
    from ..commands.knowledge import knowledge_chat_cmd

    await knowledge_chat_cmd(args)
