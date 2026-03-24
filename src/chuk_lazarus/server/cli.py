"""
CLI entry points for the Lazarus inference server.

Two entry paths
---------------
1. ``lazarus serve`` — integrated into the main lazarus CLI.
   Uses ``add_serve_parser()`` to register the subcommand, then
   ``run_serve_cmd()`` as the handler.

2. ``lazarus-serve`` — standalone script registered in pyproject.toml.
   Calls ``main()`` directly with flat argparse args.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# ── Shared serve logic ────────────────────────────────────────────────────────


def _parse_protocols(raw: str) -> list:
    """Parse a comma-separated protocol string into Protocol enum values."""
    from .app import Protocol

    names = [p.strip().lower() for p in raw.split(",") if p.strip()]
    protocols: list[Protocol] = []
    for name in names:
        try:
            protocols.append(Protocol(name))
        except ValueError:
            valid = ", ".join(p.value for p in Protocol)
            print(f"ERROR: Unknown protocol '{name}'. Valid options: {valid}", file=sys.stderr)
            sys.exit(1)
    return protocols


async def _serve_async(args: argparse.Namespace) -> None:
    """Load the model and start the server (async)."""
    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: Server dependencies not installed.\n"
            "Install with:  pip install 'chuk-lazarus[server]'\n"
            "           or:  uv add 'chuk-lazarus[server]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import create_app
    from .engine import ModelEngine

    protocols = _parse_protocols(args.protocols)
    api_key: str | None = getattr(args, "api_key", None)
    max_tokens: int = getattr(args, "max_tokens", 512)

    print(f"\nLoading model: {args.model}")
    print("=" * 60)
    engine = await ModelEngine.load(args.model, verbose=True)

    app = create_app(
        engine,
        protocols=protocols,
        api_key=api_key,
        default_max_tokens=max_tokens,
    )

    host: str = args.host
    port: int = args.port

    print("\n" + "=" * 60)
    print("Lazarus inference server ready")
    print(f"  Model     : {args.model}")
    print(f"  Protocols : {', '.join(p.value for p in protocols)}")
    print(f"  Base URL  : http://{host}:{port}")
    if "openai" in [p.value for p in protocols]:
        print(f"  OpenAI URL: http://{host}:{port}/v1")
    if api_key:
        print("  Auth      : Bearer token enabled")
    print("=" * 60 + "\n")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def run_serve_cmd(args: argparse.Namespace) -> None:
    """Synchronous handler registered with the main CLI (args.func)."""
    asyncio.run(_serve_async(args))


# ── Subparser registration (for main CLI integration) ─────────────────────────


def add_serve_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``serve`` subcommand on an existing argparse subparsers."""
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the OpenAI-compatible inference server",
    )
    _add_serve_args(serve_parser)
    serve_parser.set_defaults(func=run_serve_cmd)
    return serve_parser


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    """Add all serve arguments to an argparse parser."""
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model HuggingFace ID or local path",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port (default: 8080)",
    )
    parser.add_argument(
        "--protocols",
        default="openai",
        help="Comma-separated protocols to enable: openai,ollama,anthropic (default: openai)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Optional bearer token — if set, all requests must include Authorization: Bearer <key>",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=512,
        help="Default max_tokens when callers do not specify (default: 512)",
    )


# ── Standalone entry point ────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``lazarus-serve`` standalone script."""
    parser = argparse.ArgumentParser(
        prog="lazarus-serve",
        description="Lazarus inference server — OpenAI-compatible local LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lazarus-serve --model google/gemma-3-1b-it
  lazarus-serve --model ./checkpoints/my-model --port 8080 --api-key secret
  lazarus-serve --model gemma-3-1b --protocols openai,ollama

mcp-cli usage:
  mcp-cli --provider openai --base-url http://localhost:8080/v1 --model <model-id>
        """,
    )
    _add_serve_args(parser)
    args = parser.parse_args()
    asyncio.run(_serve_async(args))
