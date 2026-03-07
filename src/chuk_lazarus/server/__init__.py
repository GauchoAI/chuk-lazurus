"""
Lazarus inference server.

Provides an OpenAI-compatible (and extensible) HTTP API for local LLM inference.

Quick start::

    from chuk_lazarus.server.engine import ModelEngine
    from chuk_lazarus.server.app import create_app, Protocol

    engine = await ModelEngine.load("google/gemma-3-1b-it")
    app = create_app(engine, protocols=[Protocol.OPENAI])

Or via CLI::

    lazarus serve --model google/gemma-3-1b-it --port 8080
    lazarus-serve --model google/gemma-3-1b-it --port 8080
"""

from .app import Protocol, create_app
from .engine import ModelEngine

__all__ = [
    "ModelEngine",
    "Protocol",
    "create_app",
]
