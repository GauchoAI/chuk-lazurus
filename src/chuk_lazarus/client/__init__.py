"""
Lazarus client library.

Provides sync and async Python clients for the Lazarus inference server.

Usage::

    from chuk_lazarus.client import LazarusClient, AsyncLazarusClient, ChatMessage, ClientRole

    # Sync
    with LazarusClient(base_url="http://localhost:8080", api_key="secret") as client:
        response = client.chat(
            model="google/gemma-3-1b-it",
            messages=[ChatMessage(role=ClientRole.USER, content="Hello!")],
        )
        print(response.content)

    # Async
    async with AsyncLazarusClient(base_url="http://localhost:8080") as client:
        async for chunk in client.stream_chat(model=..., messages=[...]):
            print(chunk, end="", flush=True)
"""

from .client import (
    AsyncLazarusClient,
    ChatMessage,
    ChatResponse,
    ClientFinishReason,
    ClientRole,
    HealthResponse,
    LazarusClient,
    ModelInfo,
    ModelList,
)

__all__ = [
    "AsyncLazarusClient",
    "ChatMessage",
    "ChatResponse",
    "ClientFinishReason",
    "ClientRole",
    "HealthResponse",
    "LazarusClient",
    "ModelInfo",
    "ModelList",
]
