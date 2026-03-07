"""
LazarusClient — Python client library for the Lazarus inference server.

Provides sync and async clients that mirror the openai SDK shape closely
enough that mcp-cli and other OpenAI-compatible tooling work out of the box.
Uses ``httpx`` directly — no dependency on the openai package.

Usage (sync)::

    from chuk_lazarus.client import LazarusClient

    client = LazarusClient(base_url="http://localhost:8080", api_key="secret")

    # Single-turn chat
    response = client.chat(
        model="google/gemma-3-1b-it",
        messages=[ChatMessage(role=ClientRole.USER, content="Hello!")],
    )
    print(response.content)

    # Streaming
    for chunk in client.stream_chat(model=..., messages=[...]):
        print(chunk, end="", flush=True)

Usage (async)::

    from chuk_lazarus.client import AsyncLazarusClient

    async with AsyncLazarusClient(base_url="http://localhost:8080") as client:
        response = await client.chat(model=..., messages=[...])

Design rules
------------
- All public-facing types are Pydantic models — no raw dicts passed by callers.
- Roles and finish reasons are enums.
- Sync client uses ``httpx.Client``; async uses ``httpx.AsyncClient``.
- Both clients share the same request/response Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterator

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────


class ClientRole(str, Enum):
    """Role values accepted by the client."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ClientFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"


# ── Request / Response types ──────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single chat turn used when calling the client."""

    role: ClientRole
    content: str

    def _to_wire(self) -> dict:
        """Serialise to the wire dict that the server expects."""
        return {"role": self.role.value, "content": self.content}


class ChatResponse(BaseModel):
    """Result from a non-streaming chat call."""

    id: str
    model: str
    content: str
    finish_reason: ClientFinishReason
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelInfo(BaseModel):
    """A model entry from /v1/models."""

    id: str
    owned_by: str = "lazarus"


class ModelList(BaseModel):
    data: list[ModelInfo]


# ── Shared request builder ────────────────────────────────────────────────────


def _build_request_body(
    model: str,
    messages: list[ChatMessage],
    *,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    stream: bool,
    stop: str | list[str] | None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [m._to_wire() for m in messages],
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if stop is not None:
        body["stop"] = stop
    return body


def _parse_chat_response(data: dict) -> ChatResponse:
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return ChatResponse(
        id=data["id"],
        model=data["model"],
        content=choice["message"]["content"],
        finish_reason=ClientFinishReason(choice.get("finish_reason") or "stop"),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


# ── Sync client ───────────────────────────────────────────────────────────────


class LazarusClient:
    """
    Synchronous client for the Lazarus inference server.

    Parameters
    ----------
    base_url:
        Server base URL, e.g. ``http://localhost:8080``.
    api_key:
        Bearer token.  Pass ``None`` if auth is not enabled on the server.
    timeout:
        Request timeout in seconds (default: 120).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for LazarusClient. "
                "Install with: pip install 'chuk-lazarus[server]'"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    def __enter__(self) -> LazarusClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ── API ───────────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the full response."""
        body = _build_request_body(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            stop=stop,
        )
        resp = self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return _parse_chat_response(resp.json())

    def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> Iterator[str]:
        """Stream a chat completion, yielding text delta strings."""
        import json

        body = _build_request_body(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
        )
        with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    choices = payload.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

    def list_models(self) -> ModelList:
        """Return the list of models served by this instance."""
        resp = self._client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return ModelList(
            data=[ModelInfo(id=m["id"], owned_by=m.get("owned_by", "lazarus"))
                  for m in data.get("data", [])]
        )

    def health(self) -> dict:
        """Check server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()


# ── Async client ──────────────────────────────────────────────────────────────


class AsyncLazarusClient:
    """
    Async client for the Lazarus inference server.

    Supports use as an async context manager::

        async with AsyncLazarusClient(base_url="http://localhost:8080") as client:
            response = await client.chat(model=..., messages=[...])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for AsyncLazarusClient. "
                "Install with: pip install 'chuk-lazarus[server]'"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    async def __aenter__(self) -> AsyncLazarusClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── API ───────────────────────────────────────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the full response."""
        body = _build_request_body(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            stop=stop,
        )
        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return _parse_chat_response(resp.json())

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ):
        """Stream a chat completion, yielding text delta strings."""
        import json

        body = _build_request_body(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
        )
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    choices = payload.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

    async def list_models(self) -> ModelList:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return ModelList(
            data=[ModelInfo(id=m["id"], owned_by=m.get("owned_by", "lazarus"))
                  for m in data.get("data", [])]
        )

    async def health(self) -> dict:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()
