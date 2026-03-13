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
        async for chunk in client.stream_chat(model=..., messages=[...]):
            print(chunk, end="", flush=True)

Design rules
------------
- All public-facing types are Pydantic models — no raw dicts passed by callers.
- Wire serialisation uses Pydantic ``.model_dump()`` / ``.model_validate()``.
- Roles and finish reasons are enums everywhere.
- SSE stream parsing uses Pydantic model validation, not dict key access.
- All magic strings extracted to module-level constants.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from enum import Enum

from pydantic import BaseModel

# ── Constants ─────────────────────────────────────────────────────────────────

_SSE_PREFIX = "data: "
_SSE_DONE = "data: [DONE]"
_CONTENT_TYPE_JSON = "application/json"
_AUTH_HEADER = "Authorization"
_CONTENT_TYPE_HEADER = "Content-Type"
_BEARER_PREFIX = "Bearer "
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_MODELS_PATH = "/v1/models"
_HEALTH_PATH = "/health"


# ── Enumerations ──────────────────────────────────────────────────────────────


class ClientRole(str, Enum):
    """Role values accepted by the client."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ClientFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"


# ── Public request / response types ──────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single chat turn used when calling the client."""

    role: ClientRole
    content: str

    def to_wire(self) -> _WireMessage:
        """Convert to the wire format the server expects."""
        return _WireMessage(role=self.role.value, content=self.content)


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


class HealthResponse(BaseModel):
    status: str
    model: str
    protocols: list[str]


# ── Internal wire types (private — not part of the public API) ────────────────


class _WireMessage(BaseModel):
    """Wire format for a single chat message."""

    role: str
    content: str


class _ChatRequest(BaseModel):
    """Wire format for POST /v1/chat/completions."""

    model: str
    messages: list[_WireMessage]
    stream: bool
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None


class _WireUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class _WireResponseMessage(BaseModel):
    role: str
    content: str


class _WireChoice(BaseModel):
    index: int = 0
    message: _WireResponseMessage
    finish_reason: str | None = None


class _WireChatResponse(BaseModel):
    id: str
    model: str
    choices: list[_WireChoice]
    usage: _WireUsage | None = None


class _WireDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class _WireStreamChoice(BaseModel):
    index: int = 0
    delta: _WireDelta
    finish_reason: str | None = None


class _WireStreamChunk(BaseModel):
    id: str
    model: str
    choices: list[_WireStreamChoice]


class _WireModelCard(BaseModel):
    id: str
    owned_by: str = "lazarus"


class _WireModelList(BaseModel):
    data: list[_WireModelCard]


# ── Shared translation helpers ────────────────────────────────────────────────


def _build_request(
    model: str,
    messages: list[ChatMessage],
    *,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    stream: bool,
    stop: str | list[str] | None,
) -> _ChatRequest:
    return _ChatRequest(
        model=model,
        messages=[m.to_wire() for m in messages],
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )


def _parse_response(raw: dict) -> ChatResponse:
    wire = _WireChatResponse.model_validate(raw)
    choice = wire.choices[0]
    usage = wire.usage or _WireUsage()
    return ChatResponse(
        id=wire.id,
        model=wire.model,
        content=choice.message.content,
        finish_reason=ClientFinishReason(choice.finish_reason or ClientFinishReason.STOP.value),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _parse_stream_line(line: str) -> str | None:
    """Parse one SSE line and return the text delta, or None to skip."""
    if not line or line == _SSE_DONE:
        return None
    if not line.startswith(_SSE_PREFIX):
        return None
    wire = _WireStreamChunk.model_validate_json(line[len(_SSE_PREFIX) :])
    if not wire.choices:
        return None
    return wire.choices[0].delta.content  # may be None if role-only chunk


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

        headers: dict[str, str] = {_CONTENT_TYPE_HEADER: _CONTENT_TYPE_JSON}
        if api_key:
            headers[_AUTH_HEADER] = f"{_BEARER_PREFIX}{api_key}"

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
        body = _build_request(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            stop=stop,
        )
        resp = self._client.post(
            _CHAT_COMPLETIONS_PATH,
            json=body.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return _parse_response(resp.json())

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
        body = _build_request(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
        )
        with self._client.stream(
            "POST",
            _CHAT_COMPLETIONS_PATH,
            json=body.model_dump(exclude_none=True),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                content = _parse_stream_line(line)
                if content:
                    yield content

    def list_models(self) -> ModelList:
        """Return the list of models served by this instance."""
        resp = self._client.get(_MODELS_PATH)
        resp.raise_for_status()
        wire = _WireModelList.model_validate(resp.json())
        return ModelList(data=[ModelInfo(id=m.id, owned_by=m.owned_by) for m in wire.data])

    def health(self) -> HealthResponse:
        """Check server health."""
        resp = self._client.get(_HEALTH_PATH)
        resp.raise_for_status()
        return HealthResponse.model_validate(resp.json())


# ── Async client ──────────────────────────────────────────────────────────────


class AsyncLazarusClient:
    """
    Async client for the Lazarus inference server.

    Supports use as an async context manager::

        async with AsyncLazarusClient(base_url="http://localhost:8080") as client:
            response = await client.chat(model=..., messages=[...])
            async for chunk in client.stream_chat(model=..., messages=[...]):
                print(chunk, end="", flush=True)
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

        headers: dict[str, str] = {_CONTENT_TYPE_HEADER: _CONTENT_TYPE_JSON}
        if api_key:
            headers[_AUTH_HEADER] = f"{_BEARER_PREFIX}{api_key}"

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
        body = _build_request(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            stop=stop,
        )
        resp = await self._client.post(
            _CHAT_COMPLETIONS_PATH,
            json=body.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return _parse_response(resp.json())

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding text delta strings."""
        body = _build_request(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
        )
        async with self._client.stream(
            "POST",
            _CHAT_COMPLETIONS_PATH,
            json=body.model_dump(exclude_none=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                content = _parse_stream_line(line)
                if content:
                    yield content

    async def list_models(self) -> ModelList:
        resp = await self._client.get(_MODELS_PATH)
        resp.raise_for_status()
        wire = _WireModelList.model_validate(resp.json())
        return ModelList(data=[ModelInfo(id=m.id, owned_by=m.owned_by) for m in wire.data])

    async def health(self) -> HealthResponse:
        resp = await self._client.get(_HEALTH_PATH)
        resp.raise_for_status()
        return HealthResponse.model_validate(resp.json())
