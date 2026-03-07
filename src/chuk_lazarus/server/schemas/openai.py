"""
OpenAI-compatible wire types and translation helpers.

Reference: https://platform.openai.com/docs/api-reference/chat/create

Translation contract
--------------------
- ``ChatCompletionRequest.to_internal()``  → InternalRequest
- ``ChatCompletionResponse.from_internal()`` ← InternalResponse
- ``ChatCompletionChunk.text_chunk()`` / ``.finish_chunk()`` ← InternalChunk
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from .internal import (
    FinishReason,
    InternalChunk,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    MessageRole,
)

# ── Shared constants ──────────────────────────────────────────────────────────

OBJECT_CHAT_COMPLETION: Literal["chat.completion"] = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK: Literal["chat.completion.chunk"] = "chat.completion.chunk"
OBJECT_LIST: Literal["list"] = "list"
OBJECT_MODEL: Literal["model"] = "model"
OWNED_BY: Literal["lazarus"] = "lazarus"


# ── Role literals (OpenAI wire format) ────────────────────────────────────────

OpenAIRole = Literal["system", "user", "assistant", "tool", "function"]

_ROLE_MAP: dict[str, MessageRole] = {
    "system": MessageRole.SYSTEM,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    # unknown roles fall back to USER
}


# ── Request ───────────────────────────────────────────────────────────────────


class OpenAIMessage(BaseModel):
    """A single message in the OpenAI chat format."""

    role: OpenAIRole
    content: str
    name: str | None = None

    def to_internal(self) -> InternalMessage:
        return InternalMessage(
            role=_ROLE_MAP.get(self.role, MessageRole.USER),
            content=self.content,
        )


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""

    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = Field(None, ge=1)
    temperature: float | None = Field(None, ge=0.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(1, description="Only n=1 is supported")

    # Accepted but not used for local inference
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    def to_internal(self, default_max_tokens: int = 512) -> InternalRequest:
        stop: list[str] | None = None
        if self.stop is not None:
            stop = [self.stop] if isinstance(self.stop, str) else self.stop

        return InternalRequest(
            messages=[m.to_internal() for m in self.messages],
            model=self.model,
            max_tokens=self.max_tokens if self.max_tokens is not None else default_max_tokens,
            temperature=self.temperature if self.temperature is not None else 0.7,
            top_p=self.top_p if self.top_p is not None else 0.9,
            stream=self.stream,
            stop=stop,
        )


# ── Non-streaming response ────────────────────────────────────────────────────


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: OpenAIResponseMessage
    finish_reason: FinishReason
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion"] = OBJECT_CHAT_COMPLETION
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: OpenAIUsage

    @classmethod
    def from_internal(cls, response: InternalResponse) -> ChatCompletionResponse:
        return cls(
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    message=OpenAIResponseMessage(content=response.content),
                    finish_reason=response.finish_reason,
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )


# ── Streaming chunk ───────────────────────────────────────────────────────────


class DeltaMessage(BaseModel):
    """Delta content inside a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: FinishReason | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = OBJECT_CHAT_COMPLETION_CHUNK
    created: int
    model: str
    choices: list[StreamingChoice]

    def to_sse(self) -> str:
        """Serialize to a Server-Sent Events data line."""
        return f"data: {self.model_dump_json()}\n\n"

    @classmethod
    def role_chunk(cls, chunk_id: str, created: int, model: str) -> ChatCompletionChunk:
        """First chunk — announces the assistant role."""
        return cls(
            id=chunk_id,
            created=created,
            model=model,
            choices=[StreamingChoice(delta=DeltaMessage(role="assistant", content=""))],
        )

    @classmethod
    def text_chunk(
        cls, chunk_id: str, created: int, model: str, content: str
    ) -> ChatCompletionChunk:
        """Mid-stream chunk carrying generated text."""
        return cls(
            id=chunk_id,
            created=created,
            model=model,
            choices=[StreamingChoice(delta=DeltaMessage(content=content))],
        )

    @classmethod
    def finish_chunk(
        cls,
        chunk_id: str,
        created: int,
        model: str,
        finish_reason: FinishReason = FinishReason.STOP,
    ) -> ChatCompletionChunk:
        """Final chunk — empty delta, carries finish_reason."""
        return cls(
            id=chunk_id,
            created=created,
            model=model,
            choices=[StreamingChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        )


# ── Model list ────────────────────────────────────────────────────────────────


class OpenAIModelCard(BaseModel):
    id: str
    object: Literal["model"] = OBJECT_MODEL
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: Literal["lazarus"] = OWNED_BY


class ModelListResponse(BaseModel):
    object: Literal["list"] = OBJECT_LIST
    data: list[OpenAIModelCard]
