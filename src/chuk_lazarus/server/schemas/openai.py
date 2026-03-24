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
from typing import Any, Literal

from pydantic import BaseModel, Field

from .internal import (
    FinishReason,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    MessageRole,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunctionDef,
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
    "tool": MessageRole.TOOL,
}


# ── Tool definition wire types ────────────────────────────────────────────────


class OpenAIFunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None  # JSON Schema

    def to_internal(self) -> ToolFunctionDef:
        return ToolFunctionDef(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionDef

    def to_internal(self) -> Tool:
        return Tool(type=self.type, function=self.function.to_internal())


# ── Tool call wire types (in responses) ───────────────────────────────────────


class OpenAIToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction

    @classmethod
    def from_internal(cls, tc: ToolCall) -> OpenAIToolCall:
        return cls(
            id=tc.id,
            function=OpenAIToolCallFunction(
                name=tc.function.name,
                arguments=tc.function.arguments,
            ),
        )


# ── Request ───────────────────────────────────────────────────────────────────


class OpenAIMessage(BaseModel):
    """A single message in the OpenAI chat format."""

    role: OpenAIRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None

    def to_internal(self) -> InternalMessage:
        internal_tool_calls: list[ToolCall] | None = None
        if self.tool_calls:
            internal_tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in self.tool_calls
            ]
        return InternalMessage(
            role=_ROLE_MAP.get(self.role, MessageRole.USER),
            content=self.content,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_calls=internal_tool_calls,
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
    tools: list[OpenAITool] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific function
    n: int = Field(1, description="Only n=1 is supported")

    # Accepted but not used for local inference
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    def to_internal(self, default_max_tokens: int = 512) -> InternalRequest:
        stop: list[str] | None = None
        if self.stop is not None:
            stop = [self.stop] if isinstance(self.stop, str) else self.stop

        internal_tools: list[Tool] | None = None
        if self.tools:
            internal_tools = [t.to_internal() for t in self.tools]

        return InternalRequest(
            messages=[m.to_internal() for m in self.messages],
            model=self.model,
            max_tokens=self.max_tokens if self.max_tokens is not None else default_max_tokens,
            temperature=self.temperature if self.temperature is not None else 0.7,
            top_p=self.top_p if self.top_p is not None else 0.9,
            stream=self.stream,
            stop=stop,
            tools=internal_tools,
        )


# ── Non-streaming response ────────────────────────────────────────────────────


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


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
        tool_calls = (
            [OpenAIToolCall.from_internal(tc) for tc in response.tool_calls]
            if response.tool_calls
            else None
        )
        return cls(
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    message=OpenAIResponseMessage(
                        content=response.content,
                        tool_calls=tool_calls,
                    ),
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


class ToolCallDelta(BaseModel):
    """Incremental tool call delta for streaming."""

    index: int = 0
    id: str | None = None
    type: Literal["function"] | None = None
    function: OpenAIToolCallFunction | None = None


class DeltaMessage(BaseModel):
    """Delta content inside a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


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
    def tool_call_chunk(
        cls,
        chunk_id: str,
        created: int,
        model: str,
        tool_calls: list[ToolCall],
    ) -> ChatCompletionChunk:
        """Chunk announcing tool calls (no text content)."""
        deltas = [
            ToolCallDelta(
                index=i,
                id=tc.id,
                type="function",
                function=OpenAIToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for i, tc in enumerate(tool_calls)
        ]
        return cls(
            id=chunk_id,
            created=created,
            model=model,
            choices=[
                StreamingChoice(
                    delta=DeltaMessage(tool_calls=deltas),
                    finish_reason=FinishReason.TOOL_CALLS,
                )
            ],
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
