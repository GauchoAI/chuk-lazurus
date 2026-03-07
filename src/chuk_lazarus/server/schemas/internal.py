"""
Internal canonical types for the Lazarus inference server.

All protocol routers translate *to* these types before touching the engine,
and translate *from* these types before writing a response.  The engine
itself is completely unaware of any wire format.

Design rules
------------
- No raw dicts.  Every value has a named type.
- All constrained strings are enums or Literals.
- Pydantic models only — no dataclasses or NamedTuples.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    """Role of a chat participant."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(str, Enum):
    """Why the model stopped generating (internal vocabulary)."""

    EOS = "eos"          # hit an EOS / stop token
    STOP_TOKEN = "stop"  # hit a caller-supplied stop string
    MAX_TOKENS = "length"  # hit the token limit


class FinishReason(str, Enum):
    """OpenAI-compatible finish reason (used in responses)."""

    STOP = "stop"
    LENGTH = "length"


# ── Message ───────────────────────────────────────────────────────────────────


class InternalMessage(BaseModel):
    """A single turn in the conversation."""

    role: MessageRole
    content: str


# ── Request ───────────────────────────────────────────────────────────────────


class InternalRequest(BaseModel):
    """Everything the engine needs to run a generation."""

    messages: list[InternalMessage]
    model: str
    max_tokens: int = Field(512, ge=1)
    temperature: float = Field(0.7, ge=0.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None


# ── Response ──────────────────────────────────────────────────────────────────


class InternalUsage(BaseModel):
    """Token accounting for a completed generation."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class InternalResponse(BaseModel):
    """Complete (non-streaming) generation result."""

    content: str
    model: str
    finish_reason: FinishReason
    usage: InternalUsage


# ── Streaming chunk ───────────────────────────────────────────────────────────


class InternalChunk(BaseModel):
    """One incremental piece of a streaming generation.

    ``finish_reason`` is ``None`` for every chunk except the last.
    The last chunk has ``content=""`` and a non-None ``finish_reason``.
    """

    content: str
    finish_reason: FinishReason | None = None
