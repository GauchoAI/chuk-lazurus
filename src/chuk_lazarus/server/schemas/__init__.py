"""
Protocol-specific wire schemas and the internal canonical types they translate to.

Each sub-module exports:
  - Request / Response / Chunk Pydantic models  (wire format)
  - Translation helpers: .to_internal() / .from_internal()

The engine only ever sees ``internal`` types.
"""

from .internal import (
    FinishReason,
    InternalChunk,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    InternalUsage,
    MessageRole,
    StopReason,
)

__all__ = [
    "FinishReason",
    "InternalChunk",
    "InternalMessage",
    "InternalRequest",
    "InternalResponse",
    "InternalUsage",
    "MessageRole",
    "StopReason",
]
