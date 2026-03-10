"""
OpenAI-compatible API router.

Mounted at ``/v1`` by the app factory.

Endpoints
---------
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions   (streaming + non-streaming, tool calling)

Design rules
------------
- All handlers are ``async def``.
- Engine access via ``request.app.state.engine`` (injected by app factory).
- No raw dicts — all request/response types are Pydantic models.
- Streaming uses ``StreamingResponse`` + an async SSE generator.
- Tool calls detected by the engine are forwarded in OpenAI tool_call format.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..engine import ModelEngine
from ..schemas.internal import FinishReason, InternalRequest
from ..schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
    OpenAIModelCard,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI"])

_SSE_DONE = "data: [DONE]\n\n"
_STREAMING_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ── Dependency ────────────────────────────────────────────────────────────────


def _engine(request: Request) -> ModelEngine:
    """Return the ModelEngine from app state."""
    return request.app.state.engine


# ── Model listing ─────────────────────────────────────────────────────────────


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    engine = _engine(request)
    return ModelListResponse(data=[OpenAIModelCard(id=engine.model_id)])


@router.get("/models/{model_id}", response_model=OpenAIModelCard)
async def get_model(model_id: str, request: Request) -> OpenAIModelCard:
    engine = _engine(request)
    slug = engine.model_id.split("/")[-1]
    if model_id not in (engine.model_id, slug):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )
    return OpenAIModelCard(id=engine.model_id)


# ── Chat completions ──────────────────────────────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    engine = _engine(request)
    default_max_tokens: int = getattr(request.app.state, "default_max_tokens", 512)
    internal_req = body.to_internal(default_max_tokens=default_max_tokens)

    if body.stream:
        return StreamingResponse(
            _stream_sse(engine, internal_req, body.model),
            media_type="text/event-stream",
            headers=_STREAMING_HEADERS,
        )

    response = await engine.agenerate(internal_req)
    return ChatCompletionResponse.from_internal(response)


async def _stream_sse(
    engine: ModelEngine,
    internal_req: InternalRequest,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted text for a streaming chat completion.

    Text chunks are streamed as they arrive.  If the final chunk carries
    ``tool_calls``, a tool-call delta is emitted before ``[DONE]``.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Opening chunk — announces the assistant role
    yield ChatCompletionChunk.role_chunk(chunk_id, created, model_name).to_sse()

    try:
        async for chunk in engine.astream(internal_req):
            if chunk.finish_reason is not None:
                if chunk.tool_calls:
                    yield ChatCompletionChunk.tool_call_chunk(
                        chunk_id, created, model_name, chunk.tool_calls
                    ).to_sse()
                else:
                    yield ChatCompletionChunk.finish_chunk(
                        chunk_id, created, model_name, chunk.finish_reason
                    ).to_sse()
            elif chunk.content:
                yield ChatCompletionChunk.text_chunk(
                    chunk_id, created, model_name, chunk.content
                ).to_sse()
    except Exception:
        logger.exception("Error during streaming generation")
        yield ChatCompletionChunk.finish_chunk(
            chunk_id, created, model_name, FinishReason.STOP
        ).to_sse()

    yield _SSE_DONE
