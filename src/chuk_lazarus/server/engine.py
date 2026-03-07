"""
ModelEngine — format-agnostic inference engine.

Wraps a loaded ``UnifiedPipeline`` and exposes async generation against
internal canonical types.  Protocol routers never touch the pipeline directly.

Design rules
------------
- Public surface is fully async.
- ``_generate()`` is the private sync implementation — runs in a thread pool
  so it never blocks the event loop.
- ``astream()`` bridges the synchronous ``generate_stream`` generator to an
  async generator via an ``asyncio.Queue``.
- Thread-safe: a threading.Lock guards model access for non-streaming calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, AsyncGenerator

from .schemas.internal import (
    FinishReason,
    InternalChunk,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    InternalUsage,
    MessageRole,
    StopReason,
)

if TYPE_CHECKING:
    from chuk_lazarus.inference import UnifiedPipeline

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_chat_history(messages: list[InternalMessage]):
    """Convert internal messages to a ``ChatHistory`` instance."""
    from chuk_lazarus.inference.chat import ChatHistory

    history = ChatHistory()
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            history.add_system(msg.content)
        elif msg.role == MessageRole.USER:
            history.add_user(msg.content)
        elif msg.role == MessageRole.ASSISTANT:
            history.add_assistant(msg.content)
    return history


def _map_stop_reason(raw: str) -> FinishReason:
    """Map GenerationResult.stop_reason to FinishReason."""
    if raw in (StopReason.EOS, StopReason.STOP_TOKEN, "eos", "stop_token", "stop"):
        return FinishReason.STOP
    return FinishReason.LENGTH


# ── Engine ────────────────────────────────────────────────────────────────────


class ModelEngine:
    """
    Holds a single loaded model and serves concurrent inference requests.

    Usage::

        engine = await ModelEngine.load("google/gemma-3-1b-it")
        response = await engine.agenerate(request)

        async for chunk in engine.astream(request):
            ...
    """

    def __init__(self, pipeline: UnifiedPipeline, model_id: str) -> None:
        self._pipeline = pipeline
        self._model_id = model_id
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def load(cls, model_id: str, verbose: bool = True) -> ModelEngine:
        """Async factory — loads the model in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: cls._load_sync(model_id, verbose))

    @classmethod
    def _load_sync(cls, model_id: str, verbose: bool) -> ModelEngine:
        from chuk_lazarus.inference import UnifiedPipeline

        pipeline = UnifiedPipeline.from_pretrained(model_id, verbose=verbose)
        return cls(pipeline, model_id)

    # ── Private sync generation ───────────────────────────────────────────────

    def _generate(self, request: InternalRequest) -> InternalResponse:
        """Blocking generation — must be called from a thread pool."""
        history = _to_chat_history(request.messages)

        with self._lock:
            result = self._pipeline.chat_with_history(
                history,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )

        usage = InternalUsage(
            prompt_tokens=result.stats.input_tokens,
            completion_tokens=result.stats.output_tokens,
            total_tokens=result.stats.input_tokens + result.stats.output_tokens,
        )

        return InternalResponse(
            content=result.text,
            model=self._model_id,
            finish_reason=_map_stop_reason(result.stop_reason),
            usage=usage,
        )

    # ── Public async generation ───────────────────────────────────────────────

    async def agenerate(self, request: InternalRequest) -> InternalResponse:
        """Non-streaming generation (async, runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate(request))

    async def astream(
        self, request: InternalRequest
    ) -> AsyncGenerator[InternalChunk, None]:
        """
        Streaming generation — yields InternalChunk instances.

        Bridges the synchronous ``generate_stream`` generator to an async
        generator via an ``asyncio.Queue``.  The last yielded chunk always
        has a non-None ``finish_reason``.
        """
        from chuk_lazarus.inference.chat import format_history
        from chuk_lazarus.inference.generation import GenerationConfig, generate_stream

        history = _to_chat_history(request.messages)
        prompt = format_history(self._pipeline.tokenizer, history)

        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[InternalChunk | BaseException | None] = asyncio.Queue()

        def _run_sync() -> None:
            try:
                for text in generate_stream(
                    self._pipeline.model,
                    self._pipeline.tokenizer,
                    prompt,
                    config,
                ):
                    loop.call_soon_threadsafe(
                        queue.put_nowait, InternalChunk(content=text)
                    )
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_run_sync, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                yield InternalChunk(content="", finish_reason=FinishReason.STOP)
                return
            if isinstance(item, BaseException):
                raise item
            yield item
