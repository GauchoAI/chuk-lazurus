"""
FastAPI application factory for the Lazarus inference server.

Usage::

    from chuk_lazarus.server.engine import ModelEngine
    from chuk_lazarus.server.app import create_app

    engine = await ModelEngine.load("google/gemma-3-1b-it")
    app = create_app(engine, protocols=[Protocol.OPENAI])

Design rules
------------
- ``create_app`` is a plain sync factory (not async) — FastAPI itself is sync.
- Engine and config are stored on ``app.state``; routers read from there.
- Optional bearer-token auth via a middleware — off by default.
- CORS is open by default (local inference server pattern).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from .engine import ModelEngine

logger = logging.getLogger(__name__)


# ── Protocol enum ─────────────────────────────────────────────────────────────


class Protocol(str, Enum):
    """Supported wire protocols."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


# ── Error response schema ─────────────────────────────────────────────────────


class ErrorDetail(str, Enum):
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    INTERNAL = "internal_error"


def _error_response(message: str, detail: ErrorDetail, http_status: int) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={"error": {"message": message, "type": detail.value}},
    )


# ── Auth middleware ────────────────────────────────────────────────────────────


def _make_auth_middleware(api_key: str):
    """Return a Starlette middleware callable that checks the Bearer token."""

    async def auth_middleware(request: Request, call_next):
        # Health and docs endpoints are always public
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _error_response(
                "Missing or malformed Authorization header",
                ErrorDetail.UNAUTHORIZED,
                status.HTTP_401_UNAUTHORIZED,
            )
        token = auth_header.removeprefix("Bearer ").strip()
        if token != api_key:
            return _error_response(
                "Invalid API key",
                ErrorDetail.UNAUTHORIZED,
                status.HTTP_401_UNAUTHORIZED,
            )
        return await call_next(request)

    return auth_middleware


# ── Factory ───────────────────────────────────────────────────────────────────


def create_app(
    engine: ModelEngine,
    protocols: list[Protocol] | None = None,
    api_key: str | None = None,
    default_max_tokens: int = 512,
) -> FastAPI:
    """
    Build and return a configured FastAPI application.

    Parameters
    ----------
    engine:
        A loaded ``ModelEngine`` instance.
    protocols:
        Protocols to mount.  Defaults to ``[Protocol.OPENAI]``.
    api_key:
        If provided, all requests (except /health) must include
        ``Authorization: Bearer <api_key>``.
    default_max_tokens:
        Fallback ``max_tokens`` when the caller does not specify one.
    """
    protocols = protocols or [Protocol.OPENAI]

    app = FastAPI(
        title="Lazarus Inference Server",
        description="OpenAI-compatible (and more) local inference server powered by chuk-lazarus.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── State ──────────────────────────────────────────────────────────────
    app.state.engine = engine
    app.state.default_max_tokens = default_max_tokens

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Optional auth ──────────────────────────────────────────────────────
    if api_key:
        app.middleware("http")(_make_auth_middleware(api_key))

    # ── Health ─────────────────────────────────────────────────────────────
    @app.get("/health", tags=["Meta"])
    async def health() -> dict:
        return {
            "status": "ok",
            "model": engine.model_id,
            "protocols": [p.value for p in protocols],
        }

    # ── Protocol routers ───────────────────────────────────────────────────
    if Protocol.OPENAI in protocols:
        from .routers.openai import router as openai_router

        app.include_router(openai_router, prefix="/v1")
        logger.info("Mounted OpenAI router at /v1")

    if Protocol.OLLAMA in protocols:
        from .routers.ollama import router as ollama_router

        app.include_router(ollama_router)
        logger.info("Mounted Ollama router at /")

    if Protocol.ANTHROPIC in protocols:
        from .routers.anthropic import router as anthropic_router

        app.include_router(anthropic_router, prefix="/v1")
        logger.info("Mounted Anthropic router at /v1")

    return app
