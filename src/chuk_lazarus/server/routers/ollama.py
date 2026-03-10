"""
Ollama-compatible API router.

TODO: Implement Ollama protocol support.

Endpoints to implement (mounted at /):
  POST /api/chat       → streaming ndjson chat
  POST /api/generate   → raw text generation
  GET  /api/tags       → list models
  POST /api/show       → model info

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Ollama"])


# Placeholder so the router can be imported and mounted without error.
# Remove this block when implementing Ollama support.
@router.get("/api/tags")
async def ollama_tags_not_implemented():
    return {"error": "Ollama protocol not yet implemented", "models": []}
