"""
Anthropic Messages API router.

TODO: Implement Anthropic protocol support.

Endpoints to implement (mounted at /v1):
  POST /v1/messages    → chat completions (streaming + non-streaming)
  GET  /v1/models      → list models

Reference: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Anthropic"])


# Placeholder so the router can be imported and mounted without error.
# Remove this block when implementing Anthropic support.
@router.post("/messages")
async def anthropic_messages_not_implemented():
    return {"error": "Anthropic protocol not yet implemented"}
