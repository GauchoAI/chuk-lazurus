"""
Anthropic Messages API wire types and translation helpers.

TODO: Implement Anthropic protocol support.

Reference: https://docs.anthropic.com/en/api/messages

Endpoints to implement:
  POST /v1/messages       → chat completions (streaming + non-streaming)
  GET  /v1/models         → list models (uses different schema from OpenAI)
"""

from __future__ import annotations

from .internal import InternalRequest, InternalResponse  # noqa: F401  (used when implemented)

__all__: list[str] = []
