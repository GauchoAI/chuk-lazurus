"""
Ollama-compatible wire types and translation helpers.

TODO: Implement Ollama protocol support.

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md

Endpoints to implement:
  POST /api/chat          → chat completions (streaming ndjson)
  POST /api/generate      → raw text generation
  GET  /api/tags          → list models
  POST /api/show          → model info
"""

from __future__ import annotations

from .internal import InternalRequest, InternalResponse  # noqa: F401  (used when implemented)

__all__: list[str] = []
