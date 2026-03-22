"""VecInjectProvider — pluggable vector injection interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ._types import VecInjectResult


@runtime_checkable
class VecInjectProvider(Protocol):
    """Abstract retrieval + injection interface.

    A VecInjectProvider encapsulates a stored fact index, a retrieval
    mechanism (Q·K scoring at the model's retrieval head), and injection
    metadata (layer, token, coefficient).  The generation loop calls
    retrieve() once per query and vec_inject_all() at injection_layer.

    Both sync and async retrieval are supported:
      - retrieve()      — async (for CLI, server, and I/O-bound providers)
      - retrieve_sync() — sync  (for notebooks, scripts, and in-memory providers)

    In-memory providers (LocalVecInjectProvider) do zero I/O — their
    retrieve() is async only for interface compatibility. Use retrieve_sync()
    to avoid unnecessary asyncio boilerplate.

    Concrete implementations
    ------------------------
    LocalVecInjectProvider — .npz file on disk (providers/local_file.py)

    Planned
    -------
    RedisVecInjectProvider  — Redis hash of K vectors, sub-ms remote lookup
    MCPVecInjectProvider    — MCP server endpoint (chuk-mcp-kvindex)
    CompositeVecInjectProvider — fan-out across multiple providers
    """

    async def retrieve(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Retrieve facts most relevant to the query (async).

        Parameters
        ----------
        query_ids  : Tokenised query (for K-vector scoring at retrieval head).
        query_text : Raw query text (for BM25 fallback or logging).
        top_k      : Maximum number of facts to return.

        Returns
        -------
        VecInjectResult with matches sorted by descending score.
        """
        ...

    def retrieve_sync(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Retrieve facts most relevant to the query (sync).

        Same as retrieve() but without async. Preferred for in-memory
        providers where no I/O is involved.
        """
        ...

    @property
    def injection_layer(self) -> int:
        """Layer index where vec_inject_all() must be called (typically 30)."""
        ...
