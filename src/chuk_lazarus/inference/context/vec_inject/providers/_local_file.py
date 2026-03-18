"""LocalVecInjectProvider — file-backed VecInjectProvider.

Load once, retrieve fast.

At load time the entire normalised K matrix and all per-fact fields are
copied onto the Metal device and pinned with mx.eval().  Every retrieve()
call is a single fused MLX dispatch:

  query forward pass → Q vector (already on device from forward pass)
      ↓
  cosine scores  = flat_k_mx @ q_norm           [Metal matmul]
      ↓
  top-k indices  = mx.argsort(-scores)[:top_k]  [Metal sort]
      ↓
  batch gather   = flat_*_mx[top_idx]           [Metal gather × 4]
      ↓
  mx.eval()  — one sync, top-k small arrays land in Python

No numpy in the hot path.  numpy is used only at load time for NPZ I/O
and L2-normalisation (before data moves to the Metal device).

Index files produced by the prefill pipeline:
  --phases vec_inject  → vec_inject.npz        (K vectors + coefficients)
  --phases kvectors    → kv_route_index.npz    (K vectors only, legacy)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from pydantic import BaseModel

from .._types import VecInjectMatch, VecInjectMeta, VecInjectResult
from ._index_format import (
    KV_ROUTE_FILE,
    VEC_INJECT_FILE,
    VecInjectMetaKey,
    VecInjectWindowKey,
)


# ── Internal window summary (diagnostics only — not on the hot path) ──


class _WindowSummary(BaseModel):
    window_id: int
    n_facts: int


# ── Provider ──────────────────────────────────────────────────────────


class LocalVecInjectProvider:
    """File-backed vector injection provider.

    All scoring data lives on the Metal device after load().

    Attributes
    ----------
    meta                : Index metadata (layers, heads).
    has_injection       : True when injection coefficients are available.
    n_facts             : Total indexed fact positions across all windows.
    confidence_threshold: Minimum Q·K cosine score to trust routing.
                          retrieve() sets routing_confident=False below this
                          value so callers can fall back to window replay.
                          Wrong injection is catastrophically worse than no
                          injection — the 0.05% injected signal dominates the
                          99.95% structural context with no semantic protection.

    Usage
    -----
        provider = await LocalVecInjectProvider.load(checkpoint_dir, kv_gen)
        result   = await provider.retrieve(query_ids, query_text, top_k=5)
        if result.routing_confident and result.matches:
            h = vec_inject_all(h, result.matches, embed_matrix)
        else:
            # fall back to window replay
    """

    def __init__(
        self,
        meta: VecInjectMeta,
        *,
        # Device-resident arrays (pinned at construction via mx.eval)
        flat_k_mx: mx.array,              # (n_facts, head_dim) float32 L2-normalised
        flat_token_ids_mx: mx.array,      # (n_facts,) int32
        flat_coefs_mx: mx.array,          # (n_facts,) float32
        flat_wid_mx: mx.array,            # (n_facts,) int32
        flat_positions_mx: mx.array,      # (n_facts,) int32
        flat_distinctive_mx: mx.array,    # (n_facts,) int32  1=distinctive, 0=common prefix
        window_summaries: list[_WindowSummary],
        kv_gen,
        has_injection: bool,
        confidence_threshold: float = 0.15,
    ) -> None:
        self.meta = meta
        self.has_injection = has_injection
        self.confidence_threshold = confidence_threshold
        self._kv_gen = kv_gen
        self._summaries = window_summaries

        self._flat_k_mx           = flat_k_mx
        self._flat_token_ids_mx   = flat_token_ids_mx
        self._flat_coefs_mx       = flat_coefs_mx
        self._flat_wid_mx         = flat_wid_mx
        self._flat_positions_mx   = flat_positions_mx
        self._flat_distinctive_mx = flat_distinctive_mx

    @property
    def n_facts(self) -> int:
        return self._flat_k_mx.shape[0]

    @property
    def head_dim(self) -> int:
        return self._flat_k_mx.shape[1] if self.n_facts > 0 else 0

    # ── VecInjectProvider interface ───────────────────────────────────

    @property
    def injection_layer(self) -> int:
        return self.meta.injection_layer

    async def retrieve(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Route query to top-k facts via Q·K cosine scoring on Metal.

        The full critical path (forward pass → matmul → sort → gather)
        runs as one fused MLX graph.  Python only touches the final top-k
        scalars after a single mx.eval() sync.

        Parameters
        ----------
        query_ids  : Encoded query tokens.
        query_text : Raw text (available for logging; not used for scoring).
        top_k      : Maximum facts to return.
        """
        if self.n_facts == 0:
            return VecInjectResult(
                injection_layer=self.meta.injection_layer,
                routing_confident=False,
            )

        t0 = time.monotonic()

        # Q vector — lazy mx.array, stays on device, joins the graph
        q_vec = self._query_vec(query_ids)
        q_nrm = q_vec / mx.maximum(
            mx.sqrt(mx.sum(q_vec * q_vec)), mx.array(1e-9)
        )

        # Cosine scores — Metal matmul
        scores = self._flat_k_mx @ q_nrm        # (n_facts,) lazy

        # Top-k — Metal sort
        k = min(top_k, self.n_facts)
        top_idx = mx.argsort(-scores)[:k]       # (k,) lazy

        # Batch gather — Metal indexed reads, all still lazy
        top_scores      = scores[top_idx]
        top_token_ids   = self._flat_token_ids_mx[top_idx]
        top_coefs       = self._flat_coefs_mx[top_idx]
        top_wids        = self._flat_wid_mx[top_idx]
        top_positions   = self._flat_positions_mx[top_idx]
        top_distinctive = self._flat_distinctive_mx[top_idx]

        # Single sync — everything materialises in one Metal dispatch
        mx.eval(top_scores, top_token_ids, top_coefs, top_wids, top_positions, top_distinctive)
        elapsed_ms = (time.monotonic() - t0) * 1000

        matches = [
            VecInjectMatch(
                token_id=int(top_token_ids[i].item()),
                coefficient=float(top_coefs[i].item()),
                score=float(top_scores[i].item()),
                window_id=int(top_wids[i].item()),
                position=int(top_positions[i].item()),
                distinctive=bool(top_distinctive[i].item()),
            )
            for i in range(k)
        ]

        best_score = matches[0].score if matches else 0.0
        return VecInjectResult(
            matches=matches,
            retrieval_ms=elapsed_ms,
            injection_layer=self.meta.injection_layer,
            routing_confident=best_score >= self.confidence_threshold,
            top_score=best_score,
        )

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    async def load(
        cls,
        checkpoint_dir: str | Path,
        kv_gen,
        *,
        prefer_vec_inject: bool = True,
        confidence_threshold: float = 0.15,
    ) -> "LocalVecInjectProvider":
        """Load from a checkpoint directory and pin all data on Metal.

        Prefers vec_inject.npz (full routing + injection) over the legacy
        kv_route_index.npz (routing only, no coefficients).
        Pass prefer_vec_inject=False to force routing-only mode.

        Parameters
        ----------
        confidence_threshold : Minimum Q·K cosine score to trust routing.
            retrieve() returns routing_confident=False below this value.
            Default 0.15 is conservative — increase for higher precision
            (more fallbacks), decrease for higher recall (more injections).
        """
        path = Path(checkpoint_dir)
        vec_path   = path / VEC_INJECT_FILE
        route_path = path / KV_ROUTE_FILE

        if prefer_vec_inject and vec_path.exists():
            return cls._from_npz(
                vec_path, kv_gen, has_injection=True,
                confidence_threshold=confidence_threshold,
            )
        if route_path.exists():
            return cls._from_npz(
                route_path, kv_gen, has_injection=False,
                confidence_threshold=confidence_threshold,
            )

        raise FileNotFoundError(
            f"No vec-inject index found in {checkpoint_dir}. "
            "Run: lazarus context prefill --phases vec_inject  (or --phases kvectors)"
        )

    @classmethod
    def _from_npz(
        cls,
        npz_path: Path,
        kv_gen,
        *,
        has_injection: bool,
        confidence_threshold: float = 0.15,
    ) -> "LocalVecInjectProvider":
        """NPZ → numpy (normalise) → MLX device arrays.

        Uses mx.load() so bfloat16 arrays saved by mx.savez() are decoded
        correctly (numpy cannot cast the raw void-2 dtype they produce).
        """
        raw: dict[str, mx.array] = mx.load(str(npz_path))

        def _to_np_f32(key: str) -> np.ndarray:
            return np.array(raw[key].astype(mx.float32).tolist(), dtype=np.float32)

        def _to_np_i32(key: str) -> np.ndarray:
            return np.array(raw[key].astype(mx.int32).tolist(), dtype=np.int32)

        def _scalar(key: str) -> int:
            return int(raw[key].item())

        all_keys = set(raw.keys())

        meta = VecInjectMeta(
            retrieval_layer=_scalar(VecInjectMetaKey.LAYER),
            kv_head=_scalar(VecInjectMetaKey.KV_HEAD),
            query_head=(
                _scalar(VecInjectMetaKey.QUERY_HEAD)
                if VecInjectMetaKey.QUERY_HEAD in all_keys
                else _scalar(VecInjectMetaKey.KV_HEAD)
            ),
            injection_layer=(
                _scalar(VecInjectMetaKey.INJECT_LAYER)
                if VecInjectMetaKey.INJECT_LAYER in all_keys
                else _scalar(VecInjectMetaKey.LAYER) + 1
            ),
        )

        wids = sorted(set(
            wid
            for key in all_keys
            if (wid := VecInjectWindowKey.window_id_from_key(key)) is not None
        ))

        # Accumulate per-window numpy arrays before device upload
        k_rows:    list[np.ndarray] = []
        tok_rows:  list[np.ndarray] = []
        coef_rows: list[np.ndarray] = []
        wid_rows:  list[np.ndarray] = []
        pos_rows:  list[np.ndarray] = []
        dist_rows: list[np.ndarray] = []
        summaries: list[_WindowSummary] = []

        for wid in wids:
            if has_injection and VecInjectWindowKey.k_vecs(wid) in all_keys:
                k    = _to_np_f32(VecInjectWindowKey.k_vecs(wid))
                tok  = _to_np_i32(VecInjectWindowKey.token_ids(wid))
                coef = _to_np_f32(VecInjectWindowKey.coefs(wid))
                pos  = _to_np_i32(VecInjectWindowKey.positions(wid))
                # distinctive flag: 1=safe for 1D injection, 0=needs fallback
                # Legacy indexes without this key: assume all distinctive
                dist_key = VecInjectWindowKey.distinctive(wid)
                dist = (
                    _to_np_i32(dist_key)
                    if dist_key in all_keys
                    else np.ones(len(k), dtype=np.int32)
                )
            else:
                # Legacy kv_route_index.npz: flat "wN" key, no coefficients
                k    = _to_np_f32(VecInjectWindowKey.flat(wid))
                n    = len(k)
                tok  = np.zeros(n, dtype=np.int32)
                coef = np.zeros(n, dtype=np.float32)
                pos  = np.arange(n, dtype=np.int32)
                dist = np.ones(n, dtype=np.int32)

            n_facts = len(k)
            k_rows.append(k)
            tok_rows.append(tok)
            coef_rows.append(coef)
            wid_rows.append(np.full(n_facts, wid, dtype=np.int32))
            pos_rows.append(pos)
            dist_rows.append(dist)
            summaries.append(_WindowSummary(window_id=wid, n_facts=n_facts))

        if not k_rows:
            empty = np.zeros((0, 1), dtype=np.float32)
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float32)
            flat_k_mx          = mx.array(empty)
            flat_token_ids_mx  = mx.array(empty_i)
            flat_coefs_mx      = mx.array(empty_f)
            flat_wid_mx        = mx.array(empty_i)
            flat_positions_mx  = mx.array(empty_i)
            flat_distinctive_mx = mx.array(empty_i)
            mx.eval(flat_k_mx, flat_token_ids_mx, flat_coefs_mx,
                    flat_wid_mx, flat_positions_mx, flat_distinctive_mx)
            return cls(
                meta=meta,
                flat_k_mx=flat_k_mx,
                flat_token_ids_mx=flat_token_ids_mx,
                flat_coefs_mx=flat_coefs_mx,
                flat_wid_mx=flat_wid_mx,
                flat_positions_mx=flat_positions_mx,
                flat_distinctive_mx=flat_distinctive_mx,
                window_summaries=[],
                kv_gen=kv_gen,
                has_injection=has_injection,
                confidence_threshold=confidence_threshold,
            )

        # L2-normalise rows on CPU once so Metal only sees unit vectors
        flat_k = np.concatenate(k_rows, axis=0)
        norms = np.linalg.norm(flat_k, axis=1, keepdims=True)
        np.clip(norms, 1e-9, None, out=norms)
        flat_k_normed = flat_k / norms

        # Upload to Metal and pin — single mx.eval() before first retrieve()
        flat_k_mx           = mx.array(flat_k_normed,                dtype=mx.float32)
        flat_token_ids_mx   = mx.array(np.concatenate(tok_rows),     dtype=mx.int32)
        flat_coefs_mx       = mx.array(np.concatenate(coef_rows),    dtype=mx.float32)
        flat_wid_mx         = mx.array(np.concatenate(wid_rows),     dtype=mx.int32)
        flat_positions_mx   = mx.array(np.concatenate(pos_rows),     dtype=mx.int32)
        flat_distinctive_mx = mx.array(np.concatenate(dist_rows),    dtype=mx.int32)

        mx.eval(flat_k_mx, flat_token_ids_mx, flat_coefs_mx,
                flat_wid_mx, flat_positions_mx, flat_distinctive_mx)

        return cls(
            meta=meta,
            flat_k_mx=flat_k_mx,
            flat_token_ids_mx=flat_token_ids_mx,
            flat_coefs_mx=flat_coefs_mx,
            flat_wid_mx=flat_wid_mx,
            flat_positions_mx=flat_positions_mx,
            flat_distinctive_mx=flat_distinctive_mx,
            window_summaries=summaries,
            kv_gen=kv_gen,
            has_injection=has_injection,
            confidence_threshold=confidence_threshold,
        )

    # ── Internal ─────────────────────────────────────────────────────

    def _query_vec(self, query_ids: list[int]) -> mx.array:
        """Compute Q vector at the retrieval head — returns lazy mx.array.

        Not eval'd here so the caller can fuse it into the scoring matmul
        as a single Metal dispatch.
        """
        backbone = self._kv_gen.backbone
        layer_adapter = backbone.adapted_layers[self.meta.retrieval_layer]

        ids = mx.array(query_ids)[None]   # (1, S)
        B, S = ids.shape

        # Forward to retrieval layer — already on Metal
        h = self._kv_gen.prefill_to_layer(ids, target_layer=self.meta.retrieval_layer)

        # project_qkv applies pre_attn_norm output → q_norm + RoPE in one call
        x     = layer_adapter.pre_attn_norm(h[:, -1:, :])   # (1, 1, hidden)
        q, _, _ = layer_adapter.project_qkv(x, B, 1, offset=S - 1)
        # q: (1, nq, 1, head_dim)

        # Lazy — no eval — joins the downstream scoring graph
        return q[0, self.meta.query_head, 0, :].astype(mx.float32)  # (head_dim,)

    # ── Diagnostics ──────────────────────────────────────────────────

    def log_stats(self, file=sys.stderr) -> None:
        print(
            f"  LocalVecInjectProvider: {self.n_facts} facts × {self.head_dim}D "
            f"across {len(self._summaries)} windows "
            f"(L{self.meta.retrieval_layer} KV-H{self.meta.kv_head}, "
            f"inject L{self.meta.injection_layer}, "
            f"coefs={'yes' if self.has_injection else 'no'}, "
            f"conf_threshold={self.confidence_threshold:.2f}, "
            f"device=metal)",
            file=file,
        )
