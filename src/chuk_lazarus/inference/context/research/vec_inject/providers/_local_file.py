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

from .._types import SourceType, VecInjectMatch, VecInjectMeta, VecInjectResult
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
        flat_k_mx: mx.array,  # (n_facts, head_dim) float32 L2-normalised
        flat_token_ids_mx: mx.array,  # (n_facts,) int32
        flat_coefs_mx: mx.array,  # (n_facts,) float32
        flat_wid_mx: mx.array,  # (n_facts,) int32
        flat_positions_mx: mx.array,  # (n_facts,) int32
        flat_distinctive_mx: mx.array,  # (n_facts,) int32  1=distinctive, 0=common prefix
        window_summaries: list[_WindowSummary],
        kv_gen,
        has_injection: bool,
        confidence_threshold: float = 0.15,
        flat_h4_mx: mx.array | None = None,  # (n_facts, hidden_size) float32 L2-normalised
        flat_source_ids_mx: mx.array | None = None,  # (n_facts,) int32
        flat_source_types_mx: mx.array | None = None,  # (n_facts,) uint8 — 0=document, 1=generated
        residual: mx.array | None = None,  # (1, hidden_dim) float32 — running state
    ) -> None:
        self.meta = meta
        self.has_injection = has_injection
        self.confidence_threshold = confidence_threshold
        self._kv_gen = kv_gen
        self._summaries = window_summaries

        self._flat_k_mx = flat_k_mx
        self._flat_token_ids_mx = flat_token_ids_mx
        self._flat_coefs_mx = flat_coefs_mx
        self._flat_wid_mx = flat_wid_mx
        self._flat_positions_mx = flat_positions_mx
        self._flat_distinctive_mx = flat_distinctive_mx
        self._flat_h4_mx = flat_h4_mx  # None → Stage 2 not available
        self._flat_source_ids_mx = flat_source_ids_mx
        self._flat_source_types_mx = flat_source_types_mx
        self.residual = residual  # running Markov state

    def close(self) -> None:
        """Release device-resident arrays to free Metal memory."""
        self._flat_k_mx = mx.zeros((0, 1), dtype=mx.float32)
        self._flat_token_ids_mx = mx.zeros((0,), dtype=mx.int32)
        self._flat_coefs_mx = mx.zeros((0,), dtype=mx.float32)
        self._flat_wid_mx = mx.zeros((0,), dtype=mx.int32)
        self._flat_positions_mx = mx.zeros((0,), dtype=mx.int32)
        self._flat_distinctive_mx = mx.zeros((0,), dtype=mx.int32)
        self._flat_h4_mx = None
        self._flat_source_ids_mx = None
        self._flat_source_types_mx = None
        self.residual = None
        self._kv_gen = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

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

    def retrieve_sync(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Route query to top-k facts via Q·K cosine scoring on Metal (sync).

        The full critical path (forward pass → matmul → sort → gather)
        runs as one fused MLX graph.  Python only touches the final top-k
        scalars after a single mx.eval() sync.

        This is the preferred entry point for in-memory providers — no
        I/O is involved, so async adds no value.

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
        q_nrm = q_vec / mx.maximum(mx.sqrt(mx.sum(q_vec * q_vec)), mx.array(1e-9))

        # Cosine scores — Metal matmul
        scores = self._flat_k_mx @ q_nrm  # (n_facts,) lazy

        # Top-k overall — Metal sort (for diagnostics / fallback matches)
        k = min(top_k, self.n_facts)
        top_idx = mx.argsort(-scores)[:k]  # (k,) lazy

        # Best distinctive match — mask non-distinctive positions to -inf so
        # they are never selected.  Non-distinctive tokens (< 4 chars) cause
        # wrong injections because the model's prior for common prefixes
        # overwhelms the small injected coefficient.
        dist_mask = self._flat_distinctive_mx.astype(mx.float32)  # 1.0 or 0.0
        scores_dist = scores * dist_mask + (dist_mask - 1.0) * 1e9  # non-dist → -inf
        best_dist_idx_arr = mx.argmax(scores_dist)  # scalar

        # Batch gather — all still lazy
        top_scores = scores[top_idx]
        top_token_ids = self._flat_token_ids_mx[top_idx]
        top_coefs = self._flat_coefs_mx[top_idx]
        top_wids = self._flat_wid_mx[top_idx]
        top_positions = self._flat_positions_mx[top_idx]
        top_distinctive = self._flat_distinctive_mx[top_idx]

        best_dist_score_arr = scores[best_dist_idx_arr]
        best_dist_tok_arr = self._flat_token_ids_mx[best_dist_idx_arr]
        best_dist_coef_arr = self._flat_coefs_mx[best_dist_idx_arr]
        best_dist_wid_arr = self._flat_wid_mx[best_dist_idx_arr]
        best_dist_pos_arr = self._flat_positions_mx[best_dist_idx_arr]

        # Adaptive threshold — mean × 2.0 scales with score distribution.
        # Fixed threshold (e.g. 0.15) becomes useless beyond N≈50: at N=3,625
        # max Q·K scores drop below 1%, killing all injections. The adaptive
        # threshold is relative to the score distribution, not absolute.
        # self.confidence_threshold serves as a minimum floor.
        mean_score_arr = mx.mean(scores)

        # Single sync — everything materialises in one Metal dispatch
        mx.eval(
            top_scores,
            top_token_ids,
            top_coefs,
            top_wids,
            top_positions,
            top_distinctive,
            best_dist_score_arr,
            best_dist_tok_arr,
            best_dist_coef_arr,
            best_dist_wid_arr,
            best_dist_pos_arr,
            mean_score_arr,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        mean_score = float(mean_score_arr.item())
        adaptive_threshold = max(self.confidence_threshold, mean_score * 2.0)

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

        best_dist_score = float(best_dist_score_arr.item())
        stage1_confident = best_dist_score >= adaptive_threshold

        # ── Stage 1 succeeded — best distinctive K-space match above threshold ──
        if stage1_confident:
            best_dist_match = VecInjectMatch(
                token_id=int(best_dist_tok_arr.item()),
                coefficient=float(best_dist_coef_arr.item()),
                score=best_dist_score,
                window_id=int(best_dist_wid_arr.item()),
                position=int(best_dist_pos_arr.item()),
                distinctive=True,
            )
            return VecInjectResult(
                matches=[best_dist_match] + matches,
                retrieval_ms=elapsed_ms,
                injection_layer=self.meta.injection_layer,
                routing_confident=True,
                top_score=best_dist_score,
                routing_stage="kspace",
            )

        # ── Stage 2: H4 output cosine (same-template entity discrimination) ──
        # Runs when Stage 1 fails the score threshold OR has no distinctive matches.
        # Requires h4_vecs in the index.
        if self._flat_h4_mx is not None and self.meta.retrieval_layer > 0:
            h4_result = self._route_h4(query_ids, k, elapsed_ms)
            if h4_result is not None:
                return h4_result

        # ── Stage 3: fallback ────────────────────────────────────────
        top_score = float(top_scores[0].item()) if len(top_scores) > 0 else 0.0
        return VecInjectResult(
            matches=matches,
            retrieval_ms=elapsed_ms,
            injection_layer=self.meta.injection_layer,
            routing_confident=False,
            top_score=top_score,
            routing_stage="fallback",
        )

    async def retrieve(
        self,
        query_ids: list[int],
        query_text: str,
        top_k: int = 5,
    ) -> VecInjectResult:
        """Async wrapper around retrieve_sync() for protocol compatibility.

        LocalVecInjectProvider does zero I/O in retrieval — all data is
        pinned on the Metal device at load time. This method exists only
        to satisfy the async VecInjectProvider protocol. Prefer
        retrieve_sync() in non-async contexts.
        """
        return self.retrieve_sync(query_ids, query_text, top_k)

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    async def load(
        cls,
        checkpoint_dir: str | Path,
        kv_gen,
        *,
        prefer_vec_inject: bool = True,
        confidence_threshold: float = 0.15,
    ) -> LocalVecInjectProvider:
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
        vec_path = path / VEC_INJECT_FILE
        route_path = path / KV_ROUTE_FILE

        if prefer_vec_inject and vec_path.exists():
            return cls._from_npz(
                vec_path,
                kv_gen,
                has_injection=True,
                confidence_threshold=confidence_threshold,
            )
        if route_path.exists():
            return cls._from_npz(
                route_path,
                kv_gen,
                has_injection=False,
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
    ) -> LocalVecInjectProvider:
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

        wids = sorted(
            {
                wid
                for key in all_keys
                if (wid := VecInjectWindowKey.window_id_from_key(key)) is not None
            }
        )

        # Accumulate per-window numpy arrays before device upload
        k_rows: list[np.ndarray] = []
        tok_rows: list[np.ndarray] = []
        coef_rows: list[np.ndarray] = []
        wid_rows: list[np.ndarray] = []
        pos_rows: list[np.ndarray] = []
        dist_rows: list[np.ndarray] = []
        h4_rows: list[np.ndarray] = []  # Stage-2 routing vectors (may be empty)
        summaries: list[_WindowSummary] = []

        for wid in wids:
            if has_injection and VecInjectWindowKey.k_vecs(wid) in all_keys:
                k = _to_np_f32(VecInjectWindowKey.k_vecs(wid))
                tok = _to_np_i32(VecInjectWindowKey.token_ids(wid))
                coef = _to_np_f32(VecInjectWindowKey.coefs(wid))
                pos = _to_np_i32(VecInjectWindowKey.positions(wid))
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
                k = _to_np_f32(VecInjectWindowKey.flat(wid))
                n = len(k)
                tok = np.zeros(n, dtype=np.int32)
                coef = np.zeros(n, dtype=np.float32)
                pos = np.arange(n, dtype=np.int32)
                dist = np.ones(n, dtype=np.int32)

            n_facts = len(k)
            k_rows.append(k)
            tok_rows.append(tok)
            coef_rows.append(coef)
            wid_rows.append(np.full(n_facts, wid, dtype=np.int32))
            pos_rows.append(pos)
            dist_rows.append(dist)

            # H4 Stage-2 routing vectors — optional, absent in legacy indexes
            h4_key = VecInjectWindowKey.h4_vecs(wid)
            if h4_key in all_keys:
                h4_rows.append(_to_np_f32(h4_key))

            summaries.append(_WindowSummary(window_id=wid, n_facts=n_facts))

        if not k_rows:
            empty = np.zeros((0, 1), dtype=np.float32)
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float32)
            flat_k_mx = mx.array(empty)
            flat_token_ids_mx = mx.array(empty_i)
            flat_coefs_mx = mx.array(empty_f)
            flat_wid_mx = mx.array(empty_i)
            flat_positions_mx = mx.array(empty_i)
            flat_distinctive_mx = mx.array(empty_i)
            mx.eval(
                flat_k_mx,
                flat_token_ids_mx,
                flat_coefs_mx,
                flat_wid_mx,
                flat_positions_mx,
                flat_distinctive_mx,
            )
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
                flat_h4_mx=None,
            )

        # L2-normalise rows on CPU once so Metal only sees unit vectors
        flat_k = np.concatenate(k_rows, axis=0)
        norms = np.linalg.norm(flat_k, axis=1, keepdims=True)
        np.clip(norms, 1e-9, None, out=norms)
        flat_k_normed = flat_k / norms

        # H4 Stage-2 routing — only present in indexes built after 2026-03-19
        flat_h4_mx: mx.array | None = None
        if h4_rows and len(h4_rows) == len(k_rows):
            # All windows have h4_vecs — safe to concatenate
            flat_h4 = np.concatenate(h4_rows, axis=0)
            norms_h4 = np.linalg.norm(flat_h4, axis=1, keepdims=True)
            np.clip(norms_h4, 1e-9, None, out=norms_h4)
            flat_h4_normed = flat_h4 / norms_h4
            flat_h4_mx = mx.array(flat_h4_normed, dtype=mx.float32)

        # Upload to Metal and pin — single mx.eval() before first retrieve()
        flat_k_mx = mx.array(flat_k_normed, dtype=mx.float32)
        flat_token_ids_mx = mx.array(np.concatenate(tok_rows), dtype=mx.int32)
        flat_coefs_mx = mx.array(np.concatenate(coef_rows), dtype=mx.float32)
        flat_wid_mx = mx.array(np.concatenate(wid_rows), dtype=mx.int32)
        flat_positions_mx = mx.array(np.concatenate(pos_rows), dtype=mx.int32)
        flat_distinctive_mx = mx.array(np.concatenate(dist_rows), dtype=mx.int32)

        arrays_to_pin = [
            flat_k_mx,
            flat_token_ids_mx,
            flat_coefs_mx,
            flat_wid_mx,
            flat_positions_mx,
            flat_distinctive_mx,
        ]
        if flat_h4_mx is not None:
            arrays_to_pin.append(flat_h4_mx)
        mx.eval(*arrays_to_pin)

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
            flat_h4_mx=flat_h4_mx,
        )

    # ── Internal ─────────────────────────────────────────────────────

    def _query_vec(self, query_ids: list[int]) -> mx.array:
        """Compute Q vector at the retrieval head — returns lazy mx.array.

        Not eval'd here so the caller can fuse it into the scoring matmul
        as a single Metal dispatch.
        """
        backbone = self._kv_gen.backbone
        layer_adapter = backbone.adapted_layers[self.meta.retrieval_layer]

        ids = mx.array(query_ids)[None]  # (1, S)
        B, S = ids.shape

        # Forward to retrieval layer — already on Metal
        h = self._kv_gen.prefill_to_layer(ids, target_layer=self.meta.retrieval_layer)

        # project_qkv applies pre_attn_norm output → q_norm + RoPE in one call
        x = layer_adapter.pre_attn_norm(h[:, -1:, :])  # (1, 1, hidden)
        q, _, _ = layer_adapter.project_qkv(x, B, 1, offset=S - 1)
        # q: (1, nq, 1, head_dim)

        # Lazy — no eval — joins the downstream scoring graph
        return q[0, self.meta.query_head, 0, :].astype(mx.float32)  # (head_dim,)

    def _query_h4_vec(self, query_ids: list[int]) -> mx.array:
        """Compute H4's attention output at the last query position.

        Uses h entering L{retrieval_layer} (= output of L{retrieval_layer-1})
        so this reflects the ACTUAL L{retrieval_layer} forward computation,
        not the re-projected state used by _query_vec (K-space Stage 1).

        Returns lazy (hidden_size,) float32 — join into the h4_scores matmul.
        """
        backbone = self._kv_gen.backbone
        rl = self.meta.retrieval_layer
        layer_adapter = backbone.adapted_layers[rl]

        ids = mx.array(query_ids)[None]  # (1, S)
        B, S = ids.shape

        # Run to L{rl-1} — input to L{rl}
        h_pre = self._kv_gen.prefill_to_layer(ids, target_layer=rl - 1)

        x = layer_adapter.pre_attn_norm(h_pre)  # (1, S, D)
        q, k, v = layer_adapter.project_qkv(x, B, S, offset=0)

        H4 = self.meta.query_head
        kv_h = H4 // layer_adapter.n_rep
        dh = layer_adapter.head_dim

        # Last query position attending over all positions
        q_last = q[:, H4, -1:, :]  # (1, 1, dh)
        k_kv = k[:, kv_h, :, :]  # (1, S, dh)
        v_kv = v[:, kv_h, :, :]  # (1, S, dh)

        scores = mx.matmul(q_last, k_kv.transpose(0, 2, 1)) * layer_adapter.attn_scale
        attn_w = mx.softmax(scores, axis=-1)  # (1, 1, S)
        h4_out = mx.matmul(attn_w, v_kv)[:, 0, :]  # (1, dh)

        # Project through H4's O_proj columns → hidden space (via protocol)
        h4_contrib = layer_adapter.head_output_projection(h4_out, H4)  # (1, D)

        return h4_contrib[0].astype(mx.float32)  # (D,) — lazy

    def _route_h4(
        self,
        query_ids: list[int],
        top_k: int,
        stage1_ms: float,
    ) -> VecInjectResult | None:
        """Stage 2: H4 output cosine routing.

        Returns a VecInjectResult with routing_stage="h4" if confident,
        None if the H4 score distribution doesn't clear the adaptive threshold.
        """
        t1 = time.monotonic()

        q_h4 = self._query_h4_vec(query_ids)
        q_h4_nrm = q_h4 / mx.maximum(mx.sqrt(mx.sum(q_h4 * q_h4)), mx.array(1e-9))

        # H4 cosine scores against all stored fact H4 vectors (Metal matmul)
        h4_scores = self._flat_h4_mx @ q_h4_nrm  # (n_facts,) lazy

        k = min(top_k, self.n_facts)
        top_idx = mx.argsort(-h4_scores)[:k]

        # Best distinctive H4 match — same mask trick as Stage 1
        dist_mask = self._flat_distinctive_mx.astype(mx.float32)
        h4_scores_dist = h4_scores * dist_mask + (dist_mask - 1.0) * 1e9
        best_dist_idx_arr = mx.argmax(h4_scores_dist)

        top_h4_scores = h4_scores[top_idx]
        top_token_ids = self._flat_token_ids_mx[top_idx]
        top_coefs = self._flat_coefs_mx[top_idx]
        top_wids = self._flat_wid_mx[top_idx]
        top_positions = self._flat_positions_mx[top_idx]
        top_distinctive = self._flat_distinctive_mx[top_idx]

        best_dist_h4_score_arr = h4_scores[best_dist_idx_arr]
        best_dist_tok_arr = self._flat_token_ids_mx[best_dist_idx_arr]
        best_dist_coef_arr = self._flat_coefs_mx[best_dist_idx_arr]
        best_dist_wid_arr = self._flat_wid_mx[best_dist_idx_arr]
        best_dist_pos_arr = self._flat_positions_mx[best_dist_idx_arr]

        mean_h4_arr = mx.mean(h4_scores)

        mx.eval(
            top_h4_scores,
            top_token_ids,
            top_coefs,
            top_wids,
            top_positions,
            top_distinctive,
            best_dist_h4_score_arr,
            best_dist_tok_arr,
            best_dist_coef_arr,
            best_dist_wid_arr,
            best_dist_pos_arr,
            mean_h4_arr,
        )
        elapsed_ms = stage1_ms + (time.monotonic() - t1) * 1000

        mean_h4 = float(mean_h4_arr.item())
        h4_threshold = max(self.confidence_threshold, mean_h4 * 2.0)
        best_dist_h4 = float(best_dist_h4_score_arr.item())

        if best_dist_h4 < h4_threshold:
            return None

        best_dist_match = VecInjectMatch(
            token_id=int(best_dist_tok_arr.item()),
            coefficient=float(best_dist_coef_arr.item()),
            score=best_dist_h4,
            window_id=int(best_dist_wid_arr.item()),
            position=int(best_dist_pos_arr.item()),
            distinctive=True,
        )
        top_matches = [
            VecInjectMatch(
                token_id=int(top_token_ids[i].item()),
                coefficient=float(top_coefs[i].item()),
                score=float(top_h4_scores[i].item()),
                window_id=int(top_wids[i].item()),
                position=int(top_positions[i].item()),
                distinctive=bool(top_distinctive[i].item()),
            )
            for i in range(k)
        ]
        return VecInjectResult(
            matches=[best_dist_match] + top_matches,
            retrieval_ms=elapsed_ms,
            injection_layer=self.meta.injection_layer,
            routing_confident=True,
            top_score=best_dist_h4,
            routing_stage="h4",
        )

    # ── Append (living store growth) ─────────────────────────────────

    def append(
        self,
        k_vecs: mx.array,  # (N, k_dim) float32 — L2-normalised
        token_ids: mx.array,  # (N,) int32
        coefs: mx.array,  # (N,) float32
        positions: mx.array,  # (N,) int32
        distinctive: mx.array,  # (N,) int32
        source_id: int,
        source_type: SourceType = SourceType.GENERATED,
        h4_vecs: mx.array | None = None,  # (N, hidden_dim) float32 L2-normed
        window_id: int | None = None,
    ) -> int:
        """Append new entries to the live store. Returns new total fact count.

        All arrays must be on device and eval'd before calling.
        """
        n_new = k_vecs.shape[0]
        if n_new == 0:
            return self.n_facts

        wid = window_id if window_id is not None else source_id
        wid_arr = mx.full((n_new,), wid, dtype=mx.int32)
        sid_arr = mx.full((n_new,), source_id, dtype=mx.int32)
        stype_arr = mx.full((n_new,), int(source_type), dtype=mx.uint8)

        self._flat_k_mx = mx.concatenate([self._flat_k_mx, k_vecs])
        self._flat_token_ids_mx = mx.concatenate([self._flat_token_ids_mx, token_ids])
        self._flat_coefs_mx = mx.concatenate([self._flat_coefs_mx, coefs])
        self._flat_wid_mx = mx.concatenate([self._flat_wid_mx, wid_arr])
        self._flat_positions_mx = mx.concatenate([self._flat_positions_mx, positions])
        self._flat_distinctive_mx = mx.concatenate([self._flat_distinctive_mx, distinctive])

        if self._flat_source_ids_mx is not None:
            self._flat_source_ids_mx = mx.concatenate([self._flat_source_ids_mx, sid_arr])
        else:
            # First time — backfill existing entries as source_id=0, document
            old_n = self.n_facts - n_new  # before concat above
            self._flat_source_ids_mx = mx.concatenate([mx.zeros((old_n,), dtype=mx.int32), sid_arr])

        if self._flat_source_types_mx is not None:
            self._flat_source_types_mx = mx.concatenate([self._flat_source_types_mx, stype_arr])
        else:
            old_n = self.n_facts - n_new
            self._flat_source_types_mx = mx.concatenate(
                [mx.zeros((old_n,), dtype=mx.uint8), stype_arr]
            )

        if h4_vecs is not None and self._flat_h4_mx is not None:
            self._flat_h4_mx = mx.concatenate([self._flat_h4_mx, h4_vecs])

        mx.eval(
            self._flat_k_mx,
            self._flat_token_ids_mx,
            self._flat_coefs_mx,
            self._flat_wid_mx,
            self._flat_positions_mx,
            self._flat_distinctive_mx,
        )
        if self._flat_source_ids_mx is not None:
            mx.eval(self._flat_source_ids_mx, self._flat_source_types_mx)
        if self._flat_h4_mx is not None:
            mx.eval(self._flat_h4_mx)

        self._summaries.append(_WindowSummary(window_id=wid, n_facts=n_new))
        return self.n_facts

    def save(self, path: str | Path) -> None:
        """Save the full store (including appended entries) to an npz file.

        Saves in knowledge_store.npz format — flat arrays, not per-window.
        Includes the running residual if set.
        """
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}

        def _to_np(arr: mx.array, dtype) -> np.ndarray:
            return np.array(arr.tolist(), dtype=dtype)

        arrays["k_vecs"] = _to_np(self._flat_k_mx, np.float32)
        arrays["token_ids"] = _to_np(self._flat_token_ids_mx, np.int32)
        arrays["coefficients"] = _to_np(self._flat_coefs_mx, np.float32)
        arrays["positions"] = _to_np(self._flat_positions_mx, np.int32)
        arrays["distinctive"] = _to_np(self._flat_distinctive_mx, np.int32)
        arrays["source_ids"] = _to_np(
            self._flat_source_ids_mx
            if self._flat_source_ids_mx is not None
            else mx.zeros((self.n_facts,), dtype=mx.int32),
            np.int32,
        )
        arrays["source_types"] = _to_np(
            self._flat_source_types_mx
            if self._flat_source_types_mx is not None
            else mx.zeros((self.n_facts,), dtype=mx.uint8),
            np.uint8,
        )
        if self._flat_h4_mx is not None:
            arrays["h_copy_vecs"] = _to_np(self._flat_h4_mx, np.float32)
        if self.residual is not None:
            arrays["residual"] = _to_np(self.residual, np.float32)

        # Metadata scalars
        arrays[VecInjectMetaKey.LAYER] = np.array(self.meta.retrieval_layer)
        arrays[VecInjectMetaKey.KV_HEAD] = np.array(self.meta.kv_head)
        arrays[VecInjectMetaKey.QUERY_HEAD] = np.array(self.meta.query_head)
        arrays[VecInjectMetaKey.INJECT_LAYER] = np.array(self.meta.injection_layer)

        np.savez(str(path), **arrays)

    # ── Diagnostics ──────────────────────────────────────────────────

    def log_stats(self, file=sys.stderr) -> None:
        h4_note = f"h4={'yes (' + str(self._flat_h4_mx.shape[1]) + 'D)' if self._flat_h4_mx is not None else 'no'}"
        routing = "S1+S2+S3" if self._flat_h4_mx is not None else "S1+S3"
        print(
            f"  LocalVecInjectProvider: {self.n_facts} facts × {self.head_dim}D "
            f"across {len(self._summaries)} windows "
            f"(L{self.meta.retrieval_layer} KV-H{self.meta.kv_head}, "
            f"inject L{self.meta.injection_layer}, "
            f"coefs={'yes' if self.has_injection else 'no'}, "
            f"{h4_note}, routing={routing}, "
            f"conf_threshold={self.confidence_threshold:.2f}, "
            f"device=metal)",
            file=file,
        )
