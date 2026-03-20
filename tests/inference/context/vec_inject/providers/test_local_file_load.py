"""Tests for LocalVecInjectProvider._from_npz loading path.

Uses real mx.array objects (MLX is available on Apple Silicon).
mx.load is mocked to return real mx.arrays without needing a real .npz file.

Complements test_local_file.py which covers properties, adaptive threshold,
log_stats, and the load() error path.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest

from chuk_lazarus.inference.context.vec_inject.providers._local_file import (
    LocalVecInjectProvider,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_raw_npz(n_facts: int = 3, head_dim: int = 8) -> dict[str, mx.array]:
    """Return a dict as mx.load() would return for a vec_inject.npz."""
    rng = np.random.default_rng(42)
    k = rng.standard_normal((n_facts, head_dim)).astype(np.float32)
    return {
        "w0/k_vecs": mx.array(k.astype(np.float16)),
        "w0/token_ids": mx.array(np.array([1, 2, 3][:n_facts], dtype=np.int32)),
        "w0/coefs": mx.array(np.array([0.1, 0.2, 0.3][:n_facts], dtype=np.float32)),
        "w0/positions": mx.array(np.array([0, 5, 10][:n_facts], dtype=np.int32)),
        "w0/distinctive": mx.array(np.array([1, 1, 0][:n_facts], dtype=np.int32)),
        "layer": mx.array(np.array(29, dtype=np.int32)),
        "kv_head": mx.array(np.array(2, dtype=np.int32)),
        "query_head": mx.array(np.array(4, dtype=np.int32)),
        "inject_layer": mx.array(np.array(30, dtype=np.int32)),
    }


def _make_legacy_raw_npz(n_facts: int = 3, head_dim: int = 8) -> dict[str, mx.array]:
    """Return a dict as mx.load() would return for a kv_route_index.npz (legacy)."""
    rng = np.random.default_rng(7)
    k = rng.standard_normal((n_facts, head_dim)).astype(np.float32)
    return {
        "w0": mx.array(k),  # flat key — legacy kv_route format
        "layer": mx.array(np.array(29, dtype=np.int32)),
        "kv_head": mx.array(np.array(2, dtype=np.int32)),
    }


def _make_kv_gen(head_dim: int = 8, num_kv_heads: int = 4, n_rep: int = 2):
    kv_gen = MagicMock()
    backbone = MagicMock()
    layer_adapter = MagicMock()
    layer_adapter.head_dim = head_dim
    layer_adapter.num_kv_heads = num_kv_heads
    layer_adapter.n_rep = n_rep
    # Use a list so len() and [i] both work naturally
    backbone.adapted_layers = [layer_adapter] * 34
    kv_gen.backbone = backbone
    return kv_gen


# ── _from_npz: new format (vec_inject.npz) ───────────────────────────────────


class TestFromNpzNewFormat:
    def test_loads_metadata_retrieval_layer(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.meta.retrieval_layer == 29

    def test_loads_metadata_injection_layer(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.meta.injection_layer == 30

    def test_loads_metadata_kv_head(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.meta.kv_head == 2

    def test_loads_metadata_query_head(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.meta.query_head == 4

    def test_n_facts_correct(self):
        raw = _make_raw_npz(n_facts=3)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.n_facts == 3

    def test_has_injection_flag_set(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.has_injection is True

    def test_k_vectors_are_l2_normalised(self):
        """K vectors should be L2-normalised after loading (norm ≈ 1.0)."""
        raw = _make_raw_npz(n_facts=3, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        k_np = np.array(provider._flat_k_mx.tolist(), dtype=np.float32)
        norms = np.linalg.norm(k_np, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_flat_k_mx_is_real_mx_array(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert isinstance(provider._flat_k_mx, mx.array)

    def test_confidence_threshold_forwarded(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"),
                kv_gen,
                has_injection=True,
                confidence_threshold=0.25,
            )
        assert provider.confidence_threshold == pytest.approx(0.25)

    def test_head_dim_from_k_vectors(self):
        raw = _make_raw_npz(n_facts=3, head_dim=8)
        kv_gen = _make_kv_gen(head_dim=8)
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.head_dim == 8

    def test_injection_layer_fallback_when_missing(self):
        """When inject_layer key is absent, injection_layer = retrieval_layer + 1."""
        raw = _make_raw_npz()
        # Remove inject_layer key to trigger fallback
        raw.pop("inject_layer", None)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        # layer=29, so fallback inject_layer = 30
        assert provider.meta.injection_layer == 30

    def test_query_head_fallback_to_kv_head(self):
        """When query_head key is absent, query_head falls back to kv_head."""
        raw = _make_raw_npz()
        raw.pop("query_head", None)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        # kv_head=2 becomes query_head when key missing
        assert provider.meta.query_head == 2


# ── _from_npz: legacy format (kv_route_index.npz) ────────────────────────────


class TestFromNpzLegacyFormat:
    def test_legacy_loads_without_error(self):
        raw = _make_legacy_raw_npz(n_facts=3, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/kv_route_index.npz"), kv_gen, has_injection=False
            )
        assert provider is not None

    def test_legacy_n_facts_correct(self):
        raw = _make_legacy_raw_npz(n_facts=3, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/kv_route_index.npz"), kv_gen, has_injection=False
            )
        assert provider.n_facts == 3

    def test_legacy_has_injection_false(self):
        raw = _make_legacy_raw_npz(n_facts=2, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/kv_route_index.npz"), kv_gen, has_injection=False
            )
        assert provider.has_injection is False

    def test_legacy_k_vectors_are_l2_normalised(self):
        raw = _make_legacy_raw_npz(n_facts=3, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/kv_route_index.npz"), kv_gen, has_injection=False
            )
        k_np = np.array(provider._flat_k_mx.tolist(), dtype=np.float32)
        norms = np.linalg.norm(k_np, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_legacy_retrieval_layer_correct(self):
        raw = _make_legacy_raw_npz()
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/kv_route_index.npz"), kv_gen, has_injection=False
            )
        assert provider.meta.retrieval_layer == 29


# ── _from_npz: empty index ────────────────────────────────────────────────────


class TestFromNpzEmptyIndex:
    """_from_npz should handle an index with no window keys gracefully."""

    def test_empty_index_n_facts_zero(self):
        # Only metadata keys, no window keys
        raw: dict[str, mx.array] = {
            "layer": mx.array(np.array(29, dtype=np.int32)),
            "kv_head": mx.array(np.array(2, dtype=np.int32)),
            "query_head": mx.array(np.array(4, dtype=np.int32)),
            "inject_layer": mx.array(np.array(30, dtype=np.int32)),
        }
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.n_facts == 0

    def test_empty_index_head_dim_zero(self):
        raw: dict[str, mx.array] = {
            "layer": mx.array(np.array(29, dtype=np.int32)),
            "kv_head": mx.array(np.array(2, dtype=np.int32)),
            "query_head": mx.array(np.array(4, dtype=np.int32)),
            "inject_layer": mx.array(np.array(30, dtype=np.int32)),
        }
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.head_dim == 0


# ── _from_npz: multi-window ───────────────────────────────────────────────────


class TestFromNpzMultiWindow:
    def test_two_windows_n_facts_sum(self):
        """Two windows each with 2 facts → n_facts = 4."""
        rng = np.random.default_rng(99)
        raw: dict[str, mx.array] = {
            "w0/k_vecs": mx.array(rng.standard_normal((2, 8)).astype(np.float16)),
            "w0/token_ids": mx.array(np.array([1, 2], dtype=np.int32)),
            "w0/coefs": mx.array(np.array([0.1, 0.2], dtype=np.float32)),
            "w0/positions": mx.array(np.array([0, 5], dtype=np.int32)),
            "w0/distinctive": mx.array(np.array([1, 1], dtype=np.int32)),
            "w1/k_vecs": mx.array(rng.standard_normal((2, 8)).astype(np.float16)),
            "w1/token_ids": mx.array(np.array([3, 4], dtype=np.int32)),
            "w1/coefs": mx.array(np.array([0.3, 0.4], dtype=np.float32)),
            "w1/positions": mx.array(np.array([2, 7], dtype=np.int32)),
            "w1/distinctive": mx.array(np.array([0, 1], dtype=np.int32)),
            "layer": mx.array(np.array(29, dtype=np.int32)),
            "kv_head": mx.array(np.array(2, dtype=np.int32)),
            "query_head": mx.array(np.array(4, dtype=np.int32)),
            "inject_layer": mx.array(np.array(30, dtype=np.int32)),
        }
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        assert provider.n_facts == 4


# ── log_stats (with real MLX arrays) ─────────────────────────────────────────


class TestLogStatsWithRealArrays:
    def test_log_stats_shows_n_facts(self):
        raw = _make_raw_npz(n_facts=3, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        buf = io.StringIO()
        provider.log_stats(file=buf)
        output = buf.getvalue()
        assert "3" in output
        assert "LocalVecInjectProvider" in output

    def test_log_stats_shows_metal(self):
        raw = _make_raw_npz(n_facts=2, head_dim=8)
        kv_gen = _make_kv_gen()
        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            provider = LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "metal" in buf.getvalue()


# ── load() async factory paths ────────────────────────────────────────────────


class TestLoadAsync:
    """Test the async load() classmethod path selection."""

    def test_load_vec_inject_path_prefers_injection(self):
        """prefer_vec_inject=True and vec_inject.npz exists → has_injection=True."""
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()

        import asyncio

        with (
            patch(
                "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
                return_value=raw,
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            provider = asyncio.run(LocalVecInjectProvider.load("/fake/ckpt", kv_gen))

        assert provider.has_injection is True

    def test_load_vec_inject_path_returns_provider(self):
        raw = _make_raw_npz(n_facts=2)
        kv_gen = _make_kv_gen()

        import asyncio

        with (
            patch(
                "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
                return_value=raw,
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            provider = asyncio.run(LocalVecInjectProvider.load("/fake/ckpt", kv_gen))

        assert provider is not None
        assert provider.n_facts == 2

    def test_load_falls_back_to_kv_route_when_no_vec_inject(self):
        """prefer_vec_inject=True but vec_inject.npz absent → use kv_route_index.npz."""
        raw = _make_legacy_raw_npz()
        kv_gen = _make_kv_gen()

        import asyncio

        def _exists(self_path):
            return "kv_route_index" in str(self_path)

        with (
            patch(
                "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
                return_value=raw,
            ),
            patch.object(Path, "exists", _exists),
        ):
            provider = asyncio.run(LocalVecInjectProvider.load("/fake/ckpt", kv_gen))

        assert provider.has_injection is False

    def test_load_prefer_vec_inject_false_uses_route_path(self):
        """prefer_vec_inject=False skips vec_inject.npz even if it exists."""
        raw = _make_legacy_raw_npz()
        kv_gen = _make_kv_gen()

        import asyncio

        with (
            patch(
                "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
                return_value=raw,
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            provider = asyncio.run(
                LocalVecInjectProvider.load("/fake/ckpt", kv_gen, prefer_vec_inject=False)
            )

        assert provider.has_injection is False

    def test_load_confidence_threshold_forwarded(self):
        raw = _make_raw_npz()
        kv_gen = _make_kv_gen()

        import asyncio

        with (
            patch(
                "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
                return_value=raw,
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            provider = asyncio.run(
                LocalVecInjectProvider.load("/fake/ckpt", kv_gen, confidence_threshold=0.30)
            )

        assert provider.confidence_threshold == pytest.approx(0.30)


# ── retrieve() with real MLX arrays ──────────────────────────────────────────


def _make_real_provider(n_facts: int = 3, head_dim: int = 8):
    """Build a LocalVecInjectProvider with real mx.array objects."""
    from chuk_lazarus.inference.context.vec_inject._types import VecInjectMeta
    from chuk_lazarus.inference.context.vec_inject.providers._local_file import (
        _WindowSummary,
    )

    rng = np.random.default_rng(42)
    k_np = rng.standard_normal((n_facts, head_dim)).astype(np.float32)
    k_np = k_np / (np.linalg.norm(k_np, axis=-1, keepdims=True) + 1e-9)

    meta = VecInjectMeta(retrieval_layer=29, kv_head=2, query_head=4, injection_layer=30)
    summaries = [_WindowSummary(window_id=i, n_facts=1) for i in range(n_facts)]

    # Build arrays that scale to n_facts properly
    token_ids = np.arange(10, 10 + n_facts, dtype=np.int32)
    coefs = np.linspace(0.3, 0.7, n_facts, dtype=np.float32)
    wids = np.arange(n_facts, dtype=np.int32) // 2
    positions = np.arange(5, 5 + n_facts * 5, 5, dtype=np.int32)
    distinctive = np.array([1 if i % 3 != 2 else 0 for i in range(n_facts)], dtype=np.int32)

    return LocalVecInjectProvider(
        meta=meta,
        flat_k_mx=mx.array(k_np),
        flat_token_ids_mx=mx.array(token_ids),
        flat_coefs_mx=mx.array(coefs),
        flat_wid_mx=mx.array(wids),
        flat_positions_mx=mx.array(positions),
        flat_distinctive_mx=mx.array(distinctive),
        window_summaries=summaries,
        kv_gen=MagicMock(),
        has_injection=True,
        confidence_threshold=0.15,
    )


class TestRetrieve:
    """retrieve() with real MLX arrays; _query_vec is patched to avoid model."""

    def _q_vec(self, head_dim: int = 8) -> mx.array:
        return mx.array(np.random.default_rng(7).standard_normal(head_dim).astype(np.float32))

    def test_retrieve_returns_vec_inject_result(self):
        import asyncio

        from chuk_lazarus.inference.context.vec_inject._types import VecInjectResult

        provider = _make_real_provider(n_facts=3, head_dim=8)
        with patch.object(provider, "_query_vec", return_value=self._q_vec(8)):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        assert isinstance(result, VecInjectResult)

    def test_retrieve_n_facts_zero_returns_not_confident(self):
        """n_facts == 0 early exit path."""
        import asyncio

        provider = _make_real_provider(n_facts=0, head_dim=8)
        result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        assert result.routing_confident is False
        assert result.matches == []

    def test_retrieve_top_score_in_result(self):
        import asyncio

        provider = _make_real_provider(n_facts=3, head_dim=8)
        with patch.object(provider, "_query_vec", return_value=self._q_vec(8)):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        assert isinstance(result.top_score, float)

    def test_retrieve_matches_length_bounded_by_top_k(self):
        import asyncio

        provider = _make_real_provider(n_facts=5, head_dim=8)
        with patch.object(provider, "_query_vec", return_value=self._q_vec(8)):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query", top_k=2))
        assert len(result.matches) <= 2

    def test_retrieve_injection_layer_in_result(self):
        import asyncio

        provider = _make_real_provider(n_facts=3, head_dim=8)
        with patch.object(provider, "_query_vec", return_value=self._q_vec(8)):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        assert result.injection_layer == 30

    def test_retrieve_match_fields_are_populated(self):
        import asyncio

        from chuk_lazarus.inference.context.vec_inject._types import VecInjectMatch

        provider = _make_real_provider(n_facts=3, head_dim=8)
        with patch.object(provider, "_query_vec", return_value=self._q_vec(8)):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        if result.matches:
            m = result.matches[0]
            assert isinstance(m, VecInjectMatch)
            assert isinstance(m.token_id, int)
            assert isinstance(m.coefficient, float)
            assert isinstance(m.score, float)

    def test_retrieve_retrieval_ms_non_negative(self):
        import asyncio

        provider = _make_real_provider(n_facts=3, head_dim=8)
        with patch.object(provider, "_query_vec", return_value=self._q_vec(8)):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        assert result.retrieval_ms >= 0.0

    def test_retrieve_high_confidence_when_exact_match(self):
        """When query == first K vector, that window should score highest."""
        import asyncio

        provider = _make_real_provider(n_facts=3, head_dim=8)
        # Use the exact first K vector as query — cosine = 1.0
        first_k = np.array(provider._flat_k_mx[0].tolist(), dtype=np.float32)
        q = mx.array(first_k)
        with patch.object(provider, "_query_vec", return_value=q):
            result = asyncio.run(provider.retrieve([1, 2, 3], "test query"))
        assert result.top_score > 0.5


# ── _query_vec() with mocked kv_gen returning real mx arrays ─────────────────


class TestQueryVec:
    """_query_vec() calls the model; we mock to return real mx.array objects."""

    def _make_provider_with_model_mock(self, head_dim: int = 8, num_q_heads: int = 8):
        raw = _make_raw_npz(n_facts=3, head_dim=head_dim)
        hidden_size = 32

        kv_gen = MagicMock()
        backbone = MagicMock()
        layer_adapter = MagicMock()
        layer_adapter.head_dim = head_dim
        layer_adapter.num_kv_heads = 4
        layer_adapter.n_rep = 2

        # Identity norm
        layer_adapter.pre_attn_norm = lambda x: x

        # project_qkv returns real mx.arrays of correct shapes
        def fake_project_qkv(x, B, S, offset=0):
            q = mx.zeros((B, num_q_heads, S, head_dim))
            k = mx.zeros((B, 4, S, head_dim))
            v = mx.zeros((B, 4, S, head_dim))
            return q, k, v

        layer_adapter.project_qkv = fake_project_qkv
        backbone.adapted_layers = [layer_adapter] * 34
        kv_gen.backbone = backbone

        def fake_prefill_to_layer(ids, target_layer):
            S = ids.shape[1]
            return mx.zeros((1, S, hidden_size))

        kv_gen.prefill_to_layer = fake_prefill_to_layer

        with patch(
            "chuk_lazarus.inference.context.vec_inject.providers._local_file.mx.load",
            return_value=raw,
        ):
            return LocalVecInjectProvider._from_npz(
                Path("/fake/vec_inject.npz"), kv_gen, has_injection=True
            )

    def test_query_vec_returns_mx_array(self):
        provider = self._make_provider_with_model_mock()
        q = provider._query_vec([1, 2, 3])
        assert isinstance(q, mx.array)

    def test_query_vec_shape_is_head_dim(self):
        provider = self._make_provider_with_model_mock(head_dim=8)
        q = provider._query_vec([1, 2, 3])
        mx.eval(q)
        assert q.shape == (8,)

    def test_query_vec_dtype_is_float32(self):
        provider = self._make_provider_with_model_mock(head_dim=8)
        q = provider._query_vec([1, 2, 3])
        mx.eval(q)
        assert q.dtype == mx.float32
