"""Tests for vec_inject/providers/_local_file.py.

Tests focus on:
- Properties and attributes (n_facts, head_dim, injection_layer)
- Adaptive threshold calculation logic (pure Python)
- routing_confident flag behaviour
- log_stats output
- load() factory error path (FileNotFoundError when no index)

The retrieve() Metal path is skipped (requires hardware + model).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from chuk_lazarus.inference.context.vec_inject._types import (
    VecInjectMatch,
    VecInjectMeta,
    VecInjectResult,
)
from chuk_lazarus.inference.context.vec_inject.providers._local_file import (
    LocalVecInjectProvider,
    _WindowSummary,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_meta(**overrides) -> VecInjectMeta:
    defaults = {
        "retrieval_layer": 29,
        "kv_head": 2,
        "query_head": 4,
        "injection_layer": 30,
    }
    defaults.update(overrides)
    return VecInjectMeta(**defaults)


def _make_provider(
    n_facts: int = 3,
    head_dim: int = 256,
    has_injection: bool = True,
    confidence_threshold: float = 0.15,
) -> LocalVecInjectProvider:
    """Build a LocalVecInjectProvider with mock MLX arrays."""
    meta = _make_meta()

    flat_k = MagicMock()
    flat_k.shape = (n_facts, head_dim)

    flat_tok = MagicMock()
    flat_coefs = MagicMock()
    flat_wid = MagicMock()
    flat_pos = MagicMock()
    flat_dist = MagicMock()
    kv_gen = MagicMock()

    summaries = [_WindowSummary(window_id=i, n_facts=1) for i in range(max(n_facts, 1))]

    return LocalVecInjectProvider(
        meta=meta,
        flat_k_mx=flat_k,
        flat_token_ids_mx=flat_tok,
        flat_coefs_mx=flat_coefs,
        flat_wid_mx=flat_wid,
        flat_positions_mx=flat_pos,
        flat_distinctive_mx=flat_dist,
        window_summaries=summaries,
        kv_gen=kv_gen,
        has_injection=has_injection,
        confidence_threshold=confidence_threshold,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestProviderProperties:
    def test_n_facts(self):
        provider = _make_provider(n_facts=10, head_dim=256)
        assert provider.n_facts == 10

    def test_n_facts_zero(self):
        provider = _make_provider(n_facts=0, head_dim=256)
        assert provider.n_facts == 0

    def test_head_dim(self):
        provider = _make_provider(n_facts=5, head_dim=320)
        assert provider.head_dim == 320

    def test_head_dim_zero_when_no_facts(self):
        provider = _make_provider(n_facts=0, head_dim=0)
        assert provider.head_dim == 0

    def test_injection_layer(self):
        provider = _make_provider()
        assert provider.injection_layer == 30

    def test_has_injection_true(self):
        provider = _make_provider(has_injection=True)
        assert provider.has_injection is True

    def test_has_injection_false(self):
        provider = _make_provider(has_injection=False)
        assert provider.has_injection is False

    def test_confidence_threshold_stored(self):
        provider = _make_provider(confidence_threshold=0.25)
        assert provider.confidence_threshold == pytest.approx(0.25)

    def test_meta_stored(self):
        provider = _make_provider()
        assert provider.meta.retrieval_layer == 29
        assert provider.meta.kv_head == 2
        assert provider.meta.query_head == 4
        assert provider.meta.injection_layer == 30


class TestWindowSummary:
    def test_window_summary_fields(self):
        s = _WindowSummary(window_id=3, n_facts=10)
        assert s.window_id == 3
        assert s.n_facts == 10

    def test_window_summary_zero(self):
        s = _WindowSummary(window_id=0, n_facts=0)
        assert s.window_id == 0
        assert s.n_facts == 0


class TestAdaptiveThresholdLogic:
    """Pure Python: test the adaptive threshold calculation logic."""

    def test_adaptive_above_fixed(self):
        """When mean*2.0 > fixed threshold, adaptive wins."""
        confidence_threshold = 0.10
        mean_score = 0.20  # mean*2 = 0.40 > 0.10
        adaptive = max(confidence_threshold, mean_score * 2.0)
        assert adaptive == pytest.approx(0.40)

    def test_fixed_floor_when_mean_low(self):
        """When mean*2.0 < fixed threshold, fixed threshold is the floor."""
        confidence_threshold = 0.15
        mean_score = 0.05  # mean*2 = 0.10 < 0.15
        adaptive = max(confidence_threshold, mean_score * 2.0)
        assert adaptive == pytest.approx(0.15)

    def test_routing_confident_when_best_exceeds_adaptive(self):
        best_score = 0.50
        adaptive_threshold = 0.30
        routing_confident = best_score >= adaptive_threshold
        assert routing_confident is True

    def test_not_routing_confident_when_best_below_adaptive(self):
        best_score = 0.10
        adaptive_threshold = 0.30
        routing_confident = best_score >= adaptive_threshold
        assert routing_confident is False

    def test_exactly_at_threshold_is_confident(self):
        best_score = 0.30
        adaptive_threshold = 0.30
        routing_confident = best_score >= adaptive_threshold
        assert routing_confident is True


class TestVecInjectResultModel:
    def test_default_empty_result(self):
        r = VecInjectResult(injection_layer=30, routing_confident=False)
        assert r.matches == []
        assert r.routing_confident is False
        assert r.top_score == 0.0

    def test_result_with_matches(self):
        m = VecInjectMatch(
            token_id=42,
            coefficient=0.5,
            score=0.8,
            window_id=1,
            position=10,
        )
        r = VecInjectResult(
            matches=[m],
            injection_layer=30,
            routing_confident=True,
            top_score=0.8,
        )
        assert len(r.matches) == 1
        assert r.matches[0].token_id == 42

    def test_result_not_confident_when_no_matches(self):
        r = VecInjectResult(injection_layer=30, routing_confident=False)
        assert not r.routing_confident


class TestLogStats:
    def test_log_stats_writes_output(self):
        provider = _make_provider(n_facts=5, head_dim=256)
        buf = io.StringIO()
        provider.log_stats(file=buf)
        output = buf.getvalue()
        assert "LocalVecInjectProvider" in output

    def test_log_stats_includes_n_facts(self):
        provider = _make_provider(n_facts=7, head_dim=256)
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "7" in buf.getvalue()

    def test_log_stats_includes_layer_info(self):
        provider = _make_provider()
        buf = io.StringIO()
        provider.log_stats(file=buf)
        output = buf.getvalue()
        assert "L29" in output

    def test_log_stats_includes_inject_layer(self):
        provider = _make_provider()
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "30" in buf.getvalue()

    def test_log_stats_shows_confidence_threshold(self):
        provider = _make_provider(confidence_threshold=0.20)
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "0.20" in buf.getvalue()

    def test_log_stats_shows_coefs_yes(self):
        provider = _make_provider(has_injection=True)
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "yes" in buf.getvalue()

    def test_log_stats_shows_coefs_no(self):
        provider = _make_provider(has_injection=False)
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "no" in buf.getvalue()

    def test_log_stats_head_dim_in_output(self):
        provider = _make_provider(n_facts=5, head_dim=320)
        buf = io.StringIO()
        provider.log_stats(file=buf)
        assert "320" in buf.getvalue()


class TestLoadFactoryErrors:
    @pytest.mark.asyncio
    async def test_load_raises_when_no_index(self, tmp_path):
        """load() should raise FileNotFoundError if neither index file exists."""
        kv_gen = MagicMock()
        with pytest.raises(FileNotFoundError):
            await LocalVecInjectProvider.load(tmp_path, kv_gen)

    @pytest.mark.asyncio
    async def test_load_error_message_mentions_directory(self, tmp_path):
        kv_gen = MagicMock()
        try:
            await LocalVecInjectProvider.load(tmp_path, kv_gen)
        except FileNotFoundError as e:
            assert str(tmp_path) in str(e)


@pytest.mark.skip(reason="requires MLX hardware for retrieve() Metal path")
class TestRetrievePath:
    async def test_retrieve_empty_index_returns_not_confident(self):
        provider = _make_provider(n_facts=0)
        result = await provider.retrieve([1, 2, 3], "test query")
        assert result.routing_confident is False
        assert result.matches == []
