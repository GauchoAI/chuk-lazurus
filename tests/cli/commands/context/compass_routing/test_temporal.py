"""Tests for compass_routing/_temporal.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chuk_lazarus.cli.commands.context.compass_routing._temporal import (
    _temporal_stride_windows,
)


def _lib(num_windows: int) -> MagicMock:
    lib = MagicMock()
    lib.num_windows = num_windows
    return lib


class TestTemporalStrideWindows:
    def test_empty_library_returns_empty(self):
        lib = _lib(0)
        result = _temporal_stride_windows(lib)
        assert result == []

    def test_single_window(self):
        lib = _lib(1)
        result = _temporal_stride_windows(lib, k=10)
        assert len(result) == 1
        wid, score = result[0]
        assert wid == 0
        assert score == 1.0

    def test_k_capped_at_num_windows(self):
        lib = _lib(3)
        result = _temporal_stride_windows(lib, k=100)
        # At most 3 windows
        assert len(result) <= 3

    def test_returns_list_of_tuples(self):
        lib = _lib(5)
        result = _temporal_stride_windows(lib, k=3)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_scores_descend_from_one(self):
        lib = _lib(10)
        result = _temporal_stride_windows(lib, k=5)
        scores = [s for _, s in result]
        # First score should be highest (1.0)
        assert scores[0] == pytest.approx(1.0)
        # Scores should be non-increasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_no_duplicate_window_ids(self):
        lib = _lib(20)
        result = _temporal_stride_windows(lib, k=10)
        wids = [w for w, _ in result]
        assert len(wids) == len(set(wids))

    def test_window_ids_in_valid_range(self):
        n = 15
        lib = _lib(n)
        result = _temporal_stride_windows(lib, k=5)
        for wid, _ in result:
            assert 0 <= wid < n

    def test_k_equals_num_windows(self):
        lib = _lib(4)
        result = _temporal_stride_windows(lib, k=4)
        assert len(result) == 4
        wids = sorted(w for w, _ in result)
        # Should cover distinct positions
        assert len(set(wids)) == 4

    def test_default_k_is_10(self):
        lib = _lib(100)
        result = _temporal_stride_windows(lib)
        assert len(result) <= 10

    def test_k_1_returns_single_result(self):
        lib = _lib(20)
        result = _temporal_stride_windows(lib, k=1)
        assert len(result) == 1
        _, score = result[0]
        assert score == pytest.approx(1.0)

    def test_two_windows_k_2(self):
        lib = _lib(2)
        result = _temporal_stride_windows(lib, k=2)
        assert len(result) == 2
        wids = {w for w, _ in result}
        assert wids == {0, 1}

    def test_large_library_spans_document(self):
        """Selected windows should be spread across the document."""
        lib = _lib(100)
        result = _temporal_stride_windows(lib, k=5)
        wids = sorted(w for w, _ in result)
        # The range of wids should be broad
        assert max(wids) - min(wids) > 20

    def test_score_formula(self):
        """Score = 1.0 - i / max(len(unique), 1)."""
        lib = _lib(10)
        result = _temporal_stride_windows(lib, k=10)
        n = len(result)
        for i, (_, score) in enumerate(result):
            expected = 1.0 - i / max(n, 1)
            assert score == pytest.approx(expected)
