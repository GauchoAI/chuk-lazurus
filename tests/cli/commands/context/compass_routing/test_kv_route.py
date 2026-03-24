"""Tests for compass_routing/_kv_route.py.

_aggregate_scores is pure Python — tested fully.
_kv_route_score_windows requires MLX — not run without hardware.
"""

from __future__ import annotations

import pytest

# _aggregate_scores doesn't use MLX, import it directly
from chuk_lazarus.cli.commands.context.compass_routing._kv_route import (
    _aggregate_scores,
)


class TestAggregateScores:
    """Pure Python — no MLX required."""

    def test_empty_wid_map_returns_zeros(self):
        result = _aggregate_scores([], wid_map=[], num_windows=3, method="test")
        assert len(result) == 3
        for wid, score in result:
            assert score == 0.0

    def test_single_position_single_window(self):
        scores = [0.75]
        wid_map = [(0, 0)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=1, method="test")
        assert len(result) == 1
        wid, score = result[0]
        assert wid == 0
        assert score == pytest.approx(0.75)

    def test_multiple_positions_same_window_max(self):
        """Small lists (<=16 entries) use max aggregation."""
        scores = [0.3, 0.7, 0.5]
        wid_map = [(0, 0), (0, 1), (0, 2)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=1, method="test")
        assert len(result) == 1
        wid, score = result[0]
        assert score == pytest.approx(0.7)

    def test_different_windows(self):
        scores = [0.8, 0.2]
        wid_map = [(0, 0), (1, 0)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=2, method="test")
        score_by_wid = dict(result)
        assert score_by_wid[0] == pytest.approx(0.8)
        assert score_by_wid[1] == pytest.approx(0.2)

    def test_results_sorted_descending(self):
        scores = [0.1, 0.9, 0.5, 0.3]
        wid_map = [(0, 0), (1, 0), (2, 0), (3, 0)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=4, method="test")
        result_scores = [s for _, s in result]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_top_k_agg_for_large_lists(self):
        """Lists with >16 positions use top-10 average."""
        # Window 0: 17 values; top-10 are all 1.0
        scores = [1.0] * 10 + [0.0] * 7
        wid_map = [(0, i) for i in range(17)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=1, method="test")
        assert len(result) == 1
        _, score = result[0]
        # Top-10 average of ten 1.0s and seven 0.0s → 1.0
        assert score == pytest.approx(1.0)

    def test_top_k_agg_mixed_values(self):
        """Top-10 average from sorted list."""
        # 17 values: top 10 are 0.9, rest are 0.1
        scores = [0.9] * 10 + [0.1] * 7
        wid_map = [(0, i) for i in range(17)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=1, method="test")
        _, score = result[0]
        assert score == pytest.approx(0.9)

    def test_exactly_16_positions_uses_max(self):
        """Exactly 16 entries → uses max (len <= 16)."""
        scores = [float(i) / 16 for i in range(16)]
        wid_map = [(0, i) for i in range(16)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=1, method="test")
        _, score = result[0]
        assert score == pytest.approx(15.0 / 16)

    def test_mixed_windows_many_positions(self):
        """Multiple windows, some with >16 positions."""
        # Window 0: 17 scores all 0.5
        # Window 1: 3 scores of 0.9, 0.1, 0.2
        scores_w0 = [0.5] * 17
        scores_w1 = [0.9, 0.1, 0.2]
        all_scores = scores_w0 + scores_w1
        wid_map = [(0, i) for i in range(17)] + [(1, i) for i in range(3)]
        result = _aggregate_scores(all_scores, wid_map=wid_map, num_windows=2, method="test")
        score_by_wid = dict(result)
        assert score_by_wid[0] == pytest.approx(0.5)  # top-10 avg of all-0.5
        assert score_by_wid[1] == pytest.approx(0.9)  # max of 0.9, 0.1, 0.2

    def test_num_windows_determines_zero_fill_size(self):
        scores = [0.5]
        wid_map = [(0, 0)]
        # num_windows=5 but only window 0 has data
        # Returns only the windows that have data (not zero-filled for missing)
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=5, method="test")
        # Result should contain window 0 with its score
        assert any(w == 0 and s == pytest.approx(0.5) for w, s in result)

    def test_returns_list_of_tuples(self):
        result = _aggregate_scores([0.5], [(0, 0)], num_windows=1, method="test")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_negative_scores_handled(self):
        scores = [-0.5, -0.1, 0.3]
        wid_map = [(0, 0), (1, 0), (2, 0)]
        result = _aggregate_scores(scores, wid_map=wid_map, num_windows=3, method="test")
        result_scores = [s for _, s in result]
        # Should still be sorted descending
        assert result_scores == sorted(result_scores, reverse=True)
        # Best is 0.3
        assert result[0][1] == pytest.approx(0.3)
