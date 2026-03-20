"""Tests for vec_inject/_primitives.py.

vec_inject and vec_inject_all use mlx.core operations. We test the
observable behaviour by patching mlx.core at the function call level.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.inference.context.vec_inject._primitives import (
    vec_inject,
    vec_inject_all,
)
from chuk_lazarus.inference.context.vec_inject._types import (
    VecInjectMatch,
)


def _make_match(token_id: int = 5, coefficient: float = 0.5, **kwargs) -> VecInjectMatch:
    defaults = {
        "token_id": token_id,
        "coefficient": coefficient,
        "score": 0.8,
        "window_id": 0,
        "position": 10,
        "distinctive": True,
    }
    defaults.update(kwargs)
    return VecInjectMatch(**defaults)


class TestVecInjectAll:
    """Tests that don't call into MLX — use patched vec_inject."""

    def test_empty_matches_returns_h_unchanged(self):
        h = MagicMock()
        embed_matrix = MagicMock()
        result = vec_inject_all(h, matches=[], embed_matrix=embed_matrix)
        assert result is h

    def test_single_match_calls_vec_inject_once(self):
        h = MagicMock()
        h_out = MagicMock()
        embed_matrix = MagicMock()
        match = _make_match(token_id=42, coefficient=0.9)

        with patch(
            "chuk_lazarus.inference.context.vec_inject._primitives.vec_inject",
            return_value=h_out,
        ) as mock_vi:
            result = vec_inject_all(h, matches=[match], embed_matrix=embed_matrix)
            mock_vi.assert_called_once_with(h, 42, 0.9, embed_matrix)
            assert result is h_out

    def test_multiple_matches_calls_vec_inject_for_each(self):
        h0 = MagicMock()
        embed_matrix = MagicMock()
        m1 = _make_match(token_id=1, coefficient=0.1)
        m2 = _make_match(token_id=2, coefficient=0.2)
        m3 = _make_match(token_id=3, coefficient=0.3)

        call_results = [MagicMock(), MagicMock(), MagicMock()]

        with patch(
            "chuk_lazarus.inference.context.vec_inject._primitives.vec_inject",
            side_effect=call_results,
        ) as mock_vi:
            vec_inject_all(h0, matches=[m1, m2, m3], embed_matrix=embed_matrix)
            assert mock_vi.call_count == 3

    def test_chained_h_passed_forward(self):
        """Each call should receive the result of the previous call."""
        h0 = MagicMock(name="h0")
        h1 = MagicMock(name="h1")
        h2 = MagicMock(name="h2")
        embed_matrix = MagicMock()

        m1 = _make_match(token_id=10, coefficient=0.1)
        m2 = _make_match(token_id=20, coefficient=0.2)

        with patch(
            "chuk_lazarus.inference.context.vec_inject._primitives.vec_inject",
            side_effect=[h1, h2],
        ) as mock_vi:
            result = vec_inject_all(h0, matches=[m1, m2], embed_matrix=embed_matrix)

            first_call_h = mock_vi.call_args_list[0][0][0]
            second_call_h = mock_vi.call_args_list[1][0][0]
            assert first_call_h is h0
            assert second_call_h is h1
            assert result is h2

    def test_result_is_last_h_value(self):
        h0 = MagicMock()
        h_final = MagicMock(name="h_final")
        embed_matrix = MagicMock()

        matches = [_make_match(token_id=i, coefficient=float(i)) for i in range(5)]
        return_vals = [MagicMock() for _ in range(4)] + [h_final]

        with patch(
            "chuk_lazarus.inference.context.vec_inject._primitives.vec_inject",
            side_effect=return_vals,
        ):
            result = vec_inject_all(h0, matches=matches, embed_matrix=embed_matrix)
            assert result is h_final

    def test_args_passed_correctly_per_match(self):
        """token_id and coefficient from each match are passed to vec_inject."""
        h = MagicMock()
        embed_matrix = MagicMock()
        m1 = _make_match(token_id=77, coefficient=1.23)
        m2 = _make_match(token_id=88, coefficient=4.56)

        with patch(
            "chuk_lazarus.inference.context.vec_inject._primitives.vec_inject",
            side_effect=[MagicMock(), MagicMock()],
        ) as mock_vi:
            vec_inject_all(h, matches=[m1, m2], embed_matrix=embed_matrix)
            calls = mock_vi.call_args_list
            assert calls[0][0][1] == 77
            assert calls[0][0][2] == pytest.approx(1.23)
            assert calls[1][0][1] == 88
            assert calls[1][0][2] == pytest.approx(4.56)


@pytest.mark.skip(reason="requires MLX hardware for actual array operations")
class TestVecInjectMLXOperations:
    """Tests that exercise the actual MLX matmul — requires hardware."""

    def test_vec_inject_modifies_h(self):
        import mlx.core as mx

        vocab_size = 100
        hidden = 64
        embed_matrix = mx.array(
            [[float(i) / (hidden * vocab_size) for _ in range(hidden)] for i in range(vocab_size)]
        )
        h = mx.zeros((1, 1, hidden))
        result = vec_inject(h, token_id=5, coefficient=0.5, embed_matrix=embed_matrix)
        mx.eval(result)
        assert result.shape == (1, 1, hidden)
