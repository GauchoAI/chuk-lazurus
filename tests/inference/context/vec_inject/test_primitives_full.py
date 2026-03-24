"""Full MLX tests for vec_inject/_primitives.py.

Uses REAL mx.array objects (MLX is available on Apple Silicon).
Complements test_primitives.py which uses mocks.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from chuk_lazarus.inference.context.vec_inject._primitives import (
    vec_inject,
    vec_inject_all,
)
from chuk_lazarus.inference.context.vec_inject._types import VecInjectMatch

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_match(
    token_id: int = 3,
    coefficient: float = 1.0,
    **kwargs,
) -> VecInjectMatch:
    defaults = {
        "token_id": token_id,
        "coefficient": coefficient,
        "score": 0.9,
        "window_id": 0,
        "position": 0,
        "distinctive": True,
    }
    defaults.update(kwargs)
    return VecInjectMatch(**defaults)


# ── vec_inject tests ──────────────────────────────────────────────────────────


class TestVecInjectMLX:
    """Test vec_inject with real MLX arrays."""

    def test_output_shape(self):
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        result = vec_inject(h, token_id=3, coefficient=1.0, embed_matrix=embed_matrix)
        mx.eval(result)
        assert result.shape == (1, 1, 8)

    def test_injection_actually_modifies_h(self):
        """With a non-zero coefficient, result should differ from the zero h."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        result = vec_inject(h, token_id=3, coefficient=1.0, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist())
        # h was all zeros; result should have at least one non-zero element
        assert np.any(result_np != 0.0)

    def test_correct_dimension_modified(self):
        """With embed_matrix = eye(8) and token_id=3, only dim 3 gets the injection."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        # e = embed_matrix[3] = e_3 (unit vector)
        # direction = e / dot(e, e) = e / 1 = e
        # h' = h + c * direction[None, None, :] = c * e_3 at pos (0,0,3)
        result = vec_inject(h, token_id=3, coefficient=2.5, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        assert result_np[3] == pytest.approx(2.5, abs=1e-5)
        # All other dims should remain zero
        for i in range(8):
            if i != 3:
                assert result_np[i] == pytest.approx(0.0, abs=1e-5)

    def test_zero_coefficient_leaves_h_unchanged(self):
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        result = vec_inject(h, token_id=0, coefficient=0.0, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist())
        np.testing.assert_allclose(result_np, 0.0, atol=1e-6)

    def test_negative_coefficient(self):
        """Negative coefficient subtracts from h."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        result = vec_inject(h, token_id=5, coefficient=-1.0, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        assert result_np[5] == pytest.approx(-1.0, abs=1e-5)

    def test_nonzero_h_accumulates(self):
        """Injection adds to existing h, not replaces it."""
        h_np = np.zeros((1, 1, 8), dtype=np.float32)
        h_np[0, 0, 7] = 10.0
        h = mx.array(h_np)
        embed_matrix = mx.eye(8)
        result = vec_inject(h, token_id=7, coefficient=1.0, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        # dim 7 had 10.0, inject adds 1.0 → 11.0
        assert result_np[7] == pytest.approx(11.0, abs=1e-4)

    def test_returns_mx_array(self):
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        result = vec_inject(h, token_id=1, coefficient=0.5, embed_matrix=embed_matrix)
        assert isinstance(result, mx.array)

    def test_non_unit_embed_vector_direction(self):
        """direction = e / ||e||^2 so c * direction reproduces the projection."""
        h = mx.zeros((1, 1, 4))
        # Build an embed matrix where row 0 = [2, 0, 0, 0]
        e_np = np.zeros((10, 4), dtype=np.float32)
        e_np[0, 0] = 2.0
        embed_matrix = mx.array(e_np)
        # e = [2,0,0,0], ||e||^2 = 4, direction = [0.5, 0, 0, 0]
        # c=3 → injection = 3 * [0.5, 0, 0, 0] = [1.5, 0, 0, 0]
        result = vec_inject(h, token_id=0, coefficient=3.0, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        assert result_np[0] == pytest.approx(1.5, abs=1e-5)


# ── vec_inject_all tests (with real MLX) ─────────────────────────────────────


class TestVecInjectAllMLX:
    """Test vec_inject_all with real mx.array objects."""

    def test_empty_matches_returns_h_unchanged_mlx(self):
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        result = vec_inject_all(h, matches=[], embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist())
        np.testing.assert_allclose(result_np, 0.0, atol=1e-6)

    def test_single_match_same_as_vec_inject(self):
        """vec_inject_all with one match equals vec_inject called directly."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        match = _make_match(token_id=2, coefficient=0.7)

        result_all = vec_inject_all(h, matches=[match], embed_matrix=embed_matrix)
        result_direct = vec_inject(
            mx.zeros((1, 1, 8)), token_id=2, coefficient=0.7, embed_matrix=embed_matrix
        )
        mx.eval(result_all, result_direct)
        np.testing.assert_allclose(
            np.array(result_all.tolist()),
            np.array(result_direct.tolist()),
            atol=1e-6,
        )

    def test_two_matches_superposition(self):
        """Two orthogonal injections accumulate independently (linear superposition)."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        m1 = _make_match(token_id=0, coefficient=1.0)
        m2 = _make_match(token_id=1, coefficient=2.0)

        result = vec_inject_all(h, matches=[m1, m2], embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        # With identity embed, dim 0 gets +1.0 and dim 1 gets +2.0
        assert result_np[0] == pytest.approx(1.0, abs=1e-5)
        assert result_np[1] == pytest.approx(2.0, abs=1e-5)
        for i in range(2, 8):
            assert result_np[i] == pytest.approx(0.0, abs=1e-5)

    def test_three_matches_all_accumulate(self):
        """Three orthogonal injections each modify their own dimension."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        matches = [
            _make_match(token_id=0, coefficient=0.5),
            _make_match(token_id=3, coefficient=1.5),
            _make_match(token_id=7, coefficient=3.0),
        ]
        result = vec_inject_all(h, matches=matches, embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        assert result_np[0] == pytest.approx(0.5, abs=1e-5)
        assert result_np[3] == pytest.approx(1.5, abs=1e-5)
        assert result_np[7] == pytest.approx(3.0, abs=1e-5)

    def test_same_token_twice_accumulates(self):
        """Two matches for the same token_id add their contributions."""
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        m1 = _make_match(token_id=4, coefficient=1.0)
        m2 = _make_match(token_id=4, coefficient=2.0)
        result = vec_inject_all(h, matches=[m1, m2], embed_matrix=embed_matrix)
        mx.eval(result)
        result_np = np.array(result.tolist()).squeeze()
        # Both inject into dim 4 → 3.0 total
        assert result_np[4] == pytest.approx(3.0, abs=1e-5)

    def test_output_shape_preserved(self):
        h = mx.zeros((1, 1, 8))
        embed_matrix = mx.eye(8)
        matches = [_make_match(token_id=i, coefficient=0.1) for i in range(3)]
        result = vec_inject_all(h, matches=matches, embed_matrix=embed_matrix)
        mx.eval(result)
        assert result.shape == (1, 1, 8)
