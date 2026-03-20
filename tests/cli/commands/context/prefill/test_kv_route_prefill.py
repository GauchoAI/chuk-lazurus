"""Tests for prefill/_kv_route.py.

_load_fact_positions: pure Python — tested directly from source.
extract_kv_route_index: mx IS available; model/kv_gen are mocked but
mx operations run against real MLX arrays.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import mlx.core as mx

from chuk_lazarus.cli.commands.context.prefill._kv_route import (
    _load_fact_positions,
    extract_kv_route_index,
)

# Minimal model config that satisfies ArchitectureConfig.from_model_config().
_GEMMA4B_CONFIG = MagicMock()
_GEMMA4B_CONFIG.model_type = "gemma3_text"
_GEMMA4B_CONFIG.num_hidden_layers = 34

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_engine(
    num_layers: int = 34,
    hidden: int = 64,
    head_dim: int = 8,
    nkv: int = 4,
    n_rep: int = 2,
):
    """Build a minimal engine mock with real MLX forward-pass returns."""
    engine = MagicMock()
    kv_gen = MagicMock()
    backbone = MagicMock()
    layer_adapter = MagicMock()

    layer_adapter.n_rep = n_rep
    layer_adapter.num_kv_heads = nkv
    layer_adapter.head_dim = head_dim

    # Use a real list so len() and [i] both work
    backbone.adapted_layers = [layer_adapter] * num_layers
    kv_gen.backbone = backbone
    engine.kv_gen = kv_gen

    return engine, kv_gen, layer_adapter


def _wire_forward(
    kv_gen, layer_adapter, seq_len: int = 10, hidden: int = 64, head_dim: int = 8, nkv: int = 4
):
    """Attach real MLX arrays as return values for the forward pass mocks."""
    h = mx.zeros((1, seq_len, hidden))
    kv_gen.prefill_to_layer.return_value = h

    x_norm = mx.zeros((1, seq_len, hidden))
    layer_adapter.pre_attn_norm.return_value = x_norm

    nq = nkv * 2  # n_rep=2
    q = mx.zeros((1, nq, seq_len, head_dim))
    k = mx.zeros((1, nkv, seq_len, head_dim))
    v = mx.zeros((1, nkv, seq_len, head_dim))
    layer_adapter.project_qkv.return_value = (q, k, v)


# ── _load_fact_positions tests (pure Python) ─────────────────────────────────


class TestLoadFactPositions:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is None

    def test_returns_none_when_entries_empty(self, tmp_path):
        (tmp_path / "sparse_index.json").write_text(json.dumps({"entries": []}))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is None

    def test_loads_single_window(self, tmp_path):
        data = {"entries": [{"window_id": 0, "fact_spans": [{"position": 5}, {"position": 12}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=3)
        assert result is not None
        assert 0 in result
        assert result[0] == [5, 12]

    def test_skips_windows_beyond_num_archived(self, tmp_path):
        data = {
            "entries": [
                {"window_id": 10, "fact_spans": [{"position": 1}]},
            ]
        }
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is None

    def test_skips_entry_without_window_id(self, tmp_path):
        data = {"entries": [{"fact_spans": [{"position": 3}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is None

    def test_multiple_windows(self, tmp_path):
        data = {
            "entries": [
                {"window_id": 0, "fact_spans": [{"position": 1}]},
                {"window_id": 1, "fact_spans": [{"position": 2}, {"position": 9}]},
            ]
        }
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is not None
        assert result[0] == [1]
        assert result[1] == [2, 9]

    def test_boundary_window_id_equals_num_archived_excluded(self, tmp_path):
        data = {"entries": [{"window_id": 3, "fact_spans": [{"position": 5}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=3)
        assert result is None

    def test_boundary_window_id_inside_range_included(self, tmp_path):
        data = {"entries": [{"window_id": 2, "fact_spans": [{"position": 7}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=3)
        assert result is not None
        assert 2 in result

    def test_missing_entries_key_returns_none(self, tmp_path):
        (tmp_path / "sparse_index.json").write_text(json.dumps({}))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is None

    def test_span_missing_position_key_skipped(self, tmp_path):
        data = {
            "entries": [
                {
                    "window_id": 0,
                    "fact_spans": [{"position": 3}, {"no_position": 99}, {"position": 7}],
                }
            ]
        }
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is not None
        assert result[0] == [3, 7]


# ── extract_kv_route_index tests ──────────────────────────────────────────────


class TestExtractKvRouteIndex:
    def test_calls_savez_once(self, tmp_path):
        """extract_kv_route_index should call mx.savez exactly once."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))

        _wire_forward(kv_gen, layer_adapter, seq_len=10, hidden=64, head_dim=8, nkv=4)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(engine, tmp_path, num_archived=1, config=_GEMMA4B_CONFIG, lib=lib, retrieval_layer=29, query_head=4)

        mock_savez.assert_called_once()

    def test_savez_receives_layer_key(self, tmp_path):
        """The saved dict must include 'layer', 'kv_head', 'query_head' keys."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(engine, tmp_path, num_archived=1, config=_GEMMA4B_CONFIG, lib=lib, retrieval_layer=29, query_head=4)

        _path, kwargs = mock_savez.call_args[0][0], mock_savez.call_args[1]
        assert "layer" in kwargs
        assert "kv_head" in kwargs
        assert "query_head" in kwargs

    def test_savez_receives_window_key(self, tmp_path):
        """The saved dict must include a 'w0' key for window 0."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(engine, tmp_path, num_archived=1, config=_GEMMA4B_CONFIG, lib=lib, retrieval_layer=29, query_head=4)

        kwargs = mock_savez.call_args[1]
        assert "w0" in kwargs

    def test_output_path_contains_filename(self, tmp_path):
        """savez is called with 'kv_route_index.npz' in the path."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(engine, tmp_path, num_archived=1, config=_GEMMA4B_CONFIG, lib=lib, retrieval_layer=29, query_head=4)

        called_path = mock_savez.call_args[0][0]
        assert "kv_route_index.npz" in called_path

    def test_two_windows_calls_lib_twice(self, tmp_path):
        """lib.get_window_tokens should be called once per archived window."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter)

        with patch.object(mx, "savez"):
            extract_kv_route_index(engine, tmp_path, num_archived=2, config=_GEMMA4B_CONFIG, lib=lib, retrieval_layer=29, query_head=4)

        assert lib.get_window_tokens.call_count == 2

    def test_retrieval_layer_clamped(self, tmp_path):
        """retrieval_layer >= num_layers is clamped to num_layers - 1."""
        num_layers = 5
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=num_layers, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter)

        with patch.object(mx, "savez") as mock_savez:
            # Pass retrieval_layer=99, should clamp to 4
            extract_kv_route_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                retrieval_layer=99,
                query_head=4,
            )

        kwargs = mock_savez.call_args[1]
        # layer value should be clamped to num_layers - 1 = 4
        layer_val = int(kwargs["layer"].item())
        assert layer_val == num_layers - 1

    def test_kv_head_idx_computed_from_query_head_and_n_rep(self, tmp_path):
        """kv_head = query_head // n_rep, capped at nkv-1."""
        # n_rep=2, nkv=4, query_head=4 → kv_head_idx = 4//2 = 2
        engine, kv_gen, layer_adapter = _make_engine(num_layers=34, nkv=4, n_rep=2)
        layer_adapter.head_dim = 8

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                retrieval_layer=29,
                query_head=4,
            )

        kwargs = mock_savez.call_args[1]
        kv_head_val = int(kwargs["kv_head"].item())
        assert kv_head_val == 2

    def test_empty_positions_window_skipped(self, tmp_path):
        """A window that yields no valid positions produces no 'wN' key."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        lib = MagicMock()
        # Empty token list → positions will be empty after clamping
        lib.get_window_tokens.return_value = []
        _wire_forward(kv_gen, layer_adapter, seq_len=0)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(engine, tmp_path, num_archived=1, config=_GEMMA4B_CONFIG, lib=lib, retrieval_layer=29, query_head=4)

        kwargs = mock_savez.call_args[1]
        assert "w0" not in kwargs

    def test_prefill_to_layer_called_with_correct_target(self, tmp_path):
        """prefill_to_layer should be called with target_layer=retrieval_layer."""
        engine, kv_gen, layer_adapter = _make_engine(num_layers=34)
        layer_adapter.head_dim = 8

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(5))
        _wire_forward(kv_gen, layer_adapter, seq_len=5)

        with patch.object(mx, "savez"):
            extract_kv_route_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                retrieval_layer=25,
                query_head=4,
            )

        kv_gen.prefill_to_layer.assert_called_once()
        call_kwargs = kv_gen.prefill_to_layer.call_args[1]
        assert call_kwargs.get("target_layer") == 25

    def test_sparse_mode_uses_fact_positions(self, tmp_path):
        """In SPARSE mode, positions from sparse_index.json are used when available."""
        from chuk_lazarus.cli.commands.context._types import KVectorMode

        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )

        # Write a sparse index with one position for window 0
        sparse_data = {"entries": [{"window_id": 0, "fact_spans": [{"position": 3}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(sparse_data))

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        _wire_forward(kv_gen, layer_adapter, seq_len=10)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                retrieval_layer=29,
                query_head=4,
                kvector_mode=KVectorMode.SPARSE,
            )

        kwargs = mock_savez.call_args[1]
        # w0 should exist with exactly 1 fact (position 3)
        assert "w0" in kwargs
        k_arr = kwargs["w0"]
        # shape[0] == number of fact positions == 1
        assert k_arr.shape[0] == 1

    def test_lib_none_uses_archive_retrieve(self, tmp_path):
        """When lib=None, engine.archive.retrieve() is called instead."""
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )
        engine.archive.retrieve.return_value = (list(range(10)), None)
        _wire_forward(kv_gen, layer_adapter, seq_len=10, hidden=64, head_dim=8, nkv=4)

        with patch.object(mx, "savez"):
            extract_kv_route_index(engine, tmp_path, num_archived=1, config=_GEMMA4B_CONFIG, lib=None, retrieval_layer=29, query_head=4)

        engine.archive.retrieve.assert_called_once_with(0)

    def test_full_mode_uses_all_positions(self, tmp_path):
        """KVectorMode.FULL should extract every token position."""
        from chuk_lazarus.cli.commands.context._types import KVectorMode

        seq_len = 6
        engine, kv_gen, layer_adapter = _make_engine(
            num_layers=34, hidden=64, head_dim=8, nkv=4, n_rep=2
        )
        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(seq_len))
        _wire_forward(kv_gen, layer_adapter, seq_len=seq_len, hidden=64, head_dim=8, nkv=4)

        with patch.object(mx, "savez") as mock_savez:
            extract_kv_route_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                retrieval_layer=29,
                query_head=4,
                kvector_mode=KVectorMode.FULL,
            )

        kwargs = mock_savez.call_args[1]
        assert "w0" in kwargs
        # FULL mode: one entry per token position
        assert kwargs["w0"].shape[0] == seq_len
