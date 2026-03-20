"""Full tests for prefill/_vec_inject.py — MLX + mocked engine.

Complements test_vec_inject.py which uses inline copies of the pure helpers.
This file imports the helpers directly from source and tests
extract_vec_inject_index with real MLX arrays + mocked backbone/kv_gen.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import mlx.core as mx

from chuk_lazarus.cli.commands.context.prefill._vec_inject import (
    _is_distinctive_token,
    _load_fact_positions,
    extract_vec_inject_index,
)

# Minimal model config that satisfies ArchitectureConfig.from_model_config()
# so tests don't trigger copy-head discovery (which requires real weight matrices).
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
    engine = MagicMock()
    kv_gen = MagicMock()
    backbone = MagicMock()
    layer_adapter = MagicMock()

    layer_adapter.n_rep = n_rep
    layer_adapter.num_kv_heads = nkv
    layer_adapter.head_dim = head_dim

    # Real list so len() and [i] both work
    backbone.adapted_layers = [layer_adapter] * num_layers
    backbone.hidden_size = hidden
    kv_gen.backbone = backbone
    engine.kv_gen = kv_gen

    return engine, kv_gen, backbone, layer_adapter


def _wire_forward(
    kv_gen,
    backbone,
    layer_adapter,
    seq_len: int = 10,
    hidden: int = 64,
    head_dim: int = 8,
    nkv: int = 4,
    n_rep: int = 2,
):
    """Wire real MLX zeros as return values for every forward call."""
    h = mx.zeros((1, seq_len, hidden))
    kv_gen.prefill_to_layer.return_value = h

    x_norm = mx.zeros((1, seq_len, hidden))
    layer_adapter.pre_attn_norm.return_value = x_norm

    nq = nkv * n_rep
    q = mx.zeros((1, nq, seq_len, head_dim))
    k = mx.zeros((1, nkv, seq_len, head_dim))
    v = mx.zeros((1, nkv, seq_len, head_dim))
    layer_adapter.project_qkv.return_value = (q, k, v)

    # embed: returns (1, 1, hidden) for each token
    backbone.embed.return_value = mx.zeros((1, 1, hidden))

    # H4 output computation (Pass 2) — needs real floats for SDPA and O_proj
    layer_adapter.attn_scale = head_dim ** -0.5
    backbone.prefill_mask.return_value = None  # no mask for test sequences
    # o_proj.weight shape: (hidden, nq * head_dim)
    layer_adapter._block.self_attn.o_proj.weight = mx.zeros((hidden, nq * head_dim))


# ── _is_distinctive_token (imported from source) ─────────────────────────────


class TestIsDistinctiveTokenFromSource:
    """Tests import the function directly from source, not an inline copy."""

    def test_none_tokenizer_returns_true(self):
        assert _is_distinctive_token(42, None) is True

    def test_long_word_is_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "Armstrong"
        assert _is_distinctive_token(42, tok) is True

    def test_exactly_4_chars_is_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "abcd"
        assert _is_distinctive_token(1, tok) is True

    def test_3_chars_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "abc"
        assert _is_distinctive_token(1, tok) is False

    def test_single_char_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "A"
        assert _is_distinctive_token(65, tok) is False

    def test_whitespace_only_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "  "
        assert _is_distinctive_token(1, tok) is False

    def test_leading_whitespace_stripped_before_length_check(self):
        # " Par" → stripped → "Par" → 3 chars → not distinctive
        tok = MagicMock()
        tok.decode.return_value = " Par"
        assert _is_distinctive_token(1, tok) is False

    def test_leading_whitespace_long_word_distinctive(self):
        # " Paris" → stripped → "Paris" → 5 chars → distinctive
        tok = MagicMock()
        tok.decode.return_value = " Paris"
        assert _is_distinctive_token(1, tok) is True

    def test_decode_exception_returns_true(self):
        tok = MagicMock()
        tok.decode.side_effect = ValueError("bad token")
        assert _is_distinctive_token(999, tok) is True

    def test_runtime_error_returns_true(self):
        tok = MagicMock()
        tok.decode.side_effect = RuntimeError("device error")
        assert _is_distinctive_token(5, tok) is True

    def test_decode_called_with_correct_args(self):
        tok = MagicMock()
        tok.decode.return_value = "hello"
        _is_distinctive_token(77, tok)
        tok.decode.assert_called_once_with([77], skip_special_tokens=True)

    def test_empty_string_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = ""
        assert _is_distinctive_token(1, tok) is False


# ── _load_fact_positions (imported from source) ───────────────────────────────


class TestLoadFactPositionsFromSource:
    def test_missing_file_returns_none(self, tmp_path):
        assert _load_fact_positions(tmp_path, 5) is None

    def test_empty_entries_returns_none(self, tmp_path):
        (tmp_path / "sparse_index.json").write_text(json.dumps({"entries": []}))
        assert _load_fact_positions(tmp_path, 5) is None

    def test_valid_entry(self, tmp_path):
        data = {"entries": [{"window_id": 0, "fact_spans": [{"position": 4}, {"position": 8}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, 5)
        assert result == {0: [4, 8]}

    def test_out_of_range_window_skipped(self, tmp_path):
        data = {"entries": [{"window_id": 5, "fact_spans": [{"position": 1}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        assert _load_fact_positions(tmp_path, 5) is None

    def test_multiple_windows(self, tmp_path):
        data = {
            "entries": [
                {"window_id": 0, "fact_spans": [{"position": 2}]},
                {"window_id": 1, "fact_spans": [{"position": 6}, {"position": 9}]},
            ]
        }
        (tmp_path / "sparse_index.json").write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, 5)
        assert result == {0: [2], 1: [6, 9]}


# ── extract_vec_inject_index ──────────────────────────────────────────────────


class TestExtractVecInjectIndex:
    def test_calls_np_savez_once(self, tmp_path):
        """extract_vec_inject_index should call np.savez exactly once."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        mock_savez.assert_called_once()

    def test_output_filename_correct(self, tmp_path):
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        path_arg = mock_savez.call_args[0][0]
        assert "vec_inject.npz" in path_arg

    def test_savez_has_metadata_keys(self, tmp_path):
        """Saved dict must include layer, kv_head, query_head, inject_layer."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        kwargs = mock_savez.call_args[1]
        assert "layer" in kwargs
        assert "kv_head" in kwargs
        assert "query_head" in kwargs
        assert "inject_layer" in kwargs

    def test_savez_has_window_k_vecs_key(self, tmp_path):
        """Saved dict must include 'w0/k_vecs' for window 0."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        kwargs = mock_savez.call_args[1]
        assert "w0/k_vecs" in kwargs

    def test_savez_has_window_token_ids_and_coefs(self, tmp_path):
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        kwargs = mock_savez.call_args[1]
        assert "w0/token_ids" in kwargs
        assert "w0/coefs" in kwargs
        assert "w0/positions" in kwargs
        assert "w0/distinctive" in kwargs

    def test_two_windows_calls_lib_twice(self, tmp_path):
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez"):
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=2,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        assert lib.get_window_tokens.call_count == 2

    def test_retrieval_layer_clamped(self, tmp_path):
        """retrieval_layer >= num_layers is clamped to num_layers - 1."""
        num_layers = 5
        engine, kv_gen, backbone, layer_adapter = _make_engine(num_layers=num_layers)
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                retrieval_layer=99,
            )

        kwargs = mock_savez.call_args[1]
        layer_val = int(kwargs["layer"])
        assert layer_val == num_layers - 1

    def test_inject_layer_defaults_to_retrieval_plus_one(self, tmp_path):
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                retrieval_layer=20,  # inject_layer should be 21
            )

        kwargs = mock_savez.call_args[1]
        inject_layer_val = int(kwargs["inject_layer"])
        assert inject_layer_val == 21

    def test_inject_layer_explicit(self, tmp_path):
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                retrieval_layer=15,
                inject_layer=18,
            )

        kwargs = mock_savez.call_args[1]
        assert int(kwargs["inject_layer"]) == 18

    def test_distinctive_flag_set_for_long_token(self, tmp_path):
        """Tokens that decode to ≥4 chars should produce distinctive=1."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=1)

        lib = MagicMock()
        lib.get_window_tokens.return_value = [42]  # single token

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"  # 9 chars → distinctive

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        kwargs = mock_savez.call_args[1]
        distinctive_arr = kwargs["w0/distinctive"]
        assert int(distinctive_arr[0]) == 1

    def test_distinctive_flag_clear_for_short_token(self, tmp_path):
        """Tokens that decode to <4 chars should produce distinctive=0."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=1)

        lib = MagicMock()
        lib.get_window_tokens.return_value = [7]  # single token

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "A"  # 1 char → not distinctive

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        kwargs = mock_savez.call_args[1]
        distinctive_arr = kwargs["w0/distinctive"]
        assert int(distinctive_arr[0]) == 0

    def test_no_tokenizer_all_distinctive(self, tmp_path):
        """When tokenizer is None, all tokens should be marked distinctive."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=3)

        lib = MagicMock()
        lib.get_window_tokens.return_value = [1, 2, 3]

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=None,
            )

        kwargs = mock_savez.call_args[1]
        distinctive_arr = kwargs["w0/distinctive"]
        # All 3 should be 1
        assert list(map(int, distinctive_arr)) == [1, 1, 1]

    def test_empty_token_list_window_skipped(self, tmp_path):
        """A window with no tokens → no 'w0/k_vecs' key in saved dict."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=0)

        lib = MagicMock()
        lib.get_window_tokens.return_value = []

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
            )

        kwargs = mock_savez.call_args[1]
        assert "w0/k_vecs" not in kwargs

    def test_prefill_to_layer_called_with_correct_target(self, tmp_path):
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=5)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(5))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez"):
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                retrieval_layer=22,
            )

        # prefill_to_layer is called twice: Pass 1 (retrieval_layer) + Pass 2 (H4 at retrieval_layer-1)
        assert kv_gen.prefill_to_layer.call_count >= 1
        targets = [c[1].get("target_layer") for c in kv_gen.prefill_to_layer.call_args_list]
        assert 22 in targets

    def test_sparse_mode_uses_fact_positions(self, tmp_path):
        """In SPARSE mode with a sparse index, only fact positions are extracted."""
        from chuk_lazarus.cli.commands.context._types import KVectorMode

        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=10)

        sparse_data = {"entries": [{"window_id": 0, "fact_spans": [{"position": 5}]}]}
        (tmp_path / "sparse_index.json").write_text(json.dumps(sparse_data))

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                kvector_mode=KVectorMode.SPARSE,
            )

        kwargs = mock_savez.call_args[1]
        # Exactly 1 fact (position 5)
        assert kwargs["w0/k_vecs"].shape[0] == 1

    def test_kv_head_computed_correctly(self, tmp_path):
        """kv_head = query_head // n_rep, capped at nkv-1."""
        # n_rep=2, nkv=4, query_head=4 → kv_head=2
        engine, kv_gen, backbone, layer_adapter = _make_engine(nkv=4, n_rep=2)
        _wire_forward(kv_gen, backbone, layer_adapter)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(10))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                query_head=4,
            )

        kwargs = mock_savez.call_args[1]
        assert int(kwargs["kv_head"]) == 2

    def test_lib_none_uses_archive_retrieve(self, tmp_path):
        """When lib=None, engine.archive.retrieve() is called for tokens."""
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        engine.archive.retrieve.return_value = (list(range(10)), None)
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=10)

        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez"):
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=None,
                tokenizer=tokenizer,
            )

        engine.archive.retrieve.assert_called_once_with(0)

    def test_full_mode_extracts_all_positions(self, tmp_path):
        """KVectorMode.FULL should extract every token position."""
        from chuk_lazarus.cli.commands.context._types import KVectorMode

        seq_len = 5
        engine, kv_gen, backbone, layer_adapter = _make_engine()
        _wire_forward(kv_gen, backbone, layer_adapter, seq_len=seq_len)

        lib = MagicMock()
        lib.get_window_tokens.return_value = list(range(seq_len))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Armstrong"

        with patch("numpy.savez") as mock_savez:
            extract_vec_inject_index(
                engine,
                tmp_path,
                num_archived=1,
                config=_GEMMA4B_CONFIG,
                lib=lib,
                tokenizer=tokenizer,
                kvector_mode=KVectorMode.FULL,
            )

        kwargs = mock_savez.call_args[1]
        assert kwargs["w0/k_vecs"].shape[0] == seq_len
