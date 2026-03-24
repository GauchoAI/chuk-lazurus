"""Tests for vec_inject/providers/_index_format.py."""

from __future__ import annotations

from chuk_lazarus.inference.context.vec_inject.providers._index_format import (
    KV_ROUTE_FILE,
    VEC_INJECT_FILE,
    VecInjectMetaKey,
    VecInjectWindowKey,
)


class TestVecInjectMetaKey:
    def test_layer_value(self):
        assert VecInjectMetaKey.LAYER == "layer"

    def test_kv_head_value(self):
        assert VecInjectMetaKey.KV_HEAD == "kv_head"

    def test_query_head_value(self):
        assert VecInjectMetaKey.QUERY_HEAD == "query_head"

    def test_inject_layer_value(self):
        assert VecInjectMetaKey.INJECT_LAYER == "inject_layer"

    def test_all_members_are_strings(self):
        for key in VecInjectMetaKey:
            assert isinstance(str(key), str)

    def test_str_enum(self):
        # VecInjectMetaKey is a StrEnum — value IS the string
        assert VecInjectMetaKey.LAYER == "layer"


class TestVecInjectWindowKeyStaticMethods:
    def test_k_vecs_format(self):
        assert VecInjectWindowKey.k_vecs(0) == "w0/k_vecs"
        assert VecInjectWindowKey.k_vecs(5) == "w5/k_vecs"
        assert VecInjectWindowKey.k_vecs(100) == "w100/k_vecs"

    def test_token_ids_format(self):
        assert VecInjectWindowKey.token_ids(0) == "w0/token_ids"
        assert VecInjectWindowKey.token_ids(7) == "w7/token_ids"

    def test_coefs_format(self):
        assert VecInjectWindowKey.coefs(3) == "w3/coefs"
        assert VecInjectWindowKey.coefs(0) == "w0/coefs"

    def test_positions_format(self):
        assert VecInjectWindowKey.positions(2) == "w2/positions"

    def test_distinctive_format(self):
        assert VecInjectWindowKey.distinctive(4) == "w4/distinctive"
        assert VecInjectWindowKey.distinctive(0) == "w0/distinctive"

    def test_flat_format(self):
        assert VecInjectWindowKey.flat(0) == "w0"
        assert VecInjectWindowKey.flat(9) == "w9"

    def test_all_keys_are_strings(self):
        for wid in [0, 1, 10, 99]:
            assert isinstance(VecInjectWindowKey.k_vecs(wid), str)
            assert isinstance(VecInjectWindowKey.token_ids(wid), str)
            assert isinstance(VecInjectWindowKey.coefs(wid), str)
            assert isinstance(VecInjectWindowKey.positions(wid), str)
            assert isinstance(VecInjectWindowKey.distinctive(wid), str)
            assert isinstance(VecInjectWindowKey.flat(wid), str)


class TestVecInjectWindowKeyFromKey:
    def test_flat_key_wN(self):
        assert VecInjectWindowKey.window_id_from_key("w0") == 0
        assert VecInjectWindowKey.window_id_from_key("w5") == 5
        assert VecInjectWindowKey.window_id_from_key("w99") == 99

    def test_subpath_key_wN_slash_field(self):
        assert VecInjectWindowKey.window_id_from_key("w3/k_vecs") == 3
        assert VecInjectWindowKey.window_id_from_key("w10/token_ids") == 10
        assert VecInjectWindowKey.window_id_from_key("w0/coefs") == 0

    def test_meta_key_returns_none(self):
        assert VecInjectWindowKey.window_id_from_key("layer") is None
        assert VecInjectWindowKey.window_id_from_key("kv_head") is None
        assert VecInjectWindowKey.window_id_from_key("query_head") is None
        assert VecInjectWindowKey.window_id_from_key("inject_layer") is None

    def test_empty_string_returns_none(self):
        assert VecInjectWindowKey.window_id_from_key("") is None

    def test_non_window_key_returns_none(self):
        assert VecInjectWindowKey.window_id_from_key("abc") is None
        assert VecInjectWindowKey.window_id_from_key("wabc") is None

    def test_roundtrip_k_vecs(self):
        for wid in [0, 1, 42, 724]:
            key = VecInjectWindowKey.k_vecs(wid)
            recovered = VecInjectWindowKey.window_id_from_key(key)
            assert recovered == wid

    def test_roundtrip_flat(self):
        for wid in [0, 5, 100]:
            key = VecInjectWindowKey.flat(wid)
            recovered = VecInjectWindowKey.window_id_from_key(key)
            assert recovered == wid

    def test_roundtrip_token_ids(self):
        key = VecInjectWindowKey.token_ids(17)
        assert VecInjectWindowKey.window_id_from_key(key) == 17

    def test_roundtrip_coefs(self):
        key = VecInjectWindowKey.coefs(33)
        assert VecInjectWindowKey.window_id_from_key(key) == 33

    def test_roundtrip_positions(self):
        key = VecInjectWindowKey.positions(8)
        assert VecInjectWindowKey.window_id_from_key(key) == 8

    def test_roundtrip_distinctive(self):
        key = VecInjectWindowKey.distinctive(12)
        assert VecInjectWindowKey.window_id_from_key(key) == 12

    def test_w_prefix_but_non_digit_returns_none(self):
        assert VecInjectWindowKey.window_id_from_key("wabc/field") is None


class TestFileConstants:
    def test_vec_inject_filename(self):
        assert VEC_INJECT_FILE == "vec_inject.npz"

    def test_kv_route_filename(self):
        assert KV_ROUTE_FILE == "kv_route_index.npz"
