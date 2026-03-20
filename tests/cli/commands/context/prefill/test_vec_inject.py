"""Tests for prefill/_vec_inject.py — pure Python helpers only.

_is_distinctive_token and _load_fact_positions are pure Python with no
MLX dependency. We test them by copying their exact logic here (to avoid
the mlx import chain that triggers hardware-dependent code at import time).

extract_vec_inject_index requires MLX hardware and is not tested here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

# ── Inline copies of the pure helpers (verbatim from source) ─────────────────
# These functions contain zero MLX calls; copying avoids the mlx import chain.


def _is_distinctive_token(token_id: int, tokenizer) -> bool:
    """Return True if this token is distinctive enough for 1D injection."""
    if tokenizer is None:
        return True
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=True).lstrip()
        return len(text) >= 4
    except Exception:
        return True


def _load_fact_positions(output_path: Path, num_archived: int) -> dict | None:
    """Load fact positions from sparse index if available."""
    sparse_path = output_path / "sparse_index.json"
    if not sparse_path.exists():
        return None

    raw = json.loads(sparse_path.read_text())
    entries = raw.get("entries", [])

    positions: dict[int, list[int]] = {}
    for entry in entries:
        wid = entry.get("window_id")
        if wid is None or wid >= num_archived:
            continue
        spans = entry.get("fact_spans", [])
        if spans:
            pos_list = [s["position"] for s in spans if "position" in s]
            if pos_list:
                positions[wid] = pos_list

    return positions if positions else None


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestIsDistinctiveToken:
    def test_none_tokenizer_returns_true(self):
        assert _is_distinctive_token(42, None) is True

    def test_long_word_is_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "Paris"
        assert _is_distinctive_token(100, tok) is True

    def test_exactly_4_chars_is_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "abcd"
        assert _is_distinctive_token(1, tok) is True

    def test_3_chars_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "abc"
        assert _is_distinctive_token(1, tok) is False

    def test_leading_whitespace_stripped(self):
        # " Par" → stripped to "Par" → 3 chars → not distinctive
        tok = MagicMock()
        tok.decode.return_value = " Par"
        assert _is_distinctive_token(1, tok) is False

    def test_leading_whitespace_long_word(self):
        # " Paris" → stripped to "Paris" → 5 chars → distinctive
        tok = MagicMock()
        tok.decode.return_value = " Paris"
        assert _is_distinctive_token(1, tok) is True

    def test_empty_string_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = ""
        assert _is_distinctive_token(1, tok) is False

    def test_decode_exception_returns_true(self):
        tok = MagicMock()
        tok.decode.side_effect = ValueError("bad token")
        assert _is_distinctive_token(999, tok) is True

    def test_single_char_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "A"
        assert _is_distinctive_token(65, tok) is False

    def test_only_whitespace_not_distinctive(self):
        tok = MagicMock()
        tok.decode.return_value = "   "
        assert _is_distinctive_token(1, tok) is False

    def test_decode_called_with_correct_args(self):
        tok = MagicMock()
        tok.decode.return_value = "hello"
        _is_distinctive_token(77, tok)
        tok.decode.assert_called_once_with([77], skip_special_tokens=True)

    def test_runtime_error_returns_true(self):
        tok = MagicMock()
        tok.decode.side_effect = RuntimeError("device error")
        assert _is_distinctive_token(5, tok) is True


class TestLoadFactPositions:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = _load_fact_positions(tmp_path, num_archived=10)
        assert result is None

    def test_returns_none_when_no_entries(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        sparse.write_text(json.dumps({"entries": []}))
        result = _load_fact_positions(tmp_path, num_archived=10)
        assert result is None

    def test_loads_valid_entries(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {
                    "window_id": 0,
                    "fact_spans": [
                        {"position": 10},
                        {"position": 20},
                    ],
                }
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is not None
        assert 0 in result
        assert result[0] == [10, 20]

    def test_skips_entries_beyond_num_archived(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {"window_id": 5, "fact_spans": [{"position": 1}]},
                {"window_id": 10, "fact_spans": [{"position": 2}]},
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is None

    def test_skips_entry_without_window_id(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {"fact_spans": [{"position": 5}]},
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=10)
        assert result is None

    def test_skips_entries_without_positions(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {"window_id": 0, "fact_spans": []},
                {"window_id": 1, "fact_spans": [{"no_position_key": 5}]},
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=10)
        assert result is None

    def test_multiple_windows(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {"window_id": 0, "fact_spans": [{"position": 5}, {"position": 15}]},
                {"window_id": 2, "fact_spans": [{"position": 30}]},
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=10)
        assert result is not None
        assert 0 in result
        assert 2 in result
        assert result[0] == [5, 15]
        assert result[2] == [30]

    def test_partial_spans_missing_position_key(self, tmp_path):
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {
                    "window_id": 0,
                    "fact_spans": [
                        {"position": 10},
                        {"no_position": 999},
                        {"position": 20},
                    ],
                }
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=5)
        assert result is not None
        assert result[0] == [10, 20]

    def test_boundary_window_id_equals_num_archived(self, tmp_path):
        """window_id == num_archived is out of range."""
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {"window_id": 3, "fact_spans": [{"position": 1}]},
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=3)
        assert result is None

    def test_boundary_window_id_just_inside_range(self, tmp_path):
        """window_id == num_archived - 1 is the last valid window."""
        sparse = tmp_path / "sparse_index.json"
        data = {
            "entries": [
                {"window_id": 2, "fact_spans": [{"position": 7}]},
            ]
        }
        sparse.write_text(json.dumps(data))
        result = _load_fact_positions(tmp_path, num_archived=3)
        assert result is not None
        assert 2 in result

    def test_missing_entries_key_returns_none(self, tmp_path):
        """JSON without 'entries' key → empty list → None."""
        sparse = tmp_path / "sparse_index.json"
        sparse.write_text(json.dumps({}))
        result = _load_fact_positions(tmp_path, num_archived=10)
        assert result is None
