"""Tests for compass_routing/_indicator_bm25.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chuk_lazarus.cli.commands.context.compass_routing._indicator_bm25 import (
    ENGAGEMENT_INDICATORS,
    TENSION_INDICATORS,
    _indicator_bm25_score_windows,
)


def _make_lib(window_texts: list[str]) -> MagicMock:
    """Build a mock lib with num_windows and get_window_tokens."""
    lib = MagicMock()
    lib.num_windows = len(window_texts)

    # get_window_tokens(wid) → list of fake token ints (just indices)
    def _get_tokens(wid: int) -> list[int]:
        return list(range(len(window_texts[wid].split())))

    lib.get_window_tokens.side_effect = _get_tokens
    return lib


def _make_tokenizer(window_texts: list[str]) -> MagicMock:
    """Build a mock tokenizer whose decode returns the indexed window text."""
    tokenizer = MagicMock()

    def _decode(tokens, skip_special_tokens=True) -> str:
        # Tokens are just range indices; we route by position in window_texts
        # This is a simplification — just return the text based on token length
        # We encode position via the token list length matching text word count
        return " ".join(["word"] * len(tokens))

    tokenizer.decode.side_effect = _decode
    return tokenizer


def _make_fixtures(window_texts: list[str]):
    """Create lib+tokenizer where decode returns the real text per window."""
    lib = MagicMock()
    lib.num_windows = len(window_texts)

    token_store: dict[int, list[str]] = {i: text.split() for i, text in enumerate(window_texts)}

    def _get_tokens(wid: int) -> list[int]:
        # Encode words as their index in the word list (fake token ids)
        return list(range(len(token_store[wid])))

    lib.get_window_tokens.side_effect = _get_tokens

    tokenizer = MagicMock()

    def _decode(tokens, skip_special_tokens=True) -> str:
        # We need to map back — but we use the call count to track wid
        # Instead, encode the wid into the token list length via a closure
        # Simpler: return based on call order
        raise NotImplementedError("use _make_real_fixtures")

    tokenizer.decode.side_effect = _decode
    return lib, tokenizer


def _make_real_fixtures(window_texts: list[str]):
    """Fixtures where tokenizer.decode actually returns the per-window text."""
    lib = MagicMock()
    lib.num_windows = len(window_texts)

    def _get_tokens(wid: int) -> list[int]:
        return list(range(len(window_texts[wid].split()) + 1))  # +1 so len is unique

    lib.get_window_tokens.side_effect = _get_tokens

    call_counter = {"n": 0}

    def _decode(tokens, skip_special_tokens=True) -> str:
        idx = call_counter["n"]
        call_counter["n"] += 1
        return window_texts[idx]

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = _decode
    return lib, tokenizer


class TestIndicatorBm25Constants:
    def test_engagement_indicators_nonempty(self):
        assert len(ENGAGEMENT_INDICATORS) > 0

    def test_tension_indicators_nonempty(self):
        assert len(TENSION_INDICATORS) > 0

    def test_engagement_contains_expected(self):
        assert "laughter" in ENGAGEMENT_INDICATORS
        assert "funny" in ENGAGEMENT_INDICATORS

    def test_tension_contains_expected(self):
        assert "emergency" in TENSION_INDICATORS
        assert "warning" in TENSION_INDICATORS


class TestIndicatorBm25ScoreWindows:
    def test_empty_indicators_returns_zeros(self):
        lib, tok = _make_real_fixtures(["hello world", "goodbye moon"])
        result = _indicator_bm25_score_windows(lib, tok, indicators=[])
        assert len(result) == 2
        for wid, score in result:
            assert score == pytest.approx(0.0)

    def test_returns_one_result_per_window(self):
        texts = ["there was laughter in the hall", "the fuel alarm sounded", "nothing special here"]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, ENGAGEMENT_INDICATORS)
        assert len(result) == 3

    def test_result_is_sorted_descending(self):
        texts = [
            "laughter and music and funny jokes",
            "plain text with nothing",
            "birthday congratulations amazing",
        ]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, ENGAGEMENT_INDICATORS)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_matching_window_scores_higher(self):
        texts = [
            "the alarm sounded emergency abort critical",
            "a nice quiet day with no issues at all",
        ]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, TENSION_INDICATORS)
        score_by_wid = dict(result)
        assert score_by_wid[0] > score_by_wid[1]

    def test_no_matching_windows_all_zero(self):
        texts = ["the cat sat on the mat", "dogs run in the park"]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, ["xylophone", "quasar"])
        for _, score in result:
            assert score == pytest.approx(0.0)

    def test_scores_are_non_negative(self):
        texts = ["laughter and music everywhere", "emergency abort warning alarm"]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, ENGAGEMENT_INDICATORS)
        for _, score in result:
            assert score >= 0.0

    def test_single_window(self):
        texts = ["the fuel warning light came on during the emergency"]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, TENSION_INDICATORS)
        assert len(result) == 1
        _, score = result[0]
        assert score > 0

    def test_custom_k1_b_params(self):
        texts = ["laughter music funny birthday", "plain text"]
        lib, tok = _make_real_fixtures(texts)
        r1 = _indicator_bm25_score_windows(lib, tok, ENGAGEMENT_INDICATORS, k1=1.0, b=0.5)
        lib2, tok2 = _make_real_fixtures(texts)
        r2 = _indicator_bm25_score_windows(lib2, tok2, ENGAGEMENT_INDICATORS, k1=2.0, b=0.75)
        # Both should still rank the matching window first
        assert r1[0][0] == 0
        assert r2[0][0] == 0

    def test_window_ids_present_in_result(self):
        texts = ["a", "b", "c"]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, ENGAGEMENT_INDICATORS)
        wids = {w for w, _ in result}
        assert wids == {0, 1, 2}

    def test_bm25_idf_formula_respected(self):
        """Window where indicator appears in fewer docs should score higher via IDF."""
        texts = [
            "laughter laughter laughter laughter laughter",  # many repeats
            "laughter plain",  # one hit, different doc
            "plain text no indicators here at all",
        ]
        lib, tok = _make_real_fixtures(texts)
        result = _indicator_bm25_score_windows(lib, tok, ["laughter"])
        score_by_wid = dict(result)
        # Both window 0 and 1 have 'laughter'; window 0 has more term frequency
        assert score_by_wid[0] > score_by_wid[2]
        assert score_by_wid[1] > score_by_wid[2]
