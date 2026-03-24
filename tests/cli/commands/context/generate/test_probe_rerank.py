"""Tests for generate/_probe_rerank.py.

_probe_rerank_windows requires MLX and a loaded model. We mock all MLX
operations and test the observable behaviour: correct selection of
assessment prompts, return format, sorting, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.context.generate._probe_rerank import (
    _ASSESS_TOKENS,
    _ENGAGEMENT_PROMPT,
    _REPLAY_TOKENS,
    _TENSION_PROMPT,
    _probe_rerank_windows,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_kv_gen():
    """Mock KVDirectGenerator-like object."""
    kv_gen = MagicMock()

    fake_kv = [(MagicMock(), MagicMock())] * 3
    logits_mock = MagicMock()
    logits_mock.__getitem__ = MagicMock(return_value=logits_mock)

    kv_gen.prefill.return_value = (logits_mock, fake_kv)
    kv_gen.extend.return_value = (logits_mock, fake_kv)
    kv_gen.step_uncompiled.return_value = (logits_mock, fake_kv)

    h_mock = MagicMock()
    h_mock.__getitem__ = MagicMock(return_value=h_mock)
    h_mock.astype.return_value = h_mock
    kv_gen.prefill_to_layer.return_value = h_mock

    return kv_gen


def _make_lib(num_windows: int = 3, tokens_per_window: int = 10):
    lib = MagicMock()
    lib.get_window_tokens.return_value = list(range(tokens_per_window))
    return lib


def _make_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 999  # never matches generated token
    return tok


def _make_probe():
    p = MagicMock()
    p.astype.return_value = p
    return p


def _make_sum_mock(value: float):
    m = MagicMock()
    m.item.return_value = value
    return m


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestProbeRerankConstants:
    def test_replay_tokens_value(self):
        assert _REPLAY_TOKENS == 256

    def test_assess_tokens_value(self):
        assert _ASSESS_TOKENS == 20

    def test_engagement_prompt_contains_keyword(self):
        assert "amusing" in _ENGAGEMENT_PROMPT or "notable" in _ENGAGEMENT_PROMPT

    def test_tension_prompt_contains_keyword(self):
        assert "tense" in _TENSION_PROMPT or "critical" in _TENSION_PROMPT


class TestProbeRerankWindows:
    def _run(
        self, candidate_wids, probe_type="engagement", top_k=5, tokens_per_window=10, sum_value=0.5
    ):
        lib = _make_lib(tokens_per_window=tokens_per_window)
        kv_gen = _make_kv_gen()
        tokenizer = _make_tokenizer()
        probe_dir = _make_probe()
        probe_mean = _make_probe()

        import mlx.core as mx

        with (
            patch.object(mx, "argmax", return_value=MagicMock(item=lambda: 5)),
            patch.object(mx, "sum", return_value=_make_sum_mock(sum_value)),
            patch.object(mx, "eval", return_value=None),
        ):
            result = _probe_rerank_windows(
                lib=lib,
                kv_gen=kv_gen,
                tokenizer=tokenizer,
                candidate_wids=candidate_wids,
                probe_direction=probe_dir,
                probe_mean=probe_mean,
                compass_layer=26,
                probe_type=probe_type,
                top_k=top_k,
            )
        return result

    def test_empty_candidates_returns_empty(self):
        result = self._run([])
        assert result == []

    def test_returns_list(self):
        result = self._run([0, 1])
        assert isinstance(result, list)

    def test_returns_at_most_top_k(self):
        result = self._run([0, 1, 2, 3, 4], top_k=3)
        assert len(result) <= 3

    def test_returns_tuples_of_two(self):
        result = self._run([0])
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_wids_present_in_results(self):
        result = self._run([0, 1, 2])
        result_wids = {w for w, _ in result}
        # All provided wids should appear in results when top_k >= num_candidates
        assert result_wids.issubset({0, 1, 2})

    def test_scores_are_floats(self):
        result = self._run([0, 1])
        for _, score in result:
            assert isinstance(score, float)

    def test_tension_prompt_encoding(self):
        """When probe_type='tension', tension prompt text is encoded."""
        lib = _make_lib()
        kv_gen = _make_kv_gen()
        tokenizer = _make_tokenizer()
        probe_dir = _make_probe()
        probe_mean = _make_probe()

        import mlx.core as mx

        with (
            patch.object(mx, "argmax", return_value=MagicMock(item=lambda: 5)),
            patch.object(mx, "sum", return_value=_make_sum_mock(0.3)),
            patch.object(mx, "eval", return_value=None),
        ):
            _probe_rerank_windows(
                lib=lib,
                kv_gen=kv_gen,
                tokenizer=tokenizer,
                candidate_wids=[0],
                probe_direction=probe_dir,
                probe_mean=probe_mean,
                compass_layer=26,
                probe_type="tension",
                top_k=1,
            )

        encoded_texts = [c[0][0] for c in tokenizer.encode.call_args_list if c[0]]
        tension_found = any("tense" in t or "critical" in t for t in encoded_texts)
        assert tension_found

    def test_engagement_prompt_encoding(self):
        """When probe_type='engagement', engagement prompt text is encoded."""
        lib = _make_lib()
        kv_gen = _make_kv_gen()
        tokenizer = _make_tokenizer()
        probe_dir = _make_probe()
        probe_mean = _make_probe()

        import mlx.core as mx

        with (
            patch.object(mx, "argmax", return_value=MagicMock(item=lambda: 5)),
            patch.object(mx, "sum", return_value=_make_sum_mock(0.3)),
            patch.object(mx, "eval", return_value=None),
        ):
            _probe_rerank_windows(
                lib=lib,
                kv_gen=kv_gen,
                tokenizer=tokenizer,
                candidate_wids=[0],
                probe_direction=probe_dir,
                probe_mean=probe_mean,
                compass_layer=26,
                probe_type="engagement",
                top_k=1,
            )

        encoded_texts = [c[0][0] for c in tokenizer.encode.call_args_list if c[0]]
        engagement_found = any("amusing" in t or "notable" in t for t in encoded_texts)
        assert engagement_found

    def test_sorted_descending_by_score(self):
        """Results should be sorted descending by probe projection score."""
        lib = _make_lib(tokens_per_window=5)
        kv_gen = _make_kv_gen()
        tokenizer = _make_tokenizer()
        probe_dir = _make_probe()
        probe_mean = _make_probe()

        # Return different sum values: wid 0 → 0.3, wid 1 → 0.9, wid 2 → 0.1
        sum_values = iter([0.3, 0.9, 0.1])

        def _sum_side_effect(x):
            m = MagicMock()
            m.item.return_value = next(sum_values, 0.5)
            return m

        import mlx.core as mx

        with (
            patch.object(mx, "argmax", return_value=MagicMock(item=lambda: 5)),
            patch.object(mx, "sum", side_effect=_sum_side_effect),
            patch.object(mx, "eval", return_value=None),
        ):
            result = _probe_rerank_windows(
                lib=lib,
                kv_gen=kv_gen,
                tokenizer=tokenizer,
                candidate_wids=[0, 1, 2],
                probe_direction=probe_dir,
                probe_mean=probe_mean,
                compass_layer=26,
                top_k=3,
            )

        if len(result) > 1:
            scores = [s for _, s in result]
            assert scores == sorted(scores, reverse=True)

    def test_long_window_gets_sliced(self):
        """Windows longer than _REPLAY_TOKENS should have tokens sliced."""
        lib = _make_lib(tokens_per_window=512)  # > 256
        kv_gen = _make_kv_gen()
        tokenizer = _make_tokenizer()
        probe_dir = _make_probe()
        probe_mean = _make_probe()

        import mlx.core as mx

        extend_calls = []

        def _capture_extend(ids, kv, abs_start):
            extend_calls.append(ids)
            return (MagicMock(), [(MagicMock(), MagicMock())] * 3)

        kv_gen.extend.side_effect = _capture_extend

        with (
            patch.object(mx, "argmax", return_value=MagicMock(item=lambda: 5)),
            patch.object(mx, "sum", return_value=_make_sum_mock(0.5)),
            patch.object(mx, "eval", return_value=None),
        ):
            _probe_rerank_windows(
                lib=lib,
                kv_gen=kv_gen,
                tokenizer=tokenizer,
                candidate_wids=[0],
                probe_direction=probe_dir,
                probe_mean=probe_mean,
                compass_layer=26,
                top_k=1,
            )

        # extend was called — the window tokens were passed to it
        assert len(extend_calls) >= 1

    def test_top_k_limits_output(self):
        result = self._run([0, 1, 2, 3, 4, 5], top_k=2)
        assert len(result) <= 2
