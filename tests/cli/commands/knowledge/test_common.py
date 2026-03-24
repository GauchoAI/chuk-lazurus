"""Tests for knowledge CLI shared helpers (_common.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx

from chuk_lazarus.cli.commands.knowledge._common import (
    generate_plain,
    prepare_prompt,
    stop_token_ids,
)

# ── prepare_prompt ───────────────────────────────────────────────────


class TestPreparePrompt:
    def test_uses_chat_template_when_available(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = [10, 20, 30]
        result = prepare_prompt(tok, "hello")
        assert result == [10, 20, 30]
        tok.apply_chat_template.assert_called_once()

    def test_falls_back_to_encode_when_no_template(self):
        tok = MagicMock(spec=["encode"])  # has encode but no apply_chat_template
        tok.encode.return_value = [1, 2, 3]
        result = prepare_prompt(tok, "hello")
        assert result == [1, 2, 3]
        tok.encode.assert_called_once_with("hello", add_special_tokens=True)

    def test_falls_back_on_template_error(self):
        tok = MagicMock()
        tok.apply_chat_template.side_effect = RuntimeError("broken")
        tok.encode.return_value = [5, 6]
        result = prepare_prompt(tok, "hi")
        assert result == [5, 6]


# ── stop_token_ids ───────────────────────────────────────────────────


class TestStopTokenIds:
    def test_returns_eos_set(self):
        tok = MagicMock()
        tok.eos_token_id = 42
        assert stop_token_ids(tok) == {42}

    def test_returns_empty_when_no_eos(self):
        tok = MagicMock()
        tok.eos_token_id = None
        assert stop_token_ids(tok) == set()


# ── generate_plain ───────────────────────────────────────────────────


class TestGeneratePlain:
    def _make_kv_gen(self, logits_sequence: list[int]):
        """Create a mock KV generator that returns sequential tokens.

        logits_sequence: list of token IDs the model should "predict".
        """
        kv_gen = MagicMock()
        vocab_size = 100

        def make_logits(token_id: int) -> mx.array:
            logits = mx.zeros((1, 1, vocab_size))
            # Make token_id the argmax
            logits = logits.at[0, 0, token_id].add(10.0)
            return logits

        # prefill returns first token's logits
        first_logits = (
            make_logits(logits_sequence[0]) if logits_sequence else mx.zeros((1, 1, vocab_size))
        )
        kv_gen.prefill.return_value = (first_logits, "kv_store")

        # step returns subsequent tokens
        call_idx = [0]

        def step_fn(token_ids, kv_store, seq_len):
            call_idx[0] += 1
            idx = min(call_idx[0], len(logits_sequence) - 1)
            return make_logits(logits_sequence[idx]), "kv_store"

        kv_gen.step_uncompiled.side_effect = step_fn
        return kv_gen

    def test_generates_tokens(self):
        kv_gen = self._make_kv_gen([5, 6, 7])
        result = generate_plain(kv_gen, [1, 2, 3], max_tokens=3, stop_ids=set())
        assert len(result) == 3
        assert result[0] == 5

    def test_stops_on_eos(self):
        kv_gen = self._make_kv_gen([5, 6, 99, 7])
        result = generate_plain(kv_gen, [1, 2], max_tokens=10, stop_ids={99})
        assert len(result) == 2  # stops before 99
        assert 99 not in result

    def test_respects_max_tokens(self):
        kv_gen = self._make_kv_gen([5, 6, 7, 8, 9])
        result = generate_plain(kv_gen, [1], max_tokens=2, stop_ids=set())
        assert len(result) == 2

    def test_empty_generation_on_immediate_eos(self):
        kv_gen = self._make_kv_gen([99])
        result = generate_plain(kv_gen, [1, 2], max_tokens=10, stop_ids={99})
        assert result == []
