"""Tests for context CLI types."""

from argparse import Namespace
from pathlib import Path

from chuk_lazarus.cli.commands.context._types import GenerateConfig, PrefillConfig


class TestPrefillConfig:
    def _args(self, **kw) -> Namespace:
        defaults = {
            "model": "test-model",
            "input": "input.txt",
            "checkpoint": "./ckpt",
            "window_size": 512,
            "max_tokens": None,
            "no_resume": False,
        }
        defaults.update(kw)
        return Namespace(**defaults)

    def test_from_args_basic(self):
        cfg = PrefillConfig.from_args(self._args())
        assert cfg.model == "test-model"
        assert cfg.input_file == Path("input.txt")
        assert cfg.window_size == 512
        assert cfg.resume is True

    def test_no_resume_flag(self):
        cfg = PrefillConfig.from_args(self._args(no_resume=True))
        assert cfg.resume is False

    def test_max_tokens(self):
        cfg = PrefillConfig.from_args(self._args(max_tokens=1000))
        assert cfg.max_tokens == 1000


class TestGenerateConfig:
    def _args(self, **kw) -> Namespace:
        defaults = {
            "model": "test-model",
            "checkpoint": "./ckpt",
            "prompt": "What is this about?",
            "prompt_file": None,
            "max_tokens": 200,
            "temperature": 0.7,
        }
        defaults.update(kw)
        return Namespace(**defaults)

    def test_from_args_basic(self):
        cfg = GenerateConfig.from_args(self._args())
        assert cfg.model == "test-model"
        assert cfg.checkpoint == Path("./ckpt")
        assert cfg.temperature == 0.7

    def test_prompt_text_inline(self):
        cfg = GenerateConfig.from_args(self._args(prompt="Hello"))
        assert cfg.prompt_text == "Hello"

    def test_prompt_text_none(self):
        cfg = GenerateConfig.from_args(self._args(prompt=None))
        assert cfg.prompt_text is None
