"""Extended tests for context _types.py to bring coverage above 90%."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from chuk_lazarus.cli.commands.context._types import (
    GenerateConfig,
    GenerateResult,
    KVectorMode,
    PrefillConfig,
    PrefillPhase,
    PrefillResult,
    ResidualMode,
)

# ── KVectorMode ──────────────────────────────────────────────────────────────


class TestKVectorMode:
    def test_str_sparse(self):
        assert str(KVectorMode.SPARSE) == "sparse"

    def test_str_interval(self):
        assert str(KVectorMode.INTERVAL) == "interval"

    def test_str_full(self):
        assert str(KVectorMode.FULL) == "full"

    def test_values_are_strings(self):
        for mode in KVectorMode:
            assert isinstance(str(mode), str)


# ── ResidualMode ─────────────────────────────────────────────────────────────


class TestResidualMode:
    def test_value_interval(self):
        assert ResidualMode.INTERVAL.value == "interval"

    def test_value_full(self):
        assert ResidualMode.FULL.value == "full"

    def test_value_none(self):
        assert ResidualMode.NONE.value == "none"

    def test_value_darkspace(self):
        assert ResidualMode.DARKSPACE.value == "darkspace"

    def test_all_members_have_str(self):
        for mode in ResidualMode:
            assert isinstance(str(mode), str)

    def test_is_str_enum(self):
        # ResidualMode is (str, Enum) — its value IS a string
        assert isinstance(ResidualMode.INTERVAL.value, str)


# ── PrefillPhase ─────────────────────────────────────────────────────────────


class TestPrefillPhase:
    def test_str_all(self):
        assert str(PrefillPhase.ALL) == "all"

    def test_str_windows(self):
        assert str(PrefillPhase.WINDOWS) == "windows"

    def test_str_vec_inject(self):
        assert str(PrefillPhase.VEC_INJECT) == "vec_inject"

    def test_str_mode7(self):
        assert str(PrefillPhase.MODE7) == "mode7"


# ── PrefillConfig run_* properties ───────────────────────────────────────────


def _make_prefill_config(phases_str: str = "all", **overrides) -> PrefillConfig:
    """Helper to create a minimal PrefillConfig."""
    phases = {PrefillPhase(p.strip()) for p in phases_str.split(",")}
    defaults = {
        "model": "test-model",
        "input_file": Path("/tmp/input.txt"),
        "checkpoint": Path("/tmp/checkpoint"),
        "phases": phases,
    }
    defaults.update(overrides)
    return PrefillConfig(**defaults)


class TestPrefillConfigRunProperties:
    def test_run_windows_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_windows is True

    def test_run_windows_explicit(self):
        cfg = _make_prefill_config("windows")
        assert cfg.run_windows is True

    def test_run_windows_not_in_phases(self):
        cfg = _make_prefill_config("compass")
        assert cfg.run_windows is False

    def test_run_interval_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_interval is True

    def test_run_interval_explicit(self):
        cfg = _make_prefill_config("interval")
        assert cfg.run_interval is True

    def test_run_interval_missing(self):
        cfg = _make_prefill_config("compass")
        assert cfg.run_interval is False

    def test_run_compass_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_compass is True

    def test_run_compass_explicit(self):
        cfg = _make_prefill_config("compass")
        assert cfg.run_compass is True

    def test_run_compass_missing(self):
        cfg = _make_prefill_config("windows")
        assert cfg.run_compass is False

    def test_run_darkspace_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_darkspace is True

    def test_run_darkspace_explicit(self):
        cfg = _make_prefill_config("darkspace")
        assert cfg.run_darkspace is True

    def test_run_darkspace_missing(self):
        cfg = _make_prefill_config("windows")
        assert cfg.run_darkspace is False

    def test_run_pages_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_pages is True

    def test_run_pages_explicit(self):
        cfg = _make_prefill_config("pages")
        assert cfg.run_pages is True

    def test_run_pages_missing(self):
        cfg = _make_prefill_config("windows")
        assert cfg.run_pages is False

    def test_run_surprise_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_surprise is True

    def test_run_surprise_explicit(self):
        cfg = _make_prefill_config("surprise")
        assert cfg.run_surprise is True

    def test_run_sparse_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_sparse is True

    def test_run_sparse_explicit(self):
        cfg = _make_prefill_config("sparse")
        assert cfg.run_sparse is True

    def test_run_kvectors_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_kvectors is True

    def test_run_kvectors_explicit(self):
        cfg = _make_prefill_config("kvectors")
        assert cfg.run_kvectors is True

    def test_run_kvectors_via_kvectors_full(self):
        cfg = _make_prefill_config("kvectors_full")
        assert cfg.run_kvectors is True

    def test_run_kvectors_missing(self):
        cfg = _make_prefill_config("compass")
        assert cfg.run_kvectors is False

    def test_run_kvectors_full_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_kvectors_full is True

    def test_run_kvectors_full_explicit(self):
        cfg = _make_prefill_config("kvectors_full")
        assert cfg.run_kvectors_full is True

    def test_run_kvectors_full_missing(self):
        cfg = _make_prefill_config("kvectors")
        assert cfg.run_kvectors_full is False

    def test_run_vec_inject_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_vec_inject is True

    def test_run_vec_inject_explicit(self):
        cfg = _make_prefill_config("vec_inject")
        assert cfg.run_vec_inject is True

    def test_run_vec_inject_missing(self):
        cfg = _make_prefill_config("compass")
        assert cfg.run_vec_inject is False

    def test_run_mode7_all(self):
        cfg = _make_prefill_config("all")
        assert cfg.run_mode7 is True

    def test_run_mode7_explicit(self):
        cfg = _make_prefill_config("mode7")
        assert cfg.run_mode7 is True

    def test_run_mode7_missing(self):
        cfg = _make_prefill_config("compass")
        assert cfg.run_mode7 is False


class TestPrefillConfigKVectorMode:
    def test_kvector_mode_default_is_interval(self):
        cfg = _make_prefill_config("kvectors")
        assert cfg.kvector_mode == KVectorMode.INTERVAL

    def test_kvector_mode_full_when_kvectors_full(self):
        cfg = _make_prefill_config("kvectors_full")
        assert cfg.kvector_mode == KVectorMode.FULL

    def test_kvector_mode_full_via_all(self):
        # ALL includes kvectors_full → FULL mode
        cfg = _make_prefill_config("all")
        assert cfg.kvector_mode == KVectorMode.FULL


class TestPrefillConfigFromArgs:
    def _make_args(self, **kwargs) -> Namespace:
        defaults = {
            "model": "test-model",
            "input": "/tmp/input.txt",
            "checkpoint": "/tmp/checkpoint",
            "phases": "all",
            "window_size": None,
            "max_tokens": None,
            "no_resume": False,
            "name": None,
            "residual_mode": "interval",
            "frame_bank": None,
            "store_pages": False,
            "store_kv_full": False,
            "compass_layer": None,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_basic_construction(self):
        args = self._make_args()
        cfg = PrefillConfig.from_args(args)
        assert cfg.model == "test-model"

    def test_resume_true_when_no_resume_false(self):
        args = self._make_args(no_resume=False)
        cfg = PrefillConfig.from_args(args)
        assert cfg.resume is True

    def test_resume_false_when_no_resume_true(self):
        args = self._make_args(no_resume=True)
        cfg = PrefillConfig.from_args(args)
        assert cfg.resume is False

    def test_frame_bank_none(self):
        args = self._make_args(frame_bank=None)
        cfg = PrefillConfig.from_args(args)
        assert cfg.frame_bank is None

    def test_frame_bank_set(self):
        args = self._make_args(frame_bank="/tmp/frame_bank.npz")
        cfg = PrefillConfig.from_args(args)
        assert cfg.frame_bank == Path("/tmp/frame_bank.npz")

    def test_multiple_phases(self):
        args = self._make_args(phases="compass,windows")
        cfg = PrefillConfig.from_args(args)
        assert PrefillPhase.COMPASS in cfg.phases
        assert PrefillPhase.WINDOWS in cfg.phases
        assert PrefillPhase.ALL not in cfg.phases

    def test_window_size_default(self):
        args = self._make_args(window_size=None)
        cfg = PrefillConfig.from_args(args)
        from chuk_lazarus.cli.commands._constants import ContextDefaults

        assert cfg.window_size == ContextDefaults.WINDOW_SIZE


# ── PrefillResult.to_display ─────────────────────────────────────────────────


class TestPrefillResultToDisplay:
    def _make_result(self, **overrides) -> PrefillResult:
        defaults = {
            "checkpoint": "/tmp/checkpoint",
            "tokens_prefilled": 1024,
            "num_windows": 10,
            "status": "complete",
            "elapsed_seconds": 5.3,
        }
        defaults.update(overrides)
        return PrefillResult(**defaults)

    def test_contains_header(self):
        r = self._make_result()
        display = r.to_display()
        assert "Prefill Complete" in display

    def test_contains_checkpoint(self):
        r = self._make_result(checkpoint="/tmp/mylib")
        display = r.to_display()
        assert "/tmp/mylib" in display

    def test_contains_tokens_prefilled(self):
        r = self._make_result(tokens_prefilled=512)
        display = r.to_display()
        assert "512" in display

    def test_contains_num_windows(self):
        r = self._make_result(num_windows=7)
        display = r.to_display()
        assert "7" in display

    def test_contains_status(self):
        r = self._make_result(status="partial")
        display = r.to_display()
        assert "partial" in display

    def test_contains_elapsed(self):
        r = self._make_result(elapsed_seconds=12.345)
        display = r.to_display()
        assert "12.3" in display

    def test_is_multiline(self):
        r = self._make_result()
        assert "\n" in r.to_display()


# ── GenerateResult.to_display / to_stats_only ─────────────────────────────────


class TestGenerateResultToDisplay:
    def _make_result(self, **overrides) -> GenerateResult:
        defaults = {
            "response": "Hello world",
            "tokens_generated": 5,
            "context_tokens": 512,
        }
        defaults.update(overrides)
        return GenerateResult(**defaults)

    def test_contains_header(self):
        r = self._make_result()
        assert "Generated Response" in r.to_display()

    def test_contains_response(self):
        r = self._make_result(response="My response text")
        assert "My response text" in r.to_display()

    def test_contains_tokens_generated(self):
        r = self._make_result(tokens_generated=42)
        assert "42" in r.to_display()

    def test_contains_context_tokens(self):
        r = self._make_result(context_tokens=1024)
        assert "1024" in r.to_display()

    def test_stats_line_present(self):
        r = self._make_result()
        assert "Stats" in r.to_display()


class TestGenerateResultToStatsOnly:
    def _make_result(self, **overrides) -> GenerateResult:
        defaults = {
            "response": "Hello world",
            "tokens_generated": 5,
            "context_tokens": 512,
        }
        defaults.update(overrides)
        return GenerateResult(**defaults)

    def test_contains_stats(self):
        r = self._make_result(tokens_generated=10, context_tokens=300)
        stats = r.to_stats_only()
        assert "Stats" in stats

    def test_contains_tokens_generated(self):
        r = self._make_result(tokens_generated=77)
        assert "77" in r.to_stats_only()

    def test_contains_context_tokens(self):
        r = self._make_result(context_tokens=888)
        assert "888" in r.to_stats_only()

    def test_is_single_line_or_short(self):
        r = self._make_result()
        stats = r.to_stats_only()
        # to_stats_only is just the stats field — no header
        assert "Generated Response" not in stats
        assert "Hello world" not in stats

    def test_returns_string(self):
        r = self._make_result()
        assert isinstance(r.to_stats_only(), str)


# ── GenerateConfig ────────────────────────────────────────────────────────────


class TestGenerateConfigFromArgs:
    def _make_args(self, **kwargs) -> Namespace:
        defaults = {
            "model": "test-model",
            "checkpoint": None,
            "prompt": None,
            "prompt_file": None,
            "max_tokens": 200,
            "temperature": 0.7,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_basic_construction(self):
        args = self._make_args()
        cfg = GenerateConfig.from_args(args)
        assert cfg.model == "test-model"

    def test_checkpoint_none(self):
        args = self._make_args(checkpoint=None)
        cfg = GenerateConfig.from_args(args)
        assert cfg.checkpoint is None

    def test_checkpoint_set(self):
        args = self._make_args(checkpoint="/tmp/mylib")
        cfg = GenerateConfig.from_args(args)
        assert cfg.checkpoint == Path("/tmp/mylib")

    def test_prompt_text_inline(self):
        args = self._make_args(prompt="What is 2+2?")
        cfg = GenerateConfig.from_args(args)
        assert cfg.prompt_text == "What is 2+2?"

    def test_prompt_text_none_when_both_none(self):
        args = self._make_args(prompt=None, prompt_file=None)
        cfg = GenerateConfig.from_args(args)
        assert cfg.prompt_text is None

    def test_prompt_file_set(self, tmp_path):
        pf = tmp_path / "prompt.txt"
        pf.write_text("File prompt content")
        args = self._make_args(prompt=None, prompt_file=str(pf))
        cfg = GenerateConfig.from_args(args)
        assert cfg.prompt_text == "File prompt content"
