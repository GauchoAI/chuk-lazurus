"""Tests for prefill/_mode7_calibrate.py.

calibrate_mode7_probes uses lazy imports inside the function body:
  - from .....inference.context import CheckpointLibrary
  - from ..generate._probes import load_or_calibrate

We patch these at their canonical import paths.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

_CHECKPOINT_LIB_PATH = "chuk_lazarus.inference.context.CheckpointLibrary"
_LOAD_OR_CALIBRATE_PATH = "chuk_lazarus.cli.commands.context.generate._probes.load_or_calibrate"


def _run(
    tmp_path: Path,
    num_windows: int = 5,
    compass_layer: int | None = None,
    num_hidden_layers: int = 34,
    m7_available: bool = True,
    tension_available: bool = True,
    model_id: str = "test/model",
):
    """Execute calibrate_mode7_probes with mocked dependencies."""
    from chuk_lazarus.cli.commands.context.prefill._mode7_calibrate import (
        calibrate_mode7_probes,
    )

    engine = MagicMock()
    config = MagicMock()
    config.num_hidden_layers = num_hidden_layers
    tokenizer = MagicMock()

    mock_lib = MagicMock()
    mock_lib.num_windows = num_windows

    mock_probes = MagicMock()
    mock_probes.m7_available = m7_available
    mock_probes.tension_available = tension_available

    buf = io.StringIO()

    with (
        patch(_CHECKPOINT_LIB_PATH, return_value=mock_lib) as MockLib,
        patch(_LOAD_OR_CALIBRATE_PATH, return_value=mock_probes) as MockCalib,
        patch("sys.stderr", buf),
    ):
        calibrate_mode7_probes(
            engine=engine,
            output_path=tmp_path,
            num_archived=num_windows,
            config=config,
            tokenizer=tokenizer,
            model_id=model_id,
            compass_layer=compass_layer,
        )

    return MockLib, MockCalib, buf.getvalue()


class TestCalibrateMode7Probes:
    def test_default_compass_layer_77_percent(self, tmp_path):
        """When compass_layer is None, it defaults to int(num_hidden_layers * 0.77)."""
        _, MockCalib, _ = _run(tmp_path, compass_layer=None, num_hidden_layers=34)
        # Expected: int(34 * 0.77) = 26
        MockCalib.assert_called_once()
        args = MockCalib.call_args[0]
        assert args[2] == 26  # third positional arg is compass_layer

    def test_explicit_compass_layer_used(self, tmp_path):
        _, MockCalib, _ = _run(tmp_path, compass_layer=20, num_hidden_layers=34)
        args = MockCalib.call_args[0]
        assert args[2] == 20

    def test_stale_cache_files_deleted(self, tmp_path):
        """probe_cache_v*.npz files in output_path should be removed."""
        cache1 = tmp_path / ".probe_cache_v1.npz"
        cache2 = tmp_path / ".probe_cache_v2.npz"
        cache1.write_text("old")
        cache2.write_text("old")
        _run(tmp_path)
        assert not cache1.exists()
        assert not cache2.exists()

    def test_stale_cache_not_present_no_error(self, tmp_path):
        """No error if there are no stale cache files to delete."""
        _run(tmp_path)  # should complete without error

    def test_early_exit_when_no_windows(self, tmp_path):
        """When num_windows == 0, should print skip message and not call load_or_calibrate."""
        _, MockCalib, output = _run(tmp_path, num_windows=0)
        MockCalib.assert_not_called()
        assert "skip" in output.lower() or "no windows" in output.lower()

    def test_load_or_calibrate_called_with_str_output_path(self, tmp_path):
        _, MockCalib, _ = _run(tmp_path)
        args = MockCalib.call_args[0]
        assert args[4] == str(tmp_path)  # 5th positional arg is str(output_path)

    def test_load_or_calibrate_called_with_model_id(self, tmp_path):
        _, MockCalib, _ = _run(tmp_path, model_id="my/model")
        args = MockCalib.call_args[0]
        assert args[5] == "my/model"

    def test_output_mentions_calibrated(self, tmp_path):
        _, _, output = _run(tmp_path, num_windows=5)
        assert "calibrat" in output.lower()

    def test_output_mentions_m7_yes(self, tmp_path):
        _, _, output = _run(tmp_path, num_windows=5, m7_available=True)
        assert "yes" in output

    def test_output_mentions_m7_no(self, tmp_path):
        _, _, output = _run(tmp_path, num_windows=5, m7_available=False)
        assert "no" in output

    def test_output_mentions_tension_yes(self, tmp_path):
        _, _, output = _run(tmp_path, num_windows=5, tension_available=True)
        assert "yes" in output

    def test_output_mentions_tension_no(self, tmp_path):
        _, _, output = _run(tmp_path, num_windows=5, tension_available=False)
        assert "no" in output

    def test_checkpoint_library_created_from_output_path(self, tmp_path):
        MockLib, _, _ = _run(tmp_path)
        MockLib.assert_called_once_with(tmp_path)

    def test_engine_kv_gen_passed_to_load_or_calibrate(self, tmp_path):
        from chuk_lazarus.cli.commands.context.prefill._mode7_calibrate import (
            calibrate_mode7_probes,
        )

        engine = MagicMock()
        engine.kv_gen = MagicMock(name="kv_gen_sentinel")
        config = MagicMock()
        config.num_hidden_layers = 34
        tokenizer = MagicMock()

        mock_lib = MagicMock()
        mock_lib.num_windows = 2

        mock_probes = MagicMock()
        mock_probes.m7_available = True
        mock_probes.tension_available = True

        captured_args = {}

        def _capture(*args, **kwargs):
            captured_args["kv_gen"] = args[0]
            return mock_probes

        with (
            patch(_CHECKPOINT_LIB_PATH, return_value=mock_lib),
            patch(_LOAD_OR_CALIBRATE_PATH, side_effect=_capture),
            patch("sys.stderr", io.StringIO()),
        ):
            calibrate_mode7_probes(
                engine=engine,
                output_path=tmp_path,
                num_archived=2,
                config=config,
                tokenizer=tokenizer,
                model_id="test",
            )

        assert captured_args["kv_gen"] is engine.kv_gen

    def test_other_cache_files_not_deleted(self, tmp_path):
        """Files that don't match .probe_cache_v*.npz are untouched."""
        other = tmp_path / "important_data.npz"
        other.write_text("keep me")
        _run(tmp_path)
        assert other.exists()

    def test_num_windows_in_progress_message(self, tmp_path):
        """The output should mention the number of windows."""
        _, _, output = _run(tmp_path, num_windows=7)
        assert "7" in output

    def test_compass_layer_18_for_26_layers(self, tmp_path):
        """int(26 * 0.77) = 20."""
        _, MockCalib, _ = _run(tmp_path, compass_layer=None, num_hidden_layers=26)
        args = MockCalib.call_args[0]
        assert args[2] == 20  # int(26 * 0.77) = 20
