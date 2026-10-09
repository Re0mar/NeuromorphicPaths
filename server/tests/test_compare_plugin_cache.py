"""
Covers the arithmetic in examples/compare_plugin_cache.py, which settled the depth plugin's units.

The script itself runs the depth model, so it's run by hand. Its two functions are what turn a run
into a verdict, and they need nothing heavier than OpenCV.
"""

# Standard library imports
import importlib.util
from pathlib import Path

# Third party imports
import cv2
import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "compare_plugin_cache.py"
SCENE_SIZE = (1200, 1600)


def _module():
    spec = importlib.util.spec_from_file_location("compare_plugin_cache", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_conversion_at_the_plugins_resolution_is_the_measured_one() -> None:
    # Walk B's focal, at the 504 px the plugin runs at: 890.754 × 504 / 1600 / 300.
    assert _module().one_conversion_ratio(890.754334, 504, 1600) == pytest.approx(0.93529, abs=1e-5)


def test_a_map_saved_with_one_conversion_reads_back_as_that_ratio() -> None:
    module = _module()
    raw = np.random.default_rng(5).uniform(0.5, 8.0, size=(378, 504)).astype(np.float32)
    full = cv2.resize(raw, (SCENE_SIZE[1], SCENE_SIZE[0]))
    saved = cv2.resize(full * 0.9353, (400, 300), interpolation=cv2.INTER_AREA)

    assert module.saved_over_raw(saved, raw, SCENE_SIZE) == pytest.approx(0.9353, abs=1e-4)


def test_a_map_of_the_wrong_size_is_refused_not_compared() -> None:
    with pytest.raises(ValueError, match="saved map"):
        _module().saved_over_raw(np.ones((150, 200)), np.ones((378, 504)), SCENE_SIZE)


def test_a_recording_without_the_plugins_cache_exits_1(tmp_path: Path, capsys) -> None:
    assert _module().main([str(tmp_path)]) == 1
    assert "no plugin cache" in capsys.readouterr().err


@pytest.mark.parametrize("frames", ["0", "-3"])
def test_a_frame_count_that_is_not_above_zero_is_refused(frames: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _module().main([str(tmp_path), "--frames", frames])

    assert "must be above zero" in capsys.readouterr().err
