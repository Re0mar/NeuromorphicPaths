"""
Covers the three Neon scripts under examples/: that each one imports, and that what the capture
script writes is what the replay reads.

The scripts are run by hand against the glasses, so nothing else in the suite loads them. The
capture tests drive the real capture function against a fake client, then read the folder back
with the receiver's own readers. Those skip without the glasses extra, because the fake patches
the client's own classes.
"""

# Standard library imports
import asyncio
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

# Third party imports
import numpy as np
import pytest

# Local package imports
from fake_neon_client import CAMERA_MATRIX, DISTORTION, FakeNeonScript, install
from nav.sources.neon_stream import (
    GAZE_FIELDS,
    GAZE_FILENAME,
    IMU_FIELDS,
    IMU_FILENAME,
    META_FILENAME,
    SCENE_PACKETS_FILENAME,
    _read_meta,
    _read_packets,
    _read_samples,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
NEON_EXAMPLES = ("capture_neon_stream", "check_neon", "neon_connectivity")


def _load_example(name: str) -> ModuleType:
    """An example script loaded as a module, without running its main."""
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NEON_EXAMPLES)
def test_each_neon_example_script_imports(name: str) -> None:
    """A helper moved between modules broke these imports once, and only a hardware session found out."""
    module = _load_example(name)

    assert callable(module.main)


def _capture_script(monkeypatch: pytest.MonkeyPatch, script: FakeNeonScript) -> ModuleType:
    pytest.importorskip("pupil_labs.realtime_api", reason="the client comes with the glasses extra")
    install(script, monkeypatch.setattr)
    return _load_example("capture_neon_stream")


def test_a_capture_reads_back_through_the_replay_readers_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture and replay share their constants, but not their code. Only running both sides sees a format drift."""
    script = FakeNeonScript(
        scene_packets=[(1_000.0 + index * 0.01, bytes([index + 1]) + b"payload") for index in range(6)],
        gaze_samples=[(1_000.0, 10.0, 20.0), (1_000.005, 11.5, 21.5)],
        imu_samples=[(1_000.0, 0.9, 0.1, 0.2, 0.3), (1_000.01, 1.0, 0.0, 0.0, 0.0)],
        time_offset_ms=1287.0,
        round_trip_ms=9.0,
    )
    capture_module = _capture_script(monkeypatch, script)
    capture = tmp_path / "walk_1"

    asyncio.run(capture_module.capture("192.0.2.7", 8080, 0.5, capture))

    # Every packet, including the first one, which came before the stream description.
    assert list(_read_packets(capture / SCENE_PACKETS_FILENAME)) == script.scene_packets
    assert _read_samples(capture / GAZE_FILENAME, GAZE_FIELDS) == script.gaze_samples
    assert _read_samples(capture / IMU_FILENAME, IMU_FIELDS) == script.imu_samples
    calibration, offset, parameter_sets = _read_meta(capture)
    assert calibration.scene_camera_matrix == pytest.approx(np.array(CAMERA_MATRIX))
    assert calibration.scene_distortion_coefficients == pytest.approx(np.array(DISTORTION))
    assert (offset.time_offset_ms.median, offset.roundtrip_duration_ms.median) == (1287.0, 9.0)
    assert parameter_sets == script.parameter_sets


def test_a_stream_that_goes_quiet_does_not_hold_the_capture_past_its_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deadline was checked only when a sample arrived, so a silent stream kept the capture open for good."""
    script = FakeNeonScript(scene_packets=[(1_000.0 + index * 0.01, bytes([index + 1])) for index in range(6)], imu_stays_silent=True)
    capture_module = _capture_script(monkeypatch, script)

    started = time.monotonic()
    # Bounded from outside as well, so a capture that never ends fails here rather than hanging.
    asyncio.run(asyncio.wait_for(capture_module.capture("192.0.2.7", 8080, 0.5, tmp_path / "walk_1"), timeout=5.0))
    capturing_seconds = time.monotonic() - started

    assert capturing_seconds < 2.0


def test_a_capture_stopped_early_can_still_be_played_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """meta.json used to be written only after every stream returned, so Ctrl+C left packets the replay refused."""
    script = FakeNeonScript(scene_packets=[(1_000.0 + index * 0.01, bytes([index + 1])) for index in range(6)], imu_stays_silent=True)
    capture_module = _capture_script(monkeypatch, script)
    capture = tmp_path / "walk_1"

    # Cancelled from outside after a second of a 30 s capture, which is what Ctrl+C does to it.
    with pytest.raises(TimeoutError):
        asyncio.run(asyncio.wait_for(capture_module.capture("192.0.2.7", 8080, 30.0, capture), timeout=1.0))

    _, _, parameter_sets = _read_meta(capture)
    assert parameter_sets == script.parameter_sets
    assert len(list(_read_packets(capture / SCENE_PACKETS_FILENAME))) == 6
    # Written again on the way out, so the counts are the final ones.
    assert json.loads((capture / META_FILENAME).read_text(encoding="utf-8"))["counts"]["scene"] == 6
