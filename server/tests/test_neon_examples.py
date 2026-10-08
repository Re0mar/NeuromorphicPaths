"""
Covers the three Neon scripts under examples/: that each one imports, that what the capture script
writes is what the replay reads, and that check_neon.py's IMU watch passes and fails the right IMUs.

The scripts are run by hand against the glasses, so nothing else in the suite loads them. The
capture tests drive the real capture function against a fake client, then read the folder back
with the receiver's own readers. The check_neon.py tests play a short real H.264 capture through
the real device process with `--neon-replay`. All of these skip without the glasses extra, because
they need the client or PyAV.
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
    read_capture_packets,
    read_capture_samples,
)
from neon_captures import encode_h264, write_capture

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
    assert list(read_capture_packets(capture / SCENE_PACKETS_FILENAME)) == script.scene_packets
    assert read_capture_samples(capture / GAZE_FILENAME, GAZE_FIELDS) == script.gaze_samples
    assert read_capture_samples(capture / IMU_FILENAME, IMU_FIELDS) == script.imu_samples
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
    assert len(list(read_capture_packets(capture / SCENE_PACKETS_FILENAME))) == 6
    # Written again on the way out, so the counts are the final ones.
    assert json.loads((capture / META_FILENAME).read_text(encoding="utf-8"))["counts"]["scene"] == 6


# *******************************************
# check_neon.py's IMU watch
# *******************************************


def _check_on_a_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    imu: list[tuple[float, float, float, float, float]],
    source_class: type | None = None,
) -> tuple[int, str]:
    """check_neon.py run on a short real H.264 capture, through the real device process.

    `source_class` stands in for the script's `NeonLiveRgbSource` when a test needs the source to
    misbehave after the first frame.
    """
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    pytest.importorskip("pupil_labs.realtime_api.streaming.nal_unit", reason="the client comes with the glasses extra")
    stream = encode_h264(frame_count=12)
    capture = write_capture(
        tmp_path / "capture",
        [(500.0 + index / 30, picture) for index, picture in enumerate(stream.pictures)],
        imu=imu,
        parameter_sets=stream.parameter_sets,
    )
    check = _load_example("check_neon")
    # Every reading is fed within the capture's 0.4 s, so a second's watch has seen them all.
    monkeypatch.setattr(check, "IMU_WATCH_SECONDS", 1.0)
    if source_class is not None:
        monkeypatch.setattr(check, "NeonLiveRgbSource", source_class)
    code = check.main(["--neon-replay", str(capture)])
    return code, capsys.readouterr().out


def _readings(count: int, quaternion: tuple[float, float, float, float], stamped: bool = True) -> list[tuple[float, ...]]:
    return [((500.0 + index / 110) if stamped else 0.0, *quaternion) for index in range(count)]


LEVEL = (1.0, 0.0, 0.0, 0.0)
EMPTY = (0.0, 0.0, 0.0, 0.0)


def test_check_neon_passes_an_imu_with_usable_readings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    code, out = _check_on_a_capture(tmp_path, monkeypatch, capsys, _readings(44, LEVEL))

    assert code == 0
    assert "44 readings since connect, none empty" in out
    assert "gravity pose   yes" in out


def test_check_neon_warns_on_some_empty_readings_and_still_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # The positive control for the two failures below, in the same fixture shape.
    code, out = _check_on_a_capture(tmp_path, monkeypatch, capsys, _readings(40, LEVEL) + _readings(4, EMPTY))

    assert code == 0
    assert "44 readings since connect, 4 empty (9.1%)" in out
    assert "NOT READY" not in out


def test_check_neon_fails_when_every_reading_is_empty_and_names_the_imu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # The 2026-10-05 signature: zero quaternions with zero timestamps.
    code, out = _check_on_a_capture(tmp_path, monkeypatch, capsys, _readings(44, EMPTY, stamped=False))

    assert code == 1
    assert "all 44 readings since connect were empty orientations" in out
    assert "none carried a timestamp either" in out
    assert "IMU NOT READY" in out
    # The frame line says what a missing pose means now, and points at the IMU line for the verdict.
    assert "gravity pose   no, no usable IMU reading within 50 ms" in out


def test_check_neon_fails_naming_a_device_that_stops_answering_during_the_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # The first frame came, then the device process stopped answering before the counts were
    # asked for. A known failure, so the check says so and exits 1 rather than ending in a traceback.
    from nav.sources.neon_device import NeonDeviceError
    from nav.sources.neon_live import NeonLiveRgbSource

    class StoppedBeforeTheCounts(NeonLiveRgbSource):
        def imu_status(self):
            raise NeonDeviceError("the Neon process did not answer 'imu_status' within 5.0 s")

    code, out = _check_on_a_capture(tmp_path, monkeypatch, capsys, _readings(44, LEVEL), source_class=StoppedBeforeTheCounts)

    assert code == 1
    assert "the device stopped answering while the IMU was watched" in out
    assert "NeonDeviceError" not in out, "the message is the failure's own words, not its type"
    assert "did not answer 'imu_status'" in out


def test_check_neon_fails_when_no_imu_reading_arrives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    code, out = _check_on_a_capture(tmp_path, monkeypatch, capsys, [])

    assert code == 1
    assert "no readings arrived since connect" in out
    assert "IMU NOT READY" in out
