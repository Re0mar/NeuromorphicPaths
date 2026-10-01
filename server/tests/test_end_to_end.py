"""
Runs the whole pipeline through its real entry points, from a video to a frame log and back.

The video path goes through run() with the stub estimator injected at the composition root, which
is the one test-only hook. The replay path goes through main() exactly as a person would type it,
and needs no estimator at all.
"""

# Standard library imports
import dataclasses
from pathlib import Path

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.config import build_run_config
from nav.main import main
from nav.runtime.loop import HeadingBaseline, gaze_on_the_ground, wrap_angle, yaw_from_quaternion
from nav.runtime.loop import run
from nav.scene.pipeline import ScenePipeline
from nav.sources.framecodec import INDEX_FILENAME
from nav.sources.logged import LoggedDepthFrameSource
from nav.types import DepthFrame, Plane, Pose
from stubs import StubDepthEstimator

FRAME_COUNT = 30


def _write_video(target: Path, frame_count: int = FRAME_COUNT) -> Path:
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, (index * 2) % 60 : (index * 2) % 60 + 4] = 255
        writer.write(frame)
    writer.release()
    return target


def test_video_to_log_to_replay_round_trip(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "walk.avi")
    log_dir = tmp_path / "log"

    # 1. Video in, frame log out, with the stub standing in for the 1.3 GB model.
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    stub = StubDepthEstimator()
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: stub)

    assert run(config) == 0
    assert len(sorted(log_dir.glob("frame_*.bin"))) == FRAME_COUNT
    assert len((log_dir / INDEX_FILENAME).read_text(encoding="utf-8").splitlines()) == FRAME_COUNT

    # The configuration rides with the frames. A replay given a different floor gate refused
    # every frame of the first real recording, and this is how the right flags are found again.
    import json
    from nav.runtime.loop import RUN_CONFIG_FILENAME

    recorded = json.loads((log_dir / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert recorded["source_kind"] == "video_file"
    assert recorded["scene"]["floor_max_tilt_degrees"] == config.scene.floor_max_tilt_degrees
    assert "estimator_factory" not in recorded
    assert b"\r" not in (log_dir / RUN_CONFIG_FILENAME).read_bytes()

    # 2. The log reads back as what the stub produced.
    replayed = list(LoggedDepthFrameSource(log_dir).frames())
    assert len(replayed) == FRAME_COUNT
    expected = stub.estimate(np.zeros((48, 64, 3), dtype=np.uint8)).depth_meters
    assert replayed[0].depth_meters.shape == expected.shape
    valid = np.isfinite(expected)
    assert replayed[0].depth_meters[valid] == pytest.approx(expected[valid])

    # 3. The replay path through main, the way a person types it.
    assert main(["--source", "logged", "--log-dir", str(log_dir), "--sink", "none"]) == 0


def test_a_missing_video_exits_non_zero_with_the_path_in_the_error(tmp_path: Path, caplog: pytest.CaptureFixture) -> None:
    config = build_run_config(["--source", "video_file", "--path", str(tmp_path / "nothing.avi"), "--sink", "none"])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())

    with caplog.at_level("ERROR"):
        exit_code = run(config)

    assert exit_code == 1
    assert any("nothing.avi" in record.message or "nothing.avi" in str(record.exc_info) for record in caplog.records)


def test_recording_into_a_used_directory_fails_rather_than_interleaving(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "walk.avi", frame_count=3)
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "frame_000000.bin").write_bytes(b"earlier run")
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())

    assert run(config) == 1


def test_reconnect_is_refused_for_a_source_that_is_not_the_phone(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(["--source", "logged", "--log-dir", "x", "--sink", "none", "--reconnect"])

    assert "--reconnect" in capsys.readouterr().err


def test_yaw_is_zero_for_identity_and_a_quarter_turn_about_the_vertical() -> None:
    assert yaw_from_quaternion(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    # 90 degrees about camera y, which is the vertical. Forward goes to +x, which is right.
    half = np.pi / 4
    assert yaw_from_quaternion(np.array([np.cos(half), 0.0, np.sin(half), 0.0])) == pytest.approx(np.pi / 2)


def test_the_heading_baseline_follows_a_slow_curve_and_shows_a_quick_turn() -> None:
    baseline = HeadingBaseline(time_constant_seconds=5.0)
    baseline.observed(0.0, 0.0)

    quick = baseline.observed(0.3, 0.1)
    assert quick == pytest.approx(0.3, abs=0.02), "a sudden turn is almost all departure"

    for step in range(1, 200):
        last = baseline.observed(0.3, 0.1 + step * 0.1)
    assert abs(last) < 0.02, "holding the same heading for twenty seconds becomes the new straight ahead"


def test_wrap_angle_keeps_results_in_range() -> None:
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)


def test_gaze_on_the_ground_lands_where_the_pinhole_model_says() -> None:
    # Level camera 1.6 m above a flat floor, floor normal up (-y). A gaze pixel below the
    # principal point hits the floor ahead at a distance the geometry fixes: z = h * fy / (v - cy).
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.6)
    scene = ScenePipeline.__new__(ScenePipeline)
    scene._previous_plane = plane  # the only state the helper reads
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    frame = DepthFrame(0.0, np.ones((4, 4), dtype=np.float32), intrinsics, Pose(np.array([1.0, 0, 0, 0]), None, False), None, np.array([60.0, 70.0]))

    point = gaze_on_the_ground(frame, scene)

    assert point is not None
    assert point[1] == pytest.approx(1.6 * 100.0 / 20.0)  # 8 m ahead
    assert point[0] == pytest.approx(8.0 * 10.0 / 100.0)  # 0.8 m right


def test_a_gaze_above_the_horizon_gives_no_ground_point() -> None:
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.6)
    scene = ScenePipeline.__new__(ScenePipeline)
    scene._previous_plane = plane
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    frame = DepthFrame(0.0, np.ones((4, 4), dtype=np.float32), intrinsics, Pose(np.array([1.0, 0, 0, 0]), None, False), None, np.array([50.0, 10.0]))

    assert gaze_on_the_ground(frame, scene) is None
