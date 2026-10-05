"""
Runs the whole pipeline through its real entry points, from a video to a frame log and back.

The video path goes through run() with the stub estimator injected at the composition root, which
is the one test-only hook. The replay path goes through main() exactly as a person would type it,
and needs no estimator at all.
"""

# Standard library imports
import asyncio
import dataclasses
import json
import logging
import socket
import threading
import time
import urllib.request
from pathlib import Path

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.config import build_run_config
from nav.main import main
from nav.runtime.loop import RUN_CONFIG_FILENAME, HeadingBaseline, gaze_on_the_ground, wrap_angle, yaw_from_quaternion
from nav.runtime.loop import run
from nav.runtime.timing import TIMING_FILENAME, read_timing_log
from nav.scene.pipeline import ScenePipeline
from nav.sinks.web_messages import WebMessageKind
from nav.sources.framecodec import INDEX_FILENAME, decode_path, read_message
from nav.sources.logged import LoggedDepthFrameSource
from nav.types import DepthFrame, Plane, Pose
from stubs import StubDepthEstimator

FRAME_COUNT = 30


def _free_port() -> int:
    """A port nothing is listening on, for a test that needs a real one it can bind later."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


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
    recorded = json.loads((log_dir / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert recorded["source_kind"] == "video_file"
    assert recorded["scene"]["floor_max_tilt_degrees"] == config.scene.floor_max_tilt_degrees
    assert "estimator_factory" not in recorded
    assert b"\r" not in (log_dir / RUN_CONFIG_FILENAME).read_bytes()

    # 2. The log reads back as what the stub produced.
    replayed = list(LoggedDepthFrameSource(log_dir).frames())
    assert len(replayed) == FRAME_COUNT
    expected = stub.estimate(np.zeros((48, 64, 3), dtype=np.uint8)).depth
    assert replayed[0].depth_meters.shape == expected.shape
    valid = np.isfinite(expected)
    assert replayed[0].depth_meters[valid] == pytest.approx(expected[valid])

    # 3. The replay path through main, the way a person types it.
    assert main(["--source", "logged", "--log-dir", str(log_dir), "--sink", "none"]) == 0


def test_a_logged_replay_reaches_a_phone_through_main(tmp_path: Path) -> None:
    # The phone sink listens and the phone connects, so the one way to drive it through the
    # entry point a person types is a fake phone that keeps knocking until the run is listening.
    video = _write_video(tmp_path / "walk.avi")
    log_dir = tmp_path / "log"
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())
    assert run(config) == 0

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    received: list = []

    def phone() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                connection = socket.create_connection(("127.0.0.1", free_port), timeout=5.0)
            except OSError:
                time.sleep(0.05)
                continue
            try:
                received.append(decode_path(read_message(connection)))
            finally:
                connection.close()
            return

    knocking = threading.Thread(target=phone, daemon=True)
    knocking.start()
    # Real time, so the thirty frames take three seconds and the phone has time to be accepted
    # before the last path is published.
    assert main(["--source", "logged", "--log-dir", str(log_dir), "--sink", "phone_app", "--phone-port", str(free_port), "--realtime"]) == 0
    knocking.join(5.0)

    assert received, "the phone never read a path from the run"
    assert np.isfinite(received[0].first_heading_radians)
    # decode_path requires scene_information_bits and avoidance_surprise_bits, so a decoded path
    # carried both. Their values are tested through PlannerPipeline in test_planner_pipeline.py.


def test_a_logged_replay_serves_a_phone_and_a_browser_in_the_same_run(tmp_path: Path) -> None:
    # A walk puts the arrow on the phone and the depth view in a browser at once. One display
    # proves nothing about the other, so this drives both through the entry point a person types.
    video = _write_video(tmp_path / "walk.avi")
    log_dir = tmp_path / "log"
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())
    assert run(config) == 0

    phone_port, web_port = _free_port(), _free_port()
    paths: list = []
    pages: list[str] = []
    plan_views: list[dict] = []

    def phone() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                connection = socket.create_connection(("127.0.0.1", phone_port), timeout=5.0)
            except OSError:
                time.sleep(0.05)
                continue
            try:
                paths.append(decode_path(read_message(connection)))
            finally:
                connection.close()
            return

    def browser() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{web_port}/", timeout=5.0) as response:
                    pages.append(response.read().decode("utf-8"))
                break
            except OSError:
                time.sleep(0.05)
        # Then the socket, as the page itself would, reading until a plan view comes through.
        asyncio.run(read_until_a_plan_view())

    async def read_until_a_plan_view() -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{web_port}/ws") as connection:
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    message = await asyncio.wait_for(connection.receive(), 5.0)
                    if message.type == aiohttp.WSMsgType.TEXT and json.loads(message.data)["kind"] == WebMessageKind.PLAN_VIEW.value:
                        plan_views.append(json.loads(message.data))
                        return

    watchers = [threading.Thread(target=phone, daemon=True), threading.Thread(target=browser, daemon=True)]
    for watcher in watchers:
        watcher.start()

    assert main(
        [
            "--source", "logged", "--log-dir", str(log_dir),
            "--sink", "phone_app", "--phone-port", str(phone_port),
            "--sink", "web", "--web-port", str(web_port),
            "--realtime",
        ]
    ) == 0
    for watcher in watchers:
        watcher.join(5.0)

    assert paths, "the phone never read a path from the run"
    assert np.isfinite(paths[0].first_heading_radians)
    assert pages and "<canvas" in pages[0], "the browser never got the page from the same run"
    assert plan_views, "the browser never got a plan view from the same run"
    field = np.array(plan_views[0]["field"])
    assert field.shape == (len(plan_views[0]["times_seconds"]), len(plan_views[0]["grid_meters"]))
    # The loop hands the planner's body half-width to the sink, and the page draws the path twice
    # that wide. Nothing upstream of the loop reads this value, so only a run through main() can check it.
    assert plan_views[0]["path_width_meters"] == pytest.approx(2.0 * config.planner.body_half_width_meters)


def test_a_run_that_recorded_nothing_leaves_its_directory_usable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # The runtime writes run_config.json into the log directory at the end of every run, including
    # one that recorded nothing because nobody connected. While the tap judged the directory by
    # whether it was empty, that config made every later run with the same --record-to fail, and
    # the refusal named the directory the user had chosen rather than the leftover. It cost a real
    # walk on 2026-10-02.
    log_dir = tmp_path / "log"
    argv = ["--source", "arcore_tcp", "--arcore-port", str(_free_port()), "--arcore-accept-timeout", "1", "--sink", "none", "--record-to", str(log_dir)]

    assert main(argv) == 1, "nobody connects, so the run ends on the accept timeout"
    assert (log_dir / "run_config.json").is_file(), "the run wrote its configuration, which is what used to poison the directory"
    assert not list(log_dir.glob("frame_*.bin")), "and recorded no frames"

    # The exit code cannot tell these two runs apart: a refused directory and a spent accept
    # timeout both end the run at 1. What distinguishes them is whether the tap accepted the
    # directory at all, which it announces, so that is what this asserts.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(argv) == 1

    assert any("recording frames to" in record.message for record in caplog.records), "the tap refused the directory"
    assert not any("FileExistsError" in record.message for record in caplog.records), [record.message for record in caplog.records]


def test_a_missing_video_is_refused_by_the_parser_naming_the_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Refused before run() exists, so no estimator is built for a typo. The loop's own handling
    # of a source that fails after the parser let it through is in test_loop.py.
    with pytest.raises(SystemExit):
        main(["--source", "video_file", "--path", str(tmp_path / "nothing.avi"), "--sink", "none"])

    assert "nothing.avi" in capsys.readouterr().err


def test_recording_into_a_used_directory_fails_rather_than_interleaving(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    video = _write_video(tmp_path / "walk.avi", frame_count=3)
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "frame_000000.bin").write_bytes(b"earlier run")
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())

    with caplog.at_level("ERROR"):
        assert run(config) == 1

    # A used directory is a known refusal, reported by name and never as a defect in the pipeline.
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert "FileExistsError" in errors[0].message and str(log_dir) in errors[0].message
    assert "UNEXPECTED" not in errors[0].message
    assert errors[0].exc_info is None


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


class FloorlessDepthEstimator(StubDepthEstimator):
    """Depth with nothing in it, so the scene finds no floor and the worker skips every frame."""

    def estimate(self, image_rgb: np.ndarray):
        estimate = super().estimate(image_rgb)
        return dataclasses.replace(estimate, depth=np.full_like(estimate.depth, np.nan))


def _processed_count(caplog: pytest.LogCaptureFixture) -> int:
    report = next(record.message for record in caplog.records if " processed, " in record.message)
    return int(report.split(" processed")[0].split(", ")[-1])


def test_a_recorded_run_writes_one_timing_line_per_frame_it_took(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Through run(), the production entry point, from a video to the timing log beside the frames.
    video = _write_video(tmp_path / "walk.avi")
    log_dir = tmp_path / "log"
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())

    with caplog.at_level(logging.INFO):
        assert run(config) == 0

    lines = read_timing_log(log_dir)
    assert len(lines) == _processed_count(caplog)
    assert all(line.plan_done_seconds is not None for line in lines)
    assert all(line.arrival_seconds <= line.depth_ready_seconds <= line.plan_done_seconds for line in lines)
    assert all(line.capture_seconds is None for line in lines), "a file's timestamps are not on any clock"
    assert b"\r" not in (log_dir / TIMING_FILENAME).read_bytes()


def test_a_skipped_frame_still_writes_a_timing_line_with_no_floor(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    video = _write_video(tmp_path / "walk.avi", frame_count=5)
    log_dir = tmp_path / "log"
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none", "--record-to", str(log_dir)])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: FloorlessDepthEstimator())

    with caplog.at_level(logging.INFO):
        assert run(config) == 0

    lines = read_timing_log(log_dir)
    assert lines, "frames the worker skipped wrote nothing, so the floor rate would read perfect"
    assert all(line.floor_source is None and line.plan_done_seconds is None for line in lines)


def test_the_end_of_run_report_counts_floor_sources(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    video = _write_video(tmp_path / "walk.avi")
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none"])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())

    with caplog.at_level(logging.INFO):
        assert run(config) == 0

    report = [record.message for record in caplog.records if record.message.startswith("floor over")]
    assert report, "no floor-source line in the end-of-run report"
    assert f"floor over {_processed_count(caplog)} frames taken" in report[0]
    assert "fitted" in report[0]


def test_no_timing_log_is_written_without_record_to(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _write_video(tmp_path / "walk.avi", frame_count=5)
    monkeypatch.chdir(tmp_path)
    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none"])
    config = dataclasses.replace(config, estimator_factory=lambda estimator_config: StubDepthEstimator())

    assert run(config) == 0

    assert not list(tmp_path.rglob(TIMING_FILENAME))
