"""
Covers the frame loop's behavior around the sources and sinks: how it reports a failure, how it
reconnects, and that a result reaches a sink exactly once.

The round trip from a video through a frame log is in test_end_to_end.py. This file is the
loop's own contract, driven through run() with a stub estimator and the fake sender.
"""

# Standard library imports
import dataclasses
import json
import logging
import math
import socket
import threading
import time
from collections import Counter
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.config import GoalMode, RunConfig, SinkKind, SourceKind, build_run_config
from nav.main import main
from nav.planner.config import PlannerConfig
from nav.runtime.loop import EPISODES_FILENAME, RUN_CONFIG_FILENAME, FrameResult, PublisherThread, _latency_shares, _report, run
from nav.runtime.tap import RecordingTap
from nav.runtime.worker import NewestFrameWorker
from nav.sources.config import ArCoreConfig, LoggedConfig, TapConfig
from nav.sources.framecodec import INDEX_FILENAME
from nav.types import DebugView, DepthFrame, FloorSource, FrameTiming, ObstacleSet, Plane, PlannedPath, Pose
from nav.usermodel.config import UserModelConfig
from nav.usermodel.work import WorkMeter
from fake_arcore_sender import send_frames, synthetic_frames
from stubs import StubDepthEstimator


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _path(heading: float = 0.0, cost: float = 1.0) -> PlannedPath:
    return PlannedPath(0.0, np.array([0.0, 0.1]), np.array([0.0, 0.0]), heading, False, cost, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _view() -> DebugView:
    """The smallest view a result can carry: a 2 by 2 frame, no obstacles, a level floor."""
    frame = DepthFrame(
        timestamp_seconds=0.0,
        depth_meters=np.ones((2, 2), dtype=np.float32),
        intrinsics=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False),
        ground_plane=None,
        gaze_pixel=None,
    )
    return DebugView(frame, ObstacleSet(0.0, (), 0), Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30, 1.47)


def test_the_sink_is_started_before_the_source_yields_a_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both listening sinks used to open their port on their first publish, so a browser or the
    # phone's path connection found nothing until the first planned frame, which on a still
    # phone is minutes away. The page would not load on the first display run for that reason.
    order: list[str] = []

    class RecordingSink:
        def start(self) -> None:
            order.append("sink started")

        def publish(self, path: PlannedPath) -> None:
            order.append("published")

        def close(self) -> None:
            order.append("sink closed")

    class OneFrameSource:
        def frames(self):
            order.append("first frame yielded")
            yield next(synthetic_frames(1))

        def close(self) -> None:
            pass

    import nav.runtime.loop as loop_module

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RecordingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrameSource())

    assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 0
    assert "sink started" in order and "first frame yielded" in order
    assert order.index("sink started") < order.index("first frame yielded")
    assert order[-1] == "sink closed"


def test_a_source_that_fails_inside_the_loop_is_reported_as_known_not_unexpected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # A directory that exists and is not a frame log passes the parser and fails in the source.
    # That is a refusal with a name, and the log must say so without a traceback.
    empty = tmp_path / "not_a_log"
    empty.mkdir()
    config = build_run_config(["--source", "logged", "--log-dir", str(empty), "--sink", "none"])

    with caplog.at_level("ERROR"):
        assert run(config) == 1

    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert "FileNotFoundError" in errors[0].message
    assert INDEX_FILENAME in errors[0].message
    assert "UNEXPECTED" not in errors[0].message
    assert errors[0].exc_info is None


def test_a_sink_that_raises_an_unexpected_error_ends_the_run_with_exit_one_and_a_traceback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class BrokenSink:
        def start(self) -> None:
            pass

        def publish(self, path: PlannedPath) -> None:
            raise RuntimeError("a bug, not bad input")

        def close(self) -> None:
            pass

    log_dir = tmp_path / "log"
    _record_synthetic_log(log_dir, count=3)
    config = RunConfig(source_kind=SourceKind.LOGGED, sink_kinds=(SinkKind.NONE,), goal_mode=GoalMode.AHEAD, logged=LoggedConfig(log_dir=str(log_dir)))

    # The sink is swapped under the factory by building the config the loop would build, then
    # running with a sink the factory cannot produce. The loop takes what build_sink returns, so
    # the swap goes through the module's factory name.
    import nav.runtime.loop as loop_module

    original = loop_module.build_sink
    loop_module.build_sink = lambda run_config, **hooks: BrokenSink()
    try:
        with caplog.at_level("ERROR"):
            exit_code = run(config)
    finally:
        loop_module.build_sink = original

    assert exit_code == 1
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert any("UNEXPECTED RuntimeError" in record.message and record.exc_info is not None for record in errors)


def _record_synthetic_log(log_dir: Path, count: int) -> None:
    class ListSource:
        def frames(self):
            yield from synthetic_frames(count)

        def close(self) -> None:
            pass

    list(RecordingTap(ListSource(), log_dir).frames())


def test_reconnect_with_a_recording_keeps_one_log_and_the_totals_across_connections(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    port = _free_port()
    log_dir = tmp_path / "walk"
    config = RunConfig(
        source_kind=SourceKind.ARCORE_TCP,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        # Short enough that the run ends soon after the second sender, long enough for both.
        arcore=ArCoreConfig(port=port, bind_address="127.0.0.1", accept_timeout_seconds=2.0),
        tap=TapConfig(log_dir=str(log_dir)),
        reconnect=True,
    )

    def two_connections() -> None:
        time.sleep(0.3)
        send_frames("127.0.0.1", port, synthetic_frames(3))
        time.sleep(0.3)
        send_frames("127.0.0.1", port, synthetic_frames(3))

    sender = threading.Thread(target=two_connections, daemon=True)
    sender.start()
    with caplog.at_level("INFO"):
        exit_code = run(config)
    sender.join(5.0)

    # The phone came back once and then did not. Waiting was what was asked, so that is exit 0.
    assert exit_code == 0
    assert not any("UNEXPECTED" in record.message for record in caplog.records)
    # One log, six frames, numbered straight through. The second connection did not start over.
    assert sorted(path.name for path in log_dir.glob("frame_*.bin")) == [f"frame_{index:06d}.bin" for index in range(6)]
    assert len((log_dir / INDEX_FILENAME).read_text(encoding="utf-8").splitlines()) == 6
    # The totals cover both connections rather than only the last one.
    summary = next(record.message for record in caplog.records if "frames in" in record.message)
    assert summary.startswith("6 frames in")
    processed = int(summary.split("frames in, ")[1].split(" processed")[0])
    dropped = int(summary.split("processed, ")[1].split(" dropped")[0])
    assert processed + dropped == 6


def test_one_planner_serves_every_frame_of_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The alarm's hold lives on the planner and carries across frames, so a planner built per frame
    # would clear the alarm the moment its raise decision did.
    import nav.runtime.loop as loop_module

    counts = {"built": 0, "planned": 0}

    class CountingPlanner(loop_module.PlannerPipeline):
        def __init__(self, *args, **kwargs) -> None:
            counts["built"] += 1
            super().__init__(*args, **kwargs)

        def plan(self, *args, **kwargs) -> PlannedPath:
            counts["planned"] += 1
            return super().plan(*args, **kwargs)

    monkeypatch.setattr(loop_module, "PlannerPipeline", CountingPlanner)
    log_dir = tmp_path / "log"
    _record_synthetic_log(log_dir, count=6)
    config = RunConfig(source_kind=SourceKind.LOGGED, sink_kinds=(SinkKind.NONE,), goal_mode=GoalMode.AHEAD, logged=LoggedConfig(log_dir=str(log_dir)))

    assert run(config) == 0

    assert counts["planned"] >= 2, "the run must plan more than one frame for the count to mean anything"
    assert counts["built"] == 1


def test_the_views_red_point_is_the_runs_alarm_threshold_in_bits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The path turns fully red when the alarm raises, so the red point a display gets has to come
    # from the run's own threshold. A constant here would survive every sink test, because they build
    # their views by hand. 0.9 s is neither the shipped 0.7 s nor the old 1 s red point.
    import nav.runtime.loop as loop_module

    views: list[DebugView] = []

    class CapturingDebugSink:
        def start(self) -> None:
            pass

        def publish(self, path: PlannedPath) -> None:
            pass

        def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
            views.append(view)

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: CapturingDebugSink())
    log_dir = tmp_path / "log"
    _record_synthetic_log(log_dir, count=6)
    planner = dataclasses.replace(PlannerConfig(), alarm_time_to_contact_seconds=0.9)
    config = RunConfig(
        source_kind=SourceKind.LOGGED,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        planner=planner,
        logged=LoggedConfig(log_dir=str(log_dir)),
    )

    assert run(config) == 0

    assert views, "the run must publish at least one view for the check to mean anything"
    # Half of (1 s over 0.9 s) squared, in natural-log units, over ln 2.
    expected = 0.5 * (1.0 / 0.9) ** 2 / math.log(2.0)
    assert {round(view.path_red_from_bits, 9) for view in views} == {round(expected, 9)}


def test_a_result_is_published_once_however_long_it_stays_the_newest() -> None:
    published: list[PlannedPath] = []

    class CountingSink:
        def start(self) -> None:
            pass

        def publish(self, path: PlannedPath) -> None:
            published.append(path)

        def close(self) -> None:
            pass

    class StuckWorker:
        """Holds one result for as long as the test likes, like a worker mid-way through a slow frame."""

        def __init__(self) -> None:
            self.result = FrameResult(_path(0.1), np.zeros((1, 3)), np.array([-1.0, 0.0, 1.0]), _view())

        def latest_result(self):
            return self.result

        def wait_for_result(self, newer_than, timeout_seconds):
            if self.result is not newer_than:
                return self.result
            time.sleep(timeout_seconds)
            return None

    worker = StuckWorker()
    publisher = PublisherThread(CountingSink(), worker)
    publisher.start()
    try:
        assert publisher.flush(5.0)
        # Several polls with the same result in place. Each one is a chance to publish it again.
        time.sleep(0.35)
        assert len(published) == 1, "one result is one publish, however many polls see it"

        worker.result = FrameResult(_path(0.2), np.zeros((1, 3)), np.array([-1.0, 0.0, 1.0]), _view())
        assert publisher.flush(5.0)
        time.sleep(0.25)
    finally:
        publisher.stop()
    assert [path.lookahead_heading_radians for path in published] == pytest.approx([0.1, 0.2])


def test_run_config_json_carries_every_field_of_the_run_configuration(tmp_path: Path) -> None:
    # A knob added to any layer must ride with the recording, or a replay cannot be given the
    # same flags. Every top-level field except the test hook has to be in the file by name.
    log_dir = tmp_path / "log"
    config = build_run_config(["--source", "arcore_tcp", "--sink", "none", "--record-to", str(log_dir)])
    log_dir.mkdir()

    _report(NewestFrameWorker(lambda frame: None), WorkMeter(UserModelConfig()), 0, config, Counter())

    recorded = json.loads((log_dir / RUN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    expected = {field.name for field in dataclasses.fields(RunConfig)} - {"estimator_factory"}
    assert set(recorded) == expected
    assert set(recorded["scene"]) == {field.name for field in dataclasses.fields(config.scene)}
    assert set(recorded["planner"]) == {field.name for field in dataclasses.fields(config.planner)}


def test_completed_episodes_are_written_beside_the_frames(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    config = build_run_config(["--source", "arcore_tcp", "--sink", "none", "--record-to", str(log_dir)])
    meter = WorkMeter(UserModelConfig())
    # One avoidance: the planner asks for a turn at cost 9, then settles at cost 2.
    meter.observe(_path(heading=0.3, cost=9.0), 0.0, 0.0)
    meter.observe(_path(heading=0.0, cost=2.0), 0.0, 0.5)
    assert len(meter.completed_episodes()) == 1

    _report(NewestFrameWorker(lambda frame: None), meter, 0, config, Counter())

    lines = (log_dir / EPISODES_FILENAME).read_bytes()
    assert b"\r" not in lines
    episode = json.loads(lines.decode("utf-8").splitlines()[0])
    assert episode["work_bits"] == pytest.approx(7.0)
    assert episode["start_seconds"] == pytest.approx(0.0)


def test_the_verbose_shares_are_measured_between_the_right_stamps() -> None:
    """
    Each share in the --verbose line is the gap between two named stamps, worked out here by hand.

    A frame from a source that has no depth step gets arrival to plan instead, and no timing gets nothing.
    """
    timed = dataclasses.replace(
        next(synthetic_frames(1)),
        timing=FrameTiming(capture_seconds=99.95, arrival_seconds=100.0, depth_ready_seconds=100.1),
    )
    depth_in_frame = dataclasses.replace(
        timed,
        timing=FrameTiming(capture_seconds=None, arrival_seconds=100.0, depth_ready_seconds=None),
    )

    assert _latency_shares(timed, plan_done_seconds=100.13) == ", capture to arrival 50 ms, arrival to depth 100 ms, depth to plan 30 ms"
    assert _latency_shares(depth_in_frame, plan_done_seconds=100.13) == ", arrival to plan 130 ms"
    assert _latency_shares(next(synthetic_frames(1)), plan_done_seconds=100.13) == ""


def test_the_verbose_line_carries_the_observed_heading(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The mount check reads the wearer's turn off this line, so it has to be the observed one.

    Two frames a tenth of a second apart, the second turned 30 degrees about the vertical. The
    baseline moves 0.1 s / 5 s = 2 percent of the way, so the observed heading is 30 * 0.98 = 29.4.
    """
    import nav.runtime.loop as loop_module

    first, second = list(synthetic_frames(2))
    half_turn = np.radians(30.0) / 2
    # The synthetic pose is a half turn about x, (0, 1, 0, 0). A turn about world y composed on
    # top of it is (0, cos, 0, -sin) of half the angle.
    turned = dataclasses.replace(second.pose, orientation=np.array([0.0, np.cos(half_turn), 0.0, -np.sin(half_turn)]))
    frames = [first, dataclasses.replace(second, pose=turned, timestamp_seconds=first.timestamp_seconds + 0.1)]

    class OneFrameAtATimeSource:
        def frames(self):
            for frame in frames:
                yield frame
                # The worker keeps only the newest frame, so the first must be taken before the second comes.
                time.sleep(0.5)

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrameAtATimeSource())
    with caplog.at_level(logging.DEBUG, logger="nav.runtime.loop"):
        assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 0

    frame_lines = [record.message for record in caplog.records if record.message.startswith("frame ")]
    assert len(frame_lines) == 2, "both frames must be planned for the second one's heading to mean anything"
    assert "observed 0.0 deg" in frame_lines[0]
    assert "observed 29.4 deg" in frame_lines[1]


def test_verbose_raises_only_the_pipelines_loggers(tmp_path: Path) -> None:
    # The root at DEBUG drowned the per-stage timings in every HTTP library's request headers
    # during the model download. --verbose is for nav's loggers and nothing else.
    log_dir = tmp_path / "log"
    _record_synthetic_log(log_dir, count=2)
    root_level_before = logging.getLogger().level

    assert main(["--source", "logged", "--log-dir", str(log_dir), "--sink", "none", "--verbose"]) == 0

    assert logging.getLogger("nav").level == logging.DEBUG
    assert logging.getLogger().level == root_level_before
    logging.getLogger("nav").setLevel(logging.NOTSET)
