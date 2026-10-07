"""
Covers publishing each path the moment it is planned, from a thread of its own.

The loop used to publish only after the next frame arrived, so a finished path waited up to a
frame interval for nothing: about 12 ms on a recorded Pixel walk and about 200 ms on the glasses.
Every wait here has a deadline, so a missing wake fails rather than hangs.
"""

# Standard library imports
import logging
import threading
import time
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
import nav.runtime.loop as loop_module
from nav.clock import laptop_time_seconds
from nav.config import build_run_config
from nav.runtime.loop import FrameResult, PublisherThread, run
from nav.runtime.worker import NewestFrameWorker
from nav.sinks.fan_out import FanOutSink
from nav.types import PlannedPath
from fake_arcore_sender import synthetic_frames

TEST_TIMEOUT_SECONDS = 10.0
# How soon after its plan a path must reach the sink. Generous against a wake-up that takes well
# under a millisecond, and far below the frame interval the old code waited for.
PUBLISHED_WITHIN_SECONDS = 0.05


def _path(timestamp_seconds: float = 1.0) -> PlannedPath:
    return PlannedPath(timestamp_seconds, np.array([0.0, 0.1]), np.array([0.0, 0.05]), 0.1, False, 2.5, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


class RecordingSink:
    """Notes each path, the thread it was published on, and when, on the laptop clock."""

    def __init__(self) -> None:
        self.published: list[tuple[float, str, float]] = []
        self.arrived = threading.Event()

    def start(self) -> None:
        pass

    def publish(self, path: PlannedPath) -> None:
        self.published.append((path.timestamp_seconds, threading.current_thread().name, laptop_time_seconds()))
        self.arrived.set()

    def close(self) -> None:
        pass


class OneFrameThenWait:
    """Yields one frame, then holds the source open until released, like a phone between frames."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def frames(self):
        yield next(iter(synthetic_frames(1)))
        self.release.wait(TEST_TIMEOUT_SECONDS)

    def close(self) -> None:
        self.release.set()


def _run_in_thread(config) -> tuple[threading.Thread, list[int]]:
    exit_codes: list[int] = []
    thread = threading.Thread(target=lambda: exit_codes.append(run(config)), daemon=True)
    thread.start()
    return thread, exit_codes


def test_a_result_is_published_without_another_frame_arriving(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The defect itself. One frame, and the source then quiet: its path must go out at once.

    The source is held open rather than ended, because the end of a source publishes the last
    result anyway, and the old code would pass on that.
    """
    sink = RecordingSink()
    source = OneFrameThenWait()
    plan_done_stamps: list[float] = []

    def stamp_plan_done() -> float:
        # The loop stamps a frame's plan done with this, so the test reads the same stamp.
        plan_done_stamps.append(laptop_time_seconds())
        return plan_done_stamps[-1]

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: sink)
    monkeypatch.setattr(loop_module, "build_source", lambda config: source)
    monkeypatch.setattr(loop_module, "laptop_time_seconds", stamp_plan_done)
    thread, exit_codes = _run_in_thread(build_run_config(["--source", "arcore_tcp", "--sink", "none"]))
    try:
        assert sink.arrived.wait(TEST_TIMEOUT_SECONDS / 2), "the path waited for a frame that never came"
    finally:
        source.release.set()
        thread.join(TEST_TIMEOUT_SECONDS)

    assert not thread.is_alive()
    assert exit_codes == [0]
    assert [timestamp for timestamp, _, _ in sink.published] == [0.0]
    assert sink.published[0][1] == "path-publisher"
    assert len(plan_done_stamps) == 1
    assert sink.published[0][2] - plan_done_stamps[0] < PUBLISHED_WITHIN_SECONDS


def test_each_result_is_published_exactly_once() -> None:
    sink = RecordingSink()
    worker = NewestFrameWorker(lambda frame: FrameResult(_path(frame.timestamp_seconds), np.zeros((1, 3)), np.array([-1.0, 0.0, 1.0]), None))
    worker.start()
    publisher = PublisherThread(sink, worker)
    publisher.start()
    try:
        for frame in synthetic_frames(3):
            worker.submit(frame)
            assert worker.wait_until_idle(TEST_TIMEOUT_SECONDS)
            assert publisher.flush(TEST_TIMEOUT_SECONDS)
        # Nothing new, so nothing more, however long the publisher waits.
        time.sleep(0.3)
    finally:
        publisher.stop()
        worker.stop()

    assert [timestamp for timestamp, _, _ in sink.published] == pytest.approx([0.0, 1 / 30, 2 / 30])


def test_the_last_frame_of_a_source_is_published_before_the_loop_waits_for_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A phone that drops mid-walk. Its last arrow goes out before the laptop waits for it to come back."""

    class SlowSink(RecordingSink):
        # Slower than the loop gets from the worker going idle to opening the next connection, so
        # a loop that did not wait for the publish fails every time rather than now and then.
        def publish(self, path: PlannedPath) -> None:
            time.sleep(0.1)
            super().publish(path)

    sink = SlowSink()
    published_when_reconnect_began: list[list[float]] = []

    class OneFrameThenNobodyComesBack:
        def __init__(self) -> None:
            self.connections = 0

        def frames(self):
            self.connections += 1
            if self.connections == 1:
                yield next(iter(synthetic_frames(1)))
                return
            published_when_reconnect_began.append([timestamp for timestamp, _, _ in sink.published])
            raise ConnectionError("no phone within the accept timeout")

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: sink)
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrameThenNobodyComesBack())

    assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none", "--reconnect"])) == 0

    assert published_when_reconnect_began == [[0.0]]


def test_wait_for_result_returns_immediately_when_a_newer_result_already_exists() -> None:
    """Notify before wait. A result written before the publisher waits must not be missed."""
    worker = NewestFrameWorker(lambda frame: frame.timestamp_seconds)
    worker.start()
    try:
        worker.submit(next(iter(synthetic_frames(1))))
        assert worker.wait_until_idle(TEST_TIMEOUT_SECONDS)
        started = time.perf_counter()
        result = worker.wait_for_result(None, timeout_seconds=5.0)
        elapsed = time.perf_counter() - started
    finally:
        worker.stop()

    assert result == 0.0
    assert elapsed < 0.5, "it waited for a wake that had already happened"


def test_wait_for_result_times_out_with_none_when_nothing_new_comes() -> None:
    worker = NewestFrameWorker(lambda frame: frame.timestamp_seconds)
    worker.start()
    try:
        started = time.perf_counter()
        result = worker.wait_for_result(None, timeout_seconds=0.2)
        elapsed = time.perf_counter() - started
    finally:
        worker.stop()

    assert result is None
    assert 0.15 < elapsed < 2.0


def test_publisher_stop_returns_within_its_bound_when_no_result_ever_comes() -> None:
    worker = NewestFrameWorker(lambda frame: frame.timestamp_seconds)
    worker.start()
    publisher = PublisherThread(RecordingSink(), worker)
    publisher.start()
    try:
        time.sleep(0.2)
        started = time.perf_counter()
        publisher.stop()
        elapsed = time.perf_counter() - started
    finally:
        worker.stop()

    assert not publisher.is_alive()
    assert elapsed < 2.0


def test_a_worker_failure_ends_the_run_with_exit_1(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """
    Mid-stream: the frames keep coming, and the run must notice the worker has died.

    The source never ends on its own, because the end of a source checks for a failure anyway and
    would hide a loop that stopped checking per frame.
    """

    class EndlessFrames:
        def __init__(self) -> None:
            self.stopped = threading.Event()

        def frames(self):
            frames = list(synthetic_frames(30))
            while not self.stopped.is_set():
                for frame in frames:
                    time.sleep(0.01)
                    yield frame

        def close(self) -> None:
            self.stopped.set()

    source = EndlessFrames()
    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RecordingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: source)
    monkeypatch.setattr(loop_module, "gaze_on_the_ground", _raise_runtime_error)

    with caplog.at_level(logging.ERROR):
        thread, exit_codes = _run_in_thread(build_run_config(["--source", "arcore_tcp", "--sink", "none"]))
        thread.join(TEST_TIMEOUT_SECONDS / 2)
        source.stopped.set()
        ended_on_its_own = not thread.is_alive()
        thread.join(TEST_TIMEOUT_SECONDS)

    assert ended_on_its_own, "the run kept reading frames for a worker that had died"
    assert exit_codes == [1]
    assert any("UNEXPECTED RuntimeError" in record.message for record in caplog.records)


def test_a_failure_on_the_last_frame_of_a_finite_source_still_exits_1(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """One frame and the source ends. Nothing after the failure would ever call submit again."""

    class OneFrame:
        def frames(self):
            yield next(iter(synthetic_frames(1)))

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RecordingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrame())
    monkeypatch.setattr(loop_module, "gaze_on_the_ground", _raise_runtime_error)

    with caplog.at_level(logging.ERROR):
        assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 1

    assert any("UNEXPECTED RuntimeError" in record.message for record in caplog.records)


def _raise_runtime_error(*args, **kwargs):
    raise RuntimeError("a bug in the worker")


def test_a_failing_sink_is_dropped_and_publishing_continues_on_the_publisher_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSink:
        def start(self) -> None:
            pass

        def publish(self, path: PlannedPath) -> None:
            raise RuntimeError("a bug in one display")

        def close(self) -> None:
            pass

    class SomeFrames:
        def frames(self):
            for frame in synthetic_frames(5):
                time.sleep(0.05)
                yield frame

        def close(self) -> None:
            pass

    working = RecordingSink()
    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: FanOutSink([BrokenSink(), working]))
    monkeypatch.setattr(loop_module, "build_source", lambda config: SomeFrames())

    assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 0

    assert working.published, "the display that works lost its paths with the one that broke"
    assert {thread_name for _, thread_name, _ in working.published} == {"path-publisher"}


def test_phone_sink_is_called_before_web_for_the_same_result() -> None:
    """The phone's write is on the walker's path. The web page's picture is drawn after it."""
    sink = loop_module.build_sink(build_run_config(["--source", "arcore_tcp", "--sink", "phone_app", "--sink", "web"]))

    assert [type(display).__name__ for display in sink.sinks] == ["PhoneAppSink", "WebSink"]


def test_publish_wait_after_the_fix_is_small(tmp_path: Path) -> None:
    """
    Through run(), a recorded walk sent over TCP: the path goes out within milliseconds of its plan.

    The one timing assertion besides the planner's own. Its bound is generous, so a slow machine
    does not fail it, and still well under the 50 ms the old code could wait between these frames.
    """
    from test_replay_timing import _replay_through_the_sender
    from nav.runtime.timing import FrameOutcome, read_timing_log

    timing_log, _, _ = _replay_through_the_sender(tmp_path)

    waits_ms = [
        (line.sent_seconds - line.plan_done_seconds) * 1000
        for line in read_timing_log(timing_log)
        if line.outcome is FrameOutcome.PUBLISHED and line.sent_seconds is not None
    ]
    assert waits_ms
    assert float(np.median(waits_ms)) < 20.0, f"median publish wait {np.median(waits_ms):.1f} ms"


# *******************************************
# Failures and shutdown, from the audit
# *******************************************


class RaisingSink(RecordingSink):
    """A single display with a bug. Not behind a fan-out, so nothing drops it."""

    def publish(self, path: PlannedPath) -> None:
        raise RuntimeError("a bug in the only display")


def test_a_single_display_failing_mid_stream_ends_the_run_with_exit_1(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class EndlessFrames:
        def __init__(self) -> None:
            self.stopped = threading.Event()

        def frames(self):
            frames = list(synthetic_frames(30))
            while not self.stopped.is_set():
                for frame in frames:
                    time.sleep(0.01)
                    yield frame

        def close(self) -> None:
            self.stopped.set()

    source = EndlessFrames()
    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RaisingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: source)

    with caplog.at_level(logging.ERROR):
        thread, exit_codes = _run_in_thread(build_run_config(["--source", "arcore_tcp", "--sink", "none"]))
        thread.join(TEST_TIMEOUT_SECONDS / 2)
        source.stopped.set()
        ended_on_its_own = not thread.is_alive()
        thread.join(TEST_TIMEOUT_SECONDS)

    assert ended_on_its_own, "the run kept going with its only display broken"
    assert exit_codes == [1]
    assert any("publisher stopped on UNEXPECTED RuntimeError" in record.message and record.exc_info for record in caplog.records)


def test_a_single_display_failing_on_the_last_frame_exits_1_without_waiting_out_the_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the explicit checks around flush catch this one. flush itself re-raises only the worker's failures."""

    class OneFrame:
        def frames(self):
            yield next(iter(synthetic_frames(1)))

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RaisingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrame())

    started = time.perf_counter()
    exit_code = run(build_run_config(["--source", "arcore_tcp", "--sink", "none"]))
    elapsed = time.perf_counter() - started

    assert exit_code == 1
    assert elapsed < loop_module.PUBLISHER_FLUSH_SECONDS, f"took {elapsed:.1f} s, the flush timeout"


def test_a_display_failure_then_ctrl_c_still_exits_1(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The source goes quiet after the failure, so only the interrupt ends the run. It must not end as a success."""

    class OneFrameThenInterrupt:
        def __init__(self) -> None:
            self.sink: RaisingSink | None = None

        def frames(self):
            yield next(iter(synthetic_frames(1)))
            # Wait for the publisher to have failed, then interrupt the way Ctrl+C would.
            deadline = time.perf_counter() + TEST_TIMEOUT_SECONDS
            while not any("publisher stopped" in record.message for record in caplog.records) and time.perf_counter() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RaisingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrameThenInterrupt())

    with caplog.at_level(logging.INFO):
        assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 1

    assert any("before the interrupt" in record.message and record.exc_info for record in caplog.records)


def test_a_worker_failure_is_logged_as_the_workers_not_the_publishers(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class OneFrame:
        def frames(self):
            yield next(iter(synthetic_frames(1)))

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: RecordingSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrame())
    monkeypatch.setattr(loop_module, "gaze_on_the_ground", _raise_runtime_error)

    with caplog.at_level(logging.INFO):
        assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 1

    assert not any("publisher stopped on UNEXPECTED" in record.message for record in caplog.records)


def test_the_publisher_stops_before_the_displays_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """A publisher still running when a display closes can hand it a path afterwards."""
    order: list[str] = []

    class OrderedSink(RecordingSink):
        def close(self) -> None:
            order.append("displays closed")

    real_stop = PublisherThread.stop

    def recorded_stop(self, timeout_seconds: float = 5.0) -> None:
        order.append("publisher stopped")
        real_stop(self, timeout_seconds)

    class OneFrame:
        def frames(self):
            yield next(iter(synthetic_frames(1)))

        def close(self) -> None:
            pass

    monkeypatch.setattr(PublisherThread, "stop", recorded_stop)
    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: OrderedSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrame())

    assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 0

    assert order == ["publisher stopped", "displays closed"]


def test_a_display_too_slow_for_the_bounds_is_said_out_loud(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    release = threading.Event()

    class StuckSink(RecordingSink):
        def publish(self, path: PlannedPath) -> None:
            release.wait(TEST_TIMEOUT_SECONDS)

    class OneFrame:
        def frames(self):
            yield next(iter(synthetic_frames(1)))

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop_module, "PUBLISHER_FLUSH_SECONDS", 0.3)
    monkeypatch.setattr(PublisherThread.stop, "__defaults__", (0.3,))
    monkeypatch.setattr(loop_module, "build_sink", lambda config, **hooks: StuckSink())
    monkeypatch.setattr(loop_module, "build_source", lambda config: OneFrame())
    try:
        with caplog.at_level(logging.WARNING):
            assert run(build_run_config(["--source", "arcore_tcp", "--sink", "none"])) == 0
    finally:
        release.set()

    messages = [record.message for record in caplog.records]
    assert any("the last path didn't reach every display" in message for message in messages)
    assert any("publisher thread still inside a display" in message for message in messages)
