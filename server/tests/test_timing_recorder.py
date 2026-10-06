"""
Covers the timing recorder: stamps from several threads, one line per frame once its fate is known.

A fake clock drives every stamp, so each line's values are worked out by hand. The run-level tests
in test_end_to_end.py drive the same recorder through run().
"""

# Standard library imports
import json
import logging
import threading
import time
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.runtime.timing import (
    TIMING_FILENAME,
    FrameOutcome,
    StageDurations,
    TimingLog,
    TimingRecord,
    TimingRecorder,
    frame_ns_from_seconds,
    read_timing_log,
    summarize,
)
from nav.types import DepthFrame, FloorSource, FrameTiming, Pose

# Everything here finishes in well under a second. A test that blocks this long is hung.
TEST_TIMEOUT_SECONDS = 5.0
STAGES = StageDurations(scene_milliseconds=41.0, planner_milliseconds=6.0, usermodel_milliseconds=0.5)


class FakeClock:
    """Starts at 1000 s and steps a tenth of a second per reading, so each stamp is known exactly."""

    def __init__(self, start: float = 1000.0, step: float = 0.1) -> None:
        self._next = start
        self._step = step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._next
            self._next += self._step
            return value


def _frame(timestamp_seconds: float, arrival_seconds: float | None = 999.0) -> DepthFrame:
    timing = None if arrival_seconds is None else FrameTiming(capture_seconds=None, arrival_seconds=arrival_seconds, depth_ready_seconds=None)
    return DepthFrame(
        timestamp_seconds=timestamp_seconds,
        depth_meters=np.ones((2, 2), dtype=np.float32),
        intrinsics=np.eye(3),
        pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False),
        ground_plane=None,
        gaze_pixel=None,
        timing=timing,
    )


def _recorder(tmp_path: Path, **overrides) -> tuple[TimingRecorder, Path]:
    path = tmp_path / TIMING_FILENAME
    settings = {"clock": FakeClock()}
    settings.update(overrides)
    return TimingRecorder(TimingLog(path), **settings), path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _publish(recorder: TimingRecorder, frame: DepthFrame, plan_done_seconds: float, sent: bool = True) -> None:
    recorder.frame_taken(frame)
    recorder.frame_planned(frame, plan_done_seconds, FloorSource.FITTED, STAGES)
    if sent:
        recorder.path_sent(frame.timestamp_seconds)
    recorder.path_published(frame.timestamp_seconds)


# *******************************************
# One line per frame, by outcome
# *******************************************


def test_a_published_frame_writes_one_line_with_every_field(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    frame = _frame(123456.789012345, arrival_seconds=999.5)
    recorder.frame_taken(frame)  # started 1000.0
    recorder.frame_planned(frame, 1000.05, FloorSource.FITTED, STAGES)
    recorder.path_sent(frame.timestamp_seconds)  # sent 1000.1
    recorder.path_published(frame.timestamp_seconds)
    recorder.close()

    assert _lines(path) == [
        {
            "timestamp_seconds": 123456.789012345,
            "capture_seconds": None,
            "arrival_seconds": 999.5,
            "depth_ready_seconds": None,
            "plan_done_seconds": 1000.05,
            "floor_source": "fitted",
            "outcome": "published",
            "started_seconds": 1000.0,
            "scene_milliseconds": 41.0,
            "planner_milliseconds": 6.0,
            "usermodel_milliseconds": 0.5,
            "sent_seconds": pytest.approx(1000.1),
        }
    ]


def test_published_without_a_phone_client_has_null_sent_seconds(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    _publish(recorder, _frame(1.0), 1000.05, sent=False)
    recorder.close()

    (line,) = _lines(path)
    assert line["outcome"] == "published"
    assert line["sent_seconds"] is None


def test_an_unsent_result_is_closed_as_superseded_when_a_later_frame_publishes(tmp_path: Path) -> None:
    """The publisher sends only the newest result, so one planned before it and never sent is superseded."""
    recorder, path = _recorder(tmp_path)
    older, newer = _frame(1.0), _frame(2.0)
    recorder.frame_taken(older)
    recorder.frame_planned(older, 1000.05, FloorSource.FITTED, STAGES)
    _publish(recorder, newer, 1000.25)
    recorder.close()

    outcomes = {line["timestamp_seconds"]: line["outcome"] for line in _lines(path)}
    assert outcomes == {1.0: "superseded", 2.0: "published"}


def test_superseded_follows_plan_order_not_the_frame_clock(tmp_path: Path) -> None:
    """A phone that reconnects starts ARCore's clock again lower, so frame timestamps can go backwards."""
    recorder, path = _recorder(tmp_path)
    before_reconnect, after_reconnect = _frame(5000.0), _frame(3.0)
    recorder.frame_taken(before_reconnect)
    recorder.frame_planned(before_reconnect, 1000.05, FloorSource.FITTED, STAGES)
    _publish(recorder, after_reconnect, 1000.25)
    recorder.close()

    outcomes = {line["timestamp_seconds"]: line["outcome"] for line in _lines(path)}
    assert outcomes == {5000.0: "superseded", 3.0: "published"}


def test_a_dropped_frame_writes_outcome_dropped(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    recorder.frame_dropped(_frame(7.0, arrival_seconds=998.0))
    recorder.close()

    (line,) = _lines(path)
    assert line["outcome"] == "dropped"
    assert line["arrival_seconds"] == 998.0
    # The six keys the reader requires are all there, null where the frame never got that far.
    assert line["plan_done_seconds"] is None and line["floor_source"] is None and line["started_seconds"] is None


def test_a_skipped_frame_writes_outcome_skipped_and_keeps_its_null_plan_done(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    frame = _frame(8.0)
    recorder.frame_taken(frame)
    recorder.frame_skipped(frame, None)
    recorder.close()

    (line,) = _lines(path)
    assert line["outcome"] == "skipped"
    assert line["plan_done_seconds"] is None
    assert line["floor_source"] is None
    assert line["started_seconds"] == 1000.0


def test_a_frame_the_planner_refused_after_the_floor_keeps_its_floor(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    frame = _frame(9.0)
    recorder.frame_taken(frame)
    recorder.frame_skipped(frame, FloorSource.FITTED)
    recorder.close()

    (line,) = _lines(path)
    assert line["outcome"] == "skipped"
    assert line["floor_source"] == "fitted"


def test_the_log_is_lf_only_and_reads_back(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    _publish(recorder, _frame(1.0), 1000.05)
    recorder.frame_dropped(_frame(2.0))
    recorder.close()

    assert b"\r" not in path.read_bytes()
    records = read_timing_log(path)
    assert [record.outcome for record in records] == [FrameOutcome.PUBLISHED, FrameOutcome.DROPPED]


def test_frame_ns_from_seconds_recovers_integer_nanos_near_a_day_of_uptime() -> None:
    # The phone sends its ARCore nanoseconds divided by 1e9, and JSON carries the double unchanged.
    generator = np.random.default_rng(20261006)
    for original in generator.integers(10_000_000_000_000, 1_000_000_000_000_000, size=10_000):
        seconds = json.loads(json.dumps(int(original) / 1e9))
        assert frame_ns_from_seconds(seconds) == int(original)


# *******************************************
# The reader, old lines and new
# *******************************************


def test_read_timing_log_reads_lines_with_and_without_the_new_fields(tmp_path: Path) -> None:
    old_line = {
        "timestamp_seconds": 1.0,
        "capture_seconds": None,
        "arrival_seconds": 10.0,
        "depth_ready_seconds": None,
        "plan_done_seconds": 10.2,
        "floor_source": "fitted",
    }
    new_line = dict(old_line, timestamp_seconds=2.0, outcome="published", started_seconds=10.05, scene_milliseconds=40.0, planner_milliseconds=6.0, usermodel_milliseconds=0.5, sent_seconds=10.3)
    path = tmp_path / TIMING_FILENAME
    path.write_bytes((json.dumps(old_line) + "\n" + json.dumps(new_line) + "\n").encode("utf-8"))

    old, new = read_timing_log(path)

    assert old.outcome is None and old.sent_seconds is None
    assert new.outcome is FrameOutcome.PUBLISHED
    assert new.sent_seconds == 10.3
    assert new.scene_milliseconds == 40.0


def test_an_unknown_outcome_is_refused_naming_the_line(tmp_path: Path) -> None:
    path = tmp_path / TIMING_FILENAME
    line = {"timestamp_seconds": 1.0, "capture_seconds": None, "arrival_seconds": 1.0, "depth_ready_seconds": None, "plan_done_seconds": None, "floor_source": None, "outcome": "lost"}
    path.write_bytes((json.dumps(line) + "\n").encode("utf-8"))

    with pytest.raises(ValueError, match="line 1 has outcome 'lost'"):
        read_timing_log(path)


def test_a_published_line_without_its_plan_time_is_refused_naming_the_line(tmp_path: Path) -> None:
    path = tmp_path / TIMING_FILENAME
    line = {"timestamp_seconds": 1.0, "capture_seconds": None, "arrival_seconds": 1.0, "depth_ready_seconds": None, "plan_done_seconds": None, "floor_source": "fitted", "outcome": "published", "started_seconds": 1.0, "scene_milliseconds": 1.0, "planner_milliseconds": 1.0, "usermodel_milliseconds": 1.0, "sent_seconds": None}
    path.write_bytes((json.dumps(line) + "\n").encode("utf-8"))

    with pytest.raises(ValueError, match="line 1 is published and has no plan_done_seconds"):
        read_timing_log(path)


def test_summarize_ignores_dropped_and_in_flight_lines_in_rate_and_floor_counts() -> None:
    """A dropped frame was never taken, so it has no floor to count and no plan in the rate."""
    planned = [
        TimingRecord(float(index), None, 10.0 + index * 0.1, None, 10.05 + index * 0.1, FloorSource.FITTED, outcome=FrameOutcome.PUBLISHED, started_seconds=10.0 + index * 0.1, scene_milliseconds=1.0, planner_milliseconds=1.0, usermodel_milliseconds=1.0)
        for index in range(4)
    ]
    dropped = [TimingRecord(10.0 + index, None, 10.0 + index * 0.1, None, None, None, outcome=FrameOutcome.DROPPED) for index in range(3)]
    in_flight = [TimingRecord(20.0, None, 10.5, None, None, None, outcome=FrameOutcome.IN_FLIGHT)]

    summary = summarize(planned + dropped + in_flight, exclude_first_seconds=0.0)

    assert summary.frames_measured == 4
    assert summary.floor_source_counts == {FloorSource.FITTED: 4}


# *******************************************
# Refusals and limits
# *******************************************


def test_frames_open_at_close_are_written_in_flight(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    frame = _frame(1.0)
    recorder.frame_taken(frame)
    recorder.frame_planned(frame, 1000.05, FloorSource.FITTED, STAGES)
    recorder.close()

    (line,) = _lines(path)
    assert line["outcome"] == "in_flight"
    assert line["plan_done_seconds"] == 1000.05


def test_open_records_are_bounded(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path, max_open_records=10)
    for index in range(11):
        recorder.frame_taken(_frame(float(index)))
    deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    # The oldest is written out as in flight the moment the eleventh opens, before close.
    assert [line["timestamp_seconds"] for line in _lines(path)] == [0.0]
    assert _lines(path)[0]["outcome"] == "in_flight"
    recorder.close()
    assert len(_lines(path)) == 11


def test_close_returns_when_the_writer_is_stuck(tmp_path: Path) -> None:
    release = threading.Event()

    class StuckLog(TimingLog):
        def append(self, record: TimingRecord) -> None:
            release.wait(TEST_TIMEOUT_SECONDS * 2)

    recorder = TimingRecorder(StuckLog(tmp_path / TIMING_FILENAME), clock=FakeClock(), close_timeout_seconds=0.2)
    try:
        recorder.frame_dropped(_frame(1.0))
        time.sleep(0.05)
        started = time.monotonic()
        recorder.close()
        assert time.monotonic() - started < 2.0, "close waited on a stuck writer"
    finally:
        release.set()


def test_a_write_failure_stops_the_log_and_says_so_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A full disk costs the measurement and one error line, never the walk."""

    class FullDisk(TimingLog):
        def append(self, record: TimingRecord) -> None:
            raise OSError(28, "No space left on device")

    recorder = TimingRecorder(FullDisk(tmp_path / TIMING_FILENAME), clock=FakeClock())
    with caplog.at_level(logging.ERROR):
        for index in range(3):
            recorder.frame_dropped(_frame(float(index)))
        recorder.close()

    stopped = [record for record in caplog.records if "timing log stopped writing" in record.message]
    assert len(stopped) == 1


def test_an_unexpected_writer_error_is_logged_as_unexpected(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class Broken(TimingLog):
        def append(self, record: TimingRecord) -> None:
            raise RuntimeError("a bug in the writer")

    recorder = TimingRecorder(Broken(tmp_path / TIMING_FILENAME), clock=FakeClock())
    with caplog.at_level(logging.ERROR):
        recorder.frame_dropped(_frame(1.0))
        recorder.close()

    assert any("UNEXPECTED RuntimeError in the timing writer" in record.message for record in caplog.records)


def test_a_send_or_publish_for_a_frame_never_taken_writes_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Such as a frame already pushed out by the open-record bound. There is no line to finish."""
    recorder, path = _recorder(tmp_path)
    with caplog.at_level(logging.ERROR):
        recorder.path_sent(42.0)
        recorder.path_published(42.0)
        recorder.close()

    assert not path.exists()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_a_stamp_after_close_is_ignored(tmp_path: Path) -> None:
    recorder, path = _recorder(tmp_path)
    recorder.close()

    recorder.frame_dropped(_frame(1.0))

    assert not path.exists()


def test_a_recorder_without_a_log_records_nothing_and_starts_no_thread(tmp_path: Path) -> None:
    threads_before = threading.active_count()
    recorder = TimingRecorder(None, clock=FakeClock())

    _publish(recorder, _frame(1.0), 1000.05)
    # Counted before close, because close would end a writer that should never have started.
    assert threading.active_count() == threads_before
    recorder.close()

    assert not list(tmp_path.iterdir())
