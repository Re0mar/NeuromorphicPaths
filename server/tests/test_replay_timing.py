"""
Covers measuring the laptop on a recorded walk: the walk sent over TCP into a live-shaped run.

A `--source logged` replay keeps the walk's arrival times, so its shares mix two runs. Sending the
recording through fake_arcore_sender.py into `--source arcore_tcp`, with fake_path_reader.py on
the path port, runs everything a live walk runs, and the arrival is stamped fresh.
"""

# Standard library imports
import dataclasses
import importlib.util
import json
import logging
import socket
import threading
import time
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.clock import laptop_time_seconds
from nav.config import build_run_config
from nav.runtime.loop import run
from nav.runtime.tap import RecordingTap
from nav.runtime.timing import FrameOutcome, read_timing_log, summarize
from nav.sinks.config import PhoneAppConfig
from nav.sinks.phone_app import PhoneAppSink
from nav.sources.framecodec import INDEX_FILENAME
from nav.types import FrameTiming, PlannedPath
from fake_arcore_sender import recorded_frames, send_frames, synthetic_frames
from fake_path_reader import read_paths

FRAME_COUNT = 12
# A stamp from a walk days ago. A replay that kept it would read as planned days after arrival.
OLD_ARRIVAL_SECONDS = 1_700_000_000.0
TEST_TIMEOUT_SECONDS = 20.0


def _timing_report_main():
    # examples/ is not a package, so the script is loaded by path, the way a person runs it.
    spec = importlib.util.spec_from_file_location("timing_report", Path(__file__).parent.parent / "examples" / "timing_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _FrameList:
    """A source over frames already in memory, so a frame log can be written without a run."""

    def __init__(self, frames) -> None:
        self._frames = list(frames)

    def frames(self):
        yield from self._frames

    def close(self) -> None:
        pass


def _recorded_walk(target: Path) -> Path:
    """A frame log whose frames carry the arrival times of a run long gone, the way a recorded walk does."""
    old = FrameTiming(capture_seconds=None, arrival_seconds=OLD_ARRIVAL_SECONDS, depth_ready_seconds=None)
    frames = [dataclasses.replace(frame, timing=old) for frame in synthetic_frames(FRAME_COUNT)]
    list(RecordingTap(_FrameList(frames), target).frames())
    return target


def test_the_sender_replays_a_frame_log_with_its_timestamps_and_no_timing_block(tmp_path: Path) -> None:
    walk = _recorded_walk(tmp_path / "walk")

    frames = list(recorded_frames(walk, realtime=False))

    assert [frame.timestamp_seconds for frame in frames] == [frame.timestamp_seconds for frame in synthetic_frames(FRAME_COUNT)]
    assert all(frame.timing is None for frame in frames), "the phone never sends a timing block"


def _replay_through_the_sender(tmp_path: Path) -> tuple[Path, list[int], float]:
    """
    A recorded walk sent into run() over TCP, the phone sink read by the stand-in phone.

    :return: The timing log, how many paths the reader read, and when the run started on the laptop clock.
    """
    walk = _recorded_walk(tmp_path / "walk")
    depth_port, phone_port = _free_port(), _free_port()
    timing_log = tmp_path / "replay_timing.jsonl"
    config = build_run_config(
        [
            "--source", "arcore_tcp", "--arcore-port", str(depth_port), "--arcore-accept-timeout", "10",
            "--sink", "phone_app", "--phone-port", str(phone_port),
            "--timing-log", str(timing_log),
        ]
    )
    paths_read: list[int] = []
    reader = threading.Thread(target=lambda: paths_read.append(read_paths("127.0.0.1", phone_port, wait_seconds=10.0)), daemon=True)
    reader.start()

    def send_after_the_reader_is_served() -> None:
        # The first paths would go to no phone if the frames beat the reader to the sink.
        time.sleep(0.5)
        send_frames("127.0.0.1", depth_port, recorded_frames(walk, realtime=False), frame_gap_seconds=0.05, wait_seconds=10.0)

    sender = threading.Thread(target=send_after_the_reader_is_served, daemon=True)
    sender.start()
    started = laptop_time_seconds()
    assert run(config) == 0
    sender.join(TEST_TIMEOUT_SECONDS)
    reader.join(TEST_TIMEOUT_SECONDS)
    return timing_log, paths_read, started


def test_a_real_run_produces_a_log_the_report_reads(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """From the production entry points: run() writes the log, timing_report.main reads it."""
    timing_log, _, _ = _replay_through_the_sender(tmp_path)
    saved = tmp_path / "report.json"

    assert _timing_report_main()([str(timing_log), "--exclude-first-seconds", "0", "--json", str(saved)]) == 0

    printed = capsys.readouterr().out
    assert "laptop toward the phone, ms" in printed
    assert "mixes two runs" not in printed
    shares = json.loads(saved.read_text(encoding="utf-8"))["shares"]
    assert shares["laptop_publish_wait"]["frame_count"] > 0
    assert shares["laptop"]["median_milliseconds"] >= shares["laptop_publish_wait"]["median_milliseconds"]


def test_a_replay_through_the_sender_gives_fresh_arrivals_and_no_mixed_run(tmp_path: Path) -> None:
    """Through run() and the real phone sink, with the stand-ins for the phone on both ports."""
    timing_log, paths_read, started = _replay_through_the_sender(tmp_path)

    lines = read_timing_log(timing_log)
    assert len(lines) == FRAME_COUNT, "every frame sent gets exactly one line"
    assert all(line.arrival_seconds >= started for line in lines), "arrival came from the walk, not from this run"
    published = [line for line in lines if line.outcome is FrameOutcome.PUBLISHED]
    assert published
    assert all(line.sent_seconds is not None for line in published), "a published path the reader read has a send time"
    assert all(line.started_seconds <= line.plan_done_seconds <= line.sent_seconds for line in published)
    assert summarize(lines, exclude_first_seconds=0.0).mixed_run_count == 0
    assert paths_read and paths_read[0] == len(published)
    assert not (tmp_path / INDEX_FILENAME).exists(), "--timing-log measures without recording frames"


def test_a_run_with_no_phone_sink_writes_one_line_per_frame_received(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    walk = _recorded_walk(tmp_path / "walk")
    timing_log = tmp_path / "timing.jsonl"
    config = build_run_config(["--source", "logged", "--log-dir", str(walk), "--sink", "none", "--timing-log", str(timing_log)])

    with caplog.at_level(logging.INFO):
        assert run(config) == 0

    lines = read_timing_log(timing_log)
    frames_in = int(next(record.message for record in caplog.records if " frames in, " in record.message).split(" frames in")[0])
    assert len(lines) == frames_in == FRAME_COUNT
    assert all(line.outcome is not FrameOutcome.IN_FLIGHT for line in lines)
    assert all(line.sent_seconds is None for line in lines), "no phone sink, so nothing was sent"


def test_fake_path_reader_counts_paths_the_phone_sink_sent() -> None:
    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"))
    sink.start()
    paths_read: list[int] = []
    reader = threading.Thread(target=lambda: paths_read.append(read_paths("127.0.0.1", sink.port, wait_seconds=5.0)), daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
        while not sink.connected:
            assert time.monotonic() < deadline, "the reader never connected"
            time.sleep(0.01)
        path = PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), 0.1, False, 2.5, scene_information_bits=0.0, avoidance_surprise_bits=0.0)
        for _ in range(3):
            sink.publish(path)
    finally:
        sink.close()
    reader.join(TEST_TIMEOUT_SECONDS)

    assert paths_read == [3]
