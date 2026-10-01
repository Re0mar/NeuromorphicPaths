"""Covers the newest-frame worker: it drops stale frames, surfaces failures, and never hangs the loop."""

# Standard library imports
import threading
import time

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.runtime.worker import NewestFrameWorker
from nav.types import DepthFrame, Pose

INTRINSICS = np.eye(3)


def _frame(index: int) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=float(index),
        depth_meters=np.ones((2, 2), dtype=np.float32),
        intrinsics=INTRINSICS,
        pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False),
        ground_plane=None,
        gaze_pixel=None,
    )


def test_only_the_newest_frame_is_processed_when_processing_is_slow() -> None:
    processed: list[float] = []

    def slow(frame: DepthFrame) -> float:
        time.sleep(0.02)
        processed.append(frame.timestamp_seconds)
        return frame.timestamp_seconds

    worker = NewestFrameWorker(slow)
    worker.start()
    try:
        for index in range(20):
            worker.submit(_frame(index))
        worker.stop()
    finally:
        worker.stop()

    assert len(processed) < 20, "a slow worker that processes every frame is falling behind forever"
    assert worker.latest_result() == 19.0, "the last result must be from the last submitted frame"
    assert worker.dropped + worker.processed == 20


def test_a_degenerate_frame_is_skipped_and_the_worker_lives_on(caplog: pytest.LogCaptureFixture) -> None:
    def picky(frame: DepthFrame) -> float:
        if frame.timestamp_seconds == 1.0:
            raise ValueError("no floor found and no previous plane")
        return frame.timestamp_seconds

    worker = NewestFrameWorker(picky)
    worker.start()
    try:
        for index in range(3):
            worker.submit(_frame(index))
            time.sleep(0.02)
        worker.stop()
    finally:
        worker.stop()

    assert worker.latest_result() == 2.0
    assert worker.skipped == 1


def test_ten_consecutive_skips_log_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    def always_fails(frame: DepthFrame) -> float:
        raise ValueError("ceiling")

    worker = NewestFrameWorker(always_fails)
    worker.start()
    try:
        with caplog.at_level("WARNING"):
            # Wait for each skip to land before submitting the next frame. Submitting on a timer
            # lets a loaded machine drop frames before the worker sees them, and then fewer than
            # ten skips happen and the warning never fires, which is a test failing on scheduling.
            for index in range(10):
                worker.submit(_frame(index))
                deadline = time.monotonic() + 2.0
                while worker.skipped < index + 1 and time.monotonic() < deadline:
                    time.sleep(0.001)
                assert worker.skipped == index + 1, f"skip {index + 1} did not happen in time"
            worker.stop()
    finally:
        worker.stop()

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "10 consecutive" in warnings[0].message


def test_an_unexpected_exception_ends_the_thread_and_surfaces_through_latest_result() -> None:
    def broken(frame: DepthFrame) -> float:
        raise RuntimeError("a bug, not bad input")

    worker = NewestFrameWorker(broken)
    worker.start()
    worker.submit(_frame(0))
    worker.join(2.0)

    assert not worker.is_alive(), "an unexpected error must end the thread rather than loop on it"
    with pytest.raises(RuntimeError, match="a bug"):
        worker.latest_result()


def test_latest_result_is_none_before_anything_was_processed() -> None:
    worker = NewestFrameWorker(lambda frame: frame.timestamp_seconds)
    assert worker.latest_result() is None


def test_stop_returns_promptly_when_idle() -> None:
    worker = NewestFrameWorker(lambda frame: frame.timestamp_seconds)
    worker.start()

    started = time.monotonic()
    worker.stop()

    assert time.monotonic() - started < 1.0
    assert not worker.is_alive()


def test_submit_is_safe_from_another_thread() -> None:
    worker = NewestFrameWorker(lambda frame: frame.timestamp_seconds)
    worker.start()

    def hammer() -> None:
        for index in range(200):
            worker.submit(_frame(index))

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    worker.stop()

    assert worker.submitted == 800
    assert worker.dropped + worker.processed == 800
