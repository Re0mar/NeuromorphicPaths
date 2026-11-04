"""
Covers the tracker that runs in a child process and is replaced when a call hangs or the process ends.

The child processes here are real, spawned the way a run spawns them. Only the tracker inside is fake.
"""

# Standard library imports
import time

# Third party imports
import numpy as np
import pytest

# Local package imports
from fake_trackers import DyingOnSecondFrameTracker, FailingToStartTracker, HangingOnThirdFrameTracker, WalkingTracker
from nav.pose.config import PositionConfig, PositionSource
from nav.pose.isolated import IsolatedTracker
from nav.pose.tracking import TrackingOutcome
from nav.types import DepthFrame, Pose

STALL_SECONDS = 1.0
# A spawned Python process that imports numpy and Open3D takes seconds to start on this laptop.
START_WAIT_SECONDS = 60.0


def _frame(timestamp_seconds: float) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=timestamp_seconds,
        depth_meters=np.ones((4, 4), np.float32),
        intrinsics=np.eye(3),
        pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False, orientation_is_gravity_aligned=True),
        ground_plane=None,
        gaze_pixel=None,
    )


def _first_answer(tracker: IsolatedTracker, timestamp_seconds: float) -> TrackingOutcome:
    """Hand the same frame until the child answers it, since frames before it's ready get nothing."""
    deadline = time.monotonic() + START_WAIT_SECONDS
    while True:
        outcome = tracker.track(_frame(timestamp_seconds))
        if outcome.position is not None:
            return outcome
        if time.monotonic() > deadline:
            raise AssertionError("the tracker's process never answered")
        time.sleep(0.1)


def _config(max_restarts: int = 3) -> PositionConfig:
    return PositionConfig(source=PositionSource.DENSE_SLAM, stall_seconds=STALL_SECONDS, max_restarts=max_restarts)


def test_a_frame_before_the_process_is_ready_gets_no_position_and_no_wait() -> None:
    tracker = IsolatedTracker(WalkingTracker, _config())
    try:
        began = time.monotonic()
        outcome = tracker.track(_frame(0.0))

        assert outcome.position is None
        assert time.monotonic() - began < STALL_SECONDS
    finally:
        tracker.close()


def test_closing_a_process_that_isn_t_ready_yet_stops_it() -> None:
    tracker = IsolatedTracker(WalkingTracker, _config())

    tracker.close()

    assert tracker.track(_frame(0.0)).position is None


def test_the_child_answers_with_its_tracker_s_position() -> None:
    tracker = IsolatedTracker(WalkingTracker, _config())
    try:
        _first_answer(tracker, 10.0)
        outcome = tracker.track(_frame(12.5))

        assert outcome.position == pytest.approx([2.5, 0.0, 0.0])
        assert outcome.fitness == pytest.approx(0.9)
    finally:
        tracker.close()


def test_a_hung_call_is_given_up_on_time_and_the_new_world_carries_the_last_position() -> None:
    tracker = IsolatedTracker(HangingOnThirdFrameTracker, _config())
    try:
        _first_answer(tracker, 0.0)
        assert tracker.track(_frame(1.0)).position == pytest.approx([1.0, 0.0, 0.0])

        began = time.monotonic()
        hung = tracker.track(_frame(2.0))
        waited = time.monotonic() - began

        assert hung.position is None
        assert waited < STALL_SECONDS + 2.0
        assert tracker.restarts == 1
        # The new process starts its world at the camera, which the old world had at x = 1.
        assert _first_answer(tracker, 5.0).position == pytest.approx([1.0, 0.0, 0.0])
        assert tracker.track(_frame(6.0)).position == pytest.approx([2.0, 0.0, 0.0])
    finally:
        tracker.close()


def test_a_process_that_ends_on_its_own_is_replaced() -> None:
    tracker = IsolatedTracker(DyingOnSecondFrameTracker, _config())
    try:
        _first_answer(tracker, 0.0)

        outcome = tracker.track(_frame(1.0))

        assert outcome.position is None
        assert tracker.restarts == 1
        _first_answer(tracker, 2.0)
    finally:
        tracker.close()


def test_a_process_that_never_starts_is_given_up_on_after_the_restart_limit() -> None:
    tracker = IsolatedTracker(FailingToStartTracker, _config(max_restarts=1))
    try:
        deadline = time.monotonic() + START_WAIT_SECONDS
        # Each start fails, and a frame notices only once the process has ended.
        while not tracker.given_up:
            assert tracker.track(_frame(0.0)).position is None
            if time.monotonic() > deadline:
                raise AssertionError("the tracker never used up its restarts")
            time.sleep(0.1)

        assert tracker.restarts == 1
        assert tracker.track(_frame(1.0)).position is None
        # Given up: no more processes are started however many frames come.
        for step in range(5):
            tracker.track(_frame(2.0 + step))
        assert tracker.restarts == 1
    finally:
        tracker.close()
