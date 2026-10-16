"""
Covers the tracker that runs in a child process and is replaced when a call hangs.

The child processes here are real, spawned the way a run spawns them. Only the tracker inside is fake.
"""

# Standard library imports
import time

# Third party imports
import numpy as np
import pytest

# Local package imports
from fake_trackers import CountingTracker, HangingOnSecondFrameTracker
from nav.pose.config import PositionConfig, PositionSource
from nav.pose.isolated import IsolatedTracker
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


def _ready(tracker: IsolatedTracker) -> IsolatedTracker:
    """Hand frames until the child answers one, since frames before it's ready are answered with nothing."""
    deadline = time.monotonic() + START_WAIT_SECONDS
    while tracker.track(_frame(-1.0)).position is None:
        if time.monotonic() > deadline:
            raise AssertionError("the tracker's process never became ready")
        time.sleep(0.1)
    return tracker


@pytest.fixture
def config() -> PositionConfig:
    return PositionConfig(source=PositionSource.DENSE_SLAM, stall_seconds=STALL_SECONDS)


def test_a_frame_before_the_process_is_ready_gets_no_position_and_no_wait(config: PositionConfig) -> None:
    tracker = IsolatedTracker(CountingTracker, config)
    try:
        began = time.monotonic()
        outcome = tracker.track(_frame(0.0))

        assert outcome.position is None
        assert time.monotonic() - began < STALL_SECONDS
    finally:
        tracker.close()


def test_the_child_answers_with_its_tracker_s_position(config: PositionConfig) -> None:
    tracker = _ready(IsolatedTracker(CountingTracker, config))
    try:
        outcome = tracker.track(_frame(2.5))

        assert outcome.position == pytest.approx([2.5, 0.0, 0.0])
        assert outcome.fitness == pytest.approx(0.9)
    finally:
        tracker.close()


def test_a_hung_call_is_given_up_on_time_and_the_process_replaced(config: PositionConfig) -> None:
    tracker = _ready(IsolatedTracker(HangingOnSecondFrameTracker, config))
    try:
        began = time.monotonic()
        outcome = tracker.track(_frame(1.0))
        waited = time.monotonic() - began

        assert outcome.position is None
        assert waited < STALL_SECONDS + 2.0
        assert tracker.restarts == 1
        # The replacement is a fresh tracker, so its first frame is answered again.
        _ready(tracker)
    finally:
        tracker.close()
