"""
Covers the source that adds a measured position to frames that came without one.

The trackers here are fakes with real behaviour: a list of answers, one that never answers, one that
fails. What's under test is which frames get a position, what they keep, and that a hung tracker
costs the walker the position and never the arrow.
"""

# Standard library imports
import threading
from collections.abc import Iterator

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.pose.tracking import TrackingOutcome
from nav.sources.positioned import PositionedDepthSource
from nav.types import DepthFrame, Pose

ORIENTATION_WXYZ = np.array([0.9238795, 0.3826834, 0.0, 0.0])  # 45 degrees about x, so a kept orientation is visible
STALL_SECONDS = 0.2


def _frame(timestamp_seconds: float, gravity_aligned: bool = True) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=timestamp_seconds,
        depth_meters=np.ones((4, 4), np.float32),
        intrinsics=np.eye(3),
        pose=Pose(orientation=ORIENTATION_WXYZ, position=None, has_position=False, orientation_is_gravity_aligned=gravity_aligned),
        ground_plane=None,
        gaze_pixel=None,
    )


class ListSource:
    def __init__(self, frames: list[DepthFrame]) -> None:
        self._frames = frames
        self.closed = False

    def frames(self) -> Iterator[DepthFrame]:
        yield from self._frames

    def close(self) -> None:
        self.closed = True


class AnsweringTracker:
    """Answers from a list, in order, and counts the frames it was handed."""

    def __init__(self, answers: list[TrackingOutcome]) -> None:
        self._answers = list(answers)
        self.calls = 0

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        self.calls += 1
        return self._answers.pop(0)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class HangingTracker:
    """Never answers until released, the way a stalled Open3D call behaves."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        self.calls += 1
        self.release.wait()
        return TrackingOutcome(np.zeros(3), 1.0)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FailingTracker:
    def track(self, frame: DepthFrame) -> TrackingOutcome:
        raise ArithmeticError("tracker fell over")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _run(source: PositionedDepthSource) -> list[DepthFrame]:
    frames = list(source.frames())
    source.close()
    return frames


def test_a_tracked_frame_gets_the_position_and_keeps_its_orientation_and_gravity() -> None:
    tracker = AnsweringTracker([TrackingOutcome(np.array([1.0, 0.0, 2.0]), 0.8)])

    (frame,) = _run(PositionedDepthSource(ListSource([_frame(0.0)]), tracker, STALL_SECONDS))

    assert frame.pose.has_position is True
    assert frame.pose.position == pytest.approx([1.0, 0.0, 2.0])
    assert frame.pose.orientation == pytest.approx(ORIENTATION_WXYZ)
    assert frame.pose.orientation_is_gravity_aligned is True


def test_a_frame_the_tracker_lost_goes_on_without_a_position() -> None:
    tracker = AnsweringTracker([TrackingOutcome(None, 0.05)])

    (frame,) = _run(PositionedDepthSource(ListSource([_frame(0.0)]), tracker, STALL_SECONDS))

    assert frame.pose.has_position is False


def test_a_frame_without_gravity_never_reaches_the_tracker() -> None:
    tracker = AnsweringTracker([])

    (frame,) = _run(PositionedDepthSource(ListSource([_frame(0.0, gravity_aligned=False)]), tracker, STALL_SECONDS))

    assert tracker.calls == 0
    assert frame.pose.has_position is False


def test_a_hung_tracker_lets_every_frame_through_and_is_not_asked_again() -> None:
    tracker = HangingTracker()
    frames = [_frame(0.0), _frame(0.5), _frame(1.0)]

    passed = _run(PositionedDepthSource(ListSource(frames), tracker, STALL_SECONDS))
    tracker.release.set()

    assert [frame.timestamp_seconds for frame in passed] == [0.0, 0.5, 1.0]
    assert all(frame.pose.has_position is False for frame in passed)
    assert tracker.calls == 1


def test_a_tracker_failure_nobody_predicted_reaches_the_pipeline() -> None:
    source = PositionedDepthSource(ListSource([_frame(0.0)]), FailingTracker(), STALL_SECONDS)

    with pytest.raises(ArithmeticError, match="tracker fell over"):
        list(source.frames())


def test_closing_closes_the_source_underneath() -> None:
    inner = ListSource([])

    _run(PositionedDepthSource(inner, AnsweringTracker([]), STALL_SECONDS))

    assert inner.closed is True


def test_closing_frees_the_tracker() -> None:
    tracker = AnsweringTracker([])

    _run(PositionedDepthSource(ListSource([]), tracker, STALL_SECONDS))

    assert tracker.closed is True
