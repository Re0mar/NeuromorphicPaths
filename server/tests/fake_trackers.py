"""
Position trackers for tests of the isolated tracker, in a module of their own.

The isolated tracker spawns its process, and a spawned process gets its tracker by importing it, so
these have to live in a module the child can import by name, not inside a test file.
"""

# Standard library imports
import os
import time

# Third party imports
import numpy as np

# Local package imports
from nav.pose.config import PositionConfig
from nav.pose.tracking import TrackingOutcome
from nav.types import DepthFrame

# Longer than any test waits, so a hung call is never mistaken for a slow one.
HANG_SECONDS = 60.0


class WalkingTracker:
    """Puts the camera along x by the seconds since its own first frame, the way a real tracker starts at its origin."""

    def __init__(self, config: PositionConfig) -> None:
        self.frames = 0
        self._first_timestamp: float | None = None

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        self.frames += 1
        if self._first_timestamp is None:
            self._first_timestamp = frame.timestamp_seconds
        return TrackingOutcome(np.array([frame.timestamp_seconds - self._first_timestamp, 0.0, 0.0]), 0.9)

    def close(self) -> None:
        pass


class HangingOnThirdFrameTracker(WalkingTracker):
    """Answers two frames, then never answers again, the way Open3D's dense SLAM hung."""

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        if self.frames >= 2:
            time.sleep(HANG_SECONDS)
        return super().track(frame)


class DyingOnSecondFrameTracker(WalkingTracker):
    """Answers one frame, then its process ends, the way a crash inside the library ends it."""

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        if self.frames >= 1:
            os._exit(3)
        return super().track(frame)


class FailingToStartTracker:
    """Its process ends before it's ready, every time, the way running out of GPU memory at start does."""

    def __init__(self, config: PositionConfig) -> None:
        raise MemoryError("no room on the card")
