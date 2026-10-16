"""
Position trackers for tests of the isolated tracker, in a module of their own.

The isolated tracker spawns its process, and a spawned process gets its tracker by importing it, so
these have to live in a module the child can import by name, not inside a test file.
"""

# Standard library imports
import time

# Third party imports
import numpy as np

# Local package imports
from nav.pose.config import PositionConfig
from nav.pose.tracking import TrackingOutcome
from nav.types import DepthFrame

# Longer than any test waits, so a hung call is never mistaken for a slow one.
HANG_SECONDS = 60.0


class CountingTracker:
    """Puts each frame at its own timestamp along x, so the test can tell frames apart by position."""

    def __init__(self, config: PositionConfig) -> None:
        self.frames = 0

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        self.frames += 1
        return TrackingOutcome(np.array([frame.timestamp_seconds, 0.0, 0.0]), 0.9)

    def reset(self) -> None:
        self.frames = 0

    def close(self) -> None:
        pass


class HangingOnSecondFrameTracker(CountingTracker):
    """Answers the first frame, then never answers again, the way Open3D's dense SLAM hung."""

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        if self.frames >= 1:
            time.sleep(HANG_SECONDS)
        return super().track(frame)
