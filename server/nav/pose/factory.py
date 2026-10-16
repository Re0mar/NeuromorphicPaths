"""Turns a position source into the tracker that measures it."""

# Local package imports
from nav.pose.config import PositionConfig, PositionSource
from nav.pose.isolated import IsolatedTracker
from nav.pose.tracking import DENSE_SLAM_INSTALL, DenseSlamTracker, FrameToFrameOdometry, PositionTracker, dense_slam_is_available


def build_position_tracker(config: PositionConfig) -> PositionTracker:
    """
    The tracker a position source names.

    :param config: The position settings. Its source must not be ABSENT.
    :return: A fresh tracker, with no world yet.
    :rtype: PositionTracker
    :raises ValueError: For ABSENT, which has no tracker, and for dense SLAM without Open3D's CUDA build.
    """
    match config.source:
        case PositionSource.DENSE_SLAM:
            if not dense_slam_is_available():
                raise ValueError(f"--position {config.source.value} needs Open3D's CUDA build. Install it with: {DENSE_SLAM_INSTALL}")
            # In a process of its own, because Open3D's dense SLAM can hang inside a call, see isolated.py.
            return IsolatedTracker(DenseSlamTracker, config)
        case PositionSource.ODOMETRY:
            return FrameToFrameOdometry(config)
        case PositionSource.ABSENT:
            raise ValueError("position source absent has no tracker, the frames keep the pose they came with")
        case _:
            raise ValueError(f"no tracker for position source {config.source}")
