"""Covers turning a position source into its tracker, and the sources that have none or can't have one here."""

# Third party imports
import pytest

# Local package imports
from nav.pose.config import PositionConfig, PositionSource
from nav.pose.factory import build_position_tracker
from nav.pose.tracking import FrameToFrameOdometry, dense_slam_is_available


def test_odometry_is_built_in_this_process() -> None:
    assert isinstance(build_position_tracker(PositionConfig(source=PositionSource.ODOMETRY)), FrameToFrameOdometry)


def test_the_absent_source_has_no_tracker() -> None:
    with pytest.raises(ValueError, match="absent has no tracker"):
        build_position_tracker(PositionConfig(source=PositionSource.ABSENT))


@pytest.mark.skipif(dense_slam_is_available(), reason="this Open3D build has CUDA, so dense SLAM is not refused")
def test_dense_slam_is_refused_without_the_cuda_build_and_says_how_to_install_it() -> None:
    with pytest.raises(ValueError, match="open3d_cuda"):
        build_position_tracker(PositionConfig(source=PositionSource.DENSE_SLAM))
