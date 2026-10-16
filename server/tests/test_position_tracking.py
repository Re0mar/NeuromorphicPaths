"""
Covers the two position trackers, on depth rendered from a room the camera moves through.

The room is planes the test can intersect exactly, so where the camera went is known without
trusting anything the trackers compute.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.pose.config import PositionConfig, PositionSource
from nav.pose.factory import build_position_tracker
from nav.pose.tracking import DenseSlamTracker, FrameToFrameOdometry, dense_slam_is_available
from nav.types import DepthFrame, Pose

HEIGHT, WIDTH, FOCAL = 120, 160, 120.0
INTRINSICS = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
# Camera frame: x right, y down, z forward. With the identity orientation the world is the same axes.
# A floor 1.5 m below, a wall 4 m ahead, side walls 2 m either way, and a box on the floor to the left.
FLOOR_Y, FRONT_Z, SIDE_X = 1.5, 4.0, 2.0
BOX_MIN, BOX_MAX = np.array([-1.2, 0.9, 2.2]), np.array([-0.6, 1.5, 2.8])


def _room_depth(camera_position: np.ndarray) -> np.ndarray:
    """Depth along z for every pixel, from the nearest plane or box face the pixel's ray reaches."""
    columns, rows = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    rays = np.stack(((columns - WIDTH / 2.0) / FOCAL, (rows - HEIGHT / 2.0) / FOCAL, np.ones_like(columns)), axis=-1)
    nearest = np.full((HEIGHT, WIDTH), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, value in ((1, FLOOR_Y), (2, FRONT_Z), (0, -SIDE_X), (0, SIDE_X)):
            distance = (value - camera_position[axis]) / rays[..., axis]
            nearest = np.where((distance > 0) & (distance < nearest), distance, nearest)
        # The box by its slabs: a ray is inside every slab over one stretch, and enters at its start.
        entering = np.full((HEIGHT, WIDTH), -np.inf)
        leaving = np.full((HEIGHT, WIDTH), np.inf)
        for axis in range(3):
            low = (BOX_MIN[axis] - camera_position[axis]) / rays[..., axis]
            high = (BOX_MAX[axis] - camera_position[axis]) / rays[..., axis]
            entering = np.maximum(entering, np.minimum(low, high))
            leaving = np.minimum(leaving, np.maximum(low, high))
        hits_box = (entering <= leaving) & (entering > 0)
    nearest = np.where(hits_box & (entering < nearest), entering, nearest)
    return np.where(np.isfinite(nearest), nearest, np.nan).astype(np.float32)


def _frame(camera_position: np.ndarray, timestamp_seconds: float = 0.0, depth: np.ndarray | None = None) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=timestamp_seconds,
        depth_meters=_room_depth(camera_position) if depth is None else depth,
        intrinsics=INTRINSICS,
        pose=Pose(orientation=IDENTITY_WXYZ, position=None, has_position=False, orientation_is_gravity_aligned=True),
        ground_plane=None,
        gaze_pixel=None,
    )


def test_odometry_puts_the_first_frame_at_the_origin() -> None:
    odometry = FrameToFrameOdometry(PositionConfig(source=PositionSource.ODOMETRY))

    outcome = odometry.track(_frame(np.zeros(3)))

    assert outcome.position == pytest.approx(np.zeros(3))


def test_odometry_follows_a_walk_forward_and_sideways() -> None:
    odometry = FrameToFrameOdometry(PositionConfig(source=PositionSource.ODOMETRY))
    # 0.1 m forward and 0.05 m right per frame, a walking step at a few frames a second.
    path = [np.array([0.05 * step, 0.0, 0.1 * step]) for step in range(5)]

    outcomes = [odometry.track(_frame(position, timestamp_seconds=0.5 * step)) for step, position in enumerate(path)]

    assert outcomes[-1].position == pytest.approx(path[-1], abs=0.02)
    assert all(outcome.fitness > 0.5 for outcome in outcomes)


def test_odometry_gives_no_position_for_a_frame_with_no_depth() -> None:
    odometry = FrameToFrameOdometry(PositionConfig(source=PositionSource.ODOMETRY))
    odometry.track(_frame(np.zeros(3)))

    outcome = odometry.track(_frame(np.zeros(3), depth=np.full((HEIGHT, WIDTH), np.nan, np.float32)))

    assert outcome.position is None
    assert outcome.fitness == 0.0


def test_odometry_starts_over_after_a_reset() -> None:
    odometry = FrameToFrameOdometry(PositionConfig(source=PositionSource.ODOMETRY))
    odometry.track(_frame(np.zeros(3)))
    odometry.track(_frame(np.array([0.0, 0.0, 0.2])))

    odometry.reset()
    outcome = odometry.track(_frame(np.array([0.0, 0.0, 0.2])))

    assert outcome.position == pytest.approx(np.zeros(3))


def test_the_absent_source_has_no_tracker() -> None:
    with pytest.raises(ValueError, match="absent has no tracker"):
        build_position_tracker(PositionConfig(source=PositionSource.ABSENT))


@pytest.mark.skipif(dense_slam_is_available(), reason="this Open3D build has CUDA, so dense SLAM is not refused")
def test_dense_slam_is_refused_without_the_cuda_build_and_says_how_to_install_it() -> None:
    with pytest.raises(ValueError, match="open3d_cuda"):
        build_position_tracker(PositionConfig(source=PositionSource.DENSE_SLAM))


@pytest.mark.skipif(not dense_slam_is_available(), reason="dense SLAM needs Open3D's CUDA build, see server/README.md")
def test_dense_slam_follows_a_walk_forward() -> None:
    # The tracker itself, in this process. The factory wraps it in a child process, covered on its own.
    tracker = DenseSlamTracker(PositionConfig(source=PositionSource.DENSE_SLAM))
    # 3 cm a frame, walking pace at a depth sensor's 30 frames a second, the rate dense SLAM is built for.
    path = [np.array([0.0, 0.0, 0.03 * step]) for step in range(11)]

    outcomes = [tracker.track(_frame(position, timestamp_seconds=step / 30.0)) for step, position in enumerate(path)]

    # Wiring, not accuracy: the camera ends up ahead, by about the walked distance. On this room of
    # flat planes dense SLAM drifts about 0.1 m sideways over 0.3 m with a fit above 0.9, where odometry
    # lands within millimeters. Its accuracy on a real walk is measured against a reference elsewhere.
    forward = outcomes[-1].position[2]
    sideways = np.linalg.norm(outcomes[-1].position[:2])
    assert forward == pytest.approx(path[-1][2], abs=0.1)
    assert sideways < 0.2
