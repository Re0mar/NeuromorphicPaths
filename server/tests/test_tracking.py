"""
Covers the two position trackers, on depth rendered from a room the camera moves and turns through.

The room is planes and a box the test can intersect exactly, so where the camera went is known
without trusting anything the trackers compute.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.pose.config import PositionConfig, PositionSource
from nav.pose.tracking import DenseSlamTracker, FrameToFrameOdometry, dense_slam_is_available
from nav.types import DepthFrame, Pose

# Large enough that half of it, 120 x 160, still works in Open3D's raycast. At 60 x 80 it fails to launch.
HEIGHT, WIDTH, FOCAL = 240, 320, 240.0
INTRINSICS = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
# The test's world uses the camera's axes, x right, y down, z forward, so turning about y is a turn
# left or right. A floor 1.5 m below, walls 2 m to either side and behind, a wall 4 m ahead, and a
# box on the floor to the left.
FLOOR_Y, FRONT_Z, BACK_Z, SIDE_X = 1.5, 4.0, -2.0, 2.0
BOX_MIN, BOX_MAX = np.array([-1.2, 0.9, 2.2]), np.array([-0.6, 1.5, 2.8])


def _turned_about_y(degrees: float) -> np.ndarray:
    """(w, x, y, z) quaternion for a turn about the camera's y axis."""
    half = np.radians(degrees) / 2.0
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0])


def _rotation_about_y(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])


def _room_depth(camera_position: np.ndarray, yaw_degrees: float = 0.0) -> np.ndarray:
    """Depth along the camera's z for every pixel, from the nearest wall, floor or box face its ray reaches."""
    columns, rows = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    rays_camera = np.stack(((columns - WIDTH / 2.0) / FOCAL, (rows - HEIGHT / 2.0) / FOCAL, np.ones_like(columns)), axis=-1)
    # Each ray's camera z is 1, so the distance along it to a hit is that hit's depth.
    rays = rays_camera @ _rotation_about_y(yaw_degrees).T
    nearest = np.full((HEIGHT, WIDTH), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, value in ((1, FLOOR_Y), (2, FRONT_Z), (2, BACK_Z), (0, -SIDE_X), (0, SIDE_X)):
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


def _frame(
    camera_position: np.ndarray,
    timestamp_seconds: float = 0.0,
    yaw_degrees: float = 0.0,
    depth: np.ndarray | None = None,
    intrinsics: np.ndarray = INTRINSICS,
) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=timestamp_seconds,
        depth_meters=_room_depth(camera_position, yaw_degrees) if depth is None else depth,
        intrinsics=intrinsics,
        pose=Pose(orientation=_turned_about_y(yaw_degrees), position=None, has_position=False, orientation_is_gravity_aligned=True),
        ground_plane=None,
        gaze_pixel=None,
    )


def _half_size_frame(camera_position: np.ndarray, timestamp_seconds: float) -> DepthFrame:
    """The same room at half the resolution, every other pixel, with the camera matrix to match."""
    half_intrinsics = INTRINSICS * np.array([[0.5], [0.5], [1.0]])
    return _frame(camera_position, timestamp_seconds, depth=_room_depth(camera_position)[::2, ::2], intrinsics=half_intrinsics)


def _odometry() -> FrameToFrameOdometry:
    return FrameToFrameOdometry(PositionConfig(source=PositionSource.ODOMETRY))


def test_odometry_puts_the_first_frame_at_the_origin() -> None:
    outcome = _odometry().track(_frame(np.zeros(3)))

    assert outcome.position == pytest.approx(np.zeros(3))


def test_odometry_follows_a_walk_forward_and_sideways() -> None:
    odometry = _odometry()
    # 0.1 m forward and 0.05 m right per frame, a walking step at a few frames a second.
    path = [np.array([0.05 * step, 0.0, 0.1 * step]) for step in range(5)]

    outcomes = [odometry.track(_frame(position, timestamp_seconds=0.5 * step)) for step, position in enumerate(path)]

    assert outcomes[-1].position == pytest.approx(path[-1], abs=0.02)
    assert all(outcome.fitness > 0.5 for outcome in outcomes)


def test_odometry_follows_a_walk_that_turns_as_it_goes() -> None:
    # Starting turned 30 degrees and turning 3 more a frame, so every step is taken in a different
    # direction. The IMU's turn, the camera center and the step's turn into the world all have to be
    # the right way round, or the path bends away from this one. Nine frames: by the tenth the
    # camera sees almost only the side wall, and one flat plane can't pin down a step along itself.
    odometry = _odometry()
    yaws = [30.0 + 3.0 * step for step in range(9)]
    positions = [np.zeros(3)]
    for yaw in yaws[1:]:
        positions.append(positions[-1] + _rotation_about_y(yaw) @ np.array([0.0, 0.0, 0.08]))

    outcomes = [
        odometry.track(_frame(position, timestamp_seconds=0.5 * step, yaw_degrees=yaw))
        for step, (position, yaw) in enumerate(zip(positions, yaws, strict=True))
    ]

    for outcome, position in zip(outcomes, positions, strict=True):
        assert outcome.position == pytest.approx(position, abs=0.01)


def test_odometry_starts_each_match_from_the_imu_s_turn() -> None:
    # The head swings 10 degrees either side while walking ahead, so each frame turns 20 degrees from
    # the last. Started from the IMU's turn the match lands within millimeters. Started from the turn
    # the wrong way round, it was 0.3 to 1.9 m off on this same walk.
    odometry = _odometry()
    yaws = [0.0, 10.0, -10.0, 10.0, -10.0, 0.0]
    positions = [np.array([0.0, 0.0, 0.08 * step]) for step in range(len(yaws))]

    outcomes = [
        odometry.track(_frame(position, timestamp_seconds=0.5 * step, yaw_degrees=yaw))
        for step, (position, yaw) in enumerate(zip(positions, yaws, strict=True))
    ]

    for outcome, position in zip(outcomes, positions, strict=True):
        assert outcome.position == pytest.approx(position, abs=0.01)


def test_odometry_refuses_a_step_no_walker_could_make() -> None:
    odometry = _odometry()
    odometry.track(_frame(np.zeros(3), timestamp_seconds=0.0))

    # 0.3 m in a twentieth of a second is 6 m/s, twice the limit.
    outcome = odometry.track(_frame(np.array([0.0, 0.0, 0.3]), timestamp_seconds=0.05))

    assert outcome.position is None


def test_odometry_gives_no_position_for_a_frame_with_no_depth() -> None:
    odometry = _odometry()
    odometry.track(_frame(np.zeros(3)))

    outcome = odometry.track(_frame(np.zeros(3), timestamp_seconds=0.5, depth=np.full((HEIGHT, WIDTH), np.nan, np.float32)))

    assert outcome.position is None
    assert outcome.fitness == 0.0


def test_odometry_keeps_its_position_when_the_frame_size_changes() -> None:
    odometry = _odometry()
    odometry.track(_frame(np.zeros(3), timestamp_seconds=0.0))
    walked = odometry.track(_frame(np.array([0.0, 0.0, 0.2]), timestamp_seconds=0.5)).position

    outcome = odometry.track(_half_size_frame(np.array([0.0, 0.0, 0.2]), timestamp_seconds=1.0))

    assert outcome.position == pytest.approx(walked)


@pytest.mark.skipif(not dense_slam_is_available(), reason="dense SLAM needs Open3D's CUDA build, see server/README.md")
def test_dense_slam_follows_a_walk_forward() -> None:
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


@pytest.mark.skipif(not dense_slam_is_available(), reason="dense SLAM needs Open3D's CUDA build, see server/README.md")
def test_dense_slam_starts_a_new_volume_at_its_last_position_when_the_frame_size_changes() -> None:
    tracker = DenseSlamTracker(PositionConfig(source=PositionSource.DENSE_SLAM))
    outcomes = [tracker.track(_frame(np.array([0.0, 0.0, 0.03 * step]), timestamp_seconds=step / 30.0)) for step in range(4)]
    last = outcomes[-1].position

    outcome = tracker.track(_half_size_frame(np.array([0.0, 0.0, 0.12]), timestamp_seconds=4 / 30.0))

    assert outcome.position == pytest.approx(last)
