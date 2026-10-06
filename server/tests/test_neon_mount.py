"""
Covers the Neon's mount against physical poses whose answers were worked out by hand.

Every expected vector here is a literal, not a round trip through the code's own inverse. A test
that rotates a vector and rotates it back passes whatever the convention is, including the wrong
one. Pupil Labs documents the scene camera at 102 degrees about x from the IMU, 12 of which is the
camera tilted down, and the IMU's world as Z up with Y toward north.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.pose.device import pose_from_device
from nav.pose.imu_orientation import multiply_wxyz, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT, rotation_about_x_wxyz
from nav.runtime.loop import wrap_angle, yaw_from_quaternion
from nav.scene.config import SceneConfig
from nav.scene.pipeline import ScenePipeline
from nav.scene.transform import rotation_matrix_from_quaternion_wxyz
from nav.types import WORLD_UP, DepthFrame, FloorSource
from nav.walker import WalkerConfig
from synthetic_depth import CAMERA_HEIGHT_METERS, clean_scene

LEVEL_FACING_NORTH_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])
SIN_12 = np.sin(np.radians(12.0))
COS_12 = np.cos(np.radians(12.0))


def _rotation_about_z_wxyz(angle_degrees: float) -> np.ndarray:
    half_angle = np.radians(angle_degrees) / 2.0
    return np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])


def _forward_in_world(orientation_wxyz: np.ndarray) -> np.ndarray:
    return rotation_matrix_from_quaternion_wxyz(orientation_wxyz) @ CAMERA_FORWARD


def test_level_glasses_put_camera_forward_twelve_degrees_below_the_horizon() -> None:
    pose = pose_from_imu(LEVEL_FACING_NORTH_WXYZ, NEON_IMU_MOUNT)

    # Twelve degrees down, and the level part along minus Z, which is where north lands.
    assert _forward_in_world(pose.orientation) == pytest.approx([0.0, -SIN_12, -COS_12], abs=1e-9)


def test_level_glasses_give_world_up_in_the_camera_frame() -> None:
    pose = pose_from_imu(LEVEL_FACING_NORTH_WXYZ, NEON_IMU_MOUNT)
    frame = DepthFrame(
        timestamp_seconds=0.0,
        depth_meters=np.ones((4, 4), dtype=np.float32),
        intrinsics=np.eye(3),
        pose=pose,
        ground_plane=None,
        gaze_pixel=None,
    )

    # Read through the scene's own method, the one that decides what the floor is measured against.
    # Mostly image-up (minus y), tipped toward minus z because the camera looks down.
    up_camera = ScenePipeline._up_in_camera_frame(frame)

    assert up_camera == pytest.approx([0.0, -COS_12, -SIN_12], abs=1e-9)


def test_turning_the_glasses_right_a_quarter_puts_forward_east_and_still_twelve_below_the_horizon() -> None:
    """A turn about the IMU's up axis has to move only the heading. The camera's tilt stays put."""
    # Right is a negative turn about the IMU world's Z. North is +Y there, so a quarter turn right
    # faces +X, east, and the quarter turn about x into our world leaves X alone. Forward is east and
    # still 12 degrees down. Every other Neon case turns about x only, and rotations about one axis
    # commute, so those cases cannot tell the two mount rotations apart. With the two swapped this
    # comes out (1, 0, 0), level.
    pose = pose_from_imu(_rotation_about_z_wxyz(-90.0), NEON_IMU_MOUNT)

    assert _forward_in_world(pose.orientation) == pytest.approx([COS_12, -SIN_12, 0.0], abs=1e-9)


def test_pitching_the_glasses_down_thirty_degrees_puts_forward_forty_two_below_the_horizon() -> None:
    # The IMU's x points right, so a pitch down is a negative turn about it.
    pose = pose_from_imu(rotation_about_x_wxyz(-30.0), NEON_IMU_MOUNT)

    forward = _forward_in_world(pose.orientation)
    below_horizon_degrees = np.degrees(np.arcsin(-(forward @ WORLD_UP)))

    assert below_horizon_degrees == pytest.approx(42.0, abs=1e-6)


def test_a_level_neon_frame_gets_a_floor_accepted_at_the_default_tilt() -> None:
    # A floor 1.6 m below a camera pitched 12 degrees down, which is what level glasses see.
    scene = clean_scene(box_lateral_meters=None, pitch_degrees=12.0)
    frame = DepthFrame(
        timestamp_seconds=0.0,
        depth_meters=scene.depth_meters,
        intrinsics=scene.intrinsics,
        pose=pose_from_imu(LEVEL_FACING_NORTH_WXYZ, NEON_IMU_MOUNT),
        ground_plane=None,
        gaze_pixel=None,
    )
    pipeline = ScenePipeline(SceneConfig(), WalkerConfig())

    pipeline.process(frame)

    assert pipeline.last_floor_source is FloorSource.FITTED
    assert pipeline.previous_plane.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.05)


def test_a_neon_right_turn_moves_yaw_the_same_way_as_an_arcore_right_turn() -> None:
    # ARCore, level and facing minus Z. The camera's y points down, so a half turn about x.
    arcore_start = np.array([0.0, 1.0, 0.0, 0.0])
    # A right turn in a Y-up world is a negative turn about Y.
    arcore_right = np.array([np.cos(np.radians(-45.0)), 0.0, np.sin(np.radians(-45.0)), 0.0])
    arcore_turned = pose_from_device(multiply_wxyz(arcore_right, arcore_start), np.zeros(3)).orientation
    arcore_change = wrap_angle(yaw_from_quaternion(arcore_turned) - yaw_from_quaternion(arcore_start))

    # The Neon, level and facing north. A right turn in the IMU's Z-up world is a negative turn about Z.
    neon_start = pose_from_imu(LEVEL_FACING_NORTH_WXYZ, NEON_IMU_MOUNT).orientation
    neon_turned = pose_from_imu(_rotation_about_z_wxyz(-90.0), NEON_IMU_MOUNT).orientation
    neon_change = wrap_angle(yaw_from_quaternion(neon_turned) - yaw_from_quaternion(neon_start))

    # Both turned right by a quarter. Both report it with the same sign, whatever that sign is.
    assert abs(arcore_change) == pytest.approx(np.pi / 2, abs=1e-6)
    assert abs(neon_change) == pytest.approx(np.pi / 2, abs=1e-6)
    assert np.sign(neon_change) == np.sign(arcore_change)
