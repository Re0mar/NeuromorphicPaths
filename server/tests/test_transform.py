"""Covers moving points and planes into the world frame, against rotations known by hand."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.transform import camera_to_world_plane, camera_to_world_points, rotation_matrix_from_quaternion_wxyz
from nav.types import Plane, Pose

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
# 90 degrees about y: (cos 45, 0, sin 45, 0). Takes camera z to world x.
QUARTER_TURN_ABOUT_Y = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])


def test_the_identity_quaternion_is_the_identity_matrix() -> None:
    assert rotation_matrix_from_quaternion_wxyz(IDENTITY) == pytest.approx(np.eye(3))


def test_a_quarter_turn_about_y_takes_z_to_x() -> None:
    rotation = rotation_matrix_from_quaternion_wxyz(QUARTER_TURN_ABOUT_Y)

    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(np.array([1.0, 0.0, 0.0]), abs=1e-9)


def test_the_matrix_is_orthonormal_for_an_unnormalised_quaternion() -> None:
    rotation = rotation_matrix_from_quaternion_wxyz(QUARTER_TURN_ABOUT_Y * 3.0)

    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [np.zeros(4), np.array([np.nan, 0, 0, 0]), np.zeros(3)])
def test_a_quaternion_that_is_not_a_rotation_is_refused(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        rotation_matrix_from_quaternion_wxyz(bad)


def test_points_are_rotated_then_translated() -> None:
    pose = Pose(orientation=QUARTER_TURN_ABOUT_Y, position=np.array([10.0, 0.0, 0.0]), has_position=True)

    world = camera_to_world_points(np.array([[0.0, 0.0, 2.0]]), pose)

    # Two meters ahead of the camera, which points along world x from (10, 0, 0).
    assert world == pytest.approx(np.array([[12.0, 0.0, 0.0]]), abs=1e-9)


def test_a_pose_without_a_position_cannot_place_points() -> None:
    pose = Pose(orientation=IDENTITY, position=None, has_position=False)

    with pytest.raises(ValueError, match="without a position"):
        camera_to_world_points(np.zeros((1, 3)), pose)


def test_a_point_on_the_plane_stays_on_the_transformed_plane() -> None:
    # Camera floor: normal up (-y), camera 1.6 m above it. A point 1.6 m below the camera is on it.
    plane_camera = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.6)
    on_plane_camera = np.array([[0.3, 1.6, 2.0]])
    pose = Pose(orientation=QUARTER_TURN_ABOUT_Y, position=np.array([5.0, -0.2, 7.0]), has_position=True)

    plane_world = camera_to_world_plane(plane_camera, pose)
    on_plane_world = camera_to_world_points(on_plane_camera, pose)

    assert on_plane_world @ plane_world.normal + plane_world.offset_meters == pytest.approx(0.0, abs=1e-9)


def test_the_camera_keeps_its_height_above_the_transformed_plane() -> None:
    plane_camera = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.6)
    pose = Pose(orientation=QUARTER_TURN_ABOUT_Y, position=np.array([5.0, -0.2, 7.0]), has_position=True)

    plane_world = camera_to_world_plane(plane_camera, pose)

    assert pose.position @ plane_world.normal + plane_world.offset_meters == pytest.approx(1.6)
