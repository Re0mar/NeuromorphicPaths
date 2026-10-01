"""
Covers the two pose providers.

has_position is the flag the scene branches on to choose the body frame or the world frame, so
these tests are really about which frame the rest of the pipeline will work in.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.pose.device import pose_from_device
from nav.pose.imu_orientation import identity_pose, pose_from_imu


def test_imu_pose_normalizes_the_quaternion() -> None:
    pose = pose_from_imu(np.array([0.0, 0.0, 0.0, 2.0]))

    assert np.linalg.norm(pose.orientation) == pytest.approx(1.0)
    assert pose.has_position is False
    assert pose.position is None


def test_imu_pose_leaves_a_unit_quaternion_alone() -> None:
    orientation = np.array([1.0, 0.0, 0.0, 0.0])
    pose = pose_from_imu(orientation)

    assert pose.orientation == pytest.approx(orientation)


@pytest.mark.parametrize(
    "orientation",
    [
        np.zeros(4),
        np.array([np.nan, 0.0, 0.0, 0.0]),
        np.array([np.inf, 0.0, 0.0, 0.0]),
    ],
)
def test_imu_pose_refuses_a_quaternion_that_is_not_a_rotation(orientation: np.ndarray) -> None:
    # A dropped IMU reading is not a rotation. Normalizing it would tilt the floor plane by an
    # arbitrary amount, and the fit's sanity check would then reject a floor that was fine.
    with pytest.raises(ValueError, match="quaternion"):
        pose_from_imu(orientation)


def test_imu_pose_refuses_a_wrongly_shaped_quaternion() -> None:
    with pytest.raises(ValueError, match=r"\(4,\)"):
        pose_from_imu(np.array([1.0, 0.0, 0.0]))


def test_identity_pose_has_no_rotation_and_no_position() -> None:
    pose = identity_pose()

    assert pose.orientation == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]))
    assert pose.has_position is False


def test_identity_pose_cannot_be_mutated_through_a_shared_array() -> None:
    # It returns a copy of the module constant. Without that, one caller writing into the returned
    # orientation would change every later pose in the run.
    first = identity_pose()
    first.orientation[0] = 99.0

    assert identity_pose().orientation[0] == pytest.approx(1.0)


def test_device_pose_claims_a_position() -> None:
    pose = pose_from_device(np.array([0.0, 0.0, 0.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    assert pose.has_position is True
    assert pose.position == pytest.approx(np.array([1.0, 2.0, 3.0]))
    assert np.linalg.norm(pose.orientation) == pytest.approx(1.0)


def test_device_pose_refuses_a_non_finite_position() -> None:
    # Lost tracking has to arrive as has_position False. A NaN position would put the whole cloud
    # at an undefined place in the world and group ids would never match across frames again.
    with pytest.raises(ValueError, match="has_position False"):
        pose_from_device(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, np.nan, 3.0]))


def test_device_pose_refuses_a_wrongly_shaped_position() -> None:
    with pytest.raises(ValueError, match=r"position must be \(3,\)"):
        pose_from_device(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0]))
