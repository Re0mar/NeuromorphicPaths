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
from nav.pose.imu_orientation import ImuMount, identity_pose, is_usable_orientation, multiply_wxyz, pose_from_imu
from nav.pose.neon_mount import rotation_about_x_wxyz
from nav.scene.transform import rotation_matrix_from_quaternion_wxyz

IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
# A mount that changes nothing, for the tests that are about the IMU quaternion itself.
NO_MOUNT = ImuMount(camera_to_body_wxyz=IDENTITY_WXYZ, imu_world_to_world_wxyz=IDENTITY_WXYZ)


def test_imu_pose_normalizes_the_quaternion() -> None:
    pose = pose_from_imu(np.array([0.0, 0.0, 0.0, 2.0]), NO_MOUNT)

    assert np.linalg.norm(pose.orientation) == pytest.approx(1.0)
    assert pose.has_position is False
    assert pose.position is None


def test_imu_pose_leaves_a_unit_quaternion_alone() -> None:
    """With no mount, the IMU's reading is the pose. Inverting it would turn every head turn backwards."""
    # All four parts non-zero, so a conjugate, a reordering or a renormalization that moves it shows.
    orientation = np.array([0.5, 0.5, -0.5, 0.5])

    pose = pose_from_imu(orientation, NO_MOUNT)

    assert pose.orientation == pytest.approx(orientation, abs=1e-12)


def test_imu_pose_composes_the_mount_in_world_imu_camera_order() -> None:
    # Camera to body is a quarter turn about x, the IMU a quarter turn about z, our world nothing.
    # Applied rightmost first, camera forward (0, 0, 1) goes to body (0, -1, 0) under +90 about x,
    # then to (1, 0, 0) under +90 about z. Reversed order would give (0, 0, 1) then (0, -1, 0).
    mount = ImuMount(camera_to_body_wxyz=rotation_about_x_wxyz(90.0), imu_world_to_world_wxyz=IDENTITY_WXYZ)
    quarter_turn_about_z = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    pose = pose_from_imu(quarter_turn_about_z, mount)

    forward = rotation_matrix_from_quaternion_wxyz(pose.orientation) @ np.array([0.0, 0.0, 1.0])
    assert forward == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_imu_pose_applies_the_world_part_last() -> None:
    """The IMU's world only means something after the IMU has turned, so its fix-up comes last."""
    # World part a quarter turn about x, IMU a quarter turn about z, camera to body nothing. The two
    # do not commute. Right order: z leaves forward (0, 0, 1) alone, then x takes it to (0, -1, 0).
    # World part first instead: x gives (0, -1, 0), then z takes that to (1, 0, 0).
    mount = ImuMount(camera_to_body_wxyz=IDENTITY_WXYZ, imu_world_to_world_wxyz=rotation_about_x_wxyz(90.0))
    quarter_turn_about_z = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    pose = pose_from_imu(quarter_turn_about_z, mount)

    forward = rotation_matrix_from_quaternion_wxyz(pose.orientation) @ np.array([0.0, 0.0, 1.0])
    assert forward == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


def test_the_quaternion_product_is_the_product_of_the_two_rotations() -> None:
    """multiply_wxyz is what every mounted pose is built from, so it has to compose real rotations."""
    # Seeded, with every component non-zero. The Neon cases are all rotations about x, which leave
    # most of the product's terms multiplied by zero, so a sign slip in those terms passes them.
    rng = np.random.default_rng(7)
    for _ in range(5):
        left = rng.normal(size=4)
        right = rng.normal(size=4)
        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)

        product = multiply_wxyz(left, right)

        expected = rotation_matrix_from_quaternion_wxyz(left) @ rotation_matrix_from_quaternion_wxyz(right)
        assert rotation_matrix_from_quaternion_wxyz(product) == pytest.approx(expected, abs=1e-12)
        assert np.linalg.norm(product) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "orientation",
    [
        np.zeros(4),
        np.array([0.3, 0.2, 0.0, 0.0]),
        np.array([np.nan, 0.0, 0.0, 0.0]),
        np.array([np.inf, 0.0, 0.0, 0.0]),
    ],
)
def test_an_empty_imu_reading_is_not_a_usable_orientation(orientation: np.ndarray) -> None:
    """A zero or NaN reading is a missing sample. Taking it as a rotation ends the run in pose_from_imu."""
    assert is_usable_orientation(orientation) is False


@pytest.mark.parametrize(
    "orientation",
    [np.array([1.0, 0.0, 0.0, 0.0]), np.array([2.0, 0.2, 0.0, 0.0]), np.array([0.5, 0.0, 0.0, 0.0])],
)
def test_a_real_imu_reading_is_a_usable_orientation(orientation: np.ndarray) -> None:
    """Unit, unnormalized and borderline readings are rotations, and pose_from_imu normalizes them."""
    assert is_usable_orientation(orientation) is True


@pytest.mark.parametrize(
    "quaternion",
    [np.zeros(4), np.array([2.0, 0.0, 0.0, 0.0]), np.array([np.nan, 0.0, 0.0, 0.0])],
)
def test_imu_mount_refuses_a_quaternion_that_is_not_unit(quaternion: np.ndarray) -> None:
    # A mount is written by hand from a datasheet. Normalizing a typo would hide it.
    with pytest.raises(ValueError, match="camera_to_body_wxyz"):
        ImuMount(camera_to_body_wxyz=quaternion, imu_world_to_world_wxyz=IDENTITY_WXYZ)


def test_imu_mount_refuses_a_wrongly_shaped_quaternion() -> None:
    with pytest.raises(ValueError, match=r"imu_world_to_world_wxyz must be \(4,\)"):
        ImuMount(camera_to_body_wxyz=IDENTITY_WXYZ, imu_world_to_world_wxyz=np.array([1.0, 0.0, 0.0]))


def test_imu_mount_cannot_be_mutated_through_the_callers_array() -> None:
    callers_array = IDENTITY_WXYZ.copy()
    mount = ImuMount(camera_to_body_wxyz=callers_array, imu_world_to_world_wxyz=IDENTITY_WXYZ)

    callers_array[0] = 0.0

    assert mount.camera_to_body_wxyz == pytest.approx(IDENTITY_WXYZ)


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
        pose_from_imu(orientation, NO_MOUNT)


def test_imu_pose_refuses_a_wrongly_shaped_quaternion() -> None:
    with pytest.raises(ValueError, match=r"\(4,\)"):
        pose_from_imu(np.array([1.0, 0.0, 0.0]), NO_MOUNT)


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
