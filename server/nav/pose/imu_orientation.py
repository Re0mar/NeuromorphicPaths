"""
Poses that carry orientation and no position.

Both pose functions here return has_position False, which is what the scene reads to decide it must
rebuild the cloud in the body frame every frame instead of accumulating one in the world. They
live together for that reason rather than because both involve an IMU.

An IMU reports its own orientation in its own world. A Pose needs the camera's orientation in the
pipeline's world, the one with WORLD_UP up. The two fixed rotations between them are the device's
mount, and pose_from_imu will not run without one.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.types import Pose

# Quaternion with no rotation, in the (w, x, y, z) order every Pose uses.
IDENTITY_ORIENTATION_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

# How far from unit length a mount quaternion may be. Mounts are written down by hand from a
# datasheet, so anything further off is a typo rather than rounding.
MOUNT_UNIT_TOLERANCE = 1e-6

# A real orientation is a unit quaternion. Anything this short is an empty reading, not a rotation.
MINIMUM_QUATERNION_LENGTH = 0.5

# How far an IMU sample may sit from a frame's capture stamp and still be that frame's orientation.
# A Neon scene frame is 33 ms and its IMU runs at about 110 Hz, so a usable sample is within 5 ms of
# any stamp unless the stream dropped or went empty. Past this the frame gets no orientation at all,
# because an older one is wrong by however far the head turned since.
IMU_MATCH_TOLERANCE_SECONDS = 0.05


def usable_orientation_mask(orientations_wxyz: np.ndarray) -> np.ndarray:
    """
    Which IMU readings are orientations at all, rather than empty ones.

    The Neon sent nothing but zero quaternions for minutes at a time on 2026-10-05, and a recording
    can hold NaN where no sample sat near a frame. Both are skipped the same way on every route,
    so this is the one place that decides what counts as empty.

    :param orientations_wxyz: (N, 4) quaternions as the device reported them, not yet normalized.
    :return: (N,) True where a reading is finite and long enough to be a rotation.
    :rtype: np.ndarray
    """
    lengths = np.linalg.norm(np.asarray(orientations_wxyz, dtype=np.float64).reshape(-1, 4), axis=1)
    return np.isfinite(lengths) & (lengths >= MINIMUM_QUATERNION_LENGTH)


def is_usable_orientation(orientation_wxyz: np.ndarray) -> bool:
    """
    Whether one IMU reading is an orientation at all. See usable_orientation_mask.

    :param orientation_wxyz: (4,) quaternion as the device reported it, not yet normalized.
    :return: True when it is finite and long enough to be a rotation.
    :rtype: bool
    """
    return bool(usable_orientation_mask(orientation_wxyz)[0])


def orientation_at(
    stamps_seconds: np.ndarray,
    orientations_wxyz: np.ndarray,
    stamp_seconds: float,
    tolerance_seconds: float = IMU_MATCH_TOLERANCE_SECONDS,
) -> np.ndarray | None:
    """
    The orientation a frame captured at a given moment had: the usable IMU reading nearest that moment.

    The one rule for every Neon route. An empty reading is never chosen, however near. When no usable
    reading lies within the tolerance the answer is None, never an older one, because a frame posed
    from where the head was half a second earlier puts the floor off by however far it turned.

    :param stamps_seconds: (N,) reading times, ascending, on the same clock as stamp_seconds.
    :param orientations_wxyz: (N, 4) readings as the device reported them, empty ones included.
    :param stamp_seconds: When the frame was captured.
    :param tolerance_seconds: How far the chosen reading may be from stamp_seconds.
    :return: (4,) the chosen reading, a copy, or None.
    :rtype: np.ndarray | None
    :raises ValueError: When the two arrays don't pair up.
    """
    stamps = np.asarray(stamps_seconds, dtype=np.float64).reshape(-1)
    orientations = np.asarray(orientations_wxyz, dtype=np.float64).reshape(-1, 4)
    if len(stamps) != len(orientations):
        raise ValueError(f"{len(stamps)} stamps for {len(orientations)} orientations")
    usable = usable_orientation_mask(orientations)
    usable_stamps = stamps[usable]
    if usable_stamps.size == 0:
        return None
    index = int(np.searchsorted(usable_stamps, stamp_seconds))
    candidates = [i for i in (index - 1, index) if 0 <= i < usable_stamps.size]
    nearest = min(candidates, key=lambda i: abs(usable_stamps[i] - stamp_seconds))
    if abs(usable_stamps[nearest] - stamp_seconds) > tolerance_seconds:
        return None
    return orientations[usable][nearest].copy()


@dataclass(frozen=True)
class ImuMount:
    """How one device's IMU sits relative to its camera and to the pipeline's world.

    camera_to_body_wxyz rotates camera-frame vectors (x right, y down, z forward) into the IMU's
    own body frame. imu_world_to_world_wxyz rotates the IMU's world into the pipeline's world,
    the one with WORLD_UP up. Both are fixed for a given device, which is why they live as data
    rather than as code in the source.
    """

    camera_to_body_wxyz: np.ndarray
    imu_world_to_world_wxyz: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ("camera_to_body_wxyz", "imu_world_to_world_wxyz"):
            quaternion = np.array(getattr(self, field_name), dtype=np.float64)
            if quaternion.shape != (4,):
                raise ValueError(f"{field_name} must be (4,) in (w, x, y, z) order, got shape {quaternion.shape}")
            if not np.all(np.isfinite(quaternion)):
                raise ValueError(f"{field_name} must be finite, got {quaternion}")
            length = float(np.linalg.norm(quaternion))
            if abs(length - 1.0) > MOUNT_UNIT_TOLERANCE:
                raise ValueError(f"{field_name} must be a unit quaternion, got length {length}")
            # A copy, so a caller writing into the array it passed in cannot move the mount.
            object.__setattr__(self, field_name, quaternion / length)


def multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    The Hamilton product of two (w, x, y, z) quaternions.

    As rotations, the result applies right first and then left.

    :param left: (4,) quaternion.
    :param right: (4,) quaternion.
    :return: (4,) quaternion, left times right.
    :rtype: np.ndarray
    """
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return np.array(
        [
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ]
    )


def pose_from_imu(orientation_wxyz: np.ndarray, mount: ImuMount) -> Pose:
    """
    Turn an IMU quaternion into the camera's pose in the pipeline's world, with no position.

    :param orientation_wxyz: The IMU body's orientation in the IMU's world, (w, x, y, z). Need not
        be normalized.
    :param mount: The device's fixed rotations. Required, because the raw quaternion describes the
        wrong body in the wrong world and passing it through unchanged looks like it works.
    :return: Pose whose orientation rotates camera-frame vectors into the WORLD_UP world,
        has_position False, gravity-aligned.
    :rtype: Pose
    """
    if orientation_wxyz.shape != (4,):
        raise ValueError(f"orientation must be (4,) in (w, x, y, z) order, got shape {orientation_wxyz.shape}")

    length = float(np.linalg.norm(orientation_wxyz))
    # A zero or non-finite quaternion is a dropped IMU reading, not a rotation. Rotating by it
    # would tilt the floor plane by an arbitrary amount and the fit's sanity check would then
    # reject a floor that was fine.
    if not np.isfinite(length) or length == 0.0:
        raise ValueError(f"orientation must be a non-zero finite quaternion, got {orientation_wxyz}")

    # Camera into the IMU body, the body into the IMU's world, that world into ours. Rightmost
    # applies first.
    world_from_camera = multiply_wxyz(
        mount.imu_world_to_world_wxyz,
        multiply_wxyz(orientation_wxyz / length, mount.camera_to_body_wxyz),
    )

    # An IMU's orientation is measured against gravity, which is the whole reason it is worth
    # carrying without a position. The scene reads the floor's up from it.
    return Pose(
        orientation=world_from_camera / np.linalg.norm(world_from_camera),
        position=None,
        has_position=False,
        orientation_is_gravity_aligned=True,
    )


def identity_pose() -> Pose:
    """
    The pose for a source with no orientation at all, such as a plain video file.

    Downstream then works in the camera's own axes, so the floor fit has to find the ground
    rather than being told roughly where it is. Nothing here knows which way gravity points, so
    the scene falls back to the image's own up and assumes the camera is held roughly level.

    :return: Pose with no rotation, no position, and no gravity.
    :rtype: Pose
    """
    return Pose(
        orientation=IDENTITY_ORIENTATION_WXYZ.copy(),
        position=None,
        has_position=False,
        orientation_is_gravity_aligned=False,
    )
