"""
Poses that carry orientation and no position.

Both functions here return has_position False, which is what the scene reads to decide it must
rebuild the cloud in the body frame every frame instead of accumulating one in the world. They
live together for that reason rather than because both involve an IMU.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.types import Pose

# Quaternion with no rotation, in the (w, x, y, z) order every Pose uses.
IDENTITY_ORIENTATION_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


def pose_from_imu(orientation_wxyz: np.ndarray) -> Pose:
    """
    Turn an IMU quaternion into a pose with no position.

    :param orientation_wxyz: Quaternion in (w, x, y, z) order. Need not be normalized.
    :return: Pose carrying the normalized orientation, has_position False.
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

    # An IMU's orientation is measured against gravity, which is the whole reason it is worth
    # carrying without a position. The scene reads the floor's up from it.
    return Pose(
        orientation=orientation_wxyz / length,
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
