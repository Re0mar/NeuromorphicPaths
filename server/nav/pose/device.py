"""
Poses that carry a position: from a device that tracks its own, which today means ARCore on the
Pixel, or from a tracker that measured one for a pose that had none.

Lost tracking is `has_position` False on that frame, never a made-up position.
"""

# Standard library imports
import dataclasses

# Third party imports
import numpy as np

# Local package imports
from nav.types import Pose


def pose_from_device(orientation_wxyz: np.ndarray, position_xyz: np.ndarray) -> Pose:
    """
    Turn a device's own tracked pose into a Pose that claims a position.

    :param orientation_wxyz: Quaternion in (w, x, y, z) order. Need not be normalized.
    :param position_xyz: Position in meters in the device's world frame.
    :return: Pose with `has_position` True.
    :rtype: Pose
    """
    if orientation_wxyz.shape != (4,):
        raise ValueError(f"orientation must be (4,) in (w, x, y, z) order, got shape {orientation_wxyz.shape}")
    if position_xyz.shape != (3,):
        raise ValueError(f"position must be (3,) in meters, got shape {position_xyz.shape}")

    length = float(np.linalg.norm(orientation_wxyz))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError(f"orientation must be a non-zero finite quaternion, got {orientation_wxyz}")
    if not np.all(np.isfinite(position_xyz)):
        # Tracking that has lost its fix should arrive as has_position False, not as a NaN
        # position. Letting one through would put the whole cloud at an undefined place in the
        # world and the group ids would never match across frames again.
        raise ValueError(f"position must be finite, got {position_xyz}. Lost tracking is has_position False")

    return Pose(orientation=orientation_wxyz / length, position=position_xyz, has_position=True)


def pose_with_position(pose: Pose, position_xyz: np.ndarray) -> Pose:
    """
    The same pose with a measured position added, for a source whose own pose had none.

    The orientation and its gravity flag are kept as they came. On the glasses those are the IMU's,
    which stays the source of up whatever measured the position, since the floor fit depends on it.

    :param pose: The frame's pose, whatever its source.
    :param position_xyz: Position in meters, in the world the orientation rotates into.
    :return: The pose with `has_position` True.
    :rtype: Pose
    """
    if position_xyz.shape != (3,):
        raise ValueError(f"position must be (3,) in meters, got shape {position_xyz.shape}")
    if not np.all(np.isfinite(position_xyz)):
        raise ValueError(f"position must be finite, got {position_xyz}. Lost tracking is has_position False")
    return dataclasses.replace(pose, position=np.asarray(position_xyz, dtype=np.float64), has_position=True)
