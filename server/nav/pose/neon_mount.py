"""
Where the Pupil Labs Neon's IMU sits, relative to its scene camera and to the pipeline's world.

Both angles come from Pupil Labs' documentation, not from a measurement of our glasses:
https://docs.pupil-labs.com/alpha-lab/imu-transformations/

The scene camera is rotated 102 degrees about x from the IMU, written there as
np.deg2rad(-90 - 12). The extra 12 is the camera tilted down from the module's forward axis. The
IMU's world has Z up and Y toward magnetic north. Ours has Y up, so a quarter turn about x takes
one to the other and puts north on minus Z.

The first live run on the glasses is what confirms these. A floor that leans on a level head
means one of them is wrong.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.pose.imu_orientation import ImuMount

NEON_CAMERA_TO_IMU_ABOUT_X_DEGREES = -102.0
NEON_IMU_WORLD_TO_WORLD_ABOUT_X_DEGREES = -90.0


def rotation_about_x_wxyz(angle_degrees: float) -> np.ndarray:
    """
    The unit quaternion for a rotation about x, (w, x, y, z).

    :param angle_degrees: Right-handed rotation angle.
    :return: (4,) quaternion.
    :rtype: np.ndarray
    """
    half_angle = np.radians(angle_degrees) / 2.0
    return np.array([np.cos(half_angle), np.sin(half_angle), 0.0, 0.0])


NEON_IMU_MOUNT = ImuMount(
    camera_to_body_wxyz=rotation_about_x_wxyz(NEON_CAMERA_TO_IMU_ABOUT_X_DEGREES),
    imu_world_to_world_wxyz=rotation_about_x_wxyz(NEON_IMU_WORLD_TO_WORLD_ABOUT_X_DEGREES),
)
