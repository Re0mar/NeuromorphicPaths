"""
Moving points and planes between the camera frame and the world frame.

Used only when a frame's pose has a position. Without one the scene stays in the camera frame and
nothing here is called.

Convention, stated once: a Pose's orientation rotates camera-frame vectors into the world frame.
world = R camera + position. A source that gets a world-to-camera rotation from its device, which
is what a view matrix is, hands over the inverse.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.types import Plane, Pose


def rotation_matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """
    The rotation a unit quaternion (w, x, y, z) represents, as a (3, 3) matrix.

    :param quaternion: (4,) in (w, x, y, z) order. Normalized here, so near-unit input is fine.
    :return: (3, 3) rotation matrix.
    :rtype: np.ndarray
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must be (4,) in (w, x, y, z) order, got shape {quaternion.shape}")
    length = np.linalg.norm(quaternion)
    if not np.isfinite(length) or length == 0:
        raise ValueError(f"quaternion must be a non-zero finite rotation, got {quaternion}")
    w, x, y, z = quaternion / length

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_to_world_points(points_camera: np.ndarray, pose: Pose) -> np.ndarray:
    """
    Place camera-frame points in the world frame the pose describes.

    :param points_camera: (N, 3) points.
    :param pose: Must carry a position.
    :return: (N, 3) points in the world frame.
    :rtype: np.ndarray
    """
    if not pose.has_position or pose.position is None:
        raise ValueError("a pose without a position cannot place points in the world")
    rotation = rotation_matrix_from_quaternion_wxyz(pose.orientation)
    return np.asarray(points_camera) @ rotation.T + pose.position


def camera_to_world_plane(plane: Plane, pose: Pose) -> Plane:
    """
    Express a camera-frame plane in the world frame.

    A plane normal . p + offset == 0 under p -> R p + t becomes (R n) . p' + (offset - (R n) . t).

    :param plane: In the camera frame.
    :param pose: Must carry a position.
    :return: The same plane in the world frame.
    :rtype: Plane
    """
    if not pose.has_position or pose.position is None:
        raise ValueError("a pose without a position cannot place a plane in the world")
    rotation = rotation_matrix_from_quaternion_wxyz(pose.orientation)
    normal_world = rotation @ plane.normal
    return Plane(normal=normal_world, offset_meters=float(plane.offset_meters - normal_world @ pose.position))
