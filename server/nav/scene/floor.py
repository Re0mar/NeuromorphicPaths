"""
Finding the ground in a point cloud, and measuring heights against it.

A Plane here is written so that normal . point + offset is the height above the floor, with the
normal pointing up. That convention is what lets the height band be two comparisons.
"""

# Third party imports
import numpy as np
import open3d

# Local package imports
from nav.scene.config import SceneConfig
from nav.types import Plane

# Up, in camera axes where y points down.
CAMERA_UP = np.array([0.0, -1.0, 0.0])


def fit_floor(points_camera: np.ndarray, previous: Plane | None, config: SceneConfig) -> Plane:
    """
    RANSAC a plane through the points that could plausibly be floor.

    Keeps the old file's two sanity checks. A plane that is not roughly level or that sits too
    close to the camera is not the floor, however many points agree with it, and in that case the
    previous frame's plane is better than a wrong one.

    :param points_camera: (N, 3) camera-frame points.
    :param previous: Last frame's plane, returned when this frame has no believable floor.
    :param config: Candidate selection and sanity thresholds.
    :return: The floor, normal pointing up.
    :rtype: Plane
    :raises ValueError: When no floor is found and there is no previous plane to fall back on.
    """
    # Only points clearly below the camera can be floor. Without this, a wall straight ahead
    # with enough points wins the vote.
    candidates = points_camera[points_camera[:, 1] > config.floor_candidate_min_below_camera_meters]
    fitted = None
    if len(candidates) >= config.floor_min_candidate_points:
        fitted = _ransac_plane(candidates, config)

    if fitted is not None:
        return fitted
    if previous is not None:
        return previous
    raise ValueError(
        f"no floor found in {len(points_camera)} points, {len(candidates)} below the camera, and no previous plane"
    )


def _ransac_plane(candidates: np.ndarray, config: SceneConfig) -> Plane | None:
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(np.asarray(candidates, dtype=np.float64))
    (a, b, c, d), _ = cloud.segment_plane(
        distance_threshold=config.floor_ransac_distance_meters,
        ransac_n=3,
        num_iterations=config.floor_ransac_iterations,
    )
    normal = np.array([a, b, c])
    length = np.linalg.norm(normal)
    if length == 0:
        return None
    normal, offset = normal / length, d / length

    if normal @ CAMERA_UP < 0:
        normal, offset = -normal, -offset

    # Not level enough, or the camera is too close to it, means this is not the floor.
    if normal @ CAMERA_UP < np.cos(np.radians(config.floor_max_tilt_degrees)):
        return None
    if offset <= config.floor_min_offset_meters:
        return None
    return Plane(normal=normal, offset_meters=float(offset))


def ground_axes(plane: Plane, forward_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Unit axes on the floor: lateral, positive to the walker's right, and forward.

    Forward is the hint projected onto the plane, so "ahead" means where the camera points, not
    where the body happens to face. The hint is the camera's z axis in whatever frame the plane
    is in: (0, 0, 1) in the camera frame, the rotated z axis in the world frame.

    :param plane: The floor.
    :param forward_hint: (3,) direction to project. The camera frame's z axis when None.
    :return: (lateral_axis, forward_axis), each (3,).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    camera_forward = np.array([0.0, 0.0, 1.0]) if forward_hint is None else np.asarray(forward_hint, dtype=np.float64)
    normal = plane.normal
    forward = camera_forward - (camera_forward @ normal) * normal
    if np.linalg.norm(forward) < 1e-3:
        # Looking straight down. Any horizontal direction is as good as another, so take the one
        # the plane's normal is least aligned with.
        fallback = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        forward = fallback - (fallback @ normal) * normal
    forward = forward / np.linalg.norm(forward)
    lateral = np.cross(forward, normal)
    return lateral, forward


def height_above_floor(points: np.ndarray, plane: Plane) -> np.ndarray:
    """Signed height of each point above the plane, in meters."""
    return np.asarray(points) @ plane.normal + plane.offset_meters
