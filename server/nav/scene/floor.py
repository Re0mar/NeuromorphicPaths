"""
Finding the ground in a point cloud, and measuring heights against it.

A Plane here is written so that normal . point + offset is the height above the floor, with the
normal pointing up. That convention is what lets the height band be two comparisons.

Every function that needs to know which way is up takes it as a camera-frame unit vector, because
the camera is not level. A phone held in portrait sends a depth image in the sensor's landscape
orientation, so image-up points sideways and the floor's normal sits 90 degrees from it. Gravity
from the pose is up when a source places the camera in a world. Image-up, CAMERA_UP, is all a
plain video file can offer.
"""

# Third party imports
import numpy as np
import open3d

# Local package imports
from nav.scene.config import SceneConfig
from nav.types import Plane

# Image-up, in camera axes where y points down. The right "up" only when the camera is level.
CAMERA_UP = np.array([0.0, -1.0, 0.0])


def normalize_plane(plane: Plane, up_camera: np.ndarray) -> Plane:
    """
    The same plane with a unit normal pointing toward up, and the offset scaled to match.

    Every plane the scene judges goes through here first, the fit's and a source's alike, so a
    phone that sends its normal pointing down is read the same way a fit that came out upside
    down is.

    :param plane: Any plane with a non-zero normal.
    :param up_camera: Unit vector pointing up, in the camera frame.
    :return: The plane in the convention the rest of the scene assumes.
    :rtype: Plane
    :raises ValueError: When the normal has zero length, which is not a plane at all.
    """
    normal = np.asarray(plane.normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise ValueError("a plane's normal cannot have zero length")
    normal, offset = normal / length, plane.offset_meters / length
    if normal @ up_camera < 0:
        normal, offset = -normal, -offset
    return Plane(normal=normal, offset_meters=float(offset))


def plane_is_a_floor(plane: Plane, config: SceneConfig, up_camera: np.ndarray) -> str | None:
    """
    Judge a normalized plane against where a floor can be, and say which rule it broke.

    Three rules: level enough, not too close to the camera, and not too far below it. The first
    two are the old file's. The third came from the first Pixel walk, where ARCore handed over a
    plane 2.3 m down, a meter below the real floor, and nothing refused it. This is the one place
    the rules live, so the fit and a supplied plane cannot drift apart on what a floor is.

    :param plane: A plane from normalize_plane.
    :param config: The tilt limit, the minimum and the maximum camera height.
    :param up_camera: Unit vector pointing up, in the camera frame. Level is measured against it.
    :return: None when the plane passes, otherwise the rule it broke, with the numbers.
    :rtype: str | None
    """
    tilt_degrees = float(np.degrees(np.arccos(np.clip(plane.normal @ up_camera, -1.0, 1.0))))
    if tilt_degrees > config.floor_max_tilt_degrees:
        return f"leans {tilt_degrees:.1f} deg from up, limit {config.floor_max_tilt_degrees:.1f}"
    if plane.offset_meters <= config.floor_min_offset_meters:
        return f"camera {plane.offset_meters:.2f} m above it, under the minimum {config.floor_min_offset_meters:.2f}"
    if plane.offset_meters > config.floor_max_offset_meters:
        return f"camera {plane.offset_meters:.2f} m above it, over the maximum {config.floor_max_offset_meters:.2f}"
    return None


def fit_floor(points_camera: np.ndarray, previous: Plane | None, config: SceneConfig, up_camera: np.ndarray) -> Plane:
    """
    RANSAC a plane through the points that could plausibly be floor.

    Every candidate goes through plane_is_a_floor. A plane that is not roughly level, that sits
    too close to the camera, or that lies further below it than a held or worn camera can be is
    not the floor, however many points agree with it, and in that case the previous frame's plane
    is better than a wrong one.

    :param points_camera: (N, 3) camera-frame points.
    :param previous: Last frame's plane, returned when this frame has no believable floor.
    :param config: Candidate selection and sanity thresholds.
    :param up_camera: Unit vector pointing up, in the camera frame. Gravity from the pose when
        the source has one, CAMERA_UP otherwise. "Below" and "level" are both measured against it.
    :return: The floor, normal pointing up.
    :rtype: Plane
    :raises ValueError: When no floor is found and there is no previous plane to fall back on.
    """
    # Only points clearly below the camera can be floor. Without this, a wall straight ahead
    # with enough points wins the vote.
    depth_below_camera = -(points_camera @ up_camera)
    candidates = points_camera[depth_below_camera > config.floor_candidate_min_below_camera_meters]
    fitted = None
    if len(candidates) >= config.floor_min_candidate_points:
        fitted = _ransac_plane(candidates, config, up_camera)

    if fitted is not None:
        return fitted
    if previous is not None:
        return previous
    raise ValueError(
        f"no floor found in {len(points_camera)} points, {len(candidates)} below the camera, and no previous plane"
    )


def _ransac_plane(candidates: np.ndarray, config: SceneConfig, up_camera: np.ndarray) -> Plane | None:
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(np.asarray(candidates, dtype=np.float64))
    (a, b, c, d), _ = cloud.segment_plane(
        distance_threshold=config.floor_ransac_distance_meters,
        ransac_n=3,
        num_iterations=config.floor_ransac_iterations,
    )
    normal = np.array([a, b, c])
    if np.linalg.norm(normal) == 0:
        return None
    plane = normalize_plane(Plane(normal=normal, offset_meters=float(d)), up_camera)

    # Not level enough, too close, or too far down means this is not the floor.
    if plane_is_a_floor(plane, config, up_camera) is not None:
        return None
    return plane


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
