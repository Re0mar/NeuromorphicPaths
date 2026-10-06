"""
The floor under the walker, and where a point on it lands in the depth image.

Shared by the depth view, which lays the path's ribbon on the floor, and by the plan view, which
asks which floor cells the camera could see at all. Both go through the same projection, so the
ribbon and the gray cells cannot disagree about where the camera looks.

Numpy only, so the plan view's builder can use it without OpenCV.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.scene.floor import ground_axes
from nav.types import DebugView, Plane

# Nearer than this to the camera's plane, a point projects to absurdly far pixels. Polygons are
# cut here rather than dropped, so a ribbon that runs out under the camera ends at the image edge.
NEAR_PLANE_METERS = 0.05


def floor_point(floor: Plane, forward_meters: float | np.ndarray, lateral_meters: float | np.ndarray) -> np.ndarray:
    """
    Points on the floor, in the camera frame, a distance ahead of and beside the walker's feet.

    The walker stands on the floor directly below the camera.

    :param floor: The floor plane, camera frame.
    :param forward_meters: Distance ahead along the floor. Scalar or array.
    :param lateral_meters: Distance to the right along the floor. Broadcast against forward.
    :return: (..., 3) camera-frame points.
    :rtype: np.ndarray
    """
    lateral_axis, forward_axis = ground_axes(floor)
    foot = -floor.offset_meters * floor.normal
    forward = np.asarray(forward_meters, dtype=np.float64)[..., None]
    lateral = np.asarray(lateral_meters, dtype=np.float64)[..., None]
    return foot + forward * forward_axis + lateral * lateral_axis


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Where camera-frame points land in the image, unscaled, and which are in front of the camera.

    No bounds check. Whether a point outside the image is dropped or clipped is the caller's call.

    :param points_camera: (..., 3).
    :param intrinsics: (3, 3) pinhole matrix of the depth image.
    :return: (x, y, in_front), each (...). x and y are NaN where the point is not in front.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    points = np.asarray(points_camera, dtype=np.float64)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    with np.errstate(invalid="ignore"):
        in_front = np.isfinite(z) & (z > 0.0)
    safe_z = np.where(in_front, z, 1.0)
    column = np.where(in_front, intrinsics[0, 0] * x / safe_z + intrinsics[0, 2], np.nan)
    row = np.where(in_front, intrinsics[1, 1] * y / safe_z + intrinsics[1, 2], np.nan)
    return column, row, in_front


def floor_seen_mask(view: DebugView, times_seconds: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Which cells of the planner's field the camera could see on the floor.

    A cell's floor point is its lateral position at the distance walking speed covers by its time.
    It is seen when it projects in front of the camera and inside the depth image, at the image's
    own size, through the frame's own intrinsics. This is field of view only: floor behind an
    obstacle still counts as seen, which is what the view was decided to show.

    :param view: The frame, its floor and the walking speed.
    :param times_seconds: (steps,) the field's step times.
    :param grid: (cells,) the field's lateral positions.
    :return: (steps, cells) booleans.
    :rtype: np.ndarray
    """
    forward = view.walking_speed_mps * np.asarray(times_seconds, dtype=np.float64)
    forward_grid, lateral_grid = np.meshgrid(forward, np.asarray(grid, dtype=np.float64), indexing="ij")
    column, row, in_front = project_points(floor_point(view.floor, forward_grid, lateral_grid), view.frame.intrinsics)
    height, width = view.frame.depth_meters.shape
    with np.errstate(invalid="ignore"):
        return in_front & (column >= 0.0) & (column < width) & (row >= 0.0) & (row < height)


def clip_to_near_plane(polygon_camera: np.ndarray, near_meters: float = NEAR_PLANE_METERS) -> np.ndarray:
    """
    Cut a camera-frame polygon at a plane just in front of the camera, keeping the part beyond it.

    One plane of Sutherland-Hodgman clipping. A polygon wholly behind the plane comes back empty.

    :param polygon_camera: (vertices, 3), in order around the polygon.
    :param near_meters: Where the plane sits along the camera's z axis.
    :return: (vertices', 3), possibly with zero rows.
    :rtype: np.ndarray
    """
    kept = []
    count = len(polygon_camera)
    for index in range(count):
        current = polygon_camera[index]
        following = polygon_camera[(index + 1) % count]
        current_in = current[2] >= near_meters
        following_in = following[2] >= near_meters
        if current_in:
            kept.append(current)
        if current_in != following_in:
            share = (near_meters - current[2]) / (following[2] - current[2])
            kept.append(current + share * (following - current))
    return np.array(kept, dtype=np.float64).reshape(-1, 3)


def clip_segment_to_near_plane(start: np.ndarray, end: np.ndarray, near_meters: float = NEAR_PLANE_METERS) -> np.ndarray | None:
    """
    The part of a camera-frame line segment beyond a plane just in front of the camera.

    :param start: (3,).
    :param end: (3,).
    :param near_meters: Where the plane sits along the camera's z axis.
    :return: (2, 3), start then end, or None when the whole segment is behind the plane.
    :rtype: np.ndarray | None
    """
    start_in = start[2] >= near_meters
    end_in = end[2] >= near_meters
    if not start_in and not end_in:
        return None
    if start_in and end_in:
        return np.array([start, end], dtype=np.float64)
    crossing = start + (near_meters - start[2]) / (end[2] - start[2]) * (end - start)
    return np.array([start, crossing] if start_in else [crossing, end], dtype=np.float64)
