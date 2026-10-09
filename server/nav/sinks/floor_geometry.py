"""
The floor under the walker, and where a point on it lands in the depth image.

Shared by the depth view, which lays the path's ribbon on the floor, and by the plan view, which
asks which floor cells the camera could see at all, and which of those it actually saw floor at
rather than something standing in front of it. All of it goes through the same projection, so the
ribbon and the gray cells cannot disagree about where the camera looks. The risk view goes the
other way, from each pixel down to the field cell it stands over. The group rings and the picture's
quarter turns live here too, so the page can be told where the rings go without OpenCV.

No OpenCV, so the plan view's builder can use this module without OpenCV. It does load open3d, through the
scene's unprojection, which the scene has always loaded in the same process.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.floor import ground_axes, height_above_floor
from nav.scene.unproject import unproject_depth
from nav.types import DebugView, DepthFrame, Plane

# Nearer than this to the camera's plane, a point projects to absurdly far pixels. Polygons are
# cut here rather than dropped, so a ribbon that runs out under the camera ends at the image edge.
NEAR_PLANE_METERS = 0.05

# How many scatters above the floor a reading has to stand before the floor behind it counts as
# hidden. Set on recorded walks: at three, the Pixel's depth grayed 5.8 % of open floor in a
# classroom, mostly readings 8 to 15 cm up just in front of furniture. Four brought that to 2.5 %,
# and the Neon in a tiered lecture room from 3.1 % to 1.9 %, while still graying 95 to 98 % of what
# three did.
HIDDEN_SCATTER_MULTIPLE = 4.0

# An exact floor, which only synthetic depth gives, measures a scatter of 0, and then rounding alone
# would hide every cell. Real depth scatters well over a centimeter, so this only bites on perfect data.
MIN_FLOOR_SCATTER_METERS = 0.01


# The depth view is scaled up by a whole number so the Pixel's 160 by 90 is legible at 640 by
# 360 and a 640-wide estimated frame stays as it is. A fixed factor would make the estimator's
# frames four times too wide.
DEPTH_VIEW_TARGET_WIDTH = 640
GROUP_RING_MIN_RADIUS = 5
GROUP_RING_PIXELS_PER_METER = 8
GROUP_RING_MAX_RADIUS = 40
# Below this share of the floor normal lying in the image plane, the camera is looking at the floor
# and the picture has no up to turn toward. A camera pitched 72 degrees down sits right at it.
UPRIGHT_MIN_IMAGE_COMPONENT = 0.3


@dataclass(frozen=True)
class GroupRing:
    """Where one group's ring goes in the depth picture: scaled, turned upright, in whole pixels."""

    column_pixels: int
    row_pixels: int
    radius_pixels: int
    is_wall: bool


def picture_scale(depth_columns: int) -> int:
    """The whole number the depth image is scaled up by in the depth and risk pictures."""
    return max(1, DEPTH_VIEW_TARGET_WIDTH // depth_columns)


def upright_quarter_turns(floor: Plane) -> int:
    """
    How many counter-clockwise quarter turns bring the floor's down to the bottom of the picture.

    Reads the floor the scene handed over, whose normal already points up: against gravity when the
    pose knows gravity, against the image's own up otherwise. It is the plane the planner used, so
    the picture turns with the plan rather than with a second reading of the pose.

    :param floor: The floor, camera frame, normal pointing up.
    :return: 0 to 3, for np.rot90. 0 when the camera looks at the floor and the picture has no up.
    :rtype: int
    """
    normal = np.asarray(floor.normal, dtype=np.float64)
    # Down in the image, x right and y down.
    down_x, down_y = -normal[0], -normal[1]
    if np.hypot(down_x, down_y) < UPRIGHT_MIN_IMAGE_COMPONENT * np.linalg.norm(normal):
        return 0
    if abs(down_y) >= abs(down_x):
        return 0 if down_y > 0 else 2
    # A counter-clockwise turn carries the left edge to the bottom, three carry the right edge there.
    return 1 if down_x < 0 else 3


def group_rings(view: DebugView) -> list[GroupRing]:
    """
    Each obstacle group's ring in the depth picture, sized by its clearance.

    The nearest point of each group, projected, scaled by picture_scale and carried through the same
    quarter turns as the picture, so a ring lands where the group is in the picture as sent. Groups
    behind the camera or outside the image have no ring.

    :param view: The frame, its floor and the groups.
    :return: One ring per group in view, in the groups' order.
    :rtype: list[GroupRing]
    """
    rows, columns = view.frame.depth_meters.shape
    scale = picture_scale(columns)
    turns = upright_quarter_turns(view.floor)
    rings = []
    for point in view.obstacles.points:
        column, row, in_front = project_points(np.asarray(point.camera_point, dtype=np.float64), view.frame.intrinsics)
        if not bool(in_front):
            continue
        column, row = float(column) * scale, float(row) * scale
        if not (0 <= column < columns * scale and 0 <= row < rows * scale):
            continue
        turned_column, turned_row = int(column), int(row)
        width = columns * scale
        height = rows * scale
        for _ in range(turns):
            # np.rot90 puts source (row, column) at (width - 1 - column, row), and swaps the sides.
            turned_column, turned_row = turned_row, width - 1 - turned_column
            width, height = height, width
        radius = int(np.clip(GROUP_RING_MIN_RADIUS + GROUP_RING_PIXELS_PER_METER * point.clearance_meters, GROUP_RING_MIN_RADIUS, GROUP_RING_MAX_RADIUS))
        rings.append(GroupRing(turned_column, turned_row, radius, bool(point.is_wall)))
    return rings


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
    obstacle still counts as seen here. floor_hidden_mask says which of these the camera actually
    saw floor at.

    :param view: The frame, its floor and the walking speed.
    :param times_seconds: (steps,) the field's step times.
    :param grid: (cells,) the field's lateral positions.
    :return: (steps, cells) booleans.
    :rtype: np.ndarray
    """
    _, _, seen = _cell_pixels(view, times_seconds, grid)
    return seen


def floor_scatter_meters(frame: DepthFrame, floor: Plane, scene: SceneConfig) -> float:
    """
    How far this frame's floor readings spread around the floor, as an RMS height.

    The readings counted are the ones the floor fit would call floor: within the fit's inlier
    distance of the plane. The floor can be supplied, fitted or carried over, and the scatter is
    measured the same way for all three, so it never depends on a fit having run this frame.
    Display only. The planner never reads it.

    With fewer readings near the floor than the fit needs to believe one, the camera saw almost no
    floor, and the inlier distance is returned, the most the scatter could be.

    :param frame: The depth and the intrinsics.
    :param floor: The floor the scene used for this frame, camera frame.
    :param scene: The run's scene config: the inlier distance, the usable depth range and the stride.
    :return: Meters, at least MIN_FLOOR_SCATTER_METERS and at most the inlier distance.
    :rtype: float
    """
    points = unproject_depth(frame.depth_meters, frame.intrinsics, scene.depth_stride, scene)
    heights = height_above_floor(points, floor)
    # Strictly under, the same inlier test the fit applies.
    inlier_heights = heights[np.abs(heights) < scene.floor_ransac_distance_meters]
    if len(inlier_heights) < scene.floor_min_candidate_points:
        return scene.floor_ransac_distance_meters
    return max(float(np.sqrt(np.mean(inlier_heights**2))), MIN_FLOOR_SCATTER_METERS)


def floor_hidden_mask(view: DebugView, times_seconds: np.ndarray, grid: np.ndarray, scene: SceneConfig) -> np.ndarray:
    """
    Which seen cells the camera did not actually see floor at.

    A seen cell is hidden when the depth at its pixel is unusable, so nobody checked that floor, or
    when that reading stands more than HIDDEN_SCATTER_MULTIPLE scatters above the floor, so
    something stands between the camera and the floor there. A reading beyond the floor, a hole or a
    drop, never hides a cell. Display only.

    :param view: The frame, its floor and the walking speed.
    :param times_seconds: (steps,) the field's step times.
    :param grid: (cells,) the field's lateral positions.
    :param scene: The run's scene config, for the usable depth range and the scatter.
    :return: (steps, cells) booleans, never true where floor_seen_mask is false.
    :rtype: np.ndarray
    """
    column, row, seen = _cell_pixels(view, times_seconds, grid)
    hidden = np.zeros_like(seen)
    if not seen.any():
        return hidden

    depth = view.frame.depth_meters
    height, width = depth.shape
    # Seen means inside the image, so the clip only catches rounding up off the last pixel.
    pixel_columns = np.clip(np.rint(column[seen]), 0, width - 1)
    pixel_rows = np.clip(np.rint(row[seen]), 0, height - 1)
    readings = depth[pixel_rows.astype(int), pixel_columns.astype(int)].astype(np.float64)
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(readings) & (readings >= scene.min_depth_meters) & (readings <= scene.max_depth_meters)

    # The reading goes back on the ray through the pixel it was read from, as unproject_depth puts it.
    # On the cell's own ray an exact floor would read up to half a pixel's change in floor depth off
    # the floor, which near the horizon is more than the threshold.
    intrinsics = view.frame.intrinsics
    reading_points = np.column_stack(
        (
            (pixel_columns - intrinsics[0, 2]) * readings / intrinsics[0, 0],
            (pixel_rows - intrinsics[1, 2]) * readings / intrinsics[1, 1],
            readings,
        )
    )
    scatter = floor_scatter_meters(view.frame, view.floor, scene)
    with np.errstate(invalid="ignore"):
        stands_above_floor = height_above_floor(reading_points, view.floor) > HIDDEN_SCATTER_MULTIPLE * scatter
    hidden[seen] = ~usable | stands_above_floor
    return hidden


def field_cells_at_pixels(view: DebugView, times_seconds: np.ndarray, grid: np.ndarray, scene: SceneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Which cell of the planner's field each depth pixel stands over.

    The reading goes back to its point, then straight down onto the floor. Its distance ahead picks
    the step the walker reaches it at, at walking speed, and its distance to the side picks the
    nearest grid column. The reverse of floor_seen_mask, which goes from cells to pixels.
    A pixel is uncovered when its reading is outside the scene's usable depth range, when it stands
    above head height so the walker passes under it, or when it lands outside the field. Display only.

    :param view: The frame, its floor and the walking speed.
    :param times_seconds: (steps,) the field's step times, ascending.
    :param grid: (cells,) the field's lateral positions, ascending.
    :param scene: The usable depth range and the head height.
    :return: (step, cell, covered), each the depth image's shape. step and cell are 0 where uncovered.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    depth = np.asarray(view.frame.depth_meters, dtype=np.float64)
    rows, columns = depth.shape
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(depth) & (depth >= scene.min_depth_meters) & (depth <= scene.max_depth_meters)
    readings = np.where(usable, depth, 0.0)
    intrinsics = view.frame.intrinsics
    pixel_rows, pixel_columns = np.indices((rows, columns), dtype=np.float64)
    # The same unprojection floor_hidden_mask and the scene use. Depth here is distance along the camera's z axis.
    points = np.stack(
        (
            (pixel_columns - intrinsics[0, 2]) * readings / intrinsics[0, 0],
            (pixel_rows - intrinsics[1, 2]) * readings / intrinsics[1, 1],
            readings,
        ),
        axis=-1,
    )
    lateral_axis, forward_axis = ground_axes(view.floor)
    from_foot = points - (-view.floor.offset_meters * view.floor.normal)
    step, step_inside = _nearest_position(view.walking_speed_mps * np.asarray(times_seconds, dtype=np.float64), from_foot @ forward_axis)
    cell, cell_inside = _nearest_position(np.asarray(grid, dtype=np.float64), from_foot @ lateral_axis)
    covered = usable & (height_above_floor(points, view.floor) <= scene.head_height_meters) & step_inside & cell_inside
    return np.where(covered, step, 0), np.where(covered, cell, 0), covered


def _nearest_position(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The index of the nearest position for each value, and whether the value lies within half a
    # spacing of the ends. Fewer than two positions have no spacing to judge by, so nothing is inside.
    if len(positions) < 2:
        return np.zeros(values.shape, dtype=int), np.zeros(values.shape, dtype=bool)
    upper = np.clip(np.searchsorted(positions, values), 1, len(positions) - 1)
    lower = upper - 1
    nearest = np.where(values - positions[lower] <= positions[upper] - values, lower, upper)
    first_half = (positions[1] - positions[0]) / 2.0
    last_half = (positions[-1] - positions[-2]) / 2.0
    inside = (values >= positions[0] - first_half) & (values <= positions[-1] + last_half)
    return nearest, inside


def _cell_pixels(view: DebugView, times_seconds: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Where each cell's floor point lands in the depth image, and whether that is inside it. One
    # projection for both masks, so a cell can never be hidden without being seen.
    forward = view.walking_speed_mps * np.asarray(times_seconds, dtype=np.float64)
    forward_grid, lateral_grid = np.meshgrid(forward, np.asarray(grid, dtype=np.float64), indexing="ij")
    column, row, in_front = project_points(floor_point(view.floor, forward_grid, lateral_grid), view.frame.intrinsics)
    height, width = view.frame.depth_meters.shape
    with np.errstate(invalid="ignore"):
        seen = in_front & (column >= 0.0) & (column < width) & (row >= 0.0) & (row < height)
    return column, row, seen


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
