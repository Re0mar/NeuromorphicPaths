"""
The pictures the debug sinks draw: the arrow, the surprise field, the depth view and the risk view.

One module so the OpenCV window and the browser show the same thing. The depth view is the one a
person tuning the planner needs: the depth image the planner saw, each obstacle group's nearest
point projected back onto it, and the chosen path laid on the floor, all from the data the scene
and the planner produced rather than from geometry re-derived here.

Draws on blank canvases, never on a camera frame. A sink receives paths and a debug view, which is
what keeps it independent of which source is running.
"""

# Standard library imports
import logging

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.scene.config import SceneConfig
from nav.sinks.floor_geometry import (
    NEAR_PLANE_METERS,
    clip_segment_to_near_plane,
    clip_to_near_plane,
    field_cells_at_pixels,
    floor_point,
    group_rings,
    picture_scale,
    project_points,
    upright_quarter_turns,
)
from nav.sinks.path_style import BORDER_OPACITY, GROUP_RGB, WALL_RGB, path_color_rgb, path_fill_opacity
from nav.types import DebugView, PlannedPath

log = logging.getLogger(__name__)

CANVAS_HEIGHT = 480
CANVAS_WIDTH = 640
ARROW_LENGTH_PIXELS = 110
# BGR, which is what OpenCV draws in.
COLOR_CLEAR = (0, 255, 0)
COLOR_ALARM = (0, 0, 255)
COLOR_OUTLINE = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_PATH = (255, 255, 255)
COLOR_GROUP = GROUP_RGB[::-1]
COLOR_WALL = WALL_RGB[::-1]
# A dim brown rather than a gray, because the depth view is gray now and far depth is near black.
# A separate hue means a missing reading never looks like far depth.
COLOR_INVALID_DEPTH = (20, 40, 70)
FIELD_INSET_SCALE = 3
DEPTH_PERCENTILES = (2.0, 98.0)


def render_arrow(heading_radians: float, path: PlannedPath) -> np.ndarray:
    """
    The arrow canvas: heading as an arrow from the bottom center, red on alarm, a line of text.

    :param heading_radians: Where to point. Positive is right.
    :param path: For the alarm and the cost text.
    :return: A BGR canvas.
    :rtype: np.ndarray
    """
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 24, dtype=np.uint8)
    center = (CANVAS_WIDTH // 2, CANVAS_HEIGHT - 90)
    tip = (
        int(center[0] + ARROW_LENGTH_PIXELS * np.sin(heading_radians)),
        int(center[1] - ARROW_LENGTH_PIXELS * np.cos(heading_radians)),
    )
    color = COLOR_ALARM if path.alarm else COLOR_CLEAR
    cv2.arrowedLine(canvas, center, tip, COLOR_OUTLINE, 18, tipLength=0.4)
    cv2.arrowedLine(canvas, center, tip, color, 10, tipLength=0.4)

    text = f"{np.degrees(heading_radians):+.0f} deg  cost {path.cumulative_cost_bits:.2f} bits  {'ALARM' if path.alarm else 'clear'}"
    cv2.putText(canvas, text, (10, CANVAS_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    return canvas


def render_field(field: np.ndarray, grid: np.ndarray, path: PlannedPath) -> np.ndarray:
    """
    The surprise field as a color image with the chosen path drawn over it.

    Rows are future steps, now at the bottom. Columns are lateral positions, left to right.

    :param field: (steps, cells).
    :param grid: (cells,) lateral position of each column.
    :param path: Its lateral offsets are drawn as a line, one point per step.
    :return: A BGR image, enlarged for legibility.
    :rtype: np.ndarray
    """
    steps, cells = field.shape
    if len(grid) != cells:
        raise ValueError(f"field has {cells} columns and the grid has {len(grid)} entries")

    # Clip at the 98th percentile so one capped point does not turn everything else black.
    clipped = np.clip(field, 0.0, max(float(np.percentile(field, 98)), 1e-9))
    normalized = cv2.normalize(clipped.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = np.flipud(cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)).copy()

    offsets = path.lateral_offsets_meters
    if len(offsets) == steps:
        # Nearest column for each step's offset, by the grid itself rather than by any inferred
        # spacing, so a straight path draws just as correctly as a swerving one.
        columns = np.abs(grid[None, :] - offsets[:, None]).argmin(axis=1)
        rows = (steps - 1) - np.arange(steps)
        points = np.column_stack((columns, rows)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, COLOR_PATH, 1)

    return cv2.resize(image, None, fx=FIELD_INSET_SCALE, fy=FIELD_INSET_SCALE, interpolation=cv2.INTER_NEAREST)


def render_depth_view(view: DebugView, path: PlannedPath, *, rings: bool = True) -> np.ndarray:
    """
    The depth image the planner saw, with the obstacle groups and the chosen path drawn on it.

    Valid depth is mapped from the frame's own 2nd to 98th percentile onto gray, near bright and
    far dark, so a close scene and a far one both use the full range and the path's colors stand
    out against any depth. Invalid pixels are a dim brown. Each group's nearest point is a ring at
    its projected pixel, sized by clearance, magenta for a wall. The path is a ribbon on the floor
    as wide as the body: each step's lateral offset at the forward distance walking speed covers by
    then. Its fill fades to nothing at the far end, and its borders do not fade.

    The picture is turned by a quarter turn at a time so the floor is at the bottom. The Pixel sends
    its depth image the way the sensor reads it, sideways to how the phone is held, and without the
    turn "ahead" runs across the picture.

    :param view: The frame, the obstacles, the floor and where it came from.
    :param path: The path the planner chose for that frame.
    :param rings: False leaves the group rings off, for a display that draws group_rings itself.
    :return: A BGR image, the depth image scaled up by a whole number and turned upright.
    :rtype: np.ndarray
    """
    return _draw_over_and_turn(_gray_depth(view), view, path, ribbon_fill=True, rings=rings, label="")


def render_risk_view(view: DebugView, path: PlannedPath, field: np.ndarray, grid: np.ndarray, scene: SceneConfig) -> np.ndarray:
    """
    The depth view recolored by the planner's field: what each spot the camera sees costs the planner.

    Each pixel stands over one cell of the field, at the step the walker reaches it, and takes that
    cell's cost on the TURBO scale, cool for cheap and hot for costly. The scale is clipped at the
    field's 98th percentile, as the page's view from above is, so the two read the same. Pixels the
    field does not reach keep the depth view's gray: beyond the plan, behind the feet, off to the
    side, and above head height, where the walker passes under. The path keeps only its borders, so
    its blue to red stays readable over the colors. No group rings, since only the page shows this
    view and it draws group_rings over either picture. The text is the depth view's.

    :param view: The frame, the obstacles, the floor and where it came from.
    :param path: The path the planner chose. Its step times are the field's rows.
    :param field: (steps, cells) the field it planned through.
    :param grid: (cells,) the lateral position of each field column.
    :param scene: The usable depth range and the head height.
    :return: A BGR image the depth view's size.
    :rtype: np.ndarray
    :raises ValueError: When the field is not one row per path step by one column per grid cell.
    """
    expected = (len(path.times_seconds), len(grid))
    if field.shape != expected:
        raise ValueError(f"field is {field.shape}, the path and grid need {expected}")
    image = _gray_depth(view)
    step, cell, covered = field_cells_at_pixels(view, path.times_seconds, grid, scene)
    if covered.any():
        ceiling = max(float(np.percentile(field, 98)), 1e-9)
        share = np.clip(field[step[covered], cell[covered]] / ceiling, 0.0, 1.0)
        levels = np.round(share * 255).astype(np.uint8).reshape(-1, 1)
        image[covered] = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return _draw_over_and_turn(image, view, path, ribbon_fill=False, rings=False, label="  risk")


def _gray_depth(view: DebugView) -> np.ndarray:
    """The depth image at its own size, valid depth in gray with near bright, a missing reading dim brown."""
    depth = np.asarray(view.frame.depth_meters, dtype=np.float32)
    rows, columns = depth.shape
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    image = np.full((rows, columns, 3), COLOR_INVALID_DEPTH, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], DEPTH_PERCENTILES)
        span = max(float(high - low), 1e-6)
        # Invalid pixels are NaN or zero here and are masked out below, but a NaN cast to uint8
        # is undefined, so they are zeroed before the cast rather than after.
        normalized = np.where(valid, np.clip((depth - low) / span, 0.0, 1.0), 0.0)
        # Near is bright. Gray rather than a colormap, so the path's blue to red is never lost in it.
        gray = ((1.0 - normalized) * 255).astype(np.uint8)
        image[valid] = np.repeat(gray[valid][:, None], 3, axis=1)
    return image


def _draw_over_and_turn(image: np.ndarray, view: DebugView, path: PlannedPath, *, ribbon_fill: bool, rings: bool, label: str) -> np.ndarray:
    """
    Scale a depth-sized picture up, draw the path and the groups on it, turn it upright and caption it.

    :param image: BGR, the depth image's size.
    :param ribbon_fill: False draws the path's borders alone.
    :param rings: False leaves the group rings off.
    :param label: Added to the end of the caption.
    """
    rows, columns = image.shape[:2]
    scale = picture_scale(columns)
    image = cv2.resize(image, (columns * scale, rows * scale), interpolation=cv2.INTER_NEAREST)
    _draw_path_ribbon(image, view, path, scale, fill=ribbon_fill)

    # The ribbon is drawn where the sensor put it. Turned now, so the text below reads upright.
    image = np.ascontiguousarray(np.rot90(image, upright_quarter_turns(view.floor)))

    # Rings after the ribbon, so a group on the path stays visible on top of it. They are placed in
    # the turned picture, the same places the page draws them.
    for ring in group_rings(view) if rings else []:
        cv2.circle(image, (ring.column_pixels, ring.row_pixels), ring.radius_pixels, COLOR_WALL if ring.is_wall else COLOR_GROUP, 2)

    nearest = min((point.clearance_meters for point in view.obstacles.points), default=None)
    nearest_text = "nearest -" if nearest is None else f"nearest {nearest:.2f} m"
    text = f"{view.obstacles.groups_in_view} groups  {nearest_text}  floor {view.floor_source.value}{'  ALARM' if path.alarm else ''}{label}"
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_OUTLINE, 3)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    return image


def encode_png(image: np.ndarray) -> bytes:
    """
    Encode a BGR image as PNG bytes.

    :param image: What to encode.
    :return: The PNG file's bytes.
    :rtype: bytes
    :raises ValueError: When OpenCV cannot encode it, which an empty image provokes.
    """
    if image.size == 0:
        raise ValueError("cannot encode an empty image as PNG")
    try:
        encoded, buffer = cv2.imencode(".png", image)
    except cv2.error as opencv_error:
        raise ValueError(f"OpenCV could not encode the image as PNG: {opencv_error}") from opencv_error
    if not encoded:
        raise ValueError("OpenCV could not encode the image as PNG")
    return buffer.tobytes()


def _scaled_pixels(points_camera: np.ndarray, intrinsics: np.ndarray, scale: int) -> np.ndarray:
    """Pixels for points already known to be in front of the camera, shaped for cv2.fillPoly and cv2.polylines."""
    column, row, _ = project_points(points_camera, intrinsics)
    return np.round(np.column_stack((column * scale, row * scale))).astype(np.int32).reshape(-1, 1, 2)


def _draw_path_ribbon(image: np.ndarray, view: DebugView, path: PlannedPath, scale: int, *, fill: bool = True) -> None:
    """
    The path as a ribbon on the floor, the body's width, drawn onto the image in place.

    With fill off only the borders are drawn. The fill is one quadrilateral per step, faded linearly by the time at its far end, so the last
    one is clear and the ribbon is always seen running out. Its opacity is also scaled by how much
    the scene shaped the plan. The two borders are drawn at one opacity along their whole length,
    with a dark outline under each, so the direction stays readable after the fill has gone.

    Polygons are cut at the camera's near plane rather than dropped, and OpenCV clips them at the
    image edge, so a ribbon running out under the camera or off the side leaves no gap.
    """
    times = path.times_seconds
    if len(times) < 2:
        return
    forward = view.walking_speed_mps * times
    # Edges offset along the floor's lateral axis. At the sidestep angles the planner allows, that
    # is within a few centimeters of offsetting perpendicular to the path.
    left = floor_point(view.floor, forward, path.lateral_offsets_meters - view.body_half_width_meters)
    right = floor_point(view.floor, forward, path.lateral_offsets_meters + view.body_half_width_meters)
    intrinsics = view.frame.intrinsics
    color = np.array(path_color_rgb(path.avoidance_surprise_bits, view.path_red_from_bits)[::-1], dtype=np.float32)  # BGR from here on.
    fill_opacity = path_fill_opacity(path.scene_information_bits)
    end_time = float(times[-1])

    # Every edge point projected once. Only a piece that reaches inside the near plane is clipped
    # and projected again, which on a level walk is the one or two pieces under the camera.
    left_pixels, left_beyond = _edge_pixels(left, intrinsics, scale)
    right_pixels, right_beyond = _edge_pixels(right, intrinsics, scale)

    # One alpha mask for the whole fill, blended once. Segments that share an edge overwrite rather
    # than stack, so no stripe shows where two of them meet.
    fill_alpha = np.zeros(image.shape[:2], dtype=np.float32)
    for step in range(len(times) - 1 if fill else 0):
        corners = [left_beyond[step], left_beyond[step + 1], right_beyond[step + 1], right_beyond[step]]
        if all(corners):
            polygon = np.array([left_pixels[step], left_pixels[step + 1], right_pixels[step + 1], right_pixels[step]]).reshape(-1, 1, 2)
        else:
            quadrilateral = clip_to_near_plane(np.array([left[step], left[step + 1], right[step + 1], right[step]]))
            if len(quadrilateral) < 3:
                continue
            polygon = _scaled_pixels(quadrilateral, intrinsics, scale)
        fade = 1.0 - float(times[step + 1]) / end_time if end_time > 0.0 else 1.0
        cv2.fillPoly(fill_alpha, [polygon], float(fill_opacity * fade))
    _blend(image, fill_alpha, color)

    # The borders on a layer of their own, at a single opacity. Outlines first, then the colored
    # lines over them, so one segment's outline never covers the next segment's color.
    segments = []
    for points, pixels, beyond in ((left, left_pixels, left_beyond), (right, right_pixels, right_beyond)):
        for step in range(len(times) - 1):
            if beyond[step] and beyond[step + 1]:
                segments.append(pixels[step : step + 2].reshape(-1, 1, 2))
                continue
            clipped = clip_segment_to_near_plane(points[step], points[step + 1])
            if clipped is not None:
                segments.append(_scaled_pixels(clipped, intrinsics, scale))
    border_layer = np.zeros_like(image)
    border_alpha = np.zeros(image.shape[:2], dtype=np.float32)
    for line_color, thickness in ((COLOR_OUTLINE, 4), (tuple(int(channel) for channel in color), 2)):
        cv2.polylines(border_layer, segments, False, line_color, thickness)
        cv2.polylines(border_alpha, segments, False, BORDER_OPACITY, thickness)
    _blend(image, border_alpha, border_layer)


def _edge_pixels(points_camera: np.ndarray, intrinsics: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled pixels for a run of points, and which lie beyond the near plane.

    A pixel is only meaningful where its point lies beyond the plane. Elsewhere it is a placeholder
    the caller must not draw, and the caller clips that piece instead.

    :return: (pixels (n, 2) int32, beyond (n,) bool).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    beyond = points_camera[:, 2] >= NEAR_PLANE_METERS
    column, row, _ = project_points(points_camera, intrinsics)
    pixels = np.column_stack((np.nan_to_num(column) * scale, np.nan_to_num(row) * scale))
    return np.round(pixels).astype(np.int32), beyond


def _blend(image: np.ndarray, alpha: np.ndarray, paint: np.ndarray) -> None:
    """
    Paint over the image at a per-pixel opacity, in place, touching only the box the paint covers.

    :param image: The BGR image to paint on.
    :param alpha: (rows, columns) opacity, zero where nothing is painted.
    :param paint: One BGR color (3,), or a whole BGR layer the image's size.
    """
    covered_rows = np.flatnonzero(alpha.any(axis=1))
    if covered_rows.size == 0:
        return
    covered_columns = np.flatnonzero(alpha.any(axis=0))
    window = (slice(covered_rows[0], covered_rows[-1] + 1), slice(covered_columns[0], covered_columns[-1] + 1))
    weight = alpha[window][..., None]
    region = image[window].astype(np.float32)
    paint_region = paint[window].astype(np.float32) if paint.ndim == 3 else paint
    # Adding a half and truncating rounds, without a separate rounding pass over the window.
    image[window] = (region + (paint_region - region) * weight + 0.5).astype(np.uint8)
