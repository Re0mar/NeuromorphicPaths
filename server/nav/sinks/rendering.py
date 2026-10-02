"""
The pictures the debug sinks draw: the arrow, the surprise field, and the depth view.

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
from nav.scene.floor import ground_axes
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
COLOR_GROUP = (0, 200, 255)
COLOR_WALL = (255, 0, 255)
COLOR_INVALID_DEPTH = (40, 40, 40)
FIELD_INSET_SCALE = 3
# The depth view is scaled up by a whole number so the Pixel's 160 by 90 is legible at 640 by
# 360 and a 640-wide estimated frame stays as it is. A fixed factor would make the estimator's
# frames four times too wide.
DEPTH_VIEW_TARGET_WIDTH = 640
DEPTH_PERCENTILES = (2.0, 98.0)
GROUP_RING_MIN_RADIUS = 5
GROUP_RING_PIXELS_PER_METER = 8
GROUP_RING_MAX_RADIUS = 40


def render_arrow(heading_radians: float, path: PlannedPath) -> np.ndarray:
    """
    The arrow canvas: heading as an arrow from the bottom centre, red on alarm, a line of text.

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


def render_depth_view(view: DebugView, path: PlannedPath) -> np.ndarray:
    """
    The depth image the planner saw, with the obstacle groups and the chosen path drawn on it.

    Valid depth is mapped from the frame's own 2nd to 98th percentile onto a colormap, so a
    close scene and a far one both use the full range. Invalid pixels are dark grey. Each group's
    nearest point is a ring at its projected pixel, sized by clearance, magenta for a wall. The
    path is a polyline on the floor: each step's lateral offset at the forward distance walking
    speed covers by then. Points behind the camera or outside the image are skipped.

    :param view: The frame, the obstacles, the floor and where it came from.
    :param path: The path the planner chose for that frame.
    :return: A BGR image, the depth image scaled up by a whole number.
    :rtype: np.ndarray
    """
    depth = np.asarray(view.frame.depth_meters, dtype=np.float32)
    rows, columns = depth.shape
    scale = max(1, DEPTH_VIEW_TARGET_WIDTH // columns)

    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0.0)
    image = np.full((rows, columns, 3), COLOR_INVALID_DEPTH, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], DEPTH_PERCENTILES)
        span = max(float(high - low), 1e-6)
        # Invalid pixels are NaN or zero here and are masked out below, but a NaN cast to uint8
        # is undefined, so they are zeroed before the cast rather than after.
        normalized = np.where(valid, np.clip((depth - low) / span, 0.0, 1.0), 0.0)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[valid] = colored[valid]
    image = cv2.resize(image, (columns * scale, rows * scale), interpolation=cv2.INTER_NEAREST)

    intrinsics = view.frame.intrinsics
    for point in view.obstacles.points:
        pixel = _project(point.camera_point, intrinsics, scale, image.shape)
        if pixel is None:
            continue
        radius = int(np.clip(GROUP_RING_MIN_RADIUS + GROUP_RING_PIXELS_PER_METER * point.clearance_meters, GROUP_RING_MIN_RADIUS, GROUP_RING_MAX_RADIUS))
        cv2.circle(image, pixel, radius, COLOR_WALL if point.is_wall else COLOR_GROUP, 2)

    _draw_path_on_floor(image, view, path, scale)

    nearest = min((point.clearance_meters for point in view.obstacles.points), default=None)
    nearest_text = "nearest -" if nearest is None else f"nearest {nearest:.2f} m"
    text = f"{view.obstacles.groups_in_view} groups  {nearest_text}  floor {view.floor_source.value}{'  ALARM' if path.alarm else ''}"
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


def _project(point_camera: np.ndarray, intrinsics: np.ndarray, scale: int, image_shape: tuple) -> tuple[int, int] | None:
    """The scaled pixel a camera-frame point lands on, or None when it is behind the camera or outside the image."""
    x, y, z = (float(value) for value in point_camera)
    if not np.isfinite(z) or z <= 0.0:
        return None
    column = (intrinsics[0, 0] * x / z + intrinsics[0, 2]) * scale
    row = (intrinsics[1, 1] * y / z + intrinsics[1, 2]) * scale
    if not (0 <= column < image_shape[1] and 0 <= row < image_shape[0]):
        return None
    return int(column), int(row)


def _draw_path_on_floor(image: np.ndarray, view: DebugView, path: PlannedPath, scale: int) -> None:
    # The walker stands on the floor directly below the camera. Each step is that point moved
    # forward by what walking speed covers in the step's time, and sideways by its offset.
    floor = view.floor
    lateral_axis, forward_axis = ground_axes(floor)
    foot = -floor.offset_meters * floor.normal
    pixels = []
    for time_seconds, offset_meters in zip(path.times_seconds, path.lateral_offsets_meters, strict=True):
        point = foot + forward_axis * (view.walking_speed_mps * float(time_seconds)) + lateral_axis * float(offset_meters)
        pixel = _project(point, view.frame.intrinsics, scale, image.shape)
        if pixel is not None:
            pixels.append(pixel)
    if len(pixels) >= 2:
        cv2.polylines(image, [np.array(pixels, dtype=np.int32).reshape(-1, 1, 2)], False, COLOR_OUTLINE, 4)
        cv2.polylines(image, [np.array(pixels, dtype=np.int32).reshape(-1, 1, 2)], False, COLOR_PATH, 2)
