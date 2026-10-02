"""
An OpenCV window showing the arrow, the alarm, the cost, and the surprise field.

The one sink that implements DebugSink, so the field, a grid the size of the planner's horizon,
only ever goes to a window on the laptop and never to a phone or a browser.

Draws on a blank canvas rather than on the camera frame. A sink receives paths and nothing else,
which is what keeps it independent of which source is running. The old overlay drew on the video,
and the video is a source concern.
"""

# Standard library imports
import logging

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.sinks.config import DebugWindowConfig
from nav.types import PlannedPath

log = logging.getLogger(__name__)

WINDOW_NAME = "nav"
CANVAS_HEIGHT = 480
CANVAS_WIDTH = 640
ARROW_LENGTH_PIXELS = 110
# BGR, which is what OpenCV draws in.
COLOR_CLEAR = (0, 255, 0)
COLOR_ALARM = (0, 0, 255)
COLOR_OUTLINE = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_PATH = (255, 255, 255)
FIELD_INSET_SCALE = 3


class DebugWindowSink:
    """Shows each path as it arrives. Opens the window on the first publish, not on construction."""

    def __init__(self, config: DebugWindowConfig) -> None:
        self._config = config
        self._open = False
        self._displayed_heading: float | None = None
        self._latest_inset: np.ndarray | None = None

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray) -> None:
        """
        Draw the path with the surprise field inset.

        :param path: The path to show.
        :param field: (steps, cells) the planner's field, goal term included.
        :param grid: (cells,) the lateral position of each field column, in meters.
        """
        self._latest_inset = render_field(field, grid, path) if self._config.show_field else None
        self.publish(path)

    def publish(self, path: PlannedPath) -> None:
        """Draw the arrow, the alarm color and the cost. The inset only if publish_debug supplied one."""
        heading = self._display_heading(path.first_heading_radians)
        canvas = render_arrow(heading, path)

        if self._latest_inset is not None:
            height, width = self._latest_inset.shape[:2]
            canvas[10 : 10 + height, CANVAS_WIDTH - 10 - width : CANVAS_WIDTH - 10] = self._latest_inset

        cv2.imshow(WINDOW_NAME, canvas)
        self._open = True
        # One millisecond is what lets the window actually repaint. Without it nothing shows.
        cv2.waitKey(1)

    def _display_heading(self, heading: float) -> float:
        if not self._config.smooth_display:
            return heading
        # Display-only smoothing. The planner does not smooth, and the user model's relaxation
        # is the principled version of what this fakes for the eye.
        if self._displayed_heading is None:
            self._displayed_heading = heading
        else:
            weight = self._config.smoothing_weight
            self._displayed_heading = weight * self._displayed_heading + (1.0 - weight) * heading
        return self._displayed_heading

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(WINDOW_NAME)
            self._open = False


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
