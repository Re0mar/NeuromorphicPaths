"""
An OpenCV window showing the arrow, the alarm, the cost, the surprise field, and the depth view.

One of the two sinks that implement DebugSink. The field, a grid the size of the planner's
horizon, and the depth view, a whole frame, only go to a window on the laptop or to a browser
a person is looking at, never to the phone.

The pictures themselves come from nav.sinks.rendering, shared with the web sink, so the window
and the browser show the same thing.
"""

# Standard library imports
import logging

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.sinks.config import DebugWindowConfig
from nav.sinks.rendering import CANVAS_WIDTH, render_arrow, render_depth_view, render_field
from nav.types import DebugView, PlannedPath

log = logging.getLogger(__name__)

WINDOW_NAME = "nav"
CANVAS_BACKGROUND = 24


class DebugWindowSink:
    """Shows each path as it arrives. Opens the window on the first publish, not on construction."""

    def __init__(self, config: DebugWindowConfig) -> None:
        self._config = config
        self._open = False
        self._displayed_heading: float | None = None
        self._latest_inset: np.ndarray | None = None
        self._latest_depth_view: np.ndarray | None = None

    def start(self) -> None:
        """Nothing to open ahead of time. The window opens on the first publish, because an empty window is a question."""

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
        """
        Draw the path with the surprise field inset and the depth view under it.

        :param path: The path to show.
        :param field: (steps, cells) the planner's field, goal term included.
        :param grid: (cells,) the lateral position of each field column, in meters.
        :param view: The planner's input for this path.
        """
        self._latest_inset = render_field(field, grid, path) if self._config.show_field else None
        try:
            self._latest_depth_view = render_depth_view(view, path)
        except ValueError as unrenderable:
            # A frame the view cannot draw is still a frame the planner planned. The arrow and
            # the field go up, and the window keeps the last depth view it had.
            log.warning("depth view not drawn (caught %s, expected): %s", type(unrenderable).__name__, unrenderable)
        self.publish(path)

    def publish(self, path: PlannedPath) -> None:
        """Draw the arrow, the alarm color and the cost. The inset and the depth view only if publish_debug supplied them."""
        heading = self._display_heading(path.first_heading_radians)
        arrow = render_arrow(heading, path)

        if self._latest_inset is not None:
            height, width = self._latest_inset.shape[:2]
            arrow[10 : 10 + height, CANVAS_WIDTH - 10 - width : CANVAS_WIDTH - 10] = self._latest_inset

        canvas = arrow
        if self._latest_depth_view is not None:
            # The window grows to hold both: the arrow on top, the depth view under it, each at
            # its own width on a shared dark background.
            canvas = _stack(arrow, self._latest_depth_view)

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


def _stack(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Two BGR images one above the other, left aligned, on a canvas as wide as the wider one."""
    width = max(top.shape[1], bottom.shape[1])
    canvas = np.full((top.shape[0] + bottom.shape[0], width, 3), CANVAS_BACKGROUND, dtype=np.uint8)
    canvas[: top.shape[0], : top.shape[1]] = top
    canvas[top.shape[0] :, : bottom.shape[1]] = bottom
    return canvas
