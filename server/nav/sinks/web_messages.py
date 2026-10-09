"""
What the web sink sends to a browser, built without aiohttp so it can be tested as plain functions.

Every text message carries a kind, and the page dispatches on it. A path message is the path JSON
the phone gets, plus its kind. A plan view is what the page draws top-down: the planner's field,
the path through it, the obstacles, and how the path should look. Everything the page draws is
computed here, on the laptop. The page only draws.

The video socket's two text messages are here too: the stream's description, which the page's
decoder needs before the first unit, and the word that this run has no video to send.
"""

# Standard library imports
import base64
import json
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.planner.alarm import time_to_contact_from_avoidance_bits
from nav.scene.config import SceneConfig
from nav.sinks.floor_geometry import floor_hidden_mask, floor_seen_mask
from nav.sinks.path_style import BORDER_OPACITY, GROUP_RGB, WALL_RGB, path_color_rgb, path_fill_opacity
from nav.sources.scene_video import VideoDescription
from nav.types import DebugView, PlannedPath


class WebMessageKind(Enum):
    """The kinds of text message the page understands. Values are the strings on the wire."""

    PATH = "path"
    PLAN_VIEW = "plan_view"
    # On the video socket. The stream's codec and parameter sets, sent before any unit.
    VIDEO_STREAM = "video_stream"
    # On the video socket. This run's source has no video, and the socket closes after it.
    VIDEO_UNAVAILABLE = "video_unavailable"


# Every key a plan view carries, in one place. The page test checks the page reads each of these,
# so a key nothing draws from is not sent.
PLAN_VIEW_KEYS = (
    "times_seconds",
    "grid_meters",
    "walking_speed_mps",
    "path_width_meters",
    "field",
    "path_offsets_meters",
    "obstacles",
    "scene_information_bits",
    "avoidance_surprise_bits",
    "time_to_contact_seconds",
    "path_color_rgb",
    "path_fill_opacity",
    "path_border_opacity",
    "group_color_rgb",
    "wall_color_rgb",
    "floor_seen",
    "floor_hidden",
)
OBSTACLE_KEYS = ("lateral_meters", "forward_meters", "is_wall")


def plan_view_message(path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView, scene: SceneConfig) -> dict:
    """
    The top-down view's data for one planned frame.

    :param path: The path the planner chose.
    :param field: (steps, cells) the field it planned through, goal term included.
    :param grid: (cells,) the lateral position of each field column.
    :param view: The frame's obstacles, walking speed and body half-width, and the depth and floor
        the two floor masks read.
    :param scene: The run's scene config. It supplies the floor fit's inlier distance and the usable
        depth range, so the hidden floor is judged the way the scene judged the frame.
    :return: Plain Python values keyed by PLAN_VIEW_KEYS, ready for web_text_message.
    :rtype: dict
    :raises ValueError: When the field is not one row per path step by one column per grid cell.
    """
    expected = (len(path.times_seconds), len(grid))
    if field.shape != expected:
        raise ValueError(f"field is {field.shape}, the path and grid need {expected}")

    message = {
        "times_seconds": path.times_seconds.tolist(),
        "grid_meters": np.asarray(grid, dtype=np.float64).tolist(),
        "walking_speed_mps": float(view.walking_speed_mps),
        "path_width_meters": 2.0 * float(view.body_half_width_meters),
        "field": np.asarray(field, dtype=np.float64).tolist(),
        "path_offsets_meters": path.lateral_offsets_meters.tolist(),
        "obstacles": [
            {
                "lateral_meters": float(point.lateral_meters),
                "forward_meters": float(point.forward_meters),
                "is_wall": bool(point.is_wall),
            }
            for point in view.obstacles.points
        ],
        "scene_information_bits": float(path.scene_information_bits),
        "avoidance_surprise_bits": float(path.avoidance_surprise_bits),
        # None when the corridor is empty, which JSON carries as null.
        "time_to_contact_seconds": time_to_contact_from_avoidance_bits(path.avoidance_surprise_bits),
        "path_color_rgb": list(path_color_rgb(path.avoidance_surprise_bits, view.path_red_from_bits)),
        "path_fill_opacity": path_fill_opacity(path.scene_information_bits),
        "path_border_opacity": BORDER_OPACITY,
        "group_color_rgb": list(GROUP_RGB),
        "wall_color_rgb": list(WALL_RGB),
        "floor_seen": floor_seen_mask(view, path.times_seconds, grid).tolist(),
        "floor_hidden": floor_hidden_mask(view, path.times_seconds, grid, scene).tolist(),
    }
    return message


def video_stream_message(description: VideoDescription) -> dict:
    """
    What the page's decoder is configured with: the codec string and the parameter sets as base64.

    :param description: The stream as the device process described it.
    :return: Plain values, ready for web_text_message.
    :rtype: dict
    """
    return {
        "codec": description.codec,
        "parameter_sets": [base64.b64encode(parameter_set).decode("ascii") for parameter_set in description.parameter_sets],
    }


def video_unavailable_message(reason: str) -> dict:
    """The word that there is no video on this run, with the reason the page shows."""
    return {"reason": reason}


def web_text_message(kind: WebMessageKind, body: dict) -> str:
    """
    One text frame for the page: the body with its kind added.

    :param kind: What the page should do with it.
    :param body: The message's fields. Must not already carry a kind.
    :return: JSON text.
    :rtype: str
    :raises ValueError: When the body holds a non-finite number, which JSON cannot carry, or
        already has a kind.
    """
    if "kind" in body:
        raise ValueError(f"the body already carries a kind, {body['kind']!r}")
    return json.dumps({"kind": kind.value, **body}, allow_nan=False)
