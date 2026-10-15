"""
What the web sink sends to a browser, built without aiohttp so it can be tested as plain functions.

Every text message carries a kind, and the page dispatches on it. A path message is the path JSON
the phone gets, plus its kind. A plan view is what the page draws top-down: the planner's field,
the path through it, the obstacles, and how the path should look. Everything the page draws is
computed here, on the laptop. The page only draws.

The video socket's two text messages are here too: the stream's description, which the page's
decoder needs before the first unit, and the word that this run has no video to send. So are the two
messages a page sends back, saying which picture it wants in its Depth card and, on a run with a
recording, which side every page's run should plan on, and the state of the live input and that
switch the laptop announces.
"""

# Standard library imports
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.planner.alarm import time_to_contact_from_avoidance_bits
from nav.planner.config import OnOff, RunSwitchKind
from nav.route.course import Fix, check_fix
from nav.route.geodesy import LatLon
from nav.route.route import RouteState
from nav.scene.config import SceneConfig
from nav.sinks.floor_geometry import floor_hidden_mask, floor_seen_mask, group_rings
from nav.sinks.path_style import BORDER_OPACITY, GROUP_RGB, WALL_RGB, path_color_rgb, path_fill_opacity
from nav.sources.config import SourceMode
from nav.sources.scene_video import VideoDescription
from nav.sources.switching import SwitchState
from nav.types import DebugView, PlannedPath


class WebMessageKind(Enum):
    """The kinds of text message the page understands. Values are the strings on the wire."""

    PATH = "path"
    PLAN_VIEW = "plan_view"
    # On the video socket. The stream's codec and parameter sets, sent before any unit.
    VIDEO_STREAM = "video_stream"
    # On the video socket. This run's source has no video, and the socket closes after it.
    VIDEO_UNAVAILABLE = "video_unavailable"
    # Which side a run serving the page from a live input plans on, and how that input is. Never sent
    # on any other run.
    SOURCE_STATE = "source_state"
    # Where the route layer stands: its tier, how far along and how far to go, and the last fix.
    ROUTE_STATE = "route_state"
    # Whether the gaze may quiet the alarm, run-wide. Sent on every change and to each page as it connects.
    GAZE_SILENCING_STATE = "gaze_silencing_state"
    # Whether the plan leans toward where the wearer looks, and along the route. Sent like the gaze switch.
    GAZE_GOAL_STATE = "gaze_goal_state"
    ROUTE_GOAL_STATE = "route_goal_state"


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
    "group_rings",
)
OBSTACLE_KEYS = ("lateral_meters", "forward_meters", "is_wall")
# Each ring the page draws over the depth or risk picture, in that picture's own pixels.
RING_KEYS = ("column_pixels", "row_pixels", "radius_pixels", "is_wall")


class PictureKind(Enum):
    """Which picture a browser shows in its Depth card. Values are the strings the page sends."""

    DEPTH = "depth"
    RISK = "risk"


class BrowserMessageKind(Enum):
    """The kinds of text message a page sends the laptop. Values are the strings on the wire."""

    # Which picture this browser wants from now on, in a "picture" field.
    PICTURE = "picture"
    # Which side every page's run should plan on, the live input or the recording, in a "source" field.
    SOURCE = "source"
    # A phone fix from the browser's geolocation, read by parse_location.
    LOCATION = "location"
    # Where the walker wants to go, as latitude and longitude, read by parse_destination.
    DESTINATION = "destination"
    # Whether every page's run lets the gaze quiet the alarm, "off" or "on", in a "gaze_silencing" field.
    GAZE_SILENCING = "gaze_silencing"
    # Whether every page's run leans toward the gaze, and along the route, in fields of the same names.
    GAZE_GOAL = "gaze_goal"
    ROUTE_GOAL = "route_goal"


# Each message a page sends, the field its choice is in, and the vocabulary that choice comes from.
BROWSER_CHOICES: dict[BrowserMessageKind, type[Enum]] = {
    BrowserMessageKind.PICTURE: PictureKind,
    BrowserMessageKind.SOURCE: SourceMode,
    BrowserMessageKind.GAZE_SILENCING: OnOff,
    BrowserMessageKind.GAZE_GOAL: OnOff,
    BrowserMessageKind.ROUTE_GOAL: OnOff,
}
# The messages that flip a run-wide switch, which switch each one flips, and how each switch's state is
# announced. Every switch has all three, which a test checks.
SWITCH_REQUESTS: dict[BrowserMessageKind, RunSwitchKind] = {
    BrowserMessageKind.GAZE_SILENCING: RunSwitchKind.GAZE_SILENCING,
    BrowserMessageKind.GAZE_GOAL: RunSwitchKind.GAZE_GOAL,
    BrowserMessageKind.ROUTE_GOAL: RunSwitchKind.ROUTE_GOAL,
}
SWITCH_STATES: dict[RunSwitchKind, WebMessageKind] = {
    RunSwitchKind.GAZE_SILENCING: WebMessageKind.GAZE_SILENCING_STATE,
    RunSwitchKind.GAZE_GOAL: WebMessageKind.GAZE_GOAL_STATE,
    RunSwitchKind.ROUTE_GOAL: WebMessageKind.ROUTE_GOAL_STATE,
}


@dataclass(frozen=True)
class SwitchRequest:
    """A page asking for one run-wide switch to be on or off. Every switch's choice is an OnOff, so it says which."""

    switch: RunSwitchKind
    enabled: bool


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
        "group_rings": [
            {
                "column_pixels": ring.column_pixels,
                "row_pixels": ring.row_pixels,
                "radius_pixels": ring.radius_pixels,
                "is_wall": ring.is_wall,
            }
            for ring in group_rings(view)
        ],
    }
    return message


def browser_choice(text: str) -> PictureKind | SourceMode | OnOff:
    """
    What a page asked for, read from one text frame it sent: a picture, a side to plan on, or the gaze switch.

    :param text: The frame, JSON with a kind of "picture", "source" or "gaze_silencing" and the field of the same name.
    :return: The choice. The caller tells them apart by type, OnOff being the gaze switch's alone.
    :rtype: PictureKind | SourceMode | OnOff
    :raises ValueError: When the frame is not JSON, not a message this laptop reads, or names a
        choice that isn't one.
    """
    try:
        message = json.loads(text)
    except json.JSONDecodeError as not_json:
        raise ValueError(f"the page sent a frame that is not JSON: {not_json}") from not_json
    kind = message.get("kind") if isinstance(message, dict) else None
    try:
        message_kind = BrowserMessageKind(kind)
    except ValueError as unknown_kind:
        raise ValueError(f"the page sent a message this laptop does not read: {text[:80]!r}") from unknown_kind
    try:
        return BROWSER_CHOICES[message_kind](message.get(message_kind.value))
    except ValueError as unknown:
        raise ValueError(f"the page asked for a {message_kind.value} this laptop does not have: {message.get(message_kind.value)!r}") from unknown


# The fields a location message carries besides its kind, every one required. Heading and speed may be
# null, which the browser sends when it doesn't know them, and stay None rather than becoming 0.
LOCATION_FIELDS = ("latitude", "longitude", "accuracy_meters", "heading_degrees", "speed_mps", "timestamp_ms")
DESTINATION_FIELDS = ("latitude", "longitude")


def browser_message(text: str, clock: Callable[[], float] = laptop_time_seconds) -> PictureKind | SourceMode | SwitchRequest | Fix | LatLon:
    """
    Whatever one text frame from a page says: a choice, a fix, or a destination.

    :param text: The frame.
    :param clock: The laptop's clock, read once when a fix arrives. Replaced only by tests.
    :return: The choice, a switch request, the fix stamped with the laptop's time, or the destination.
        The caller tells them apart by type.
    :rtype: PictureKind | SourceMode | SwitchRequest | Fix | LatLon
    :raises ValueError: When the frame is not a message this laptop reads, or breaks one of its rules.
    """
    message = _json_object(text)
    try:
        kind = BrowserMessageKind(message.get("kind"))
    except ValueError as unknown_kind:
        raise ValueError(f"the page sent a message this laptop does not read: {text[:80]!r}") from unknown_kind
    match kind:
        case BrowserMessageKind.LOCATION:
            return parse_location(message, clock())
        case BrowserMessageKind.DESTINATION:
            return parse_destination(message)
        case _ if kind in SWITCH_REQUESTS:
            return SwitchRequest(SWITCH_REQUESTS[kind], browser_choice(text) is OnOff.ON)
        case _:
            return browser_choice(text)


def parse_location(message: dict, received_seconds: float) -> Fix:
    """
    A fix from a page's location message, stamped with when the laptop received it.

    Each field's type is checked before it is converted, so a true, a string or a fractional
    timestamp is refused rather than coerced.

    :param message: The decoded message.
    :param received_seconds: The laptop's clock now.
    :return: The fix.
    :rtype: Fix
    :raises ValueError: Naming the field and the rule it broke.
    """
    _refuse_unknown_keys(message, LOCATION_FIELDS, "location")
    timestamp = message["timestamp_ms"]
    if type(timestamp) is not int:
        raise ValueError(f"location timestamp_ms must be an integer, got {timestamp!r}")
    fix = Fix(
        point=_point(message, "location"),
        accuracy_meters=_number(message, "accuracy_meters", "location"),
        heading_degrees=_optional_number(message, "heading_degrees", "location"),
        speed_mps=_optional_number(message, "speed_mps", "location"),
        timestamp_ms=timestamp,
        received_seconds=received_seconds,
    )
    try:
        check_fix(fix)
    except ValueError as broken_rule:
        raise ValueError(f"location {broken_rule}") from broken_rule
    return fix


def parse_destination(message: dict) -> LatLon:
    """
    Where the walker wants to go, from a page's destination message.

    :raises ValueError: Naming the field and the rule it broke.
    """
    _refuse_unknown_keys(message, DESTINATION_FIELDS, "destination")
    return _point(message, "destination")


def route_state_message(
    state: RouteState,
    along_meters: float | None,
    remaining_meters: float | None,
    bearing_degrees: float | None,
    fix_age_seconds: float | None,
    accuracy_meters: float | None,
    reason: str,
) -> dict:
    """What every page shows of the route layer this frame. Unknowns are None, which JSON carries as null."""
    return {
        "state": state.value,
        "along_meters": along_meters,
        "remaining_meters": remaining_meters,
        "bearing_degrees": bearing_degrees,
        "fix_age_seconds": fix_age_seconds,
        "accuracy_meters": accuracy_meters,
        "reason": reason,
    }


def _json_object(text: str) -> dict:
    try:
        message = json.loads(text)
    except json.JSONDecodeError as not_json:
        raise ValueError(f"the page sent a frame that is not JSON: {not_json}") from not_json
    if not isinstance(message, dict):
        raise ValueError(f"the page sent a frame that is not an object: {text[:80]!r}")
    return message


def _refuse_unknown_keys(message: dict, fields: tuple[str, ...], kind: str) -> None:
    unknown = sorted(set(message) - set(fields) - {"kind"})
    if unknown:
        raise ValueError(f"{kind} message carries fields this laptop does not read: {unknown}")
    missing = [field for field in fields if field not in message]
    if missing:
        raise ValueError(f"{kind} message is missing {missing}")


def _number(message: dict, field: str, kind: str) -> float:
    value = message[field]
    # bool is an int in Python, so it is refused before the number check would let it through.
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{kind} {field} must be a number, got {value!r}")
    return float(value)


def _optional_number(message: dict, field: str, kind: str) -> float | None:
    return None if message[field] is None else _number(message, field, kind)


def _point(message: dict, kind: str) -> LatLon:
    latitude = _number(message, "latitude", kind)
    longitude = _number(message, "longitude", kind)
    try:
        return LatLon(latitude, longitude)
    except ValueError as out_of_range:
        raise ValueError(f"{kind} {out_of_range}") from out_of_range


def source_state_message(state: SwitchState) -> dict:
    """What every page shows of a run with a live input: the side in use, the input, why there is none, and what the run offers."""
    return {
        "mode": state.mode.value,
        "live": state.live.value,
        "live_detail": state.live_detail,
        "has_recording": state.has_recording,
        "live_has_video": state.live_has_video,
    }


def switch_state_message(enabled: bool) -> dict:
    """A run-wide switch's state, spelled the way the page sends it."""
    return {"state": (OnOff.ON if enabled else OnOff.OFF).value}


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
