"""
Whether the walker has seen the thing the alarm is about, read from where they looked.

The alarm is feedback, and a beep about a chair the walker just looked at tells them nothing new.
This module decides whether the danger was looked at. It never changes the alarm's own decision or
the planned path. It only feeds what the display is told to play.

The test is an angle. Each gaze sample is a ray out of the camera, and the danger is the part of the
picture its object covers. The sample is on the danger when the angle from the ray to the nearest
part of that outline is under a tolerance. No floor is needed, and a look at a chair's seat counts
as well as a look at its feet.

The scene hands over 0.25 m pieces, and the alarm's danger is the nearest piece, usually an
object's base. A walker looks at the object's body. So the object is the danger's piece plus every
piece chained to it within a short distance on the floor, and a look at any of them is a look at it.

A look also counts for a while afterwards. People look at an obstacle while it is still a few
meters off, and by the time it raises the alarm they are looking past it. So a look is remembered
as a position in the walker's frame, moved toward the walker at walking speed, and a later danger
at that position is still noticed until the credit runs out.
"""

# Standard library imports
from collections.abc import Sequence

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.types import GAZE_CARRY_WINDOW_SECONDS, CameraOutline, ObstaclePoint

# A look further off than this is not at the thing in any useful sense, and a tolerance this wide
# would accept most of the picture. Above zero because a zero tolerance accepts nothing.
MAXIMUM_TOLERANCE_DEGREES = 45.0


def tolerance_radians(tolerance_degrees: float) -> float:
    """
    An angular tolerance in degrees, checked and converted. The one place degrees become radians.

    :param tolerance_degrees: Above zero and under MAXIMUM_TOLERANCE_DEGREES.
    :return: The same angle in radians.
    :rtype: float
    :raises ValueError: For a tolerance out of that range or not finite.
    """
    _check_tolerance(tolerance_degrees)
    return float(np.radians(tolerance_degrees))


def check_attention_config(config: PlannerConfig) -> None:
    """
    Refuse an attention configuration that cannot work, so a run fails at startup rather than on its first frame.

    The credit's own values are refused by AttentionCredit when the pipeline builds it.

    :param config: The attention block and the alarm's threshold.
    :raises ValueError: Naming the first field that is out of range.
    """
    _check_tolerance(config.attention_tolerance_degrees)
    window = config.attention_window_seconds
    if not np.isfinite(window) or not 0.0 <= window <= GAZE_CARRY_WINDOW_SECONDS:
        raise ValueError(
            f"attention_window_seconds must be from 0 to the {GAZE_CARRY_WINDOW_SECONDS} s of gaze a frame carries, got {window}"
        )
    for field_name in ("attention_link_meters", "attention_cap_meters"):
        value = getattr(config, field_name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be above zero, got {value}")
    dwell = config.attention_dwell_seconds
    if not np.isfinite(dwell) or dwell < 0.0:
        raise ValueError(f"attention_dwell_seconds must be zero or more, got {dwell}")
    floor = config.attention_floor_seconds
    if not np.isfinite(floor) or not 0.0 < floor < config.alarm_time_to_contact_seconds:
        # A floor at or over the alarm's threshold would mean the gaze could never quiet anything.
        raise ValueError(
            f"attention_floor_seconds must be above zero and under alarm_time_to_contact_seconds "
            f"{config.alarm_time_to_contact_seconds}, got {floor}"
        )


def alarm_audible(alarm: bool, eyes_on_danger: bool, time_to_contact_seconds: float | None, silencing_on: bool, config: PlannerConfig) -> bool:
    """
    Whether the display should make the alarm heard this frame.

    Heard whenever the alarm is up, except with the switch on, the eyes on the danger, and contact
    no nearer than the floor. A raised alarm with nothing left in the corridor is the hold, and its
    time to contact is None, which is above the floor.

    :param alarm: The alarm, hold included. Never changed here.
    :param eyes_on_danger: Whether the gaze was on the danger's object, credit included.
    :param time_to_contact_seconds: From corridor_time_to_contact, or None with the corridor empty.
    :param silencing_on: The run's switch.
    :param config: The floor.
    :return: False only when the alarm is up and quieted.
    :rtype: bool
    """
    above_floor = time_to_contact_seconds is None or time_to_contact_seconds >= config.attention_floor_seconds
    return alarm and not (silencing_on and eyes_on_danger and above_floor)


def _check_tolerance(tolerance_degrees: float) -> None:
    if not np.isfinite(tolerance_degrees) or not 0.0 < tolerance_degrees < MAXIMUM_TOLERANCE_DEGREES:
        raise ValueError(
            f"attention tolerance must be above 0 and under {MAXIMUM_TOLERANCE_DEGREES:g} degrees, got {tolerance_degrees}"
        )


def pixel_ray(pixel: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    The ray out of the camera through a pixel, in the camera frame, z forward. Not normalized.

    :param pixel: (2,) x and y, or (N, 2) for several, at the resolution intrinsics describes.
    :param intrinsics: (3, 3) pinhole matrix.
    :return: (3,) or (N, 3), with z equal to 1, so x and y are the ray's tangents.
    :rtype: np.ndarray
    """
    pixel = np.asarray(pixel, dtype=np.float64)
    focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
    principal_x, principal_y = intrinsics[0, 2], intrinsics[1, 2]
    return np.stack(
        [
            (pixel[..., 0] - principal_x) / focal_x,
            (pixel[..., 1] - principal_y) / focal_y,
            np.ones_like(pixel[..., 0]),
        ],
        axis=-1,
    )


def angle_between_radians(ray_a: np.ndarray, ray_b: np.ndarray) -> np.ndarray:
    """
    The angle between two rays, or between each of several rays and one.

    Dot products are summed elementwise rather than with a matrix product, so a replay gets the
    same angle to the last bit and a sample sitting on the tolerance lands on the same side.

    :param ray_a: (3,) or (N, 3).
    :param ray_b: (3,) or (N, 3), broadcast against ray_a. Neither may have zero length.
    :return: Radians in [0, pi], one per ray pair.
    :rtype: np.ndarray
    """
    dot = np.sum(ray_a * ray_b, axis=-1)
    length_a = np.sqrt(np.sum(ray_a * ray_a, axis=-1))
    length_b = np.sqrt(np.sum(ray_b * ray_b, axis=-1))
    # Clipped because rounding can put the cosine of two parallel rays a hair over 1.
    return np.arccos(np.clip(dot / (length_a * length_b), -1.0, 1.0))


def samples_in_window(offsets_seconds: np.ndarray, window_seconds: float) -> np.ndarray:
    """
    Which carried samples are within the window of the frame's stamp, either side, inclusive.

    :param offsets_seconds: (N,) seconds from the frame's stamp.
    :param window_seconds: Zero or more.
    :return: (N,) bool.
    :rtype: np.ndarray
    """
    return np.abs(offsets_seconds) <= window_seconds


def angle_to_outline_radians(rays: np.ndarray, outline: CameraOutline) -> np.ndarray:
    """
    The angle from each ray to the nearest part of an outline. Zero for a ray inside it.

    :param rays: (N, 3) from pixel_ray, z equal to 1.
    :param outline: The group's outline.
    :return: (N,) radians.
    :rtype: np.ndarray
    """
    # On the z = 1 plane the outline is a rectangle, and the nearest part of it to a ray's point
    # is that point clamped into the rectangle.
    nearest = np.stack(
        [
            np.clip(rays[:, 0], outline.left_tangent, outline.right_tangent),
            np.clip(rays[:, 1], outline.top_tangent, outline.bottom_tangent),
            np.ones(len(rays)),
        ],
        axis=-1,
    )
    return angle_between_radians(rays, nearest)


def danger_object(
    danger: ObstaclePoint,
    points: Sequence[ObstaclePoint],
    link_meters: float,
    cap_meters: float,
) -> tuple[ObstaclePoint, ...]:
    """
    The danger's piece and the pieces chained to it, each within link_meters of the next on the floor.

    Chained, so a chair's seat and back join its base. Capped at cap_meters from the danger, because
    unchecked chaining runs along rows of seats and walls until the object covers half the picture,
    and then any gaze at all is on it.

    :param danger: The alarm's nearest corridor point.
    :param points: Every group this frame, the danger among them or not.
    :param link_meters: The furthest two neighboring pieces of one object may be apart, above zero.
    :param cap_meters: The furthest a piece may be from the danger and still be part of it, above zero.
    :return: The danger first, then the pieces chained to it within the cap, each once.
    :rtype: tuple[ObstaclePoint, ...]
    """
    others = [
        point for point in points
        if point is not danger
        and np.hypot(point.lateral_meters - danger.lateral_meters, point.forward_meters - danger.forward_meters) <= cap_meters
    ]
    if not others:
        return (danger,)
    positions = np.array([[point.lateral_meters, point.forward_meters] for point in others])
    joined = np.zeros(len(others), dtype=bool)
    frontier = [np.array([danger.lateral_meters, danger.forward_meters])]
    while frontier:
        here = frontier.pop()
        near = ~joined & (np.hypot(positions[:, 0] - here[0], positions[:, 1] - here[1]) <= link_meters)
        joined |= near
        frontier.extend(positions[near])
    return (danger, *(point for point, member in zip(others, joined) if member))


def longest_dwell_seconds(offsets_seconds: np.ndarray, on_object: np.ndarray, gap_seconds: float) -> float | None:
    """
    The longest unbroken stretch of samples on the object, from its first sample to its last.

    A stretch breaks at a sample off the object, or at a gap between samples longer than gap_seconds,
    since a dropout says nothing about where the eye was.

    :param offsets_seconds: (N,) ascending.
    :param on_object: (N,) whether each sample was on it.
    :param gap_seconds: The longest gap a stretch survives.
    :return: Seconds, 0.0 for a single sample on it, None when no sample is on it.
    :rtype: float | None
    """
    longest: float | None = None
    run_start: float | None = None
    previous: float | None = None
    for offset, on in zip(offsets_seconds.tolist(), on_object.tolist()):
        if not on or (previous is not None and run_start is not None and offset - previous > gap_seconds):
            run_start = None
        if on:
            if run_start is None:
                run_start = offset
            longest = max(longest if longest is not None else 0.0, offset - run_start)
        previous = offset
    return longest


def gaze_on_outlines(
    gaze_pixels: np.ndarray | None,
    offsets_seconds: np.ndarray | None,
    window_seconds: float,
    intrinsics: np.ndarray,
    outlines: Sequence[CameraOutline | None],
    tolerance_radians: float,
    dwell_seconds: float = 0.0,
    gap_seconds: float = 0.02,
) -> bool:
    """
    Whether the gaze rested on any of the outlines, within the tolerance, for at least the dwell.

    Every sample is compared against the outlines as seen in the frame the samples came with, never
    against a later frame's, since the head moves between frames. A dwell of zero accepts a single
    sample, so a glance passing over the object counts. Above zero it takes a steady look.

    :param gaze_pixels: (N, 2) at the resolution of intrinsics, or None when the frame has no gaze.
    :param offsets_seconds: (N,) seconds from the frame's stamp, ascending, or None with gaze_pixels.
    :param window_seconds: How far from the stamp a sample may be and still count.
    :param intrinsics: (3, 3) for the frame the samples came with.
    :param outlines: The danger's object, one outline per piece, as that frame's camera saw it.
    :param tolerance_radians: From tolerance_radians().
    :param dwell_seconds: How long the gaze must stay on the object without a break, zero or more.
    :param gap_seconds: The longest gap between samples a steady look survives.
    :return: True when the gaze stayed on the object long enough. No sample is not noticed.
    :rtype: bool
    """
    if gaze_pixels is None or offsets_seconds is None or len(gaze_pixels) == 0:
        # No gaze means not noticed, which keeps the alarm sounding. Unknown fails safe.
        return False
    # Hand-built points and the golden fixtures carry no outline. They are never noticed, by rule.
    known = [outline for outline in outlines if outline is not None]
    if not known:
        return False
    inside = samples_in_window(offsets_seconds, window_seconds)
    if not np.any(inside):
        return False
    rays = pixel_ray(gaze_pixels[inside], intrinsics)
    # A sample is on the object when it is on any of its pieces, so a look sliding from the seat to
    # the back of a chair stays one steady look.
    on_object = np.zeros(len(rays), dtype=bool)
    for outline in known:
        on_object |= angle_to_outline_radians(rays, outline) < tolerance_radians
    dwell = longest_dwell_seconds(offsets_seconds[inside], on_object, gap_seconds)
    return dwell is not None and dwell >= dwell_seconds


class AttentionCredit:
    """
    The last danger the walker looked at, remembered in the walker's frame for a while.

    The position is moved toward the walker at walking speed, which assumes the walker carries on
    straight, as the previous-plan prior does. A turn moves the remembered position off the danger
    and the credit stops matching, which is the safe way for it to fail.

    Runs on frame time only. A stamp earlier than the last one forgets, as a new run does.
    """

    def __init__(self, credit_seconds: float, match_meters: float, walking_speed_mps: float) -> None:
        """
        :param credit_seconds: How long a look counts for, zero or more. Zero is the looking frame only.
        :param match_meters: How near the advanced position a later danger must be, above zero.
        :param walking_speed_mps: How fast the remembered position comes toward the walker, above zero.
        :raises ValueError: Naming the first value out of range.
        """
        if not np.isfinite(credit_seconds) or credit_seconds < 0.0:
            raise ValueError(f"attention credit_seconds must be zero or more, got {credit_seconds}")
        if not np.isfinite(match_meters) or match_meters <= 0.0:
            raise ValueError(f"attention match_meters must be above zero, got {match_meters}")
        if not np.isfinite(walking_speed_mps) or walking_speed_mps <= 0.0:
            raise ValueError(f"attention walking_speed_mps must be above zero, got {walking_speed_mps}")
        self._credit_seconds = credit_seconds
        self._match_meters = match_meters
        self._walking_speed_mps = walking_speed_mps
        # (lateral, forward) of the danger when it was looked at, and the stamp of that look.
        self._remembered: tuple[float, float, float] | None = None
        self._last_stamp: float | None = None

    def update(self, danger: ObstaclePoint | None, looked_now: bool, timestamp_seconds: float) -> bool:
        """
        Take this frame's danger and whether it was looked at, and say whether it counts as noticed.

        :param danger: The alarm's nearest corridor point this frame, or None for an empty corridor.
        :param looked_now: Whether a gaze sample this frame was on it.
        :param timestamp_seconds: The frame's stamp.
        :return: True when it was looked at now, or matches a look still in credit.
        :rtype: bool
        """
        if self._last_stamp is not None and timestamp_seconds < self._last_stamp:
            self.forget()
        self._last_stamp = timestamp_seconds
        if danger is not None and looked_now:
            self._remembered = (danger.lateral_meters, danger.forward_meters, timestamp_seconds)
            return True
        if danger is None or self._remembered is None:
            # An empty corridor keeps the credit, so a danger that drops out of view for a frame
            # and comes back is still the one that was looked at.
            return False
        lateral, forward, looked_at = self._remembered
        elapsed = timestamp_seconds - looked_at
        if elapsed > self._credit_seconds:
            # Only the look goes. The stamp stays, so a backward stamp next frame is still caught.
            self._remembered = None
            return False
        # Worked out from the look each time rather than stepped per frame, so nothing drifts.
        advanced_forward = forward - self._walking_speed_mps * elapsed
        distance = np.hypot(danger.lateral_meters - lateral, danger.forward_meters - advanced_forward)
        return bool(distance <= self._match_meters)

    def forget(self) -> None:
        """Start afresh, as a new run does: no look remembered and no stamp seen."""
        self._remembered = None
        self._last_stamp = None
