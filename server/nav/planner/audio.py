"""
The stereo cue: how loud each ear should be, from the planned heading and the alarm.

Two cues, both decided here on the laptop so the page and the phone only apply numbers.

The steering cue. The ear on the far side from the planned heading goes quieter, and the walker
turns away from the quiet ear. How much quieter is the course's surprise of the heading error: a
turn of theta against a tolerance sigma costs U = half of (theta over sigma) squared, and the far
ear's gain is e to the minus U, an exact Gaussian in sigma. Sigma is the planner's own goal
tolerance seen as an angle, atan(goal_tolerance_meters / goal_distance_meters), 20.6 degrees with
the shipped values. At that sigma an ordinary 4 degree head wobble leaves the far ear at 98 %, and
the 35.5 degree sidestep limit takes it to 22.5 %. The gain never goes under a floor, because an
ear that goes fully silent reads as broken headphones rather than as a cue.

The danger cue. While the alarm is up, the ear on the danger's side drops to the floor. The side
comes from the corridor point that raised the alarm, as a pan from -1 (left) to +1 (right) that
reaches full at the body's half-width, the corridor's edge. Dead ahead ducks both ears.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig


def check_audio_config(config: PlannerConfig) -> None:
    """
    Refuse a floor or a goal geometry the cue can't use.

    :raises ValueError: Naming the field.
    """
    floor = config.far_ear_floor_gain
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError(f"far_ear_floor_gain must be from 0 to 1, got {floor}")
    if not np.isfinite(config.goal_distance_meters) or config.goal_distance_meters <= 0.0:
        raise ValueError(f"goal_distance_meters must be above zero, got {config.goal_distance_meters}")
    if not np.isfinite(config.goal_tolerance_meters) or config.goal_tolerance_meters <= 0.0:
        raise ValueError(f"goal_tolerance_meters must be above zero, got {config.goal_tolerance_meters}")


def heading_tolerance_radians(config: PlannerConfig) -> float:
    """
    The goal tolerance as an angle: how far off the heading can be before the cue counts it.

    :return: atan(goal_tolerance_meters / goal_distance_meters), radians.
    :rtype: float
    """
    check_audio_config(config)
    return float(np.arctan(config.goal_tolerance_meters / config.goal_distance_meters))


def far_ear_gain(heading_radians: float, config: PlannerConfig) -> float:
    """
    How loud the ear on the far side from the heading is, from 1 straight ahead down to the floor.

    :param heading_radians: The planned heading, either sign. Only its size matters here.
    :return: e to the minus half of (heading over tolerance) squared, no lower than the floor.
    :rtype: float
    """
    sigma = heading_tolerance_radians(config)
    surprise = 0.5 * (heading_radians / sigma) ** 2
    return float(max(config.far_ear_floor_gain, np.exp(-surprise)))


def alarm_pan(lateral_meters: float, config: PlannerConfig) -> float:
    """
    Where the danger is, left to right, from the lateral position of the point that raised the alarm.

    :param lateral_meters: The point's lateral position, positive right.
    :return: -1 at or past the corridor's left edge, +1 at or past its right edge, 0 dead ahead.
    :rtype: float
    """
    if not np.isfinite(lateral_meters):
        raise ValueError(f"the danger's lateral position must be finite, got {lateral_meters}")
    return float(np.clip(lateral_meters / config.body_half_width_meters, -1.0, 1.0))


def ear_gains(heading_radians: float, alarm: bool, alarm_pan_value: float | None, config: PlannerConfig) -> tuple[float, float]:
    """
    The left and right ear gains for one frame.

    :param heading_radians: The planned heading, positive right.
    :param alarm: Whether the alarm is up this frame, hold included.
    :param alarm_pan_value: Where the danger is while the alarm is up, from alarm_pan, or None.
    :return: (left, right), each from the floor to 1.
    :rtype: tuple[float, float]
    """
    if not np.isfinite(heading_radians):
        raise ValueError(f"the heading must be finite, got {heading_radians}")
    left = right = 1.0
    far = far_ear_gain(heading_radians, config)
    # Heading right means the far ear is the left one, and the walker turns away from it.
    if heading_radians > 0.0:
        left = far
    elif heading_radians < 0.0:
        right = far
    if alarm and alarm_pan_value is not None:
        floor = config.far_ear_floor_gain
        if alarm_pan_value < 0.0:
            left = min(left, floor)
        elif alarm_pan_value > 0.0:
            right = min(right, floor)
        else:
            left = right = min(left, right, floor)
    return left, right
