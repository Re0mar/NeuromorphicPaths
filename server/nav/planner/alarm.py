"""
When the display turns red: something is in the walker's way and close, or was a moment ago.

Only groups in the walker's corridor count. The corridor is the strip the footprint sweeps as the
walker carries on straight, its radius plus a margin to each side, and only ahead. Furniture
being walked past is not in it, so it no longer raises the alarm.

The alarm reads the obstacles and the footprint, never the planned path. An alarm that trusted
the plan would go quiet exactly when the plan was wrong, and catching that is what it is for.

Once raised it stays up for a minimum time. A signal computed at 30 Hz chatters near its threshold
whatever it measures, and a display that flickers five times a second tells the walker nothing.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig


def check_alarm_config(config: PlannerConfig) -> None:
    """
    Refuse an alarm configuration that cannot work, so a run fails at startup rather than on its first frame.

    :param config: The margin, walking speed and threshold. The hold is checked by AlarmHold.
    :raises ValueError: Naming the first field that is out of range.
    """
    _check_margin(config)
    _check_walking_speed(config)
    _check_threshold(config)


def _check_margin(config: PlannerConfig) -> None:
    margin = config.corridor_margin_meters
    if not np.isfinite(margin) or margin < 0:
        raise ValueError(f"corridor_margin_meters must be zero or more, got {margin}")


def _check_walking_speed(config: PlannerConfig) -> None:
    speed = config.walking_speed_mps
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError(f"walking_speed_mps must be above zero, got {speed}")


def _check_threshold(config: PlannerConfig) -> None:
    threshold = config.alarm_time_to_contact_seconds
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"alarm_time_to_contact_seconds must be above zero, got {threshold}")


def corridor_points(obstacles: ObstacleSet, config: PlannerConfig, walker: WalkerConfig) -> tuple[ObstaclePoint, ...]:
    """
    The groups in the walker's corridor: ahead, and within the footprint radius plus the margin.

    Each group is represented by its nearest point. That is enough because the scene makes one
    group per occupied 0.25 m cell, so a table reaching into the corridor is several groups and
    the ones inside it are found.

    :param obstacles: This frame's groups, walker relative.
    :param config: The corridor margin.
    :param walker: The footprint radius.
    :return: The points inside the corridor, in their original order.
    :rtype: tuple[ObstaclePoint, ...]
    :raises ValueError: When the margin is negative or not a number.
    """
    _check_margin(config)
    half_width = walker.radius_meters + config.corridor_margin_meters
    # Strictly ahead. A group level with the walker or behind it is being passed, not approached.
    return tuple(point for point in obstacles.points if point.forward_meters > 0.0 and abs(point.lateral_meters) <= half_width)


def corridor_time_to_contact(obstacles: ObstacleSet, config: PlannerConfig, walker: WalkerConfig) -> float | None:
    """
    Seconds until the walker reaches the nearest group in its corridor, at walking pace.

    This uses the planner's walking speed rather than each group's measured closing rate. The
    closing rate is a two-frame difference, so a centimeter of depth jitter reads as a third of a
    meter per second. The cost is that something moving toward the walker faster than walking pace,
    such as a person, is seen as no more urgent than if it stood still.

    :param obstacles: This frame's groups.
    :param config: Walking speed and the corridor margin.
    :param walker: The footprint radius.
    :return: The smallest clearance over walking speed, or None when the corridor is empty.
    :rtype: float | None
    :raises ValueError: When the walking speed is not above zero.
    """
    _check_walking_speed(config)
    in_the_way = corridor_points(obstacles, config, walker)
    if not in_the_way:
        return None
    return min(point.clearance_meters for point in in_the_way) / config.walking_speed_mps


def alarm_raised(obstacles: ObstacleSet, config: PlannerConfig, walker: WalkerConfig) -> bool:
    """
    Whether this frame alone calls for the alarm, before any hold is applied.

    :param obstacles: This frame's groups.
    :param config: The threshold, walking speed and corridor margin.
    :param walker: The footprint radius.
    :return: True when something in the corridor is under the threshold from contact.
    :rtype: bool
    :raises ValueError: When the threshold is not above zero.
    """
    _check_threshold(config)
    contact = corridor_time_to_contact(obstacles, config, walker)
    return contact is not None and contact < config.alarm_time_to_contact_seconds


class AlarmHold:
    """Keeps a raised alarm up for a minimum time, measured on frame timestamps."""

    def __init__(self, hold_seconds: float) -> None:
        if not np.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError(f"alarm_hold_seconds must be zero or more, got {hold_seconds}")
        self._hold_seconds = hold_seconds
        self._raised_at: float | None = None

    def update(self, raise_now: bool, timestamp_seconds: float) -> bool:
        """
        Feed one frame's raise decision and get back whether the alarm is shown.

        Runs on the frame's own timestamp, never on the wall clock, so a replay of a recording
        gives the same answer the live run did.

        :param raise_now: This frame's decision, from alarm_raised.
        :param timestamp_seconds: This frame's time.
        :return: Whether the alarm is up.
        :rtype: bool
        :raises ValueError: When the timestamp is not a finite number.
        """
        if not np.isfinite(timestamp_seconds):
            raise ValueError(f"timestamp must be finite, got {timestamp_seconds}")
        # A clock that went backward is a phone reconnect with a fresh session or a new recording
        # segment. A duplicated frame carries an equal timestamp, which is not backward.
        if self._raised_at is not None and timestamp_seconds < self._raised_at:
            self._raised_at = None
        if raise_now:
            if self._raised_at is None:
                self._raised_at = timestamp_seconds
            return True
        # The minimum runs from when the alarm was raised, not from the last frame that asked for
        # it. A long gap forward needs no case of its own, because the hold has expired by then.
        if self._raised_at is not None and timestamp_seconds - self._raised_at < self._hold_seconds:
            return True
        self._raised_at = None
        return False
