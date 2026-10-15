"""
Phone fixes, which way the walker is going, and which fix belongs with which frame.

The course is the phone's own heading, its course over ground, read only while the walker moves fast
enough for it to mean something. Standing still, a phone's heading points anywhere. A missing heading
or speed is no course, never a course of 0, which would read as walking due north.

Fixes and frames come from different clocks: the phone's, the glasses', the laptop's. They can differ
by seconds, more than a fix stays fresh. So a fix is paired with a frame on the laptop's clock alone,
the time the laptop received each, and the phone's own stamp is only for telling fixes apart.
"""

# Standard library imports
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

# Local package imports
from nav.route.geodesy import LatLon, distance_meters, wrap_degrees
from nav.route.route import RouteConfig

# Fixes kept for pairing. A phone sends about one a second, so this is minutes of history, far more
# than the few seconds a fix stays fresh.
FIX_HISTORY_LENGTH = 256


@dataclass(frozen=True)
class Fix:
    """One location reading from the phone."""

    point: LatLon
    accuracy_meters: float
    heading_degrees: float | None
    speed_mps: float | None
    timestamp_ms: int  # The phone's clock, epoch milliseconds. For telling fixes apart, never for pairing.
    received_seconds: float  # The laptop's clock on receipt. The one pairing reads.


class FixVerdict(Enum):
    """What FixHistory.add did with a fix."""

    KEPT = "kept"
    TOO_FAST = "too fast"  # Implies a walk faster than anyone walks from the previous fix.
    HEADING_JUMP = "heading jump"  # The heading turned faster than a walker turns.
    SAME_TIME = "same time"  # Received at the same laptop moment as the previous one, so no speed can be judged.
    CLOCK_BACK = "clock back"  # The laptop's clock went back, a replay restarting. The history starts over from it.


def course_degrees(fix: Fix, config: RouteConfig) -> float | None:
    """
    The walker's direction of travel from a fix, degrees from north, or None when the fix can't say.

    :return: The fix's heading at walking speed or more. None when slower, or when the heading or
        the speed is missing.
    :rtype: float | None
    """
    if fix.heading_degrees is None or fix.speed_mps is None:
        return None
    if fix.speed_mps < config.course_min_speed_mps:
        return None
    return fix.heading_degrees


class FixHistory:
    """The recent fixes, with the implausible ones dropped and counted, and pairing on the laptop's clock."""

    def __init__(self, config: RouteConfig) -> None:
        self._config = config
        self._fixes: deque[Fix] = deque(maxlen=FIX_HISTORY_LENGTH)
        self.counts: Counter[FixVerdict] = Counter()

    @property
    def latest(self) -> Fix | None:
        return self._fixes[-1] if self._fixes else None

    def add(self, fix: Fix) -> FixVerdict:
        """Keep a fix unless it is implausible next to the previous one. Every verdict is counted."""
        verdict = self._judge(fix)
        self.counts[verdict] += 1
        if verdict is FixVerdict.CLOCK_BACK:
            self._fixes.clear()
            self._fixes.append(fix)
        elif verdict is FixVerdict.KEPT:
            self._fixes.append(fix)
        return verdict

    def nearest(self, received_seconds: float, max_age_seconds: float) -> Fix | None:
        """
        The fix received nearest a moment on the laptop's clock, if one is within the maximum age of it.

        :param received_seconds: The frame's arrival on the laptop.
        :param max_age_seconds: How far apart the two may be.
        :return: The fix, or None when nothing is close enough.
        :rtype: Fix | None
        """
        best = min(self._fixes, key=lambda fix: abs(fix.received_seconds - received_seconds), default=None)
        if best is None or abs(best.received_seconds - received_seconds) > max_age_seconds:
            return None
        return best

    def _judge(self, fix: Fix) -> FixVerdict:
        previous = self.latest
        if previous is None:
            return FixVerdict.KEPT
        elapsed = fix.received_seconds - previous.received_seconds
        if elapsed < 0:
            return FixVerdict.CLOCK_BACK
        if elapsed == 0:
            return FixVerdict.SAME_TIME
        if distance_meters(previous.point, fix.point) / elapsed > self._config.max_plausible_speed_mps:
            return FixVerdict.TOO_FAST
        if previous.heading_degrees is not None and fix.heading_degrees is not None:
            turned = abs(wrap_degrees(fix.heading_degrees - previous.heading_degrees))
            if turned / elapsed > self._config.max_heading_rate_degrees_per_second:
                return FixVerdict.HEADING_JUMP
        return FixVerdict.KEPT


def check_fix(fix: Fix) -> None:
    """
    Refuse a fix whose numbers can't be real. The page's message parser applies the same rules first.

    :raises ValueError: Naming the field.
    """
    if not math.isfinite(fix.accuracy_meters) or fix.accuracy_meters < 0:
        raise ValueError(f"accuracy_meters must be zero or more, got {fix.accuracy_meters}")
    if fix.heading_degrees is not None and (not math.isfinite(fix.heading_degrees) or not 0.0 <= fix.heading_degrees < 360.0):
        raise ValueError(f"heading_degrees must be in [0, 360) or None, got {fix.heading_degrees}")
    if fix.speed_mps is not None and (not math.isfinite(fix.speed_mps) or fix.speed_mps < 0):
        raise ValueError(f"speed_mps must be zero or more or None, got {fix.speed_mps}")
    if fix.timestamp_ms <= 0:
        raise ValueError(f"timestamp_ms must be a positive integer, got {fix.timestamp_ms}")
    if not math.isfinite(fix.received_seconds):
        raise ValueError(f"received_seconds must be finite, got {fix.received_seconds}")
