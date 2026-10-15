"""
A smoothed bearing, so GPS noise doesn't swing the route goal from side to side every second.

Bearings are averaged as unit vectors, not as numbers, so 350 and 10 degrees average to 0 and not 180.
Each new sample pulls the vector toward itself by one minus e to the minus elapsed time over the time
constant, so the smoothing is the same whether fixes arrive once a second or five times.
"""

# Standard library imports
import math

# Local package imports
from nav.route.geodesy import bearing_of_vector, vector_of_bearing


class CircularExponentialFilter:
    """An exponential moving average of a bearing, on the timestamps it is fed."""

    def __init__(self, time_constant_seconds: float) -> None:
        if not math.isfinite(time_constant_seconds) or time_constant_seconds < 0:
            raise ValueError(f"bearing time constant must be zero or more, got {time_constant_seconds}")
        self._time_constant = time_constant_seconds
        self._vector: tuple[float, float] | None = None
        self._timestamp: float | None = None
        self._last_bearing: float | None = None

    def update(self, bearing: float, timestamp_seconds: float) -> None:
        """Take one bearing in degrees. A timestamp earlier than the last starts the average over from it."""
        sample = vector_of_bearing(bearing)
        self._last_bearing = bearing
        if self._vector is None or self._timestamp is None or timestamp_seconds < self._timestamp:
            self._vector = sample
            self._timestamp = timestamp_seconds
            return
        elapsed = timestamp_seconds - self._timestamp
        # A time constant of zero follows each sample exactly.
        pull = 1.0 if self._time_constant == 0 else 1.0 - math.exp(-elapsed / self._time_constant)
        self._vector = (
            (1.0 - pull) * self._vector[0] + pull * sample[0],
            (1.0 - pull) * self._vector[1] + pull * sample[1],
        )
        self._timestamp = timestamp_seconds

    def value(self) -> float | None:
        """The smoothed bearing in degrees, [0, 360), or None before the first sample."""
        if self._vector is None:
            return None
        # Two opposite samples can cancel to nothing, which has no direction. The latest sample stands in.
        if math.hypot(*self._vector) < 1e-9:
            return self._last_bearing
        return bearing_of_vector(*self._vector)

    def reset(self) -> None:
        self._vector = None
        self._timestamp = None
        self._last_bearing = None
