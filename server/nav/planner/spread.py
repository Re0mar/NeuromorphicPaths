"""
A prior's spread, widened by what the frame says.

Every prior in the planner has the professor's form, half of (distance over a spread) squared. When a
reading of the frame should loosen it, its variance grows with that reading: the goal's with the
avoidance surprise, the previous plan's with its age. Variances add, so any number of them still
leave one spread and one half-squared term, a log probability like his.

Each growth is a pair: how many square meters of variance per unit of the reading, and the reading.
A later reason to loosen a prior is one more pair, not a new formula.
"""

# Standard library imports
import math
from collections.abc import Iterable


def spread_variance(base_meters: float, growths: Iterable[tuple[float, float]]) -> float:
    """
    A prior's variance: its base spread squared, plus each growth rate times its reading.

    :param base_meters: The spread with every reading at zero.
    :param growths: (square meters of variance per unit, the reading), in the order they are added.
    :return: Square meters.
    :rtype: float
    :raises ValueError: When the base is not above zero, or a rate or a reading is negative or not finite.
    """
    if not math.isfinite(base_meters) or base_meters <= 0:
        raise ValueError(f"a spread must be above zero, got {base_meters}")
    variance = base_meters**2
    for rate, reading in growths:
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"a variance growth must be finite and zero or more, got {rate}")
        if not math.isfinite(reading) or reading < 0:
            raise ValueError(f"a reading that widens a spread must be finite and zero or more, got {reading}")
        variance += rate * reading
    return variance


def spread_meters(base_meters: float, growths: Iterable[tuple[float, float]]) -> float:
    """The widened spread itself, the root of spread_variance."""
    return math.sqrt(spread_variance(base_meters, growths))
