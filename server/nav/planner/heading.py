"""
Where the arrow points: the direction the planned path is heading over the next second or so.

The arrow used to be the angle of the path's first 0.1 s step. One step can move at most one
lateral cell, so that angle could only ever be straight ahead or a full sidestep either way, three
values and nothing between. Reading the path a little further out gives the arrow every angle the
path can take, and a one-cell wiggle at the start no longer flips it from one side to the other.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.surprise import step_count


def lookahead_step_index(config: PlannerConfig) -> int:
    """
    Which step of the path the arrow reads, from the configured lookahead.

    :param config: The lookahead, the time step and the horizon.
    :return: The step index, at least 1 and never past the horizon's last step.
    :rtype: int
    :raises ValueError: When the lookahead is not a positive number, rounds to no steps at all, or
        reaches past the horizon.
    """
    lookahead = config.heading_lookahead_seconds
    if not np.isfinite(lookahead) or lookahead <= 0:
        raise ValueError(f"heading_lookahead_seconds must be a positive number, got {lookahead}")
    # round, not int: 0.3 / 0.1 is 2.9999999999999996 in floating point, and int would make it 2.
    index = int(round(lookahead / config.time_step_seconds))
    last_step = step_count(config) - 1
    if index < 1:
        raise ValueError(f"heading_lookahead_seconds of {lookahead} s is under half a {config.time_step_seconds} s step")
    if index > last_step:
        raise ValueError(f"heading_lookahead_seconds of {lookahead} s reaches past the {config.horizon_seconds} s horizon")
    return index


def lookahead_heading(offsets: np.ndarray, step_index: int, config: PlannerConfig) -> float:
    """
    The angle from where the walker is now to where the path is at the lookahead. Positive is right.

    :param offsets: The path's lateral offset per step, from the dynamic program.
    :param step_index: From lookahead_step_index.
    :param config: The time step and walking speed, which give the forward distance covered.
    :return: Heading in radians.
    :rtype: float
    """
    # The dynamic program returns one offset per step of the horizon, and the pipeline checked the
    # index against the horizon when it was built, so the path always reaches the index.
    forward_meters = step_index * config.time_step_seconds * config.walking_speed_mps
    return float(np.arctan2(offsets[step_index] - offsets[0], forward_meters))
