"""
Where the walker is trying to go, and how that enters the planner.

The goal is a prior on the terminal row only. It picks among paths that are all safe rather than
pulling the walker through something in the way, which is what an attractive term on every row
did in the old planner. With nothing in view, every path is equally safe and the goal decides.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import GoalMode, PlannerConfig


def goal_position(mode: GoalMode, config: PlannerConfig, gaze_ground_point: np.ndarray | None) -> np.ndarray:
    """
    The goal as (lateral, forward) in the walker's ground frame.

    :param mode: Straight ahead, or where the wearer is looking.
    :param config: Goal distance and the grid the goal is clipped into.
    :param gaze_ground_point: (2,) lateral and forward of the gaze on the floor, or None when
        the source has no gaze or it did not land on the floor this frame.
    :return: (2,) goal position, inside the grid.
    :rtype: np.ndarray
    """
    match mode:
        case GoalMode.AHEAD:
            return np.array([0.0, config.goal_distance_meters])
        case GoalMode.GAZE:
            if gaze_ground_point is None:
                # No gaze this frame. Straight ahead is the only honest answer.
                return np.array([0.0, config.goal_distance_meters])
            # The old file clipped the goal into the grid so a glance at the horizon did not put
            # the goal fifty meters out and flatten the terminal term.
            lateral = float(np.clip(gaze_ground_point[0], -config.grid_half_width_meters, config.grid_half_width_meters))
            forward = float(np.clip(gaze_ground_point[1], 0.0, config.goal_distance_meters))
            return np.array([lateral, forward])
        case _:
            raise ValueError(f"no goal rule for {mode}")


def goal_term(grid: np.ndarray, goal: np.ndarray, config: PlannerConfig) -> np.ndarray:
    """
    The prior added to the terminal row: half of (lateral distance to the goal over tolerance) squared.

    :param grid: The lateral candidates.
    :param goal: (2,) goal, only its lateral component matters here.
    :param config: The tolerance, half the corridor width by default.
    :return: (len(grid),) term.
    :rtype: np.ndarray
    """
    if config.goal_tolerance_meters <= 0:
        raise ValueError("goal tolerance must be positive")
    return 0.5 * ((grid - goal[0]) / config.goal_tolerance_meters) ** 2
