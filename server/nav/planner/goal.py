"""
Where the walker is trying to go, and how that enters the planner.

The goal is a prior on the terminal row only: half of (distance from the goal over a tolerance)
squared. It picks where the plan ends, rather than pulling the walker through something in the way,
which is what an attractive term on every row did in the old planner.

It used to be added to the field as if it were a rate, so the dynamic program multiplied it by the
0.1 s time step like every other row. With the 1.5 m tolerance it then couldn't move an empty-field
plan by one cell. One 0.1 m sidestep costs 0.325, and the goal row saved at most 0.013 between
neighboring cells. It only broke ties.

Now it is a cost paid once. terminal_cost_as_rate divides the row by the time step before the
dynamic program multiplies it back. The plan moves one more cell toward a goal d away only while
that saves more than the sidestep costs: half of (d squared minus (d - 0.1) squared) over the
tolerance squared, against 0.325. So a goal 1.5 m to the side is reached to about 1.5 minus 3.25
times the tolerance squared. At 0.4 m that is 1.0 m.

A goal that strong would pull the plan back into a post on the walker's line. So the tolerance
widens with the course's avoidance surprise, half of (1 s over the time to contact) squared: the
sooner something in the corridor is reached, the less ending off the goal's line costs. With nothing
in the corridor it is the base tolerance.

The goal has a forward component too, which goal_position clips and returns. Nothing reads it yet.
It's kept for a term that weighs how far ahead the goal is.
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


def goal_tolerance(config: PlannerConfig, avoidance_nats: float) -> float:
    """
    The goal's tolerance this frame: the base tolerance, widened the sooner something in the corridor is reached.

    The variance grows with the course's avoidance surprise, U = half of (1 s over tau) squared:
    tolerance squared = goal_tolerance_meters squared + goal_tolerance_growth_square_meters_per_nat * U.
    So the goal pulls less the closer contact is, and with an empty corridor it is the base tolerance.

    :param config: The base tolerance and the growth.
    :param avoidance_nats: This frame's avoidance surprise, from avoidance_surprise_nats.
    :return: Meters, above zero.
    :rtype: float
    :raises ValueError: When the base is not above zero, the growth is negative, or the surprise is negative.
    """
    check_goal_config(config)
    if not np.isfinite(avoidance_nats) or avoidance_nats < 0:
        raise ValueError(f"avoidance surprise must be finite and zero or more, got {avoidance_nats}")
    return float(np.sqrt(config.goal_tolerance_meters**2 + config.goal_tolerance_growth_square_meters_per_nat * avoidance_nats))


def check_goal_config(config: PlannerConfig) -> None:
    """
    Refuse a base tolerance or growth the goal can't use.

    :raises ValueError: Naming the field.
    """
    if not np.isfinite(config.goal_tolerance_meters) or config.goal_tolerance_meters <= 0:
        raise ValueError(f"goal_tolerance_meters must be above zero, got {config.goal_tolerance_meters}")
    growth = config.goal_tolerance_growth_square_meters_per_nat
    if not np.isfinite(growth) or growth < 0:
        raise ValueError(f"goal_tolerance_growth_square_meters_per_nat must be zero or more, got {growth}")


def goal_term(grid: np.ndarray, goal: np.ndarray, tolerance_meters: float) -> np.ndarray:
    """
    The prior added to the terminal row: half of (lateral distance to the goal over tolerance) squared.

    :param grid: The lateral candidates.
    :param goal: (2,) goal, only its lateral component matters here.
    :param tolerance_meters: This frame's tolerance, from goal_tolerance.
    :return: (len(grid),) term.
    :rtype: np.ndarray
    """
    if not np.isfinite(tolerance_meters) or tolerance_meters <= 0:
        raise ValueError(f"goal tolerance must be above zero, got {tolerance_meters}")
    return 0.5 * ((grid - goal[0]) / tolerance_meters) ** 2


def terminal_cost_as_rate(goal_row: np.ndarray, config: PlannerConfig) -> np.ndarray:
    """
    The goal row divided by the time step, ready for the dynamic program.

    Every other row of the field is a rate, and the dynamic program multiplies each row by the time step.
    The goal is a cost paid once at the end, not a rate. Dividing it here means that multiplication gives
    the cost back in full. The pipeline and the band breakdown's re-plans both call this, so the factor
    lives in one place.

    :param goal_row: (len(grid),) the goal term from goal_term.
    :param config: The time step.
    :return: (len(grid),) the row to put on the terminal row of the field the dynamic program reads.
    :rtype: np.ndarray
    """
    return goal_row / config.time_step_seconds
