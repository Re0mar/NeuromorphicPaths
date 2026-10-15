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

Two sources can say where the goal is, each behind a run-wide switch: where the wearer looks, and the
walking route. Each one with something to say this frame gives a center and a tolerance, and they
combine into one half-squared term by precision. With neither, the goal is straight ahead.

The goal has a forward component too, which the sources fill in and combine. Nothing reads it yet.
It's kept for a term that weighs how far ahead the goal is.
"""

# Standard library imports
import math
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import GoalSwitches, PlannerConfig
from nav.planner.spread import spread_meters


@dataclass(frozen=True)
class RouteGoal:
    """
    Where the route says to go this frame, in the walker's own terms. The route layer converts into this, so the planner never imports it.

    relative_bearing_radians is the bearing to the route's look-ahead point minus the walker's course,
    positive right. tolerance_meters is the route goal's own spread, already widened for a poor fix.
    """

    relative_bearing_radians: float
    tolerance_meters: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_bearing_radians) or abs(self.relative_bearing_radians) > math.pi:
            raise ValueError(f"relative_bearing_radians must be within pi of zero, got {self.relative_bearing_radians}")
        if not math.isfinite(self.tolerance_meters) or self.tolerance_meters <= 0:
            raise ValueError(f"tolerance_meters must be above zero, got {self.tolerance_meters}")


@dataclass(frozen=True)
class GoalContribution:
    """One source's say on where the plan should end: a point in the walker's ground frame and how sure it is."""

    lateral_meters: float
    forward_meters: float
    tolerance_meters: float


def gaze_goal_contribution(gaze_ground_point: np.ndarray | None, config: PlannerConfig) -> GoalContribution | None:
    """
    The gaze's goal: where the wearer is looking on the floor, clipped into the grid.

    :param gaze_ground_point: (2,) lateral and forward of the gaze on the floor, or None when the source
        has no gaze or it missed the floor this frame.
    :param config: The grid, the goal distance and the base tolerance.
    :return: The contribution, or None without a gaze point.
    :rtype: GoalContribution | None
    """
    if gaze_ground_point is None:
        return None
    # The old file clipped the goal into the grid so a glance at the horizon did not put the goal fifty
    # meters out and flatten the terminal term.
    lateral = float(np.clip(gaze_ground_point[0], -config.grid_half_width_meters, config.grid_half_width_meters))
    forward = float(np.clip(gaze_ground_point[1], 0.0, config.goal_distance_meters))
    return GoalContribution(lateral, forward, config.goal_tolerance_meters)


def route_goal_contribution(route_goal: RouteGoal | None, config: PlannerConfig) -> GoalContribution | None:
    """
    The route's goal: the point at the goal distance that lies in the route's direction from the walker.

    The goal row sits at the goal distance ahead, so the sideways offset that points there along a
    relative bearing b is the goal distance times tan b. Past route_goal_max_bearing_degrees that runs
    toward infinity, so the goal sits at the grid's edge on b's side instead.

    example: 30 degrees right at a 4 m goal distance -> 4 x tan 30 = 2.31 m right

    :param route_goal: The route layer's bearing and tolerance, or None when it has no goal this frame.
    :param config: The goal distance, the grid and the cutoff.
    :return: The contribution, at the route's own tolerance, or None.
    :rtype: GoalContribution | None
    """
    if route_goal is None:
        return None
    bearing = route_goal.relative_bearing_radians
    edge = config.grid_half_width_meters
    if abs(bearing) <= np.radians(config.route_goal_max_bearing_degrees):
        lateral = float(np.clip(config.goal_distance_meters * np.tan(bearing), -edge, edge))
    else:
        lateral = float(np.copysign(edge, bearing))
    return GoalContribution(lateral, config.goal_distance_meters, route_goal.tolerance_meters)


def combine_goals(contributions: list[GoalContribution | None], config: PlannerConfig) -> GoalContribution:
    """
    One goal from every source that has something to say.

    Two half-squared terms on one axis add to one half-squared term. Its center is the mean of the two
    centers weighted by each one's precision, one over its tolerance squared, and its precision is the
    sum of theirs. So a sure source pulls the center toward itself, and two sources together are surer
    than either.

    example: 1.0 m right at 1.0 m and 1.0 m left at 0.5 m -> 0.6 m left at 0.447 m

    :param contributions: One per source, None for a source with nothing this frame.
    :param config: The goal distance and base tolerance for straight ahead.
    :return: The combined goal. Straight ahead at the base tolerance when nothing contributes.
    :rtype: GoalContribution
    """
    present = [contribution for contribution in contributions if contribution is not None]
    if not present:
        return GoalContribution(0.0, config.goal_distance_meters, config.goal_tolerance_meters)
    # One source is itself, with no arithmetic, so a single goal comes out to the bit as it went in.
    if len(present) == 1:
        return present[0]
    precisions = [1.0 / contribution.tolerance_meters**2 for contribution in present]
    total = sum(precisions)
    lateral = sum(precision * contribution.lateral_meters for precision, contribution in zip(precisions, present)) / total
    forward = sum(precision * contribution.forward_meters for precision, contribution in zip(precisions, present)) / total
    return GoalContribution(lateral, forward, math.sqrt(1.0 / total))


def goal_for_frame(
    switches: GoalSwitches,
    gaze_ground_point: np.ndarray | None,
    route_goal: RouteGoal | None,
    config: PlannerConfig,
) -> tuple[np.ndarray, float]:
    """
    The goal this frame plans toward and its tolerance before any widening, from the switched-on sources.

    The widening near danger is a fact about the frame, not about a source, so the caller applies it
    once to this combined tolerance, with widened_goal_tolerance.

    :param switches: Which sources are on.
    :param gaze_ground_point: The gaze on the floor, read only when the gaze goal is on.
    :param route_goal: The route layer's goal, read only when the route goal is on.
    :param config: The planner's constants.
    :return: ((2,) lateral and forward, the combined tolerance in meters).
    :rtype: tuple[np.ndarray, float]
    """
    combined = combine_goals(
        [
            gaze_goal_contribution(gaze_ground_point, config) if switches.gaze_goal else None,
            route_goal_contribution(route_goal, config) if switches.route_goal else None,
        ],
        config,
    )
    goal = np.array([combined.lateral_meters, combined.forward_meters])
    return goal, combined.tolerance_meters


def widened_goal_tolerance(tolerance_meters: float, config: PlannerConfig, avoidance_nats: float) -> float:
    """
    A goal tolerance, widened the sooner something in the corridor is reached.

    tolerance squared = tolerance squared + goal_tolerance_growth_square_meters_per_nat * U, with U the
    course's avoidance surprise, half of (1 s over tau) squared. With an empty corridor it is unchanged.

    :raises ValueError: When the growth or the surprise is negative or not finite.
    """
    check_goal_config(config)
    if not np.isfinite(avoidance_nats) or avoidance_nats < 0:
        raise ValueError(f"avoidance surprise must be finite and zero or more, got {avoidance_nats}")
    return spread_meters(tolerance_meters, [(config.goal_tolerance_growth_square_meters_per_nat, avoidance_nats)])


def goal_tolerance(config: PlannerConfig, avoidance_nats: float) -> float:
    """The base goal tolerance widened for this frame. Straight ahead's tolerance, when no source contributes."""
    return widened_goal_tolerance(config.goal_tolerance_meters, config, avoidance_nats)


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
