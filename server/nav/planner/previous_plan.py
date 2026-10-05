"""
A prior toward the plan the walker was already following, so a near-tie doesn't flip sides every frame.

The professor's dynamic program picks the cheapest end cell and remembers nothing between frames.
When going left and going right around something centered ahead cost nearly the same, any small
change between frames can tip the choice, and on the recorded walks the arrow swung from one sidestep
limit to the other 65 to 127 times a minute. This adds memory in his own form: a prior surprise, half
of (distance from the previous plan over a spread) squared, added to the field like the goal prior.

The previous plan is held in the walker's own frame, not in the world. It means "keep the same
offsets relative to my forward", so it never resists a deliberate turn, and it needs no position,
which the Neon glasses don't stream. Between frames it is advanced the way the planner already
assumes the walker moves: forward at walking speed, and sideways along the plan itself.

Small adjustments cost almost nothing, since the cost is quadratic, and only a swing to the other
side is expensive. There is no clock. A side holds until the other side is cheaper by more than the
cost of switching, so a near-tie stays put and a side that becomes blocked is left at once.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig


def check_previous_plan_config(config: PlannerConfig) -> None:
    """
    Refuse a spread, reach or gap the prior can't use.

    :raises ValueError: Naming the field.
    """
    if not np.isfinite(config.previous_plan_spread_meters) or config.previous_plan_spread_meters <= 0:
        raise ValueError(f"previous_plan_spread_meters must be above zero, got {config.previous_plan_spread_meters}")
    if not np.isfinite(config.previous_plan_prior_seconds) or config.previous_plan_prior_seconds <= 0:
        raise ValueError(f"previous_plan_prior_seconds must be above zero, got {config.previous_plan_prior_seconds}")
    if not np.isfinite(config.previous_plan_max_gap_seconds) or config.previous_plan_max_gap_seconds < 0:
        raise ValueError(f"previous_plan_max_gap_seconds must be zero or more, got {config.previous_plan_max_gap_seconds}")


class PreviousPlanPrior:
    """Remembers the last plan and prices every lateral position by its distance from it, row by row."""

    def __init__(self, config: PlannerConfig, times_seconds: np.ndarray) -> None:
        check_previous_plan_config(config)
        self._config = config
        self._times = np.asarray(times_seconds, dtype=np.float64)
        self._forward = self._times * config.walking_speed_mps
        self._offsets: np.ndarray | None = None
        self._timestamp: float | None = None
        self._goal: np.ndarray | None = None

    def field(self, grid: np.ndarray, timestamp_seconds: float, goal: np.ndarray) -> np.ndarray | None:
        """
        The prior's cost rate at every step and lateral position, or None when there is nothing to remember.

        Nothing is remembered on the first frame, after a clock that went backwards (a reconnect or a
        new recording segment), or after a gap longer than previous_plan_max_gap_seconds, when the old
        plan is too stale to say anything about this one. Nor after the goal moved farther than the
        goal's own tolerance: that is a different place to go, such as a wearer looking at the other of
        two gaps, and holding the old plan would keep them from it.

        :param grid: The lateral candidates.
        :param timestamp_seconds: This frame's time.
        :param goal: (2,) this frame's goal, lateral and forward.
        :return: (steps, len(grid)) cost per second, zero on rows the prior doesn't reach.
        :rtype: np.ndarray | None
        """
        if self._offsets is None or self._timestamp is None or self._goal is None:
            return None
        elapsed = timestamp_seconds - self._timestamp
        goal_moved = float(np.hypot(*(np.asarray(goal, dtype=np.float64) - self._goal)))
        if elapsed < 0 or elapsed > self._config.previous_plan_max_gap_seconds or goal_moved > self._config.goal_tolerance_meters:
            self.forget()
            return None
        # Where each of this frame's rows lies on the previous plan. The planner assumes the walker
        # walks at walking speed and follows the plan, so since then they have gone that much further
        # along it, sideways too. Every plan starts at the walker, so the sideways part they already
        # walked comes off. Without it the prior would ask for the old offsets again from wherever the
        # walker now is, and a sidestep would never end.
        walked = self._config.walking_speed_mps * elapsed
        source = self._forward + walked
        expected = np.interp(source, self._forward, self._offsets) - np.interp(walked, self._forward, self._offsets)
        reached = source <= self._forward[-1]
        # Row 0 is where the walker stands, which the dynamic program fixes. Beyond the prior's reach,
        # the rows stay free to replan as new things come into view.
        rows = reached & (self._times > 0) & (self._times <= self._config.previous_plan_prior_seconds + 1e-9)
        prior = np.zeros((len(self._times), len(grid)), dtype=np.float64)
        spread = self._config.previous_plan_spread_meters
        prior[rows] = 0.5 * ((grid[None, :] - expected[rows, None]) / spread) ** 2
        return prior

    def remember(self, offsets: np.ndarray, timestamp_seconds: float, goal: np.ndarray) -> None:
        """Keep this frame's plan, and the goal it was made for, for the next frame."""
        self._offsets = np.asarray(offsets, dtype=np.float64).copy()
        self._timestamp = float(timestamp_seconds)
        self._goal = np.asarray(goal, dtype=np.float64).copy()

    def forget(self) -> None:
        self._offsets = None
        self._timestamp = None
        self._goal = None
