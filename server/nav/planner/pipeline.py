"""An ObstacleSet in, a PlannedPath out. Field, goal prior, dynamic program, heading, alarm."""

# Standard library imports
import logging
import time

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.dynamic_programming import plan
from nav.planner.goal import goal_position, goal_term
from nav.planner.surprise import lateral_grid, step_count, surprise_field, time_to_contact
from nav.types import ObstacleSet, PlannedPath
from nav.walker import WalkerConfig

log = logging.getLogger(__name__)


class PlannerPipeline:
    """Holds the grid, which never changes, and the last field, which the debug window draws."""

    def __init__(self, config: PlannerConfig, walker: WalkerConfig) -> None:
        self._config = config
        self._walker = walker
        self._grid = lateral_grid(config)
        self._times = np.arange(step_count(config)) * config.time_step_seconds
        self._last_field: np.ndarray | None = None

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def last_field(self) -> np.ndarray | None:
        """The field from the most recent plan, goal term included.

        Debug only. The runtime hands it to a DebugSink and nothing else reads it, so nothing else
        grows a dependency on a grid the size of the horizon.
        """
        return self._last_field

    def plan(
        self,
        obstacles: ObstacleSet,
        start_lateral_meters: float = 0.0,
        goal_mode: GoalMode = GoalMode.AHEAD,
        gaze_ground_point: np.ndarray | None = None,
    ) -> PlannedPath:
        """
        Choose the path for this frame.

        :param obstacles: This frame's groups, walker relative.
        :param start_lateral_meters: Where the walker is on the grid. Zero, the walker is the origin.
        :param goal_mode: Straight ahead or the gaze.
        :param gaze_ground_point: The gaze on the floor, when the mode wants it.
        :return: The chosen path, its first heading, and the alarm.
        :rtype: PlannedPath
        """
        config = self._config
        started = time.perf_counter()

        field = surprise_field(obstacles, self._grid, config, self._walker)
        goal = goal_position(goal_mode, config, gaze_ground_point)
        field[-1] += goal_term(self._grid, goal, config)
        self._last_field = field
        after_field = time.perf_counter()

        offsets, cost = plan(field, start_lateral_meters, self._grid, config)
        after_plan = time.perf_counter()

        # Where the arrow points now: the first lateral step against the forward distance the
        # walker covers in one time step. Positive is right.
        if len(offsets) > 1:
            first_heading = float(np.arctan2(offsets[1] - offsets[0], config.time_step_seconds * config.walking_speed_mps))
        else:
            first_heading = 0.0

        contact = time_to_contact(obstacles)
        alarm = contact is not None and contact < config.alarm_time_to_contact_seconds

        log.debug(
            "planner %.1f ms: field %.1f, dp %.1f, %d groups, cost %.2f, heading %.1f deg, alarm %s",
            (after_plan - started) * 1000,
            (after_field - started) * 1000,
            (after_plan - after_field) * 1000,
            obstacles.groups_in_view,
            cost,
            np.degrees(first_heading),
            alarm,
        )

        return PlannedPath(
            timestamp_seconds=obstacles.timestamp_seconds,
            times_seconds=self._times.copy(),
            lateral_offsets_meters=np.asarray(offsets, dtype=np.float64),
            first_heading_radians=first_heading,
            alarm=alarm,
            cumulative_cost_bits=cost,
        )
