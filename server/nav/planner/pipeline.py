"""An ObstacleSet in, a PlannedPath out. Field, goal prior, dynamic program, heading, alarm."""

# Standard library imports
import logging
import time

# Third party imports
import numpy as np

# Local package imports
from nav.planner.alarm import AlarmHold, alarm_raised, check_alarm_config
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.dynamic_programming import plan
from nav.planner.goal import goal_position, goal_term
from nav.planner.heading import lookahead_heading, lookahead_step_index
from nav.planner.surprise import lateral_grid, step_count, surprise_field
from nav.types import ObstacleSet, PlannedPath
from nav.walker import WalkerConfig

log = logging.getLogger(__name__)


class PlannerPipeline:
    """Holds the grid, which never changes, the last field, which the debug window draws, and the alarm's hold.

    The hold carries across frames, so one pipeline has to serve a whole run. A pipeline built per
    frame would let the alarm clear the moment its raise decision did.
    """

    def __init__(self, config: PlannerConfig, walker: WalkerConfig) -> None:
        self._config = config
        self._walker = walker
        self._grid = lateral_grid(config)
        self._times = np.arange(step_count(config)) * config.time_step_seconds
        # Checked here so a bad lookahead stops the run at startup rather than on its first frame.
        self._lookahead_index = lookahead_step_index(config)
        check_alarm_config(config)
        self._alarm_hold = AlarmHold(config.alarm_hold_seconds)
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
        :return: The chosen path, its heading at the lookahead, and the alarm.
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

        # Where the arrow points: at where the path is a lookahead from now, not at its first step.
        heading = lookahead_heading(offsets, self._lookahead_index, config)

        alarm = self._alarm_hold.update(alarm_raised(obstacles, config), obstacles.timestamp_seconds)

        log.debug(
            "planner %.1f ms: field %.1f, dp %.1f, %d groups, cost %.2f, heading %.1f deg, alarm %s",
            (after_plan - started) * 1000,
            (after_field - started) * 1000,
            (after_plan - after_field) * 1000,
            obstacles.groups_in_view,
            cost,
            np.degrees(heading),
            alarm,
        )

        return PlannedPath(
            timestamp_seconds=obstacles.timestamp_seconds,
            times_seconds=self._times.copy(),
            lateral_offsets_meters=np.asarray(offsets, dtype=np.float64),
            first_heading_radians=heading,
            alarm=alarm,
            cumulative_cost_bits=cost,
        )
