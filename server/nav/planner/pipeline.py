"""An ObstacleSet in, a PlannedPath out. Field, goal prior, previous-plan prior, dynamic program, heading, alarm."""

# Standard library imports
import logging
import time

# Third party imports
import numpy as np

# Local package imports
from nav.planner.alarm import AlarmHold, alarm_raised, avoidance_surprise_bits, check_alarm_config, nearest_corridor_point
from nav.planner.audio import alarm_pan, check_audio_config, ear_gains
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.contact import ContactSurprise, check_contact_config
from nav.planner.dynamic_programming import plan, start_cell_index
from nav.planner.field import CostTerm, cost_field, lateral_grid, step_count
from nav.planner.goal import goal_position, goal_term
from nav.planner.heading import lookahead_heading, lookahead_step_index
from nav.planner.information import scene_information_bits
from nav.planner.previous_plan import PreviousPlanPrior, check_previous_plan_config
from nav.planner.surprise import CollisionSurprise
from nav.planner.units import bits_from_nats
from nav.types import ObstacleSet, PlannedPath
from nav.walker import WalkerConfig

log = logging.getLogger(__name__)


def planner_terms(config: PlannerConfig) -> tuple[CostTerm, ...]:
    """The cost terms a plan is built from: the collision surprise always, and the contact surprise unless switched off."""
    if config.contact_term_enabled:
        return (CollisionSurprise(), ContactSurprise())
    return (CollisionSurprise(),)


class PlannerPipeline:
    """Holds what a run keeps between frames: the grid and the cost terms, the last field, the alarm's hold, the previous plan.

    The grid and the terms never change. The last field is there for the debug window to draw and for
    the band breakdown to split by term. The previous plan is what the previous-plan prior pulls toward
    on the next frame.

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
        check_audio_config(config)
        # Where the danger is while the alarm is up, kept through the hold. None while it is down.
        self._alarm_pan: float | None = None
        check_contact_config(config)
        self._terms = planner_terms(config)
        self._alarm_hold = AlarmHold(config.alarm_hold_seconds)
        # Checked at construction even when off, so switching it on can't start a run with a bad constant.
        check_previous_plan_config(config)
        self._previous_plan = PreviousPlanPrior(config, self._times) if config.previous_plan_prior_enabled else None
        self._last_field: np.ndarray | None = None

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def last_field(self) -> np.ndarray | None:
        """The field from the most recent plan: every cost term, the goal term on its last row, and the previous-plan prior.

        For looking at, not for planning with. The runtime hands it to a DebugSink, and the evaluation's
        band breakdown reads it through replay to split a frame's cost by term. Nothing on the live path
        reads it, so nothing there grows a dependency on a grid the size of the horizon.
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

        field = cost_field(obstacles, self._grid, config, self._walker, self._terms)
        goal = goal_position(goal_mode, config, gaze_ground_point)
        goal_row = goal_term(self._grid, goal, config)
        field[-1] += goal_row
        previous_plan_field = None
        if self._previous_plan is not None:
            previous_plan_field = self._previous_plan.field(self._grid, obstacles.timestamp_seconds, goal)
            if previous_plan_field is not None:
                field += previous_plan_field
        self._last_field = field
        after_field = time.perf_counter()

        offsets, cost = plan(field, start_lateral_meters, self._grid, config)
        after_plan = time.perf_counter()
        if previous_plan_field is not None:
            # The user model's work figure reads this cost. Holding a side is the walker's memory,
            # not work the scene imposed, so the prior's charge along the chosen path comes off.
            cells = np.argmin(np.abs(self._grid[None, :] - offsets[:, None]), axis=1)
            cost -= float(np.sum(previous_plan_field[np.arange(len(offsets)), cells]) * config.time_step_seconds)
        # Everything above sums natural logs. Here they become the bits the course quotes, once, after
        # the prior's charge came off in the same units. Nothing before this line is in bits.
        cost_bits = float(bits_from_nats(cost))
        if self._previous_plan is not None:
            self._previous_plan.remember(offsets, obstacles.timestamp_seconds, goal)

        # The KL comparison's prior is the planner with nothing in view. It keeps the goal, or the scene gets
        # credit for a turn the gaze caused. It keeps the previous-plan prior too, since that's the walker's
        # memory and not something seen this frame, so holding a side doesn't count as scene information.
        prior_field = np.zeros_like(field)
        prior_field[-1] += goal_row
        if previous_plan_field is not None:
            prior_field += previous_plan_field
        start_cell = start_cell_index(self._grid, start_lateral_meters)
        information = scene_information_bits(field, prior_field, start_cell, self._grid, config, self._lookahead_index)
        avoidance = avoidance_surprise_bits(obstacles, config)
        after_information = time.perf_counter()

        # Where the arrow points: at where the path is a lookahead from now, not at its first step.
        heading = lookahead_heading(offsets, self._lookahead_index, config)

        raised_now = alarm_raised(obstacles, config)
        alarm = self._alarm_hold.update(raised_now, obstacles.timestamp_seconds)
        # The danger's side is read on the frames that raise the alarm and kept through the hold, so
        # the cue stays on the side the walker last heard it rather than jumping to center the
        # moment the point leaves the corridor.
        if raised_now:
            danger = nearest_corridor_point(obstacles, config)
            self._alarm_pan = None if danger is None else alarm_pan(danger.lateral_meters, config)
        elif not alarm:
            self._alarm_pan = None
        ear_left, ear_right = ear_gains(heading, alarm, self._alarm_pan, config)

        log.debug(
            "planner %.1f ms: field %.1f, dp %.1f, info %.1f, %d groups, cost %.2f bits, heading %.1f deg, "
            "information %.2f bits, avoidance %.2f bits, alarm %s",
            (after_information - started) * 1000,
            (after_field - started) * 1000,
            (after_plan - after_field) * 1000,
            (after_information - after_plan) * 1000,
            obstacles.groups_in_view,
            cost_bits,
            np.degrees(heading),
            information,
            avoidance,
            alarm,
        )

        return PlannedPath(
            timestamp_seconds=obstacles.timestamp_seconds,
            times_seconds=self._times.copy(),
            lateral_offsets_meters=np.asarray(offsets, dtype=np.float64),
            lookahead_heading_radians=heading,
            alarm=alarm,
            cumulative_cost_bits=cost_bits,
            scene_information_bits=information,
            avoidance_surprise_bits=avoidance,
            alarm_pan=self._alarm_pan,
            ear_gain_left=ear_left,
            ear_gain_right=ear_right,
        )
