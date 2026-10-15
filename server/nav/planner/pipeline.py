"""An ObstacleSet in, a PlannedPath out. Field, goal prior, previous-plan prior, dynamic program, heading, alarm."""

# Standard library imports
import logging
import time

# Third party imports
import numpy as np

# Local package imports
from nav.planner.alarm import (
    AlarmHold,
    alarm_raised,
    avoidance_surprise_nats,
    check_alarm_config,
    corridor_time_to_contact,
    nearest_corridor_point,
)
from nav.planner.attention import (
    AttentionCredit,
    alarm_audible,
    check_attention_config,
    danger_object,
    gaze_on_outlines,
    tolerance_radians,
)
from nav.planner.audio import alarm_pan, check_audio_config, ear_gains
from nav.planner.config import GoalSwitches, PlannerConfig, goal_switches_of
from nav.planner.contact import ContactSurprise, check_contact_config
from nav.planner.dynamic_programming import plan, start_cell_index
from nav.planner.field import CostTerm, cost_field, lateral_grid, step_count
from nav.planner.field_terms import GOAL, PREVIOUS_PLAN, SCENE, TermField, charges_left_out, last_row_only, sum_terms
from nav.planner.goal import RouteGoal, check_goal_config, goal_for_frame, goal_term, widened_goal_tolerance
from nav.planner.heading import lookahead_heading, lookahead_step_index
from nav.planner.information import scene_information_bits
from nav.planner.previous_plan import PreviousPlanPrior, check_previous_plan_config
from nav.planner.surprise import CollisionSurprise
from nav.planner.units import bits_from_nats
from nav.types import GazeSamples, ObstacleSet, PlannedPath
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
        check_goal_config(config)
        self._terms = planner_terms(config)
        self._alarm_hold = AlarmHold(config.alarm_hold_seconds)
        # Checked at construction even when off, so switching it on can't start a run with a bad constant.
        check_previous_plan_config(config)
        self._previous_plan = PreviousPlanPrior(config, self._times) if config.previous_plan_prior_enabled else None
        self._last_field: np.ndarray | None = None
        check_attention_config(config)
        self._attention_tolerance = tolerance_radians(config.attention_tolerance_degrees)
        self._attention = AttentionCredit(config.attention_credit_seconds, config.attention_match_meters, config.walking_speed_mps)
        # Whether the eyes are on the danger, kept through the hold the way the pan is.
        self._danger_noticed = False
        # The run's switch. Read once a frame, so a page flipping it changes the next plan's sound.
        self._gaze_silences_alarm = config.gaze_silences_alarm
        # Which of the goal's sources are on. Run-wide switches too, set by the loop before each plan.
        self._goal_switches = goal_switches_of(config)
        # Whether this alarm has been under the attention floor since it went up. Once it has, it stays
        # heard until it clears, so a near danger dropping out of the picture can't quiet it.
        self._alarm_reached_floor = False

    @property
    def gaze_silences_alarm(self) -> bool:
        return self._gaze_silences_alarm

    def set_gaze_silences_alarm(self, enabled: bool) -> None:
        """Turn the gaze switch on or off for every plan from here on. The path itself never changes with it."""
        self._gaze_silences_alarm = enabled

    @property
    def goal_switches(self) -> GoalSwitches:
        return self._goal_switches

    def set_goal_switches(self, switches: GoalSwitches) -> None:
        """Choose the goal's sources for every plan from here on."""
        self._goal_switches = switches

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def last_field(self) -> np.ndarray | None:
        """The field from the most recent plan: every cost term, the goal term on its last row, and the previous-plan prior.

        Every row is a rate here, the goal row included. The plan was made from a copy of this in which each
        term is charged as field_terms declares it, so the goal's row is divided by the time step and paid
        once as a cost. Everything else in that copy is the same.

        For looking at, not for planning with. The runtime hands it to a DebugSink, and the evaluation's
        band breakdown reads it through replay to split a frame's cost by term. Nothing on the live path
        reads it, so nothing there grows a dependency on a grid the size of the horizon.
        """
        return self._last_field

    def plan(
        self,
        obstacles: ObstacleSet,
        start_lateral_meters: float = 0.0,
        gaze_ground_point: np.ndarray | None = None,
        gaze: GazeSamples | None = None,
        route_goal: RouteGoal | None = None,
    ) -> PlannedPath:
        """
        Choose the path for this frame.

        :param obstacles: This frame's groups, walker relative.
        :param start_lateral_meters: Where the walker is on the grid. Zero, the walker is the origin.
        :param gaze_ground_point: The gaze on the floor, read when the gaze goal is on.
        :param gaze: The gaze samples the frame carried, for whether the alarm is heard. None without gaze.
        :param route_goal: The route layer's goal this frame, read when the route goal is on.
        :return: The chosen path, its heading at the lookahead, and the alarm.
        :rtype: PlannedPath
        """
        config = self._config
        started = time.perf_counter()

        scene_field = cost_field(obstacles, self._grid, config, self._walker, self._terms)
        # The goal gives way to something in the corridor: its spread widens with the course's avoidance
        # surprise, so the sooner contact is, the less ending off the goal's line costs.
        avoidance_nats = avoidance_surprise_nats(obstacles, config)
        goal, combined_tolerance = goal_for_frame(self._goal_switches, gaze_ground_point, route_goal, config)
        steps = len(self._times)
        goal_row = goal_term(self._grid, goal, widened_goal_tolerance(combined_tolerance, config, avoidance_nats))
        # With nothing in view the corridor is empty, so the prior's goal is the same goal at its unwidened
        # tolerance. The goal giving way to something seen is the scene's doing and counts as information.
        prior_goal_row = goal_term(self._grid, goal, widened_goal_tolerance(combined_tolerance, config, 0.0))
        terms = [
            TermField(SCENE, scene_field, None),
            TermField(GOAL, last_row_only(goal_row, steps), last_row_only(prior_goal_row, steps)),
        ]
        if self._previous_plan is not None:
            previous_plan_field = self._previous_plan.field(self._grid, obstacles.timestamp_seconds, goal)
            if previous_plan_field is not None:
                # The walker's memory, not something seen this frame, so the prior keeps it too and
                # holding a side doesn't count as scene information.
                terms.append(TermField(PREVIOUS_PLAN, previous_plan_field, previous_plan_field))
        summed = sum_terms(terms, config)
        self._last_field = summed.kept
        after_field = time.perf_counter()

        offsets, cost = plan(summed.planned, start_lateral_meters, self._grid, config)
        after_plan = time.perf_counter()
        # The user model's work figure reads this cost. The walker's own priors are not work the scene
        # imposed, so their charges along the chosen path come off, one by one in the terms' order.
        for charge in charges_left_out(terms, offsets, self._grid, config):
            cost -= charge
        # Everything above sums natural logs. Here they become the bits the course quotes, once, after
        # the priors' charges came off in the same units. Nothing before this line is in bits.
        cost_bits = float(bits_from_nats(cost))
        if self._previous_plan is not None:
            self._previous_plan.remember(offsets, obstacles.timestamp_seconds, goal)

        # The KL comparison's prior is the planner with nothing in view: every term's declared prior.
        start_cell = start_cell_index(self._grid, start_lateral_meters)
        information = scene_information_bits(summed.planned, summed.prior, start_cell, self._grid, config, self._lookahead_index)
        avoidance = float(bits_from_nats(avoidance_nats))
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
        audible = self._attend(obstacles, gaze, alarm)
        # The danger's ear drops only while the alarm is heard, so a quieted alarm leaves the steering
        # cue as it is. ear_gains calls its argument alarm, and it is handed the heard one on purpose.
        ear_left, ear_right = ear_gains(heading, audible, self._alarm_pan, config)

        log.debug(
            "planner %.1f ms: field %.1f, dp %.1f, info %.1f, %d groups, cost %.2f bits, heading %.1f deg, "
            "information %.2f bits, avoidance %.2f bits, alarm %s, eyes on danger %s, heard %s",
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
            self._danger_noticed,
            audible,
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
            danger_noticed=self._danger_noticed,
            alarm_audible=audible,
        )

    def _attend(self, obstacles: ObstacleSet, gaze: GazeSamples | None, alarm: bool) -> bool:
        """
        Decide whether the eyes are on the danger and whether the alarm is heard. Runs every frame, switch or not.

        :return: Whether the alarm is heard this frame.
        :rtype: bool
        """
        config = self._config
        danger = nearest_corridor_point(obstacles, config)
        looked = False
        if danger is not None and gaze is not None:
            pieces = danger_object(danger, obstacles.points, config.attention_link_meters, config.attention_cap_meters)
            looked = gaze_on_outlines(
                gaze.pixels,
                gaze.offsets_seconds,
                config.attention_window_seconds,
                gaze.intrinsics,
                [piece.camera_outline for piece in pieces],
                self._attention_tolerance,
                config.attention_dwell_seconds,
            )
        # No gaze is not looked, which keeps the alarm heard. Unknown fails safe.
        noticed_now = self._attention.update(danger, looked, obstacles.timestamp_seconds)
        if danger is not None:
            self._danger_noticed = noticed_now
        elif not alarm:
            self._danger_noticed = False
        # Otherwise the hold with an empty corridor, which keeps the last answer like the pan.
        contact = corridor_time_to_contact(obstacles, config)
        if not alarm:
            self._alarm_reached_floor = False
        elif contact is not None and contact < config.attention_floor_seconds:
            self._alarm_reached_floor = True
        # The hold, or a farther danger looked at during it, never quiets an alarm that was under the floor.
        effective_contact = 0.0 if self._alarm_reached_floor else contact
        return alarm_audible(alarm, self._danger_noticed, effective_contact, self._gaze_silences_alarm, config)
