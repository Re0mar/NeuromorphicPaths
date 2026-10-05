"""
The band breakdown, on scenes built by hand with the answers worked out here.

The command test writes a synthetic walk through the real tap, as test_evaluation_replay.py does.
"""

# Standard library imports
import dataclasses
import math
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation.band_attribution import (
    FrameAttribution,
    PinCandidate,
    band_attribution,
    path_term_costs,
    sidestep_is_the_right_plan,
    summarize_attribution,
    term_fields,
    turned_into_travel,
    walls_joined,
)
from nav.evaluation.check_planner import main
from nav.evaluation.config import PlannerNumbersConfig
from nav.evaluation.planner_numbers import PlannerInput
from nav.evaluation.replay import replayed_frames
from nav.planner.config import GoalMode, PlannerConfig
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig
from synthetic_walks import write_recording

PLANNER = PlannerConfig()
CONFIG = PlannerNumbersConfig()
WALKER = WalkerConfig()
CELL = 0.25


def point(lateral: float, forward: float, group: int, is_wall: bool = False) -> ObstaclePoint:
    return ObstaclePoint(lateral, forward, group, float(np.hypot(lateral, forward)) - WALKER.radius_meters, 0.03, None, None, is_wall, np.zeros(3))


def scene(*points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(0.0, points, len(points))


def wall(start: float, stop: float, forward: float, first_group: int = 0) -> list[ObstaclePoint]:
    return [point(float(x), forward, first_group + index, is_wall=True) for index, x in enumerate(np.arange(start, stop + 1e-9, CELL))]


def replayed(obstacles: ObstacleSet):
    return replayed_frames([PlannerInput(0.0, obstacles, None, None, None, None)], PLANNER, WALKER, GoalMode.AHEAD, keep_fields=True)


# A wall from 3 m left to 1 m right, 4 m ahead: the nearest corridor point is in the band, and the
# planner's arrow goes to its limit for the open floor on the right.
PINNED_BAND = scene(*wall(-3.0, 1.0, 4.0))


# *******************************************
# The breakdown and the five tests
# *******************************************


def test_the_term_breakdown_adds_up_to_the_plans_cost() -> None:
    frame = replayed(PINNED_BAND)[0]
    costs = path_term_costs(term_fields(frame, frame.input.obstacles, PLANNER, WALKER, GoalMode.AHEAD), frame.path.lateral_offsets_meters, PLANNER)
    # The plan's cost leaves out the previous-plan prior, which a first frame doesn't have anyway.
    assert costs["previous plan"] == pytest.approx(0.0, abs=1e-12)
    assert sum(costs.values()) == pytest.approx(frame.path.cumulative_cost_bits, rel=1e-9)


def test_the_breakdown_adds_up_on_a_frame_with_a_previous_plan() -> None:
    # The open side moves from the right to the left between the frames, so the second plan crosses over and
    # the prior toward it charges something along the chosen path. The plan's cost leaves that charge
    # out, as the user model's work figure needs, and the breakdown shows it as its own term.
    first = ObstacleSet(0.0, PINNED_BAND.points, PINNED_BAND.groups_in_view)
    switched = scene(*wall(-1.0, 3.0, 4.0))
    second = ObstacleSet(1.0 / 30.0, switched.points, switched.groups_in_view)
    inputs = [PlannerInput(rows.timestamp_seconds, rows, None, None, None, None) for rows in (first, second)]
    frame = replayed_frames(inputs, PLANNER, WALKER, GoalMode.AHEAD, keep_fields=True)[1]
    fields = term_fields(frame, frame.input.obstacles, PLANNER, WALKER, GoalMode.AHEAD)
    costs = path_term_costs(fields, frame.path.lateral_offsets_meters, PLANNER)
    assert costs["previous plan"] > 0.0
    without_prior = sum(value for name, value in costs.items() if name != "previous plan")
    assert without_prior == pytest.approx(frame.path.cumulative_cost_bits, rel=1e-9)


def test_with_the_contact_term_off_the_breakdown_charges_nothing_to_contact() -> None:
    without_contact = dataclasses.replace(PLANNER, contact_term_enabled=False)
    frame = replayed_frames([PlannerInput(0.0, PINNED_BAND, None, None, None, None)], without_contact, WALKER, GoalMode.AHEAD, keep_fields=True)[0]
    fields = term_fields(frame, frame.input.obstacles, without_contact, WALKER, GoalMode.AHEAD)
    # A first frame has no prior, so whatever is left of the field after the other terms must be nothing.
    # Taking a contact term the planner never added would leave minus that term here, over the whole grid.
    assert np.max(np.abs(fields["previous plan"])) == pytest.approx(0.0, abs=1e-9)
    # The same holds in a re-plan, which rebuilds the real scene's terms beside the changed one.
    joined = term_fields(frame, walls_joined(frame.input.obstacles, CELL), without_contact, WALKER, GoalMode.AHEAD)
    assert np.max(np.abs(joined["previous plan"])) == pytest.approx(0.0, abs=1e-9)
    costs = path_term_costs(fields, frame.path.lateral_offsets_meters, without_contact)
    assert costs["contact"] == 0.0
    assert sum(costs.values()) == pytest.approx(frame.path.cumulative_cost_bits, rel=1e-9)


def test_an_obstacle_too_close_to_clear_in_time_needs_the_full_sidestep() -> None:
    # A post 0.7 m ahead takes 0.5 s to reach. Clearing it takes 0.30 m, 0.6 m/s, so under the limit
    # on its own, but a wall beside it pushes the nearest gap 0.6 m out: 1.2 m/s, past the 1.0 m/s a
    # walker can sidestep. Nothing gentler than the full sidestep gets there.
    obstacles = scene(point(0.0, 0.7, 1), point(-0.25, 0.7, 2), point(0.25, 0.7, 3), point(-0.5, 0.7, 4))
    assert sidestep_is_the_right_plan(obstacles, PLANNER, CONFIG)


def test_a_wide_obstacle_close_ahead_needs_the_full_sidestep() -> None:
    # A wall 3 m wide, 1.5 m ahead. The body clears it only past 1.5 + 0.30 = 1.8 m to either side,
    # with 1.5 / 1.4 = 1.07 s to get there, which is faster than the 1.0 m/s sidestep allows.
    assert sidestep_is_the_right_plan(scene(*wall(-1.5, 1.5, 1.5)), PLANNER, CONFIG)


def test_a_post_off_to_the_side_far_out_is_not_a_right_plan() -> None:
    # 0.4 m to the side already clears the 0.30 m body, so walking straight on is the slowest clearing path.
    assert not sidestep_is_the_right_plan(scene(point(0.4, 4.0, 1)), PLANNER, CONFIG)


def test_a_post_dead_ahead_in_the_band_needs_only_a_gentle_sidestep() -> None:
    # 0.30 m to clear in 4 / 1.4 = 2.86 s is 0.105 m/s, and a second ahead that reads as about 4.3
    # degrees, far from the 35.5 degree limit.
    assert not sidestep_is_the_right_plan(scene(point(0.0, 4.0, 1)), PLANNER, CONFIG)


def test_a_wall_split_into_cells_is_joined_into_one_group() -> None:
    joined = walls_joined(scene(*wall(-0.5, 0.5, 4.0, first_group=10), point(2.0, 2.0, 99)), CELL)
    wall_groups = {p.group_id for p in joined.points if p.is_wall}
    assert wall_groups == {10}
    assert {p.group_id for p in joined.points if not p.is_wall} == {99}
    assert joined.groups_in_view == 2


def test_rotating_the_scene_into_the_travel_direction_moves_the_obstacles_by_the_offset() -> None:
    # Travel 30 degrees right of the phone. A point straight ahead of the phone 3 m out is, to the
    # walker, 3 sin 30 = 1.5 m to the left and 3 cos 30 = 2.60 m ahead.
    turned = turned_into_travel(scene(point(0.0, 3.0, 1)), math.radians(30.0))
    assert turned.points[0].lateral_meters == pytest.approx(-1.5)
    assert turned.points[0].forward_meters == pytest.approx(3.0 * math.cos(math.radians(30.0)))


def test_a_pinned_band_frame_is_attributed_with_its_terms() -> None:
    attribution = band_attribution(replayed(PINNED_BAND), lambda time: 0.0, PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    assert attribution.band_frames == 1
    assert attribution.pinned_band_frames == 1
    frame = attribution.frames[0]
    assert set(frame.term_difference) == {"surprise", "contact", "goal", "previous plan", "kinetic"}
    # The sidestep costs kinetic effort, so the chosen path pays more there than walking straight.
    assert frame.term_difference["kinetic"] > 0.0
    # 17 wall cells each count as their own obstacle, and joined into one the open right side stops
    # being worth the limit. Spreading the goal over every row does the same. With the phone along the
    # walker's travel, turning into the travel frame changes nothing, and a first frame has no prior.
    assert frame.explained == {PinCandidate.WALL_CELLS, PinCandidate.GOAL_LAST_ROW}


def test_a_phone_turned_off_the_walking_direction_explains_the_frame() -> None:
    # Turned 40 degrees, the wall leaves the open floor ahead of the walker, so the arrow no longer
    # needs its limit.
    attribution = band_attribution(replayed(PINNED_BAND), lambda time: math.radians(40.0), PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    assert PinCandidate.PHONE_POINTING in attribution.frames[0].explained


def test_a_sidestep_held_by_the_previous_plan_is_explained_by_it() -> None:
    # First frame pins right. The second wall is 3 m wide and centered, so either side clears it and
    # the prior is what keeps the plan at its limit on the right.
    centered = scene(*wall(-1.5, 1.5, 4.0))
    later = ObstacleSet(1.0 / 30.0, centered.points, centered.groups_in_view)
    inputs = [PlannerInput(rows.timestamp_seconds, rows, None, None, None, None) for rows in (PINNED_BAND, later)]
    frames = replayed_frames(inputs, PLANNER, WALKER, GoalMode.AHEAD, keep_fields=True)
    first, second = band_attribution(frames, lambda time: 0.0, PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG).frames
    assert PinCandidate.PREVIOUS_PLAN not in first.explained
    assert PinCandidate.PREVIOUS_PLAN in second.explained


def test_attribution_counts_overlapping_candidates_in_each() -> None:
    def frame(explained: set[PinCandidate], unknown: set[PinCandidate] = frozenset()) -> FrameAttribution:
        return FrameAttribution(0.0, 3.5, 35.5, {}, frozenset(explained), frozenset(unknown))

    frames = [
        frame({PinCandidate.WALL_CELLS, PinCandidate.GOAL_LAST_ROW}),
        frame({PinCandidate.PHONE_POINTING}),
        frame({PinCandidate.PHONE_POINTING, PinCandidate.WALL_CELLS}),
        frame(set()),
        frame(set(), {PinCandidate.PHONE_POINTING}),
        frame({PinCandidate.RIGHT_PLAN, PinCandidate.PREVIOUS_PLAN}),
    ]
    summary = summarize_attribution(frames, band_frames=20)
    assert summary.explained_by == {
        PinCandidate.PHONE_POINTING: 2,
        PinCandidate.WALL_CELLS: 2,
        PinCandidate.GOAL_LAST_ROW: 1,
        PinCandidate.RIGHT_PLAN: 1,
        PinCandidate.PREVIOUS_PLAN: 1,
    }
    assert summary.only_on_hold == 2
    # Right plan and the previous plan are the planner working as built, so they aren't fixable.
    assert summary.fixable == 1
    assert summary.several == 3
    assert summary.unexplained == 1
    assert summary.unknown_by[PinCandidate.PHONE_POINTING] == 1
    assert summary.pinned_band_frames == 6


def test_band_command_prints_the_attribution_on_a_recording(tmp_path: Path, capsys) -> None:
    recording = write_recording(tmp_path / "straight", lambda time: 0.0, 22.0)
    code = main(["band", str(recording)])
    out = capsys.readouterr().out
    assert code == 0
    assert "band frames have the arrow at its limit" in out


# *******************************************
# Refusals
# *******************************************


def test_an_unknown_travel_direction_is_unknown_not_explained() -> None:
    attribution = band_attribution(replayed(PINNED_BAND), lambda time: None, PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    frame = attribution.frames[0]
    assert PinCandidate.PHONE_POINTING in frame.unknown
    assert PinCandidate.PHONE_POINTING not in frame.explained
    assert attribution.unknown_by[PinCandidate.PHONE_POINTING] == 1


def test_a_phone_pointing_sideways_is_untested_not_explained() -> None:
    # Past 90 degrees the phone points sideways or behind, which is the track's heading gone wrong.
    # At 60 degrees the same scene is explained by the phone, so the cap is what changes the answer.
    turned = band_attribution(replayed(PINNED_BAND), lambda time: math.radians(100.0), PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    assert PinCandidate.PHONE_POINTING in turned.frames[0].unknown
    assert PinCandidate.PHONE_POINTING not in turned.frames[0].explained
    within = band_attribution(replayed(PINNED_BAND), lambda time: math.radians(60.0), PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    assert PinCandidate.PHONE_POINTING in within.frames[0].explained


def test_a_frame_outside_the_band_is_not_attributed() -> None:
    attribution = band_attribution(replayed(scene(point(0.0, 2.0, 1))), lambda time: 0.0, PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    assert attribution.band_frames == 0
    assert attribution.frames == []


def test_wall_joining_never_joins_cells_that_do_not_touch() -> None:
    # Two wall runs with one empty cell between them, as a doorway leaves: they stay two groups.
    runs = wall(-1.0, -0.5, 3.0, first_group=0) + wall(0.0, 0.5, 3.0, first_group=10)
    joined = walls_joined(scene(*runs), CELL)
    assert len({p.group_id for p in joined.points}) == 2


def test_term_fields_refuses_a_frame_without_its_field() -> None:
    frame = replayed_frames([PlannerInput(0.0, PINNED_BAND, None, None, None, None)], PLANNER, WALKER, GoalMode.AHEAD)[0]
    with pytest.raises(ValueError, match="replayed without its field"):
        term_fields(frame, frame.input.obstacles, PLANNER, WALKER, GoalMode.AHEAD)


def test_the_attribution_counts_walls_over_every_band_frame() -> None:
    attribution = band_attribution(replayed(PINNED_BAND), lambda time: 0.0, PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    assert attribution.band_frames_with_walls == 1
    assert attribution.band_wall_points == len(PINNED_BAND.points) == attribution.band_points


def test_a_frame_keeps_its_phone_offset_and_whether_something_sits_beside_it() -> None:
    attribution = band_attribution(replayed(PINNED_BAND), lambda time: math.radians(10.0), PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG)
    frame = attribution.frames[0]
    assert frame.phone_offset_degrees == pytest.approx(10.0)
    # Every point of this wall is 4 m out, so nothing nearer than 3 m sits beside the line.
    assert frame.near_beside is False
    beside = scene(*wall(-3.0, 1.0, 4.0), point(0.8, 2.0, 99))
    beside_frame = band_attribution(replayed(beside), lambda time: 0.0, PLANNER, WALKER, GoalMode.AHEAD, CELL, CONFIG).frames
    assert beside_frame and beside_frame[0].near_beside is True
