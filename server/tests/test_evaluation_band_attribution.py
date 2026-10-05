"""
The band breakdown, on scenes built by hand with the answers worked out here.

The command test writes a synthetic walk through the real tap, as test_evaluation_replay.py does.
"""

# Standard library imports
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
# The breakdown and the four tests
# *******************************************


def test_the_term_breakdown_adds_up_to_the_plans_cost() -> None:
    frame = replayed(PINNED_BAND)[0]
    costs = path_term_costs(term_fields(frame, frame.input.obstacles, PLANNER, WALKER, GoalMode.AHEAD), frame.path.lateral_offsets_meters, PLANNER)
    assert sum(costs.values()) == pytest.approx(frame.path.cumulative_cost_bits, rel=1e-9)


def test_the_breakdown_adds_up_on_a_frame_with_a_previous_plan() -> None:
    # The second frame carries the prior toward the first frame's plan, so it is in the sum too.
    inputs = [PlannerInput(time, ObstacleSet(time, PINNED_BAND.points, PINNED_BAND.groups_in_view), None, None, None, None) for time in (0.0, 1.0 / 30.0)]
    frame = replayed_frames(inputs, PLANNER, WALKER, GoalMode.AHEAD, keep_fields=True)[1]
    fields = term_fields(frame, frame.input.obstacles, PLANNER, WALKER, GoalMode.AHEAD)
    assert np.any(fields["previous plan"] != 0.0)
    costs = path_term_costs(fields, frame.path.lateral_offsets_meters, PLANNER)
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
    # With the phone along the walker's travel, turning into the travel frame changes nothing.
    assert PinCandidate.PHONE_POINTING not in frame.explained


def test_attribution_counts_overlapping_candidates_in_each() -> None:
    def frame(explained: set[PinCandidate], unknown: set[PinCandidate] = frozenset()) -> FrameAttribution:
        return FrameAttribution(0.0, 3.5, 35.5, {}, frozenset(explained), frozenset(unknown))

    frames = [
        frame({PinCandidate.WALL_CELLS, PinCandidate.GOAL_LAST_ROW}),
        frame({PinCandidate.PHONE_POINTING}),
        frame({PinCandidate.PHONE_POINTING, PinCandidate.WALL_CELLS}),
        frame(set()),
        frame(set(), {PinCandidate.PHONE_POINTING}),
    ]
    summary = summarize_attribution(frames, band_frames=20)
    assert summary.explained_by == {PinCandidate.PHONE_POINTING: 2, PinCandidate.WALL_CELLS: 2, PinCandidate.GOAL_LAST_ROW: 1, PinCandidate.RIGHT_PLAN: 0}
    assert summary.only_on_hold == 2
    assert summary.fixable == 1
    assert summary.several == 2
    assert summary.unexplained == 1
    assert summary.unknown_by[PinCandidate.PHONE_POINTING] == 1
    assert summary.pinned_band_frames == 5


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
