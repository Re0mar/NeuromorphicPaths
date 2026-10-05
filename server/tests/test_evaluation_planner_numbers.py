"""
The planner's whole-walk numbers, on frames built by hand with answers worked out here.

Nothing here needs a recording from frame_logs/. The command tests write a synthetic walk through the
real tap, as test_evaluation_replay.py does.
"""

# Standard library imports
import dataclasses
import math
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation import replay as replay_module
from nav.evaluation.check_planner import format_numbers, main
from nav.evaluation.config import PlannerNumbersConfig
from nav.evaluation.planner_numbers import (
    ClearanceBand,
    PairCause,
    PlannerNumbers,
    PlannerInput,
    ReplayedFrame,
    clearance_band,
    disagreement_pairs,
    horizon_reach_meters,
    is_pinned,
    in_restated_band,
    nearest_beside_clearance,
    nearest_heading_clearance,
    pair_cause,
    plan_disagreement,
    plan_forward_distances,
    share,
    sidestep_limit_degrees,
    whole_walk_numbers,
)
from nav.evaluation.replay import replayed_frames
from nav.planner.alarm import alarm_raised
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.types import ObstaclePoint, ObstacleSet, PlannedPath
from nav.walker import WalkerConfig
from synthetic_walks import write_recording

PLANNER = PlannerConfig()
CONFIG = PlannerNumbersConfig()
STEPS = len(plan_forward_distances(PLANNER))
ORIGIN = np.zeros(3)
LATERAL = np.array([1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])


def point(lateral: float, forward: float, clearance: float, group: int = 1, noise: float = 0.05) -> ObstaclePoint:
    return ObstaclePoint(lateral, forward, group, clearance, noise, None, None, False, np.zeros(3))


def planned_input(
    time: float,
    points: tuple[ObstaclePoint, ...] = (),
    origin: np.ndarray | None = ORIGIN,
    axes: tuple[np.ndarray, np.ndarray] = (LATERAL, FORWARD),
) -> PlannerInput:
    has_world = origin is not None
    return PlannerInput(
        timestamp_seconds=time,
        obstacles=ObstacleSet(time, points, len(points)),
        origin=origin,
        lateral_axis=axes[0] if has_world else None,
        forward_axis=axes[1] if has_world else None,
        gaze_ground_point=None,
    )


def frame(
    time: float,
    heading_degrees: float = 0.0,
    offsets: np.ndarray | float = 0.0,
    alarm: bool = False,
    raised: bool = False,
    points: tuple[ObstaclePoint, ...] = (),
    origin: np.ndarray | None = ORIGIN,
    axes: tuple[np.ndarray, np.ndarray] = (LATERAL, FORWARD),
) -> ReplayedFrame:
    lateral = np.broadcast_to(np.asarray(offsets, dtype=np.float64), (STEPS,)).copy()
    path = PlannedPath(time, np.arange(STEPS) * PLANNER.time_step_seconds, lateral, math.radians(heading_degrees), alarm, 0.0)
    return ReplayedFrame(planned_input(time, points, origin, axes), path, raised)


def turned_left(degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """The walker frame's lateral and forward axes after turning left by this much."""
    angle = math.radians(degrees)
    return math.cos(angle) * LATERAL + math.sin(angle) * FORWARD, math.cos(angle) * FORWARD - math.sin(angle) * LATERAL


def run(arguments: list, capsys) -> tuple[int, str, str]:
    code = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    """A straight 22 s walk past the synthetic scene's box."""
    return write_recording(tmp_path_factory.mktemp("walk") / "straight", lambda time: 0.0, 22.0)


# *******************************************
# The three borrowed values and the bands
# *******************************************


def test_the_sidestep_limit_is_the_full_sidestep_at_walking_pace() -> None:
    assert sidestep_limit_degrees(PLANNER) == pytest.approx(math.degrees(math.atan2(1.0, 1.4)))
    assert round(sidestep_limit_degrees(PLANNER), 1) == 35.5


def test_a_heading_within_tolerance_of_the_limit_is_pinned_on_either_side() -> None:
    limit = sidestep_limit_degrees(PLANNER)
    assert is_pinned(limit + 0.49, limit, CONFIG)
    assert is_pinned(limit - 0.49, limit, CONFIG)
    assert is_pinned(-limit, limit, CONFIG)
    assert not is_pinned(limit - 0.51, limit, CONFIG)
    assert not is_pinned(0.0, limit, CONFIG)


def test_band_edges_match_the_scorecard() -> None:
    reach = 1.4 * 3.8
    assert horizon_reach_meters(PLANNER) == pytest.approx(reach)
    assert clearance_band(-0.2, PLANNER, CONFIG) is ClearanceBand.CLOSE
    assert clearance_band(0.99, PLANNER, CONFIG) is ClearanceBand.CLOSE
    assert clearance_band(1.0, PLANNER, CONFIG) is ClearanceBand.NEAR
    assert clearance_band(3.0, PLANNER, CONFIG) is ClearanceBand.BAND
    assert clearance_band(reach - 1e-9, PLANNER, CONFIG) is ClearanceBand.BAND
    assert clearance_band(reach, PLANNER, CONFIG) is ClearanceBand.CLEAR
    assert clearance_band(float("inf"), PLANNER, CONFIG) is ClearanceBand.CLEAR


def test_only_points_ahead_in_the_heading_corridor_count() -> None:
    obstacles = ObstacleSet(0.0, (point(0.51, 2.0, 0.4), point(0.0, -0.1, 0.2), point(-0.3, 4.0, 3.5)), 3)
    assert nearest_heading_clearance(obstacles, CONFIG) == 3.5
    assert nearest_heading_clearance(ObstacleSet(0.0, (), 0), CONFIG) == float("inf")


def test_the_heading_corridor_is_wider_than_the_alarms() -> None:
    # 0.40 m to the side is outside the alarm's 0.30 m body and inside the heading's 0.50 m corridor.
    obstacles = ObstacleSet(0.0, (point(0.40, 3.0, 2.5), point(0.1, 5.0, 4.5)), 2)
    assert nearest_heading_clearance(obstacles, CONFIG) == 2.5


def test_whole_walk_numbers_counts_pinned_frames_per_band() -> None:
    limit = sidestep_limit_degrees(PLANNER)
    band_point = (point(0.0, 4.0, 3.5),)
    frames = [
        frame(0.0, limit, points=band_point),
        frame(0.1, 0.0, points=band_point),
        frame(0.2, -limit),  # clear, pinned
        frame(0.3, 10.0, points=(point(0.0, 1.0, 0.5),)),  # close, not pinned
    ]
    numbers = whole_walk_numbers(frames, PLANNER, CONFIG)
    assert numbers.frames_by_band == {ClearanceBand.CLOSE: 1, ClearanceBand.NEAR: 0, ClearanceBand.BAND: 2, ClearanceBand.CLEAR: 1}
    assert numbers.pinned_by_band == {ClearanceBand.CLOSE: 0, ClearanceBand.NEAR: 0, ClearanceBand.BAND: 1, ClearanceBand.CLEAR: 1}
    assert numbers.pinned_frames == 2


def test_distinct_headings_count_at_two_decimals() -> None:
    frames = [frame(0.0, 10.001), frame(0.1, 10.004), frame(0.2, 10.006)]
    assert whole_walk_numbers(frames, PLANNER, CONFIG).distinct_headings == 2


# *******************************************
# The alarm
# *******************************************


def test_alarm_changes_count_every_flip() -> None:
    frames = [frame(0.0, alarm=True), frame(0.1, alarm=True), frame(0.2, alarm=False), frame(0.3, alarm=True)]
    numbers = whole_walk_numbers(frames, PLANNER, CONFIG)
    assert numbers.alarm_changes == 2
    assert numbers.alarm_on_frames == 3


def test_raise_decisions_past_close_are_counted() -> None:
    near = (point(0.0, 0.8, 0.5),)
    far = (point(0.0, 2.0, 1.6),)
    frames = [frame(0.0, raised=True, points=near), frame(0.1, raised=True, points=far), frame(0.2, raised=False, points=far)]
    assert whole_walk_numbers(frames, PLANNER, CONFIG).raised_past_close == 1


def test_longest_hold_is_measured_from_the_last_raise() -> None:
    frames = [
        frame(0.0, alarm=True, raised=True),
        frame(0.1, alarm=True, raised=True),
        frame(0.3, alarm=True, raised=False),
        frame(0.5, alarm=True, raised=False),
        frame(0.6, alarm=False, raised=False),
    ]
    assert whole_walk_numbers(frames, PLANNER, CONFIG).longest_hold_seconds == pytest.approx(0.4)


def test_near_noise_reads_heading_corridor_points_within_two_meters() -> None:
    points = (point(0.0, 1.5, 1.0, noise=0.03), point(0.0, 3.0, 2.5, noise=0.9), point(0.8, 1.0, 1.0, noise=0.9))
    numbers = whole_walk_numbers([frame(0.0, points=points)], PLANNER, CONFIG)
    assert numbers.near_noise_median_meters == pytest.approx(0.03)
    assert numbers.near_noise_readings == 1


# *******************************************
# Consecutive plans placed in the world
# *******************************************


def test_identical_plans_from_the_same_place_disagree_by_zero() -> None:
    assert plan_disagreement(frame(0.0, offsets=0.2), frame(0.1, offsets=0.2), plan_forward_distances(PLANNER)) == pytest.approx(0.0)


def test_a_walker_who_moved_forward_with_the_same_straight_plan_disagrees_by_zero() -> None:
    later = frame(0.1, origin=ORIGIN + 0.5 * FORWARD)
    assert plan_disagreement(frame(0.0), later, plan_forward_distances(PLANNER)) == pytest.approx(0.0)


def test_a_plan_shifted_sideways_disagrees_by_the_shift() -> None:
    assert plan_disagreement(frame(0.0), frame(0.1, offsets=0.3), plan_forward_distances(PLANNER)) == pytest.approx(0.3)


def test_disagreement_percentiles_cover_every_pair() -> None:
    frames = [frame(0.1 * index, offsets=0.1 * index) for index in range(11)]
    numbers = whole_walk_numbers(frames, PLANNER, CONFIG)
    assert numbers.disagreement_pairs == 10
    assert numbers.disagreement_median_meters == pytest.approx(0.1)
    assert numbers.disagreement_p90_meters == pytest.approx(0.1)


# *******************************************
# The replay's single planning loop
# *******************************************


def test_replayed_frames_make_the_loops_call() -> None:
    inputs = [planned_input(0.0, (point(0.0, 1.2, 0.85),)), planned_input(0.1, (point(0.0, 1.1, 0.75),))]
    frames = replayed_frames(inputs, PLANNER, WalkerConfig(), GoalMode.AHEAD)
    planner = PlannerPipeline(PLANNER, WalkerConfig())
    for row, replayed in zip(inputs, frames, strict=True):
        direct = planner.plan(row.obstacles, 0.0, GoalMode.AHEAD, row.gaze_ground_point)
        assert replayed.path.first_heading_radians == direct.first_heading_radians
        assert np.array_equal(replayed.path.lateral_offsets_meters, direct.lateral_offsets_meters)
        assert replayed.path.alarm == direct.alarm
        assert replayed.raise_decision == alarm_raised(row.obstacles, PLANNER)
        assert replayed.input is row


def test_a_replayed_frame_holds_the_path_plan_returned(monkeypatch) -> None:
    returned = []
    original = PlannerPipeline.plan

    def recording_plan(self, *arguments):
        path = original(self, *arguments)
        returned.append(path)
        return path

    monkeypatch.setattr(PlannerPipeline, "plan", recording_plan)
    frames = replayed_frames([planned_input(0.0), planned_input(0.1)], PLANNER, WalkerConfig(), GoalMode.AHEAD)
    assert [frame.path for frame in frames] == returned
    assert all(frame.path is path for frame, path in zip(frames, returned))


# *******************************************
# The command, through its real entry point
# *******************************************


def test_numbers_command_prints_every_figure_on_a_recording(recording: Path, capsys) -> None:
    code, out, _ = run(["numbers", recording], capsys)
    assert code == 0
    for line in (
        "nearer than 1.00 m",
        "3.00 to 5.32 m",
        "clear past 5.32 m, or empty",
        "all planned frames",
        "distinct headings",
        "alarm on",
        "raise decisions with the nearest alarm-corridor point past 1.0 m",
        "longest the alarm stayed up",
        "consecutive plans disagree by median",
        "verdicts:",
    ):
        assert line in out
    assert "lateral_kinetic_weight = 6.5" in out


def test_numbers_command_marks_an_override(recording: Path, capsys) -> None:
    code, out, _ = run(["numbers", recording, "--set", "lateral_kinetic_weight=0.055"], capsys)
    assert code == 0
    assert "lateral_kinetic_weight = 0.055  <- --set" in out


# *******************************************
# Refusals
# *******************************************


def test_moving_the_alarm_corridor_moves_no_heading_number() -> None:
    limit = sidestep_limit_degrees(PLANNER)
    side_point = (point(0.25, 1.5, 0.9), point(0.45, 4.0, 3.5))
    frames = [frame(0.1 * index, limit if index % 2 else 5.0, offsets=0.05 * index, raised=True, points=side_point) for index in range(6)]
    wide = whole_walk_numbers(frames, PLANNER, CONFIG)
    narrow = whole_walk_numbers(frames, dataclasses.replace(PLANNER, body_half_width_meters=0.20), CONFIG)
    assert narrow.raised_past_close != wide.raised_past_close
    for name in ("pinned_by_band", "frames_by_band", "pinned_frames", "distinct_headings", "disagreement_median_meters", "disagreement_p90_meters"):
        assert getattr(narrow, name) == getattr(wide, name), name


def test_a_share_over_too_few_frames_is_not_a_result() -> None:
    assert share(50, 99, CONFIG) is None
    assert share(50, 100, CONFIG) == 0.5


def test_no_frames_is_refused() -> None:
    with pytest.raises(ValueError, match="no replayed frames"):
        whole_walk_numbers([], PLANNER, CONFIG)


def test_plans_that_do_not_overlap_give_no_disagreement() -> None:
    later = frame(0.1, origin=ORIGIN + 10.0 * FORWARD)
    assert plan_disagreement(frame(0.0), later, plan_forward_distances(PLANNER)) is None


def test_a_pair_without_a_world_frame_is_counted_not_scored() -> None:
    frames = [frame(0.0), frame(0.1, origin=None), frame(0.2)]
    numbers = whole_walk_numbers(frames, PLANNER, CONFIG)
    assert numbers.pairs_without_world == 2
    assert numbers.disagreement_pairs == 0


def test_plan_disagreement_refuses_a_frame_without_a_world_pose() -> None:
    with pytest.raises(ValueError, match="no world pose"):
        plan_disagreement(frame(0.0), frame(0.1, origin=None), plan_forward_distances(PLANNER))


def test_equal_timestamps_are_counted_as_duplicates() -> None:
    numbers = whole_walk_numbers([frame(0.0), frame(0.0), frame(0.1)], PLANNER, CONFIG)
    assert numbers.duplicate_pairs == 1
    assert numbers.disagreement_pairs == 2


def test_a_close_edge_past_the_band_is_refused() -> None:
    with pytest.raises(ValueError, match="close_meters"):
        PlannerNumbersConfig(close_meters=3.0)


def test_a_target_over_one_is_refused() -> None:
    with pytest.raises(ValueError, match="band_target_share"):
        PlannerNumbersConfig(band_target_share=1.5)


def test_a_fractional_frame_count_is_refused() -> None:
    with pytest.raises(ValueError, match="min_frames_for_a_share must be a whole number"):
        PlannerNumbersConfig(min_frames_for_a_share=99.5)


def test_a_negative_or_zero_corridor_is_refused() -> None:
    with pytest.raises(ValueError, match="heading_corridor_half_width_meters"):
        PlannerNumbersConfig(heading_corridor_half_width_meters=-0.5)
    with pytest.raises(ValueError, match="heading_corridor_half_width_meters must be above zero"):
        PlannerNumbersConfig(heading_corridor_half_width_meters=0.0)


def test_numbers_command_refuses_an_unknown_planner_field(recording: Path, capsys) -> None:
    code, _, err = run(["numbers", recording, "--set", "no_such_field=1"], capsys)
    assert code == 1
    assert "no_such_field" in err


def test_numbers_command_refuses_a_missing_log(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nowhere"
    code, _, err = run(["numbers", missing], capsys)
    assert code == 1
    assert "nowhere" in err


def test_numbers_command_refuses_a_segment_that_does_not_exist(recording: Path, capsys) -> None:
    code, _, err = run(["numbers", recording, "--segment", "4"], capsys)
    assert code == 2
    assert "has 1 segments, so there is no segment 4" in err


def test_numbers_command_with_no_planned_frame_exits_3(tmp_path: Path, capsys) -> None:
    floorless = write_recording(tmp_path / "floorless", lambda time: 0.0, 3.0, floorless_until_seconds=10.0)
    code, _, err = run(["numbers", floorless], capsys)
    assert code == 3
    assert "no planned frame" in err


def test_the_replay_module_plans_in_one_place() -> None:
    # The turn scores and the numbers must come from one loop. planner_pass reaches the planner only
    # through replayed_frames.
    source = Path(replay_module.__file__).read_text(encoding="utf-8")
    assert source.count(".plan(") == 1


# *******************************************
# What a disagreeing pair is
# *******************************************

POST = (point(0.0, 3.0, 2.65),)


def test_plans_on_opposite_sides_of_a_post_both_frames_saw_are_a_side_flip() -> None:
    cause, deciding = pair_cause(frame(0.0, offsets=-0.8, points=POST), frame(0.033, offsets=0.8, points=POST), PLANNER, CONFIG)
    assert cause is PairCause.SIDE_FLIP
    assert deciding == (0.0, 3.0)


def test_a_post_only_the_later_frame_saw_is_a_new_obstacle() -> None:
    cause, deciding = pair_cause(frame(0.0, offsets=-0.8), frame(0.033, offsets=0.8, points=POST), PLANNER, CONFIG)
    assert cause is PairCause.NEW_OBSTACLE
    assert deciding == (0.0, 3.0)


def test_a_phone_that_turned_between_frames_is_an_axis_turn() -> None:
    # The same plan in each walker frame, so lining up the axes leaves nothing to disagree about.
    earlier = frame(0.0, offsets=0.5, points=POST)
    later = frame(0.033, offsets=0.5, points=POST, axes=turned_left(20.0))
    assert plan_disagreement(earlier, later, plan_forward_distances(PLANNER)) > 0.5
    assert pair_cause(earlier, later, PLANNER, CONFIG) == (PairCause.AXIS_TURN, None)


def test_an_origin_jump_faster_than_walking_is_a_frame_jump() -> None:
    later = frame(0.033, offsets=0.8, points=POST, origin=ORIGIN + 1.0 * LATERAL)
    assert pair_cause(frame(0.0, offsets=-0.8, points=POST), later, PLANNER, CONFIG) == (PairCause.FRAME_JUMP, None)


def test_plans_on_the_same_side_that_differ_are_a_same_side_shift() -> None:
    assert pair_cause(frame(0.0, offsets=0.5, points=POST), frame(0.033, offsets=1.2, points=POST), PLANNER, CONFIG) == (PairCause.SAME_SIDE_SHIFT, None)


def test_the_first_matching_cause_wins() -> None:
    # A side flip too, but the frame jumped, and a jump says nothing about the plan.
    later = frame(0.033, offsets=0.8, points=POST, origin=ORIGIN + 0.2 * FORWARD + 1.0 * LATERAL)
    cause, _ = pair_cause(frame(0.0, offsets=-0.8, points=POST), later, PLANNER, CONFIG)
    assert cause is PairCause.FRAME_JUMP


def test_a_phone_turn_with_a_real_flip_stays_a_flip() -> None:
    # The phone turned 5 degrees, but the plan also changed sides, so lining up the axes leaves most of the gap.
    earlier = frame(0.0, offsets=-0.8, points=POST)
    later = frame(0.033, offsets=0.8, points=POST, axes=turned_left(5.0))
    cause, _ = pair_cause(earlier, later, PLANNER, CONFIG)
    assert cause in (PairCause.SIDE_FLIP, PairCause.NEW_OBSTACLE)


def test_disagreement_pairs_keeps_only_pairs_at_or_above_the_percentile() -> None:
    offsets = [0.0, 0.1, 0.1, 0.5, 0.5, 0.6]  # gaps 0.1, 0.0, 0.4, 0.0, 0.1
    frames = [frame(0.033 * index, offsets=value) for index, value in enumerate(offsets)]
    threshold, pairs = disagreement_pairs(frames, PLANNER, CONFIG, 80.0)
    assert threshold == pytest.approx(float(np.percentile([0.1, 0.0, 0.4, 0.0, 0.1], 80.0)))
    assert [round(pair.disagreement_meters, 6) for pair in pairs] == [0.4]
    assert pairs[0].earlier_seconds == pytest.approx(0.066)


def test_a_left_turn_reads_as_a_positive_axis_turn() -> None:
    frames = [frame(0.0, offsets=0.5), frame(0.033, offsets=0.5, axes=turned_left(20.0))]
    _, pairs = disagreement_pairs(frames, PLANNER, CONFIG, 0.0)
    assert pairs[0].axis_turn_degrees == pytest.approx(20.0)


def test_flips_command_prints_the_cause_counts_on_a_recording(recording: Path, capsys) -> None:
    code, out, _ = run(["flips", recording, "--largest", 3], capsys)
    assert code == 0
    assert "th percentile of consecutive-plan disagreement is" in out
    for cause in PairCause:
        assert f"  {cause.value:<16}" in out
    assert "largest, to open in the recording:" in out


def test_a_plan_through_the_point_has_no_side() -> None:
    # The earlier plan passes 0.2 m from the post, inside side_clearance_meters, so it has no side to flip from.
    assert pair_cause(frame(0.0, offsets=-0.2, points=POST), frame(0.033, offsets=0.8, points=POST), PLANNER, CONFIG)[0] is PairCause.SAME_SIDE_SHIFT


def test_a_pair_without_a_world_frame_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="a pair needs both frames placed in the world"):
        pair_cause(frame(0.0), frame(0.033, origin=None), PLANNER, CONFIG)


def test_group_ids_are_not_used_to_match_obstacles() -> None:
    # The same post under two different ids is still the same post.
    renamed = (point(0.0, 3.0, 2.65, group=99),)
    assert pair_cause(frame(0.0, offsets=-0.8, points=POST), frame(0.033, offsets=0.8, points=renamed), PLANNER, CONFIG)[0] is PairCause.SIDE_FLIP
    # And a post elsewhere that happens to share the id is not.
    elsewhere = (point(2.0, 1.0, 1.9, group=1),)
    assert pair_cause(frame(0.0, offsets=-0.8, points=elsewhere), frame(0.033, offsets=0.8, points=POST), PLANNER, CONFIG)[0] is PairCause.NEW_OBSTACLE


def test_disagreement_pairs_refuses_a_percentile_outside_0_to_100() -> None:
    with pytest.raises(ValueError, match="a percentile is from 0 to 100, got 101"):
        disagreement_pairs([frame(0.0), frame(0.1)], PLANNER, CONFIG, 101.0)


def test_flips_command_refuses_a_percentile_outside_0_to_100(recording: Path, capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["flips", str(recording), "--percentile", "150"])
    assert stopped.value.code == 2
    assert "--percentile is from 0 to 100, got 150" in capsys.readouterr().err


# *******************************************
# The restated band
# *******************************************


def test_something_near_just_beside_the_corridor_takes_a_frame_out_of_the_restated_band() -> None:
    # The nearest corridor point is 3.5 m out, so the plain band counts it. A point 0.8 m to the side
    # 2 m out is nearer than 3 m within 1.0 m of the line, so the restated band doesn't.
    beside = ObstacleSet(0.0, (point(0.0, 4.0, 3.5), point(0.8, 2.0, 1.8, group=2)), 2)
    assert clearance_band(nearest_heading_clearance(beside, CONFIG), PLANNER, CONFIG) is ClearanceBand.BAND
    assert nearest_beside_clearance(beside, CONFIG) == 1.8
    assert not in_restated_band(beside, PLANNER, CONFIG)


def test_a_frame_with_its_sides_clear_stays_in_the_restated_band() -> None:
    clear_sides = ObstacleSet(0.0, (point(0.0, 4.0, 3.5), point(1.2, 2.0, 1.9, group=2), point(0.8, 4.0, 3.2, group=3)), 3)
    assert in_restated_band(clear_sides, PLANNER, CONFIG)


def test_the_beside_edges_are_inclusive_at_one_meter_and_at_three_meters() -> None:
    at_the_edge = ObstacleSet(0.0, (point(0.0, 4.0, 3.5), point(1.0, 2.0, 1.8, group=2)), 2)
    assert not in_restated_band(at_the_edge, PLANNER, CONFIG)
    just_past = ObstacleSet(0.0, (point(0.0, 4.0, 3.5), point(1.01, 2.0, 1.8, group=2)), 2)
    assert in_restated_band(just_past, PLANNER, CONFIG)
    exactly_three = ObstacleSet(0.0, (point(0.0, 4.0, 3.5), point(0.8, 3.4, 3.0, group=2)), 2)
    assert in_restated_band(exactly_three, PLANNER, CONFIG)


def test_whole_walk_numbers_keep_the_plain_band_beside_the_restated_one() -> None:
    limit = sidestep_limit_degrees(PLANNER)
    plain_only = (point(0.0, 4.0, 3.5), point(0.8, 2.0, 1.8, group=2))
    both = (point(0.0, 4.0, 3.5),)
    frames = [frame(0.0, limit, points=plain_only), frame(0.1, limit, points=both), frame(0.2, 0.0, points=both)]
    numbers = whole_walk_numbers(frames, PLANNER, CONFIG)
    assert numbers.frames_by_band[ClearanceBand.BAND] == 3
    assert numbers.pinned_by_band[ClearanceBand.BAND] == 2
    assert numbers.restated_band_frames == 2
    assert numbers.restated_band_pinned == 1


def test_a_beside_reach_narrower_than_the_corridor_is_refused() -> None:
    with pytest.raises(ValueError, match="beside_meters of 0.3 m must reach at least the heading corridor"):
        PlannerNumbersConfig(beside_meters=0.3)


def test_the_numbers_command_judges_the_band_on_the_restated_figure(recording: Path, capsys) -> None:
    code, out, _ = run(["numbers", recording], capsys)
    assert code == 0
    assert "restated band, 3.00 to 5.32 m, nothing nearer within 1 m: pinned" in out
    assert "restated band pinned" in out


def test_a_clear_frame_with_clear_sides_is_not_in_the_restated_band() -> None:
    # Nothing ahead at all within the corridor and nothing beside it: clear, not band.
    assert not in_restated_band(ObstacleSet(0.0, (point(2.0, 3.0, 3.2),), 1), PLANNER, CONFIG)


def test_the_band_verdict_reads_the_restated_figure_not_the_plain_one() -> None:
    numbers = PlannerNumbers(
        frames=400,
        sidestep_limit_degrees=35.5,
        horizon_reach_meters=5.32,
        pinned_by_band={ClearanceBand.CLOSE: 0, ClearanceBand.NEAR: 0, ClearanceBand.BAND: 20, ClearanceBand.CLEAR: 0},
        frames_by_band={ClearanceBand.CLOSE: 0, ClearanceBand.NEAR: 0, ClearanceBand.BAND: 200, ClearanceBand.CLEAR: 200},
        restated_band_frames=120,
        restated_band_pinned=70,
        pinned_frames=20,
        distinct_headings=10,
        alarm_on_frames=0,
        alarm_changes=0,
        raised_past_close=0,
        longest_hold_seconds=0.0,
        near_noise_median_meters=None,
        near_noise_readings=0,
        disagreement_median_meters=None,
        disagreement_p90_meters=None,
        disagreement_pairs=0,
        pairs_without_world=0,
        duplicate_pairs=0,
    )
    text = format_numbers("made up", numbers, PLANNER, CONFIG)
    # The plain band is 10 % of 200 and would pass. The restated band is 58.3 % of 120 and fails.
    assert "FAIL  restated band pinned 58.3 % of 120, target at most 50 %" in text
