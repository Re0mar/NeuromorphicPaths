"""Covers what raises the alarm and how long it is held, on hand-built groups and written-out timestamps."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.alarm import (
    AlarmHold,
    GapRatioAlarm,
    alarm_raised,
    avoidance_surprise_bits,
    avoidance_surprise_bits_at,
    corridor_points,
    corridor_time_to_contact,
    path_red_from_bits,
    time_to_contact_from_avoidance_bits,
)
from nav.planner.config import AlarmRule, PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)
EDGE = CONFIG.body_half_width_meters  # 0.30 m with the defaults.


def _point(lateral: float, forward: float, group: int = 1, closing: float | None = None) -> ObstaclePoint:
    clearance = max(0.0, float(np.hypot(lateral, forward)) - WALKER.radius_meters)
    # A point built by hand has no camera frame. The planner never reads camera_point.
    return ObstaclePoint(lateral, forward, group, clearance, 0.1, closing, None, False, np.zeros(3))


def _set(*points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(timestamp_seconds=0.0, points=tuple(points), groups_in_view=len({p.group_id for p in points}))


def _hold_run(hold: AlarmHold, script: list[tuple[bool, float]]) -> list[bool]:
    # Timestamps are written out by each test, never generated, so equal and backward ones can be expressed.
    return [hold.update(raise_now, timestamp) for raise_now, timestamp in script]


def test_a_post_dead_ahead_inside_the_threshold_raises() -> None:
    # 0.9 m ahead is 0.55 m of clearance, 0.39 s at 1.4 m/s, under 0.7 s.
    assert alarm_raised(_set(_point(0.0, 0.9)), CONFIG) is True


def test_the_same_post_beyond_the_threshold_does_not_raise() -> None:
    # 1.4 m ahead is 1.05 m of clearance, 0.75 s, over 0.7 s.
    assert alarm_raised(_set(_point(0.0, 1.4)), CONFIG) is False


def test_contact_exactly_at_the_threshold_does_not_raise() -> None:
    # Under the threshold raises, at it does not. Speed 2.0 and threshold 0.5 put 1.0 m of
    # clearance at exactly 0.5 s in floating point, so the comparison itself is what is tested.
    config = replace(CONFIG, walking_speed_mps=2.0, alarm_time_to_contact_seconds=0.5)
    at_the_threshold = ObstaclePoint(0.0, 1.35, 1, 1.0, 0.1, None, None, False, np.zeros(3))
    just_under = ObstaclePoint(0.0, 1.35, 1, 0.999, 0.1, None, None, False, np.zeros(3))

    assert alarm_raised(_set(at_the_threshold), config) is False
    assert alarm_raised(_set(just_under), config) is True


def test_a_post_beside_the_corridor_does_not_raise() -> None:
    assert alarm_raised(_set(_point(0.31, 0.8)), CONFIG) is False


def test_the_same_post_just_inside_the_edge_raises() -> None:
    assert alarm_raised(_set(_point(0.29, 0.8)), CONFIG) is True


def test_a_doorway_the_body_fits_through_stays_quiet() -> None:
    # Sides 0.40 m either side of center, 0.8 m ahead: an opening the walker's body clears walking
    # straight, but inside the planner's 0.35 m footprint plus a margin. That corridor turned the
    # screen red all the way through a doorway on the apartment walk of 2026-10-03.
    doorway = _set(_point(-0.40, 0.8, group=1), _point(0.40, 0.8, group=2))
    assert alarm_raised(doorway, CONFIG) is False


def test_a_doorway_side_that_would_hit_the_body_raises() -> None:
    narrow = _set(_point(-0.25, 0.8, group=1), _point(0.40, 0.8, group=2))
    assert alarm_raised(narrow, CONFIG) is True


def test_the_corridor_edge_is_inclusive() -> None:
    on_the_edge = _point(EDGE, 0.8)
    assert corridor_points(_set(on_the_edge), CONFIG) == (on_the_edge,)
    assert corridor_points(_set(_point(-EDGE, 0.8)), CONFIG) != ()


@pytest.mark.parametrize("forward", [0.0, -0.5])
def test_a_group_level_with_or_behind_the_walker_never_raises(forward: float) -> None:
    assert alarm_raised(_set(_point(0.0, forward)), CONFIG) is False


def test_a_group_just_ahead_of_the_walker_raises() -> None:
    assert alarm_raised(_set(_point(0.0, 0.05)), CONFIG) is True


def test_the_nearest_corridor_group_sets_the_contact_time() -> None:
    near = _point(0.1, 1.0, group=1)
    far = _point(-0.1, 2.0, group=2)
    beside_and_nearer = _point(0.6, 0.1, group=3)
    assert beside_and_nearer.clearance_meters < near.clearance_meters, "the decoy must be nearer to prove it is ignored"

    expected = near.clearance_meters / CONFIG.walking_speed_mps
    assert corridor_time_to_contact(_set(far, beside_and_nearer, near), CONFIG) == pytest.approx(expected)


def test_an_empty_scene_has_no_contact_time() -> None:
    assert corridor_time_to_contact(_set(), CONFIG) is None
    assert alarm_raised(_set(), CONFIG) is False


@pytest.mark.parametrize("closing", [-2.0, None, 0.0, 5.0])
def test_the_alarm_ignores_the_closing_rate(closing: float | None) -> None:
    assert alarm_raised(_set(_point(0.0, 0.9, closing=closing)), CONFIG) is True
    assert alarm_raised(_set(_point(0.0, 1.4, closing=closing)), CONFIG) is False


def test_the_threshold_at_walking_pace_stays_under_a_meter() -> None:
    # Something more than a meter of clearance away must never raise the alarm. The two constants
    # are tuned separately, so this holds them to that together.
    assert CONFIG.alarm_time_to_contact_seconds * CONFIG.walking_speed_mps <= 1.0


@pytest.mark.parametrize("half_width", [0.0, -0.3, float("nan")])
def test_a_bad_body_half_width_is_refused(half_width: float) -> None:
    with pytest.raises(ValueError, match="body_half_width_meters"):
        corridor_points(_set(_point(0.0, 1.0)), replace(CONFIG, body_half_width_meters=half_width))


@pytest.mark.parametrize("speed", [0.0, -1.4])
def test_a_walking_speed_that_is_not_positive_is_refused(speed: float) -> None:
    with pytest.raises(ValueError, match="walking_speed_mps"):
        corridor_time_to_contact(_set(_point(0.0, 1.0)), replace(CONFIG, walking_speed_mps=speed))


@pytest.mark.parametrize("threshold", [0.0, -0.7])
def test_a_threshold_that_is_not_positive_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match="alarm_time_to_contact_seconds"):
        alarm_raised(_set(_point(0.0, 1.0)), replace(CONFIG, alarm_time_to_contact_seconds=threshold))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("body_half_width_meters", 0.0),
        ("walking_speed_mps", 0.0),
        ("alarm_time_to_contact_seconds", 0.0),
        ("alarm_hold_seconds", -0.1),
    ],
)
def test_the_pipeline_refuses_a_bad_alarm_config_when_built(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        PlannerPipeline(replace(CONFIG, **{field: value}), WALKER)


def test_a_raised_alarm_stays_up_for_the_hold() -> None:
    script = [(True, 0.0), (False, 0.1), (False, 0.3), (False, 0.49), (False, 0.5)]
    assert _hold_run(AlarmHold(0.5), script) == [True, True, True, True, False]


def test_a_long_raise_clears_on_the_first_false_frame() -> None:
    script = [(True, 0.0), (True, 1.0), (True, 3.0), (False, 3.1)]
    assert _hold_run(AlarmHold(0.5), script) == [True, True, True, False]


def test_an_unraised_alarm_stays_down() -> None:
    assert _hold_run(AlarmHold(0.5), [(False, 0.0), (False, 0.1)]) == [False, False]


def test_a_duplicate_timestamp_changes_nothing() -> None:
    script = [(True, 0.0), (False, 0.2), (False, 0.2), (False, 0.6)]
    assert _hold_run(AlarmHold(0.5), script) == [True, True, True, False]


def test_a_backward_timestamp_resets_the_hold() -> None:
    script = [(True, 10.0), (False, 10.1), (False, 2.0)]
    assert _hold_run(AlarmHold(0.5), script) == [True, True, False]


def test_a_backward_timestamp_that_raises_starts_a_new_hold() -> None:
    script = [(True, 10.0), (True, 2.0), (False, 2.3), (False, 2.5)]
    assert _hold_run(AlarmHold(0.5), script) == [True, True, True, False]


def test_a_large_forward_gap_expires_the_hold() -> None:
    assert _hold_run(AlarmHold(0.5), [(True, 0.0), (False, 60.0)]) == [True, False]


def test_a_zero_hold_follows_the_decision_exactly() -> None:
    decisions = [True, False, True, True, False, False, True]
    script = [(raise_now, 0.1 * index) for index, raise_now in enumerate(decisions)]
    assert _hold_run(AlarmHold(0.0), script) == decisions


@pytest.mark.parametrize("hold", [-0.1, float("nan"), float("inf")])
def test_a_bad_hold_is_refused(hold: float) -> None:
    with pytest.raises(ValueError, match="alarm_hold_seconds"):
        AlarmHold(hold)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf")])
def test_a_non_finite_timestamp_is_refused(timestamp: float) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        AlarmHold(0.5).update(True, timestamp)


def _at_clearance(clearance: float, lateral: float = 0.0) -> ObstaclePoint:
    # Clearance given directly, so the time to contact is exact rather than derived from a footprint.
    return ObstaclePoint(lateral, 2.0, 1, clearance, 0.1, None, None, False, np.zeros(3))


def test_avoidance_surprise_is_zero_with_an_empty_corridor() -> None:
    assert avoidance_surprise_bits(_set(), CONFIG) == 0.0
    # The same fixture with a group moved into the corridor reads above zero, so the zero above is
    # the empty corridor and not a function that always says zero.
    assert avoidance_surprise_bits(_set(_at_clearance(1.0)), CONFIG) > 0.0


def test_avoidance_surprise_is_0_72_bits_at_one_second_to_contact() -> None:
    # 1.4 m of clearance at 1.4 m/s is one second, and (1 / 1)^2 / (2 ln 2) is 0.7213 bits.
    assert avoidance_surprise_bits(_set(_at_clearance(1.4)), CONFIG) == pytest.approx(1.0 / (2.0 * np.log(2.0)))


def test_avoidance_surprise_stays_finite_at_zero_clearance() -> None:
    floored_seconds = CONFIG.clearance_epsilon_meters / CONFIG.walking_speed_mps
    expected = (1.0 / floored_seconds) ** 2 / (2.0 * np.log(2.0))

    assert avoidance_surprise_bits(_set(_at_clearance(0.0)), CONFIG) == pytest.approx(expected)


def test_avoidance_surprise_reads_the_nearest_group_in_the_corridor() -> None:
    near_and_far = _set(_at_clearance(2.8), ObstaclePoint(0.0, 1.0, 2, 1.4, 0.1, None, None, False, np.zeros(3)))

    assert avoidance_surprise_bits(near_and_far, CONFIG) == pytest.approx(1.0 / (2.0 * np.log(2.0)))


def test_a_group_outside_the_corridor_adds_no_avoidance_surprise() -> None:
    assert avoidance_surprise_bits(_set(_at_clearance(0.1, lateral=EDGE + 0.05)), CONFIG) == 0.0
    assert avoidance_surprise_bits(_set(_at_clearance(0.1, lateral=EDGE - 0.05)), CONFIG) > 0.0


def test_avoidance_surprise_refuses_a_walking_speed_of_zero() -> None:
    # Time to contact divides by walking speed. The refusal names the setting, where a missing check
    # would surface as a bare ZeroDivisionError.
    with pytest.raises(ValueError, match="walking_speed_mps must be above zero, got 0.0"):
        avoidance_surprise_bits(_set(_at_clearance(1.0)), replace(CONFIG, walking_speed_mps=0.0))


def test_the_avoidance_formula_is_0_72_bits_at_one_second() -> None:
    # Half of (1 s over 1 s) squared is 0.5 nats, and 0.5 / ln 2 is 0.7213 bits.
    assert avoidance_surprise_bits_at(1.0) == pytest.approx(0.7213, abs=1e-4)


@pytest.mark.parametrize("seconds", [0.05, 0.7, 1.0, 3.2, 40.0])
def test_the_time_to_contact_reads_back_from_the_surprise(seconds: float) -> None:
    assert time_to_contact_from_avoidance_bits(avoidance_surprise_bits_at(seconds)) == pytest.approx(seconds, rel=1e-12)


def test_no_surprise_reads_back_as_no_contact_ahead() -> None:
    assert time_to_contact_from_avoidance_bits(0.0) is None


@pytest.mark.parametrize("bits", [-0.1, float("nan"), float("inf")])
def test_a_surprise_that_no_time_gives_is_refused(bits: float) -> None:
    with pytest.raises(ValueError, match="avoidance surprise must be finite and zero or more"):
        time_to_contact_from_avoidance_bits(bits)


def test_the_path_turns_red_at_the_alarms_own_threshold() -> None:
    # 0.7 s at the shipped settings: half of (1 / 0.7) squared is 1.0204 nats, over ln 2 that is 1.4721 bits.
    assert path_red_from_bits(CONFIG) == pytest.approx(avoidance_surprise_bits_at(CONFIG.alarm_time_to_contact_seconds))
    assert path_red_from_bits(CONFIG) == pytest.approx(1.4721, abs=1e-4)
    # It follows the threshold rather than holding a number of its own.
    assert path_red_from_bits(replace(CONFIG, alarm_time_to_contact_seconds=1.0)) == pytest.approx(avoidance_surprise_bits_at(1.0))


def test_the_avoidance_surprise_and_the_red_point_share_one_formula() -> None:
    # A group whose time to contact is exactly the alarm threshold reads exactly the red point.
    clearance = CONFIG.alarm_time_to_contact_seconds * CONFIG.walking_speed_mps
    point = ObstaclePoint(0.0, clearance + WALKER.radius_meters, 1, clearance, 0.1, None, None, False, np.zeros(3))

    assert avoidance_surprise_bits(_set(point), CONFIG) == pytest.approx(path_red_from_bits(CONFIG))


@pytest.mark.parametrize("seconds", [0.0, -0.5, float("nan"), float("inf")])
def test_the_avoidance_formula_refuses_a_time_that_is_not_above_zero(seconds: float) -> None:
    with pytest.raises(ValueError, match="time to contact must be above zero"):
        avoidance_surprise_bits_at(seconds)


def test_a_red_point_from_a_threshold_of_zero_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="alarm_time_to_contact_seconds must be above zero"):
        path_red_from_bits(replace(CONFIG, alarm_time_to_contact_seconds=0.0))


# The gap-ratio rule. Frames come 0.3 s apart and the window is 2.95 s, so a full window holds 10
# gaps. For evenly spaced gaps the sample standard deviation is the step times sqrt(n(n+1)/12),
# which is 3.028 steps for 10.
GAP_RATIO = replace(
    CONFIG,
    alarm_rule=AlarmRule.GAP_RATIO,
    alarm_ratio_window_seconds=2.95,
    alarm_ratio_multiplier=1.0,
    alarm_ratio_margin_meters=0.0,
)
FRAME_SECONDS = 0.3


def _frame_at(seconds: float, gap_meters: float | None) -> ObstacleSet:
    points = () if gap_meters is None else (_point(0.0, gap_meters + WALKER.radius_meters),)
    return ObstacleSet(timestamp_seconds=seconds, points=points, groups_in_view=len(points))


def _first_raise(config: PlannerConfig, start_gap: float, speed_mps: float, frames: int) -> tuple[int, float] | None:
    alarm = GapRatioAlarm(config)
    for index in range(frames):
        gap = start_gap - speed_mps * FRAME_SECONDS * index
        if alarm.raised(_frame_at(index * FRAME_SECONDS, gap), config):
            return index, gap
    return None


def test_a_steady_approach_raises_where_the_spread_reaches_the_gap() -> None:
    # 0.5 m/s is a 0.15 m step, a spread of 0.454 m. From 3.0 m the gap is 0.60 m at frame 16 and
    # 0.45 m at frame 17, the first at or under the spread.
    index, gap = _first_raise(GAP_RATIO, 3.0, 0.5, 30)
    assert index == 17
    assert gap == pytest.approx(0.45)


def test_the_multiplier_moves_the_raise_out_in_proportion() -> None:
    # Doubled, the spread is 0.908 m, reached at frame 14, 0.90 m out.
    index, gap = _first_raise(replace(GAP_RATIO, alarm_ratio_multiplier=2.0), 3.0, 0.5, 30)
    assert index == 14
    assert gap == pytest.approx(0.90)


def test_the_time_to_contact_at_the_raise_does_not_depend_on_walking_speed() -> None:
    # Twice the speed doubles both the step and the spread, so the raise comes at twice the gap and
    # the same 0.9 s from contact. Nothing in the rule assumes a speed.
    _, slow_gap = _first_raise(GAP_RATIO, 3.0, 0.5, 30)
    _, fast_gap = _first_raise(GAP_RATIO, 6.0, 1.0, 30)
    assert slow_gap / 0.5 == pytest.approx(0.9)
    assert fast_gap / 1.0 == pytest.approx(0.9)


def test_standing_still_close_to_something_never_raises_the_gap_rule() -> None:
    # The distance rule calls 0.5 m an alarm whatever the walker does. With no change in the gap
    # there is no spread, so the gap rule stays down.
    alarm = GapRatioAlarm(GAP_RATIO)
    frames = [_frame_at(index * FRAME_SECONDS, 0.5) for index in range(20)]
    assert not any(alarm.raised(frame, GAP_RATIO) for frame in frames)
    assert alarm_raised(frames[-1], CONFIG)


def test_too_few_gaps_never_raise_the_gap_rule() -> None:
    # Two gaps falling from 2 m to almost nothing would be a huge spread, but two samples say nothing.
    alarm = GapRatioAlarm(GAP_RATIO)
    assert not alarm.raised(_frame_at(0.0, 2.0), GAP_RATIO)
    assert not alarm.raised(_frame_at(0.3, 0.05), GAP_RATIO)


def test_an_empty_corridor_adds_no_gap() -> None:
    alarm = GapRatioAlarm(GAP_RATIO)
    alarm.raised(_frame_at(0.0, 2.0), GAP_RATIO)
    alarm.raised(_frame_at(0.3, 1.0), GAP_RATIO)
    assert not alarm.raised(_frame_at(0.6, None), GAP_RATIO)
    # Only now is there a third gap, and 2.0, 1.0, 0.05 has a spread far above 0.05.
    assert alarm.raised(_frame_at(0.9, 0.05), GAP_RATIO)


def test_gaps_older_than_the_window_are_dropped() -> None:
    # A fall from 2 m to 0.05 m, then 3 s of standing at 0.05 m. Once the fall has aged out the spread
    # is zero, so the rule goes down even though the walker is still that close.
    alarm = GapRatioAlarm(GAP_RATIO)
    for index, gap in enumerate([2.0, 1.0, 0.05]):
        alarm.raised(_frame_at(index * FRAME_SECONDS, gap), GAP_RATIO)
    assert alarm.raised(_frame_at(0.9, 0.05), GAP_RATIO)
    assert not alarm.raised(_frame_at(0.9 + 3.0, 0.05), GAP_RATIO)


def test_a_clock_that_went_backward_starts_the_gaps_over() -> None:
    alarm = GapRatioAlarm(GAP_RATIO)
    for index, gap in enumerate([2.0, 1.0, 0.05]):
        alarm.raised(_frame_at(10.0 + index * FRAME_SECONDS, gap), GAP_RATIO)
    assert not alarm.raised(_frame_at(1.0, 0.05), GAP_RATIO)


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"alarm_ratio_window_seconds": 0.0}, "alarm_ratio_window_seconds"),
        ({"alarm_ratio_window_seconds": float("nan")}, "alarm_ratio_window_seconds"),
        ({"alarm_ratio_multiplier": 0.0}, "alarm_ratio_multiplier"),
        ({"alarm_ratio_min_samples": 1}, "alarm_ratio_min_samples"),
        ({"alarm_ratio_margin_meters": -0.1}, "alarm_ratio_margin_meters"),
        ({"alarm_ratio_prior_meters": -0.1}, "alarm_ratio_prior_meters"),
    ],
)
def test_gap_rule_settings_it_cannot_use_are_refused_at_startup(change: dict, named: str) -> None:
    with pytest.raises(ValueError, match=named):
        PlannerPipeline(replace(GAP_RATIO, **change), WALKER)


def test_the_gap_rule_is_the_default_and_distance_is_there_on_request() -> None:
    standing = [_frame_at(index * FRAME_SECONDS, 0.5) for index in range(12)]
    distance = PlannerPipeline(replace(CONFIG, alarm_rule=AlarmRule.DISTANCE), WALKER)
    gap_ratio = PlannerPipeline(CONFIG, WALKER)
    assert CONFIG.alarm_rule is AlarmRule.GAP_RATIO
    assert all(distance.plan(frame).alarm for frame in standing)
    assert not any(gap_ratio.plan(frame).alarm for frame in standing)


def test_the_margin_raises_where_the_spread_reaches_the_gap_less_the_margin() -> None:
    # The 0.454 m spread now meets the gap less 0.2 m, so the gap at the raise is at most 0.654 m. From
    # 3.0 m that is 0.60 m at frame 16, one frame before the plain rule's 0.45 m.
    index, gap = _first_raise(replace(GAP_RATIO, alarm_ratio_margin_meters=0.2), 3.0, 0.5, 30)
    assert index == 16
    assert gap == pytest.approx(0.60)


def test_inside_the_margin_a_wobble_raises_the_gap_rule_and_stillness_does_not() -> None:
    # Inside a 0.3 m margin the gap less the margin is under the 0.06 m epsilon, so the rule raises
    # on any spread of 0.06 m or more. A gap wobbling 0.1 to 0.3 m has a spread of about 0.11 m and
    # raises it. The same wobble 1 m away doesn't, and a gap that never changes never does.
    config = replace(GAP_RATIO, alarm_ratio_margin_meters=0.3)

    def raised_on(gaps: list[float]) -> list[bool]:
        alarm = GapRatioAlarm(config)
        return [alarm.raised(_frame_at(index * FRAME_SECONDS, gap), config) for index, gap in enumerate(gaps)]

    assert raised_on([0.1, 0.3] * 3)[-1]
    assert not any(raised_on([0.9, 1.1] * 3))
    assert not any(raised_on([0.2] * 6))


def test_a_prior_raises_the_alarm_inside_its_distance_with_no_change_in_the_gap() -> None:
    # Standing still gives no spread, so only the prior speaks: 0.4 m is inside a 0.5 m prior, 0.6 m isn't.
    config = replace(GAP_RATIO, alarm_ratio_prior_meters=0.5)
    inside, outside = GapRatioAlarm(config), GapRatioAlarm(config)
    assert all(inside.raised(_frame_at(index * FRAME_SECONDS, 0.4), config) for index in range(5))
    assert not any(outside.raised(_frame_at(index * FRAME_SECONDS, 0.6), config) for index in range(5))


def test_the_prior_and_the_spread_add_as_squares() -> None:
    # The 0.454 m spread and a 0.3 m prior reach sqrt(0.454^2 + 0.3^2) = 0.544 m together. The 0.60 m
    # gap is still above that, so the raise stays at frame 17, 0.45 m. A 0.5 m prior reaches 0.675 m,
    # which moves the raise to frame 16, 0.60 m.
    index_small, _ = _first_raise(replace(GAP_RATIO, alarm_ratio_prior_meters=0.3), 3.0, 0.5, 30)
    index_large, gap_large = _first_raise(replace(GAP_RATIO, alarm_ratio_prior_meters=0.5), 3.0, 0.5, 30)
    assert index_small == 17
    assert index_large == 16
    assert gap_large == pytest.approx(0.60)


def test_with_no_prior_the_rule_is_the_spread_alone() -> None:
    assert _first_raise(GAP_RATIO, 3.0, 0.5, 30) == _first_raise(replace(GAP_RATIO, alarm_ratio_prior_meters=0.0), 3.0, 0.5, 30)
