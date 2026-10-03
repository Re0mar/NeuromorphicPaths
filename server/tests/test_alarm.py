"""Covers what raises the alarm and how long it is held, on hand-built groups and written-out timestamps."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.alarm import AlarmHold, alarm_raised, corridor_points, corridor_time_to_contact
from nav.planner.config import PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)
EDGE = CONFIG.alarm_body_half_width_meters  # 0.30 m with the defaults.


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
    with pytest.raises(ValueError, match="alarm_body_half_width_meters"):
        corridor_points(_set(_point(0.0, 1.0)), replace(CONFIG, alarm_body_half_width_meters=half_width))


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
        ("alarm_body_half_width_meters", 0.0),
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
