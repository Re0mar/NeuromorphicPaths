"""Covers the planner end to end on synthetic obstacle sets, against behaviors stated in advance."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.contact import ContactSurprise
from nav.planner.dynamic_programming import plan
from nav.planner.field import cost_field
from nav.planner.goal import goal_position, goal_term
from nav.planner.heading import lookahead_step_index
from nav.planner.pipeline import PlannerPipeline, planner_terms
from nav.planner.surprise import CollisionSurprise, surprise_field
from nav.sources.framecodec import decode_path, encode_path
from nav.types import ObstaclePoint, ObstacleSet, PlannedPath
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)


def _point(lateral: float, forward: float, group: int, noise: float = 0.2, closing: float | None = None, is_wall: bool = False) -> ObstaclePoint:
    clearance = max(0.0, float(np.hypot(lateral, forward)) - WALKER.radius_meters)
    # A point built by hand has no camera frame. The planner never reads camera_point.
    return ObstaclePoint(lateral, forward, group, clearance, noise, closing, None, is_wall, np.zeros(3))


def _set(*points: ObstaclePoint, timestamp: float = 0.0) -> ObstacleSet:
    return ObstacleSet(timestamp_seconds=timestamp, points=tuple(points), groups_in_view=len({p.group_id for p in points}))


def _wall_across(forward: float, closing: float | None = None, gap: tuple[float, float] | None = None) -> ObstacleSet:
    points = []
    group = 0
    for lateral in np.arange(-3.0, 3.01, 0.25):
        if gap is not None and gap[0] <= lateral <= gap[1]:
            continue
        points.append(_point(float(lateral), forward, group, noise=0.3, closing=closing))
        group += 1
    return _set(*points)


def test_an_empty_scene_plans_straight_ahead_with_no_alarm() -> None:
    path = PlannerPipeline(CONFIG, WALKER).plan(_set())

    assert isinstance(path, PlannedPath)
    assert path.lateral_offsets_meters == pytest.approx(np.zeros_like(path.lateral_offsets_meters))
    assert path.first_heading_radians == pytest.approx(0.0)
    assert path.alarm is False
    assert len(path.times_seconds) == len(path.lateral_offsets_meters)


def test_a_wall_a_meter_ahead_raises_the_alarm() -> None:
    # 0.65 m of clearance dead ahead is 0.46 s at walking pace, under the threshold.
    assert PlannerPipeline(CONFIG, WALKER).plan(_wall_across(1.0)).alarm is True


def test_the_same_wall_further_ahead_raises_no_alarm() -> None:
    # 1.15 m of clearance is 0.82 s at walking pace, over the threshold.
    assert PlannerPipeline(CONFIG, WALKER).plan(_wall_across(1.5)).alarm is False


def test_the_pipeline_holds_the_alarm_across_frames() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)
    in_the_way = _wall_across(1.0)

    raised = pipeline.plan(_set(*in_the_way.points, timestamp=0.0))
    held = pipeline.plan(_set(timestamp=0.1))
    cleared = pipeline.plan(_set(timestamp=0.6))

    assert (raised.alarm, held.alarm, cleared.alarm) == (True, True, False)


def test_cost_rises_monotonically_as_a_group_is_moved_closer() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)
    costs = [pipeline.plan(_set(_point(0.0, forward, group=1, noise=0.3))).cumulative_cost_bits for forward in (5.0, 4.0, 3.0)]

    assert costs[0] < costs[1] < costs[2]


def test_a_wall_dead_ahead_with_equal_gaps_makes_the_path_leave_centre() -> None:
    # The old planner needed a nudged initial heading to break this symmetry. Dynamic programming
    # has no local minimum to escape, so the path commits to one side without help.
    wall = _wall_across(2.5, gap=None)
    points = [p for p in wall.points if abs(p.lateral_meters) < 1.0 or abs(p.lateral_meters) > 2.0]
    symmetric = _set(*points)

    path = PlannerPipeline(CONFIG, WALKER).plan(symmetric)

    assert np.max(np.abs(path.lateral_offsets_meters)) > 0.5
    assert path.first_heading_radians != pytest.approx(0.0)


def test_the_path_goes_through_the_gap_and_not_the_wall() -> None:
    gapped = _wall_across(2.5, gap=(0.75, 1.75))

    path = PlannerPipeline(CONFIG, WALKER).plan(gapped)

    # Once past the wall's forward distance the path should sit inside the gap.
    k_past = int(np.ceil(2.5 / (CONFIG.walking_speed_mps * CONFIG.time_step_seconds)))
    assert np.all((path.lateral_offsets_meters[k_past:] > 0.5) & (path.lateral_offsets_meters[k_past:] < 2.0))


# Median noise of real groups inside the walker's corridor and within 2 m of clearance, replayed
# over the 2026-10-02 classroom walk. pixel_walk_3's median is 0.0958 m on the seeded floor fit. The lower one
# pushes the plan least, so a too-heavy kinetic weight fails here first.
MEASURED_NEAR_NOISE_METERS = 0.0337


@pytest.mark.parametrize("forward", [1.0, 1.5])
def test_a_post_close_ahead_is_cleared_before_the_walker_reaches_it(forward: float) -> None:
    post = _set(_point(0.0, forward, group=1, noise=MEASURED_NEAR_NOISE_METERS))
    path = PlannerPipeline(CONFIG, WALKER).plan(post)

    k_reach = int(np.ceil(forward / (CONFIG.walking_speed_mps * CONFIG.time_step_seconds)))
    assert abs(path.lateral_offsets_meters[k_reach]) > WALKER.radius_meters, path.lateral_offsets_meters[: k_reach + 1]


def test_a_gap_close_ahead_is_still_taken() -> None:
    gapped = _wall_across(1.5, gap=(0.75, 1.75))
    path = PlannerPipeline(CONFIG, WALKER).plan(gapped)

    k_reach = int(np.ceil(1.5 / (CONFIG.walking_speed_mps * CONFIG.time_step_seconds)))
    assert np.all((path.lateral_offsets_meters[k_reach:] > 0.5) & (path.lateral_offsets_meters[k_reach:] < 2.0))


def test_the_gaze_goal_picks_the_gap_on_the_gaze_side_when_two_gaps_are_equal() -> None:
    two_gaps = _wall_across(2.5, gap=None)
    points = [p for p in two_gaps.points if not (1.0 <= abs(p.lateral_meters) <= 1.5)]
    scene = _set(*points)
    pipeline = PlannerPipeline(CONFIG, WALKER)

    left = pipeline.plan(scene, goal_mode=GoalMode.GAZE, gaze_ground_point=np.array([-1.25, 4.0]))
    right = pipeline.plan(scene, goal_mode=GoalMode.GAZE, gaze_ground_point=np.array([1.25, 4.0]))

    assert left.lateral_offsets_meters[-1] < 0.0
    assert right.lateral_offsets_meters[-1] > 0.0


def test_the_gaze_goal_falls_back_to_ahead_without_a_gaze() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)

    ahead = pipeline.plan(_set(), goal_mode=GoalMode.AHEAD)
    no_gaze = pipeline.plan(_set(), goal_mode=GoalMode.GAZE, gaze_ground_point=None)

    assert no_gaze.lateral_offsets_meters == pytest.approx(ahead.lateral_offsets_meters)


def test_the_path_heading_reads_the_path_at_the_lookahead() -> None:
    # This gap's path sidesteps for half a second and then holds, so its first step and its
    # lookahead read different angles. A gap the path runs at full speed for a whole second
    # reads the same both ways and could not tell the two rules apart.
    wall = _wall_across(1.5, gap=(0.0, 1.0))
    path = PlannerPipeline(CONFIG, WALKER).plan(wall)

    index = lookahead_step_index(CONFIG)
    offsets = path.lateral_offsets_meters
    first_step = np.arctan2(offsets[1] - offsets[0], 0.1 * 1.4)
    # Worked from the returned path rather than through lookahead_heading, so a wrong index or
    # forward distance inside it cannot agree with itself here.
    expected = np.arctan2(offsets[10] - offsets[0], 10 * 0.1 * 1.4)
    assert index == 10
    assert abs(first_step - expected) > np.radians(5.0), "the scene no longer separates the two readings"
    assert path.first_heading_radians == pytest.approx(expected)
    assert path.first_heading_radians > 0.0, "the gap is on the right"


def test_the_heading_takes_more_than_three_values_across_scenes() -> None:
    # A first-step reading can only be straight, or a full sidestep either way. Sliding a gap
    # across the wall gives paths that reach the lookahead at many different offsets.
    pipeline = PlannerPipeline(CONFIG, WALKER)
    headings = set()
    for gap_start in np.arange(-2.0, 1.01, 0.25):
        path = pipeline.plan(_wall_across(2.5, gap=(float(gap_start), float(gap_start) + 1.0)))
        headings.add(round(float(np.degrees(path.first_heading_radians)), 2))

    assert len(headings) > 3, headings


def test_no_planned_heading_exceeds_the_sidestep_limit() -> None:
    # The wire format promises the heading stays within the sidestep limit. That holds only while the
    # dynamic program moves at most the limit's worth of cells per step, so plan real scenes and check.
    limit = np.arctan2(CONFIG.max_lateral_speed_mps, CONFIG.walking_speed_mps)
    pipeline = PlannerPipeline(CONFIG, WALKER)
    headings = [
        pipeline.plan(_wall_across(forward, gap=(float(gap_start), float(gap_start) + 1.0))).first_heading_radians
        for forward in (1.5, 2.5)
        for gap_start in np.arange(-2.0, 1.01, 0.5)
    ]

    assert max(abs(heading) for heading in headings) == pytest.approx(limit), "the sweep must reach the limit to test it"
    assert all(abs(heading) <= limit + 1e-9 for heading in headings)


def test_last_field_is_the_field_the_plan_was_made_from() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)
    assert pipeline.last_field is None

    path = pipeline.plan(_set(_point(0.0, 3.0, group=1)))

    assert pipeline.last_field is not None
    assert pipeline.last_field.shape == (len(path.times_seconds), len(pipeline.grid))
    # The goal term is in it, so the terminal row is not flat even with one post far ahead.
    assert pipeline.last_field[-1].max() > pipeline.last_field[-1].min()


def test_the_path_timestamp_is_the_obstacle_sets() -> None:
    path = PlannerPipeline(CONFIG, WALKER).plan(_set(timestamp=12.5))

    assert path.timestamp_seconds == pytest.approx(12.5)


# The contact term. Above the shipped weight, so this guards a weight the shipped config does not
# already cover. His term alone walks into both posts here, and the contact term still clears them.
WEIGHT_ONLY_CONTACT_HOLDS = 7.0
# pixel_walk_3's median near noise on the seeded floor fit, beside the classroom's MEASURED_NEAR_NOISE_METERS.
# More noise makes his term dodge harder, so the classroom's value is the hard case and this one
# checks the margin does not depend on it.
OTHER_WALK_NEAR_NOISE_METERS = 0.0958


def test_the_shipping_terms_are_his_surprise_and_contact() -> None:
    terms = planner_terms(PlannerConfig())

    assert [type(term) for term in terms] == [CollisionSurprise, ContactSurprise]


def test_switching_contact_off_leaves_his_term_alone() -> None:
    terms = planner_terms(replace(CONFIG, contact_term_enabled=False))

    assert [type(term) for term in terms] == [CollisionSurprise]


@pytest.mark.parametrize(
    "scene",
    [_wall_across(1.0), _wall_across(2.5, gap=(0.75, 1.75)), _set(_point(0.0, 1.5, group=1, noise=MEASURED_NEAR_NOISE_METERS))],
    ids=["wall", "gap", "post"],
)
def test_with_contact_off_the_plan_is_what_his_field_alone_gives(scene: ObstacleSet) -> None:
    config = replace(CONFIG, contact_term_enabled=False)
    pipeline = PlannerPipeline(config, WALKER)
    field = surprise_field(scene, pipeline.grid, config, WALKER)
    field[-1] += goal_term(pipeline.grid, goal_position(GoalMode.AHEAD, config, None), config)
    expected_offsets, expected_cost = plan(field, 0.0, pipeline.grid, config)

    path = pipeline.plan(scene)

    np.testing.assert_array_equal(path.lateral_offsets_meters, expected_offsets)
    assert path.cumulative_cost_bits == expected_cost


def test_last_field_includes_the_contact_term() -> None:
    scene = _set(_point(0.0, 1.5, group=1, noise=MEASURED_NEAR_NOISE_METERS))
    with_contact = PlannerPipeline(CONFIG, WALKER)
    without = PlannerPipeline(replace(CONFIG, contact_term_enabled=False), WALKER)
    with_contact.plan(scene)
    without.plan(scene)

    contact_only = cost_field(scene, with_contact.grid, CONFIG, WALKER, terms=(ContactSurprise(),))

    np.testing.assert_allclose(with_contact.last_field - without.last_field, contact_only, atol=1e-12)
    assert contact_only.max() > 0.0


@pytest.mark.parametrize("forward", [1.0, 1.5])
def test_a_post_at_a_raised_weight_is_cleared_only_because_of_contact(forward: float) -> None:
    raised = replace(CONFIG, lateral_kinetic_weight=WEIGHT_ONLY_CONTACT_HOLDS)
    post = _set(_point(0.0, forward, group=1, noise=MEASURED_NEAR_NOISE_METERS))
    k_reach = int(np.ceil(forward / (CONFIG.walking_speed_mps * CONFIG.time_step_seconds)))

    his_alone = PlannerPipeline(replace(raised, contact_term_enabled=False), WALKER).plan(post)
    with_contact = PlannerPipeline(raised, WALKER).plan(post)

    # Both halves in one test, so the scene cannot quietly stop telling the two apart.
    assert abs(his_alone.lateral_offsets_meters[k_reach]) <= WALKER.radius_meters, "the scene no longer needs the contact term"
    assert abs(with_contact.lateral_offsets_meters[k_reach]) > WALKER.radius_meters, with_contact.lateral_offsets_meters[: k_reach + 1]
    assert abs(with_contact.lateral_offsets_meters[k_reach]) > CONFIG.body_half_width_meters


def test_a_bad_contact_constant_stops_the_pipeline_at_construction() -> None:
    with pytest.raises(ValueError, match="walker_sway_meters"):
        PlannerPipeline(replace(CONFIG, walker_sway_meters=0.0), WALKER)


def test_the_alarm_does_not_depend_on_the_contact_term() -> None:
    scenes = [
        _set(*_wall_across(1.0).points, timestamp=0.0),
        _set(timestamp=0.1),
        _set(*_wall_across(1.5, gap=(0.75, 1.75)).points, timestamp=0.2),
        _set(timestamp=0.8),
        _set(_point(0.0, 0.8, group=1, noise=MEASURED_NEAR_NOISE_METERS), timestamp=0.9),
    ]
    with_contact = PlannerPipeline(CONFIG, WALKER)
    without = PlannerPipeline(replace(CONFIG, contact_term_enabled=False), WALKER)

    alarms_with = [with_contact.plan(scene).alarm for scene in scenes]
    alarms_without = [without.plan(scene).alarm for scene in scenes]

    assert alarms_with == alarms_without
    assert True in alarms_with and False in alarms_with, "the scenes must raise and clear the alarm"


def test_the_cost_stays_finite_with_the_walker_inside_an_obstacle() -> None:
    inside = _set(_point(0.0, 0.1, group=1, noise=0.0))

    path = PlannerPipeline(CONFIG, WALKER).plan(inside)
    decoded = decode_path(encode_path(path))

    assert np.isfinite(path.cumulative_cost_bits)
    assert decoded.cumulative_cost_bits == pytest.approx(path.cumulative_cost_bits)


@pytest.mark.parametrize("forward", [1.0, 1.5])
@pytest.mark.parametrize("noise", [MEASURED_NEAR_NOISE_METERS, OTHER_WALK_NEAR_NOISE_METERS])
@pytest.mark.parametrize("sway", [0.05, 0.20])
def test_the_shipped_weight_clears_the_post_under_the_margin_conditions(sway: float, noise: float, forward: float) -> None:
    # The margin on the shipped weight: half to twice the assumed sway, at either walk's near noise.
    # Twice the sway at the classroom's noise is the corner closest to failing, and it hits at a weight of 7.
    config = replace(CONFIG, walker_sway_meters=sway)
    post = _set(_point(0.0, forward, group=1, noise=noise))

    path = PlannerPipeline(config, WALKER).plan(post)

    k_reach = int(np.ceil(forward / (CONFIG.walking_speed_mps * CONFIG.time_step_seconds)))
    assert abs(path.lateral_offsets_meters[k_reach]) > WALKER.radius_meters, path.lateral_offsets_meters[: k_reach + 1]


def test_without_contact_the_shipped_weight_walks_into_the_post() -> None:
    # The shipped weight depends on the contact term. Removing it should fail here, not on a walk.
    post = _set(_point(0.0, 1.5, group=1, noise=MEASURED_NEAR_NOISE_METERS))

    path = PlannerPipeline(replace(CONFIG, contact_term_enabled=False), WALKER).plan(post)

    k_reach = int(np.ceil(1.5 / (CONFIG.walking_speed_mps * CONFIG.time_step_seconds)))
    assert abs(path.lateral_offsets_meters[k_reach]) <= WALKER.radius_meters
