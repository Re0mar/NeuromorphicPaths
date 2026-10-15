"""Covers the goal on its own: where it is, how it is clipped, and the prior it adds."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import GoalSwitches, PlannerConfig, goal_switches_from_legacy_mode
from nav.planner.goal import GoalContribution, RouteGoal, combine_goals, goal_for_frame, goal_term, goal_tolerance, route_goal_contribution
from nav.planner.field import lateral_grid

CONFIG = PlannerConfig()


def goal_of(switches: GoalSwitches, gaze: np.ndarray | None = None, route: RouteGoal | None = None) -> tuple[np.ndarray, float]:
    return goal_for_frame(switches, gaze, route, CONFIG)


def test_both_switches_off_is_straight_ahead_at_the_default_tolerance() -> None:
    goal, tolerance = goal_of(GoalSwitches(), np.array([2.0, 1.0]), RouteGoal(0.5, 1.0))
    # A gaze and a route are there, and both are ignored with their switches off.
    assert goal == pytest.approx([0.0, CONFIG.goal_distance_meters])
    assert tolerance == CONFIG.goal_tolerance_meters


def test_a_gaze_switch_off_ignores_the_gaze() -> None:
    goal, _ = goal_of(GoalSwitches(route_goal=True), np.array([2.0, 1.0]), None)
    assert goal == pytest.approx([0.0, CONFIG.goal_distance_meters])


def test_gaze_without_a_point_falls_back_to_ahead() -> None:
    goal, tolerance = goal_of(GoalSwitches(gaze_goal=True), None)
    assert goal == pytest.approx([0.0, CONFIG.goal_distance_meters])
    assert tolerance == CONFIG.goal_tolerance_meters


def test_gaze_inside_the_grid_is_used_as_it_is() -> None:
    goal, _ = goal_of(GoalSwitches(gaze_goal=True), np.array([-1.25, 3.0]))
    assert goal == pytest.approx([-1.25, 3.0])


def test_a_glance_at_the_horizon_is_clipped_into_the_grid() -> None:
    # Fifty meters out would flatten the terminal term to nothing. The goal stays at the goal
    # distance forward and inside the grid laterally, so the prior still picks among safe paths.
    far_right, _ = goal_of(GoalSwitches(gaze_goal=True), np.array([50.0, 50.0]))
    assert far_right == pytest.approx([CONFIG.grid_half_width_meters, CONFIG.goal_distance_meters])
    far_left, _ = goal_of(GoalSwitches(gaze_goal=True), np.array([-50.0, -5.0]))
    assert far_left == pytest.approx([-CONFIG.grid_half_width_meters, 0.0])


def test_one_contribution_gives_its_own_center_and_tolerance() -> None:
    goal, tolerance = goal_of(GoalSwitches(route_goal=True), route=RouteGoal(0.0, 1.7))
    assert goal == pytest.approx([0.0, CONFIG.goal_distance_meters])
    assert tolerance == 1.7


def test_two_contributions_combine_by_precision() -> None:
    # Precisions 1 and 4. The center is (1 x 1.0 + 4 x -1.0) / 5 = -0.6, the tolerance the root of 1/5.
    combined = combine_goals([GoalContribution(1.0, 4.0, 1.0), GoalContribution(-1.0, 4.0, 0.5)], CONFIG)
    assert combined.lateral_meters == pytest.approx(-0.6)
    assert combined.tolerance_meters == pytest.approx(0.4472, abs=1e-4)


def test_a_contribution_with_no_position_drops_out() -> None:
    alone = GoalContribution(0.8, 4.0, 0.9)
    assert combine_goals([None, alone], CONFIG) == alone


def test_the_gaze_and_the_route_combine_when_both_are_on() -> None:
    # The gaze 1.0 m right at the 0.4 m base tolerance, the route straight ahead at 0.8 m. Precisions
    # 6.25 and 1.5625: (6.25 x 1.0) / 7.8125 = 0.8 m right, at the root of 1 / 7.8125 = 0.358 m.
    goal, tolerance = goal_of(GoalSwitches(gaze_goal=True, route_goal=True), np.array([1.0, 4.0]), RouteGoal(0.0, 0.8))
    assert goal[0] == pytest.approx(0.8)
    assert tolerance == pytest.approx(0.3578, abs=1e-4)


def test_a_route_goal_at_30_degrees_right_gives_the_tangent_lateral() -> None:
    contribution = route_goal_contribution(RouteGoal(np.radians(30.0), 1.0), CONFIG)
    assert contribution is not None
    # 4.0 x tan 30 = 2.309 m.
    assert contribution.lateral_meters == pytest.approx(2.309, abs=1e-3)
    assert contribution.forward_meters == CONFIG.goal_distance_meters


def test_a_route_goal_behind_the_walker_sits_at_the_grid_edge() -> None:
    # 120 degrees left. The tangent there would put the goal 6.9 m to the right, behind becoming ahead.
    contribution = route_goal_contribution(RouteGoal(np.radians(-120.0), 1.0), CONFIG)
    assert contribution is not None and contribution.lateral_meters == -CONFIG.grid_half_width_meters


def test_a_route_goal_past_the_cutoff_reads_the_cutoff_from_the_config() -> None:
    # 50 degrees is inside the 60 degree default and past a 45 degree cutoff no default holds.
    inside = route_goal_contribution(RouteGoal(np.radians(50.0), 1.0), CONFIG)
    past = route_goal_contribution(RouteGoal(np.radians(50.0), 1.0), replace(CONFIG, route_goal_max_bearing_degrees=45.0))
    assert inside is not None and inside.lateral_meters == CONFIG.grid_half_width_meters  # 4 x tan 50 = 4.77, clipped
    assert past is not None and past.lateral_meters == CONFIG.grid_half_width_meters
    near = route_goal_contribution(RouteGoal(np.radians(20.0), 1.0), replace(CONFIG, route_goal_max_bearing_degrees=15.0))
    assert near is not None and near.lateral_meters == CONFIG.grid_half_width_meters


def test_the_combined_center_is_clipped_into_the_grid() -> None:
    # Each source is clipped first, and a weighted mean of points in the grid is in the grid.
    goal, _ = goal_of(GoalSwitches(gaze_goal=True, route_goal=True), np.array([40.0, 4.0]), RouteGoal(np.radians(170.0), 0.3))
    assert abs(goal[0]) <= CONFIG.grid_half_width_meters


@pytest.mark.parametrize("gaze_goal", [False, True])
@pytest.mark.parametrize("route_goal", [False, True])
def test_every_switch_combination_produces_a_goal(gaze_goal: bool, route_goal: bool) -> None:
    goal, tolerance = goal_of(GoalSwitches(gaze_goal, route_goal), np.array([0.5, 2.0]), RouteGoal(0.2, 1.1))
    assert goal.shape == (2,) and tolerance > 0.0


def test_an_old_goal_mode_maps_to_the_switches() -> None:
    assert goal_switches_from_legacy_mode("ahead") == GoalSwitches()
    assert goal_switches_from_legacy_mode("gaze") == GoalSwitches(gaze_goal=True)


def test_an_unknown_legacy_goal_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="sideways"):
        goal_switches_from_legacy_mode("sideways")


def test_the_terminal_prior_is_zero_at_the_goal_and_grows_with_distance() -> None:
    grid = lateral_grid(CONFIG)
    term = goal_term(grid, np.array([1.0, 4.0]), CONFIG.goal_tolerance_meters)

    assert term[int(np.argmin(np.abs(grid - 1.0)))] == pytest.approx(0.0, abs=1e-12)
    assert term[0] > term[int(np.argmin(np.abs(grid - 0.5)))] > 0.0
    # Half of (distance over tolerance) squared: one tolerance away costs one half.
    one_tolerance_away = int(np.argmin(np.abs(grid - (1.0 + CONFIG.goal_tolerance_meters))))
    assert term[one_tolerance_away] == pytest.approx(0.5, abs=1e-6)


def test_a_non_positive_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        goal_term(lateral_grid(CONFIG), np.array([0.0, 4.0]), 0.0)
    with pytest.raises(ValueError, match="goal_tolerance_meters"):
        goal_tolerance(PlannerConfig(goal_tolerance_meters=0.0), 0.0)


def test_with_an_empty_corridor_the_goal_keeps_its_base_tolerance() -> None:
    config = PlannerConfig(goal_tolerance_meters=0.37, goal_tolerance_growth_square_meters_per_nat=2.9)
    assert goal_tolerance(config, 0.0) == pytest.approx(0.37)


def test_the_goals_variance_grows_with_the_avoidance_surprise() -> None:
    # Worked by hand: 0.37 squared is 0.1369, plus 2.9 times 0.74 is 2.146, and the root of 2.2829 is 1.5109.
    config = PlannerConfig(goal_tolerance_meters=0.37, goal_tolerance_growth_square_meters_per_nat=2.9)
    assert goal_tolerance(config, 0.74) == pytest.approx(1.5109, abs=1e-4)


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_a_negative_growth_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="goal_tolerance_growth_square_meters_per_nat"):
        goal_tolerance(PlannerConfig(goal_tolerance_growth_square_meters_per_nat=value), 0.0)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_a_negative_or_non_finite_avoidance_surprise_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="avoidance surprise"):
        goal_tolerance(CONFIG, value)


@pytest.mark.parametrize("bearing, tolerance, field", [(3.2, 1.0, "relative_bearing_radians"), (float("nan"), 1.0, "relative_bearing_radians"), (0.5, 0.0, "tolerance_meters")])
def test_a_route_goal_with_impossible_numbers_is_refused_by_name(bearing: float, tolerance: float, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RouteGoal(bearing, tolerance)
