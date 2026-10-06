"""Covers the goal on its own: where it is, how it is clipped, and the prior it adds."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.goal import goal_position, goal_term
from nav.planner.field import lateral_grid

CONFIG = PlannerConfig()


def test_ahead_is_straight_ahead_at_the_goal_distance() -> None:
    assert goal_position(GoalMode.AHEAD, CONFIG, None) == pytest.approx([0.0, CONFIG.goal_distance_meters])
    # A gaze point is ignored in AHEAD mode, which is what makes the mode a choice.
    assert goal_position(GoalMode.AHEAD, CONFIG, np.array([2.0, 1.0])) == pytest.approx([0.0, CONFIG.goal_distance_meters])


def test_gaze_without_a_point_falls_back_to_ahead() -> None:
    assert goal_position(GoalMode.GAZE, CONFIG, None) == pytest.approx([0.0, CONFIG.goal_distance_meters])


def test_gaze_inside_the_grid_is_used_as_it_is() -> None:
    assert goal_position(GoalMode.GAZE, CONFIG, np.array([-1.25, 3.0])) == pytest.approx([-1.25, 3.0])


def test_a_glance_at_the_horizon_is_clipped_into_the_grid() -> None:
    # Fifty meters out would flatten the terminal term to nothing. The goal stays at the goal
    # distance forward and inside the grid laterally, so the prior still picks among safe paths.
    far_right = goal_position(GoalMode.GAZE, CONFIG, np.array([50.0, 50.0]))
    assert far_right == pytest.approx([CONFIG.grid_half_width_meters, CONFIG.goal_distance_meters])

    far_left = goal_position(GoalMode.GAZE, CONFIG, np.array([-50.0, -5.0]))
    assert far_left == pytest.approx([-CONFIG.grid_half_width_meters, 0.0])


def test_every_goal_mode_produces_a_goal() -> None:
    # A member added without a rule fails here rather than in the first run that selects it.
    for mode in GoalMode:
        assert goal_position(mode, CONFIG, np.array([0.5, 2.0])).shape == (2,)


def test_the_terminal_prior_is_zero_at_the_goal_and_grows_with_distance() -> None:
    grid = lateral_grid(CONFIG)
    term = goal_term(grid, np.array([1.0, 4.0]), CONFIG)

    assert term[int(np.argmin(np.abs(grid - 1.0)))] == pytest.approx(0.0, abs=1e-12)
    assert term[0] > term[int(np.argmin(np.abs(grid - 0.5)))] > 0.0
    # Half of (distance over tolerance) squared: one tolerance away costs one half.
    one_tolerance_away = int(np.argmin(np.abs(grid - (1.0 + CONFIG.goal_tolerance_meters))))
    assert term[one_tolerance_away] == pytest.approx(0.5, abs=1e-6)


def test_a_non_positive_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        goal_term(lateral_grid(CONFIG), np.array([0.0, 4.0]), PlannerConfig(goal_tolerance_meters=0.0))
