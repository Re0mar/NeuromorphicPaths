"""
The field-term accumulator: how each declaration is charged, summed and left out of the work figure.

The fields here are small and written by hand, so each expected value is read off the arrays.
"""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.field_terms import (
    GOAL,
    PREVIOUS_PLAN,
    SCENE,
    Charging,
    TermDeclaration,
    TermField,
    WorkFigure,
    as_planned,
    charges_left_out,
    last_row_only,
    sum_terms,
)

# A time step no default holds, so a factor read from the wrong place shows.
CONFIG = replace(PlannerConfig(), time_step_seconds=0.25)
GRID = np.array([-0.1, 0.0, 0.1])


def test_a_rate_is_planned_as_it_is() -> None:
    field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(as_planned(field, SCENE, CONFIG), field)


def test_a_terminal_cost_is_divided_by_the_time_step_on_its_last_row_only() -> None:
    field = last_row_only(np.array([1.0, 2.0, 3.0]), 3)
    planned = as_planned(field, GOAL, CONFIG)
    np.testing.assert_array_equal(planned[:-1], np.zeros((2, 3)))
    np.testing.assert_array_equal(planned[-1], [4.0, 8.0, 12.0])
    # The term's own field is left as it was, for the kept sum.
    np.testing.assert_array_equal(field[-1], [1.0, 2.0, 3.0])


def test_the_three_sums_read_each_declaration() -> None:
    scene = np.ones((3, 3))
    goal = last_row_only(np.array([2.0, 0.0, 2.0]), 3)
    goal_prior = last_row_only(np.array([1.0, 0.0, 1.0]), 3)
    previous = np.full((3, 3), 0.5)
    summed = sum_terms([TermField(SCENE, scene, None), TermField(GOAL, goal, goal_prior), TermField(PREVIOUS_PLAN, previous, previous)], CONFIG)
    # Kept: every term as it is. The last row is 1 + 2 + 0.5 at the edges.
    np.testing.assert_array_equal(summed.kept[-1], [3.5, 1.5, 3.5])
    np.testing.assert_array_equal(summed.kept[0], [1.5, 1.5, 1.5])
    # Planned: the goal's 2 is 8 once divided by the 0.25 s step.
    np.testing.assert_array_equal(summed.planned[-1], [9.5, 1.5, 9.5])
    # Prior: no scene, the goal's prior row divided the same way, and the previous plan.
    np.testing.assert_array_equal(summed.prior[-1], [4.5, 0.5, 4.5])
    np.testing.assert_array_equal(summed.prior[0], [0.5, 0.5, 0.5])


def test_the_sum_matches_adding_the_terms_by_hand_to_the_bit() -> None:
    rng = np.random.default_rng(7)
    scene, previous = rng.random((4, 3)), rng.random((4, 3))
    goal_row = rng.random(3)
    summed = sum_terms([TermField(SCENE, scene, None), TermField(GOAL, last_row_only(goal_row, 4), None), TermField(PREVIOUS_PLAN, previous, previous)], CONFIG)
    by_hand = scene.copy()
    by_hand[-1] += goal_row / CONFIG.time_step_seconds
    by_hand += previous
    np.testing.assert_array_equal(summed.planned, by_hand)


def test_the_work_figure_leaves_out_what_its_declaration_says() -> None:
    scene = np.full((3, 3), 9.0)
    goal = last_row_only(np.array([0.0, 1.0, 2.0]), 3)
    previous = np.full((3, 3), 0.5)
    offsets = np.array([0.0, 0.1, 0.1])
    charges = charges_left_out([TermField(SCENE, scene, None), TermField(GOAL, goal, None), TermField(PREVIOUS_PLAN, previous, previous)], offsets, GRID, CONFIG)
    # The scene is counted, so only the goal and the previous plan come back, in that order. The goal
    # ends at 0.1 m, worth 2, paid in full. The previous plan's three rows of 0.5 are rates, 3 x 0.5 x 0.25.
    assert charges == pytest.approx([2.0, 0.375])


def test_a_new_term_needs_only_its_declaration() -> None:
    # A preference charged per second and counted as work, the shape a terrain term would take.
    terrain = TermDeclaration("terrain", Charging.RATE, WorkFigure.COUNTED)
    field = np.full((2, 3), 0.25)
    summed = sum_terms([TermField(SCENE, np.zeros((2, 3)), None), TermField(terrain, field, None)], CONFIG)
    np.testing.assert_array_equal(summed.planned, field)
    assert charges_left_out([TermField(terrain, field, None)], np.zeros(2), GRID, CONFIG) == []


def test_no_terms_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one term"):
        sum_terms([], CONFIG)


def test_terms_of_different_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="goal"):
        sum_terms([TermField(SCENE, np.zeros((3, 3)), None), TermField(GOAL, np.zeros((2, 3)), None)], CONFIG)
