"""
Covers how much the scene shaped the plan, on hand-built fields at the default grid.

The posterior here is always built the way the pipeline builds it, an obstacle field with the goal
row added, and the prior is the same goal row on a zero field. The pinned-cell cases compute their
expected value from the prior's own distribution rather than quoting a number, so a change to the
kinetic weight moves the expectation with it.
"""

# Standard library imports
import math

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.dynamic_programming import backward_costs, forward_costs, start_cell_index
from nav.planner.field import lateral_grid, step_count
from nav.planner.goal import goal_term
from nav.planner.heading import lookahead_step_index
from nav.planner.information import path_cost_through_cells, scene_information_bits, softmin_distribution

CONFIG = PlannerConfig()
GRID = lateral_grid(CONFIG)
STEPS = step_count(CONFIG)
STEP = lookahead_step_index(CONFIG)
START = start_cell_index(GRID, 0.0)
# Far beyond anything a real surprise reaches, so a blocked cell carries no probability at all.
BLOCKED = 1.0e4


def _goal_only(goal_lateral: float = 0.0) -> np.ndarray:
    field = np.zeros((STEPS, len(GRID)))
    field[-1] += goal_term(GRID, np.array([goal_lateral, CONFIG.goal_distance_meters]), CONFIG)
    return field


def _prior_distribution(prior_field: np.ndarray) -> np.ndarray:
    return softmin_distribution(path_cost_through_cells(prior_field, START, GRID, CONFIG, STEP))


def _pinned_to(cell: int) -> np.ndarray:
    """The posterior field with every cell at the lookahead blocked but one."""
    field = _goal_only()
    blocked = np.ones(len(GRID), dtype=bool)
    blocked[cell] = False
    field[STEP, blocked] += BLOCKED
    return field


def test_an_empty_scene_carries_no_information() -> None:
    assert scene_information_bits(_goal_only(), _goal_only(), START, GRID, CONFIG, STEP) == pytest.approx(0.0, abs=1e-12)


def test_a_gaze_goal_in_an_empty_scene_carries_no_information() -> None:
    # The case the prior keeps the goal for. With the goal off to the right and nothing in view, the
    # plan bends right because of the gaze, and the scene did nothing.
    gaze = _goal_only(goal_lateral=1.5)

    assert scene_information_bits(gaze, gaze, START, GRID, CONFIG, STEP) == pytest.approx(0.0, abs=1e-12)
    # A prior without the goal would credit that bend to the scene. Shown here so the fixture is
    # known to reach the rule: the two priors give different answers on it. The difference is small
    # with today's weights, about 0.0003 bits, because the goal row is paid for one time step only.
    kinetic_only = np.zeros((STEPS, len(GRID)))
    assert scene_information_bits(gaze, kinetic_only, START, GRID, CONFIG, STEP) > 1.0e-5


def test_a_wall_beside_the_path_carries_information() -> None:
    field = _goal_only()
    field[5:, GRID >= 0.2] += 50.0
    prior = _prior_distribution(_goal_only())
    # The most any posterior can read: all of it on the cell the prior thinks least likely.
    ceiling = -math.log2(float(prior[prior > 0.0].min()))

    information = scene_information_bits(field, _goal_only(), START, GRID, CONFIG, STEP)

    assert 0.0 < information <= ceiling


def test_a_plan_pinned_to_one_cell_reads_minus_log2_of_that_cells_prior() -> None:
    edge = START - STEP
    expected = -math.log2(float(_prior_distribution(_goal_only())[edge]))

    information = scene_information_bits(_pinned_to(edge), _goal_only(), START, GRID, CONFIG, STEP)

    assert information == pytest.approx(expected, rel=1e-9)


def test_the_uniform_bound_is_not_a_ceiling() -> None:
    # log2 of the 21 reachable cells is the most a KL can read against a uniform prior. This prior
    # favors the center, so a plan pinned to the edge reads more. A clamp to the uniform bound
    # anywhere would fail here.
    edge = START - STEP
    reachable = 2 * STEP + 1

    information = scene_information_bits(_pinned_to(edge), _goal_only(), START, GRID, CONFIG, STEP)

    assert information > math.log2(reachable)


def test_softmin_is_stable_for_large_costs() -> None:
    costs = np.array([3.0, 4.0, 5.5, np.inf])

    assert softmin_distribution(costs + 1.0e4) == pytest.approx(softmin_distribution(costs))
    assert softmin_distribution(costs).sum() == pytest.approx(1.0)
    assert softmin_distribution(costs)[-1] == 0.0


def test_softmin_refuses_a_row_with_no_reachable_cell() -> None:
    with pytest.raises(ValueError, match="no cell is reachable"):
        softmin_distribution(np.full(5, np.inf))


def test_a_posterior_cell_the_prior_cannot_reach_is_refused_not_infinite() -> None:
    # A prior cost so large its probability underflows to exactly zero, where the posterior still
    # puts some. The divergence there is infinite, and returning inf would reach the wire.
    prior = _goal_only()
    prior[STEP, START] += 1.0e7

    with pytest.raises(ValueError, match="divergence is infinite"):
        scene_information_bits(_goal_only(), prior, START, GRID, CONFIG, STEP)


def test_fields_of_different_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match=r"posterior field \(39, 61\) and prior field \(38, 61\) must match"):
        scene_information_bits(_goal_only(), _goal_only()[:-1], START, GRID, CONFIG, STEP)


def test_a_step_past_the_horizon_is_refused() -> None:
    with pytest.raises(ValueError, match="step 39 is not one of the field's 39 steps"):
        path_cost_through_cells(_goal_only(), START, GRID, CONFIG, STEPS)


def test_the_halved_passes_match_the_full_passes_at_every_step() -> None:
    # path_cost_through_cells runs each pass over only its half of the field. On a random field that
    # must equal forward plus backward over the whole field, at every step, the first and last included.
    field = np.random.default_rng(12).random((STEPS, len(GRID))) * 10.0
    forward, _ = forward_costs(field, START, GRID, CONFIG)
    to_go = backward_costs(field, GRID, CONFIG)

    for step in range(STEPS):
        assert path_cost_through_cells(field, START, GRID, CONFIG, step) == pytest.approx(forward[step] + to_go[step]), f"step {step}"


def test_the_distribution_is_in_base_two() -> None:
    # The number is quoted in bits, so a cost one bit higher must carry half the probability. Base e
    # would pass every other test here, because they compare the code against itself.
    assert softmin_distribution(np.array([0.0, 1.0, np.inf])) == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 0.0])


def test_a_negative_step_is_refused_naming_the_step() -> None:
    # A negative step would otherwise slice an empty field and fail later as "at least one step",
    # which names the wrong thing.
    with pytest.raises(ValueError, match="step -1 is not one of the field's 39 steps"):
        path_cost_through_cells(_goal_only(), START, GRID, CONFIG, -1)
