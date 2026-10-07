"""
Covers the dynamic program against a pure Python reference and against paths known by hand.

plan_reference is the obvious double loop from the slides. It is slow, which is why the shipping
version is vectorised, and it is kept here because the vectorised version must agree with it to
floating point on random fields.
"""

# Standard library imports
import time
from collections.abc import Callable

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.dynamic_programming import (
    backward_costs,
    forward_costs,
    plan,
    reachable_cell_offset,
    start_cell_index,
)
from nav.planner.heading import lookahead_step_index
from nav.planner.information import path_cost_through_cells, scene_information_bits
from nav.planner.field import lateral_grid, step_count

CONFIG = PlannerConfig()
GRID = lateral_grid(CONFIG)
STEPS = step_count(CONFIG)

# The speed tests keep the fastest of several batches, the way timeit does. One batch's mean also
# times whatever else the machine was doing, so a busy full-suite run read 15 ms for a plan that
# takes about 3 ms.
TIMING_BATCHES = 5
CALLS_PER_BATCH = 10


def fastest_milliseconds_per_call(function: Callable[[], None]) -> tuple[float, list[float]]:
    """
    Time a function in several batches and return the fastest batch's milliseconds per call.

    :param function: The call to time. It runs once first to warm up.
    :return: The fastest batch's per-call time, and every batch's, for the failure message.
    """
    function()
    batch_milliseconds = []
    for _ in range(TIMING_BATCHES):
        started = time.perf_counter()
        for _ in range(CALLS_PER_BATCH):
            function()
        batch_milliseconds.append((time.perf_counter() - started) * 1000.0 / CALLS_PER_BATCH)
    return min(batch_milliseconds), batch_milliseconds


def plan_reference(field: np.ndarray, start: float, grid: np.ndarray, config: PlannerConfig) -> tuple[np.ndarray, float]:
    """The double loop. One cell at a time, one predecessor at a time."""
    steps, cells = field.shape
    dt = config.time_step_seconds
    spacing = grid[1] - grid[0]
    max_offset = reachable_cell_offset(config, grid)
    cost = np.full((steps, cells), np.inf)
    parent = np.full((steps, cells), -1, dtype=int)
    start_cell = int(np.argmin(np.abs(grid - start)))
    cost[0, start_cell] = field[0, start_cell] * dt
    for k in range(1, steps):
        for ix in range(cells):
            best, best_parent = np.inf, -1
            for prev in range(max(0, ix - max_offset), min(cells, ix + max_offset + 1)):
                speed = (grid[ix] - grid[prev]) / dt
                candidate = cost[k - 1, prev] + (field[k, ix] + 0.5 * config.lateral_kinetic_weight * speed**2) * dt
                if candidate < best:
                    best, best_parent = candidate, prev
            cost[k, ix] = best
            parent[k, ix] = best_parent
    terminal = int(np.argmin(cost[-1]))
    indices = [terminal]
    for k in range(steps - 1, 0, -1):
        indices.append(parent[k, indices[-1]])
    return grid[indices[::-1]], float(cost[-1, terminal])


def _field_with_wall(lateral_from: float, lateral_to: float, value: float = 100.0) -> np.ndarray:
    field = np.zeros((STEPS, len(GRID)))
    blocked = (GRID >= lateral_from) & (GRID <= lateral_to)
    field[5:, blocked] = value
    return field


def test_a_zero_field_keeps_the_path_at_the_start() -> None:
    offsets, cost = plan(np.zeros((STEPS, len(GRID))), 0.0, GRID, CONFIG)

    assert offsets == pytest.approx(np.zeros(STEPS))
    assert cost == pytest.approx(0.0)


def test_the_path_starts_at_the_nearest_grid_cell_to_the_start() -> None:
    offsets, _ = plan(np.zeros((STEPS, len(GRID))), 0.52, GRID, CONFIG)

    assert offsets[0] == pytest.approx(0.5)


def test_a_gap_between_two_walls_is_found_and_never_entered() -> None:
    # Walls from -3 to -0.6 and from 0.6 to 3, so the only cheap lane is the centre gap. Start at
    # the left edge of the left wall, where staying would cost the wall.
    field = _field_with_wall(-3.0, -0.6) + _field_with_wall(0.6, 3.0)

    offsets, _ = plan(field, -1.0, GRID, CONFIG)

    assert np.all(np.abs(offsets[10:]) < 0.6), "the path must reach the gap and stay in it"


def test_the_lateral_step_never_exceeds_the_speed_limit() -> None:
    generator = np.random.default_rng(3)
    field = generator.random((STEPS, len(GRID))) * 50.0

    offsets, _ = plan(field, 0.0, GRID, CONFIG)

    max_step = CONFIG.max_lateral_speed_mps * CONFIG.time_step_seconds
    assert np.all(np.abs(np.diff(offsets)) <= max_step + 1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 11])
def test_the_vectorised_plan_matches_the_reference_on_random_fields(seed: int) -> None:
    generator = np.random.default_rng(seed)
    field = generator.random((12, 25)) * 10.0
    grid = np.linspace(-1.2, 1.2, 25)
    start = float(generator.uniform(-1.0, 1.0))

    offsets, cost = plan(field, start, grid, CONFIG)
    reference_offsets, reference_cost = plan_reference(field, start, grid, CONFIG)

    assert cost == pytest.approx(reference_cost)
    # Ties between equal-cost paths may break differently, so compare cost first and the path
    # only where the cost is unique. On a continuous random field ties are measure zero.
    assert offsets == pytest.approx(reference_offsets)


def test_the_vectorised_plan_matches_the_reference_on_the_default_grid() -> None:
    generator = np.random.default_rng(5)
    field = generator.random((STEPS, len(GRID))) * 10.0

    offsets, cost = plan(field, 0.0, GRID, CONFIG)
    reference_offsets, reference_cost = plan_reference(field, 0.0, GRID, CONFIG)

    assert cost == pytest.approx(reference_cost)
    assert offsets == pytest.approx(reference_offsets)


def test_the_default_grid_plans_in_under_ten_milliseconds() -> None:
    generator = np.random.default_rng(9)
    field = generator.random((STEPS, len(GRID))) * 10.0

    fastest_ms, batch_ms = fastest_milliseconds_per_call(lambda: plan(field, 0.0, GRID, CONFIG))

    assert fastest_ms < 10.0, f"{fastest_ms:.1f} ms per plan at best, batches {[round(ms, 1) for ms in batch_ms]}"


def test_a_start_outside_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the grid"):
        plan(np.zeros((STEPS, len(GRID))), CONFIG.grid_half_width_meters + 1.0, GRID, CONFIG)


def test_a_field_that_does_not_match_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="columns"):
        plan(np.zeros((STEPS, len(GRID) - 1)), 0.0, GRID, CONFIG)


def test_a_non_finite_field_is_refused() -> None:
    field = np.zeros((STEPS, len(GRID)))
    field[3, 3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        plan(field, 0.0, GRID, CONFIG)


def test_reachable_offset_is_at_least_one_cell() -> None:
    crawl = PlannerConfig(max_lateral_speed_mps=0.0001)

    assert reachable_cell_offset(crawl, GRID) == 1


@pytest.mark.parametrize(("speed", "cells"), [(1.0, 1), (2.0, 2), (3.0, 3), (2.5, 2)])
def test_reachable_offset_is_exact_at_whole_cells_per_step(speed: float, cells: int) -> None:
    # linspace puts the grid spacing a few ulp above 0.1, and a whole-number ratio floored
    # through that came out one cell short: 2.0 m/s at 0.1 s over 0.1 m gave 1 cell, not 2. The
    # default 1.0 m/s only looked right because of the clamp at one.
    assert reachable_cell_offset(PlannerConfig(max_lateral_speed_mps=speed), GRID) == cells


def backward_reference(field: np.ndarray, grid: np.ndarray, config: PlannerConfig) -> np.ndarray:
    """The backward pass as a double loop. One cell at a time, one successor at a time."""
    steps, cells = field.shape
    dt = config.time_step_seconds
    max_offset = reachable_cell_offset(config, grid)
    to_go = np.zeros((steps, cells))
    for k in range(steps - 2, -1, -1):
        for ix in range(cells):
            best = np.inf
            for following in range(max(0, ix - max_offset), min(cells, ix + max_offset + 1)):
                speed = (grid[following] - grid[ix]) / dt
                candidate = to_go[k + 1, following] + (field[k + 1, following] + 0.5 * config.lateral_kinetic_weight * speed**2) * dt
                best = min(best, candidate)
            to_go[k, ix] = best
    return to_go


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 11])
def test_forward_and_backward_costs_meet_at_the_plans_cost_on_every_step(seed: int) -> None:
    # Forward cost includes a cell's own field cost and the backward cost does not, so their sum is
    # the cheapest path through that cell with the cell counted once. The cheapest of those at any
    # step is the cheapest path overall. Counting the cell twice would come out high by its cost.
    generator = np.random.default_rng(seed)
    field = generator.random((12, 25)) * 10.0
    grid = np.linspace(-1.2, 1.2, 25)
    start = float(generator.uniform(-1.0, 1.0))

    _, plan_cost = plan(field, start, grid, CONFIG)
    forward, _ = forward_costs(field, start_cell_index(grid, start), grid, CONFIG)
    to_go = backward_costs(field, grid, CONFIG)

    for k in range(field.shape[0]):
        assert np.min(forward[k] + to_go[k]) == pytest.approx(plan_cost), f"step {k}"


def test_the_cell_the_path_passes_through_holds_that_minimum() -> None:
    generator = np.random.default_rng(4)
    field = generator.random((STEPS, len(GRID))) * 10.0
    step = lookahead_step_index(CONFIG)

    offsets, plan_cost = plan(field, 0.0, GRID, CONFIG)
    through = path_cost_through_cells(field, start_cell_index(GRID, 0.0), GRID, CONFIG, step)
    path_cell = int(np.argmin(np.abs(GRID - offsets[step])))

    assert through[path_cell] == pytest.approx(plan_cost)
    assert through[path_cell] == pytest.approx(np.min(through))


@pytest.mark.parametrize("seed", [0, 3, 8])
def test_backward_costs_match_a_plain_double_loop_reference(seed: int) -> None:
    generator = np.random.default_rng(seed)
    field = generator.random((12, 25)) * 10.0
    grid = np.linspace(-1.2, 1.2, 25)

    assert backward_costs(field, grid, CONFIG) == pytest.approx(backward_reference(field, grid, CONFIG))


def test_unreachable_cells_cost_infinity_going_forward() -> None:
    forward, _ = forward_costs(np.zeros((STEPS, len(GRID))), start_cell_index(GRID, 0.0), GRID, CONFIG)
    step = lookahead_step_index(CONFIG)

    assert int(np.isfinite(forward[step]).sum()) == 2 * step + 1


def test_the_default_grid_plans_and_measures_information_in_under_twenty_milliseconds() -> None:
    # The pipeline runs the plan, then the posterior's and the prior's forward and backward passes.
    # That's about three times the plan alone, so this budget is twice the plan's 10 ms. Idle on the
    # group laptop it takes about 8 ms, and the old 10 ms failed whenever the machine was busy.
    generator = np.random.default_rng(9)
    field = generator.random((STEPS, len(GRID))) * 10.0
    prior = np.zeros_like(field)
    start_cell = start_cell_index(GRID, 0.0)
    step = lookahead_step_index(CONFIG)

    def one_frame() -> None:
        plan(field, 0.0, GRID, CONFIG)
        scene_information_bits(field, prior, start_cell, GRID, CONFIG, step)

    fastest_ms, batch_ms = fastest_milliseconds_per_call(one_frame)

    assert fastest_ms < 20.0, (
        f"{fastest_ms:.1f} ms per plan and information at best, batches {[round(ms, 1) for ms in batch_ms]}"
    )


def test_the_fastest_batch_still_catches_a_call_slower_than_the_budget() -> None:
    # Taking the fastest batch must not hide a call that's slow every time. Busy-waiting is used
    # because time.sleep on Windows rounds up to the 15.6 ms timer tick.
    def twelve_milliseconds() -> None:
        deadline = time.perf_counter() + 0.012
        while time.perf_counter() < deadline:
            pass

    fastest_ms, _ = fastest_milliseconds_per_call(twelve_milliseconds)

    assert fastest_ms >= 12.0


def test_backward_costs_refuses_a_field_that_does_not_match_the_grid() -> None:
    with pytest.raises(ValueError, match="field has 60 columns and the grid has 61 cells"):
        backward_costs(np.zeros((STEPS, len(GRID) - 1)), GRID, CONFIG)


def test_backward_costs_refuses_a_non_finite_field() -> None:
    field = np.zeros((STEPS, len(GRID)))
    field[3, 3] = np.inf

    with pytest.raises(ValueError, match="field contains a non-finite value"):
        backward_costs(field, GRID, CONFIG)


def test_forward_costs_refuses_a_start_cell_off_the_grid() -> None:
    with pytest.raises(ValueError, match="start cell 61 is not one of the grid's 61 cells"):
        forward_costs(np.zeros((STEPS, len(GRID))), len(GRID), GRID, CONFIG)


def test_a_field_of_the_wrong_shape_is_reported_before_a_start_off_the_grid() -> None:
    # Both are wrong here. The shape is the more basic fault, and plan checks it first so its error
    # names the shape rather than whatever the start check trips on.
    with pytest.raises(ValueError, match="field has 60 columns and the grid has 61 cells"):
        plan(np.zeros((STEPS, len(GRID) - 1)), CONFIG.grid_half_width_meters + 1.0, GRID, CONFIG)
