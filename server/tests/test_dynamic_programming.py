"""
Covers the dynamic program against a pure Python reference and against paths known by hand.

plan_reference is the obvious double loop from the slides. It is slow, which is why the shipping
version is vectorised, and it is kept here because the vectorised version must agree with it to
floating point on random fields.
"""

# Standard library imports
import time

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.dynamic_programming import plan, reachable_cell_offset
from nav.planner.surprise import lateral_grid, step_count

CONFIG = PlannerConfig()
GRID = lateral_grid(CONFIG)
STEPS = step_count(CONFIG)


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
    plan(field, 0.0, GRID, CONFIG)  # warm up

    started = time.perf_counter()
    for _ in range(10):
        plan(field, 0.0, GRID, CONFIG)
    per_call_ms = (time.perf_counter() - started) * 100

    assert per_call_ms < 10.0, f"{per_call_ms:.1f} ms per plan"


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
