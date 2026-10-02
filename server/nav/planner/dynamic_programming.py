"""
The minimum accumulated cost path through the surprise field.

Each step costs the field value at the position reached plus a lateral kinetic term, half a
weight times lateral speed squared, both integrated over the time step. A step may move at most
as many cells sideways as the lateral speed limit allows. Dynamic programming over (step, cell)
with parent pointers gives the cheapest path to every terminal cell, and the cheapest terminal
wins.

This is the professor's formulation, written from the slides. The vectorized inner loop below
gives the same result as the obvious double loop, which the test file keeps as the reference.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig


# linspace puts the grid spacing a few ulp off the configured value, so an exact ratio such as 2.0
# cells per step floored to 1 without this. A millionth of a cell is far below any real speed.
REACH_ROUNDING_TOLERANCE_CELLS = 1e-6


def reachable_cell_offset(config: PlannerConfig, grid: np.ndarray) -> int:
    """How many cells sideways one time step may move, at most. Never less than one."""
    spacing = float(grid[1] - grid[0]) if len(grid) > 1 else config.grid_spacing_meters
    cells_per_step = config.max_lateral_speed_mps * config.time_step_seconds / spacing
    return max(1, int(np.floor(cells_per_step + REACH_ROUNDING_TOLERANCE_CELLS)))


def plan(field: np.ndarray, start_lateral_meters: float, grid: np.ndarray, config: PlannerConfig) -> tuple[np.ndarray, float]:
    """
    Find the cheapest lateral path through the field.

    :param field: (steps, cells) surprise, from surprise_field, with any goal term already added.
    :param start_lateral_meters: Where the walker is now. Must lie inside the grid.
    :param grid: The lateral candidates the field's columns correspond to.
    :param config: Time step, kinetic weight, lateral speed limit.
    :return: (lateral offset per step, accumulated cost of the chosen path).
    :rtype: tuple[np.ndarray, float]
    :raises ValueError: When the start is outside the grid or the field does not match it.
    """
    steps, cells = field.shape
    if cells != len(grid):
        raise ValueError(f"field has {cells} columns and the grid has {len(grid)} cells")
    if steps < 1:
        raise ValueError("field must have at least one step")
    if not grid[0] <= start_lateral_meters <= grid[-1]:
        raise ValueError(f"start {start_lateral_meters} m is outside the grid [{grid[0]}, {grid[-1]}]")
    if not np.all(np.isfinite(field)):
        raise ValueError("field contains a non-finite value")

    dt = config.time_step_seconds
    max_offset = reachable_cell_offset(config, grid)
    spacing = float(grid[1] - grid[0]) if cells > 1 else config.grid_spacing_meters

    cost = np.full((steps, cells), np.inf)
    parent = np.full((steps, cells), -1, dtype=np.int32)

    start_cell = int(np.argmin(np.abs(grid - start_lateral_meters)))
    cost[0, start_cell] = field[0, start_cell] * dt

    # Each candidate offset d is a lateral speed of d spacing / dt, and the kinetic term for it is
    # the same for every cell, so it is one number per offset.
    offsets = np.arange(-max_offset, max_offset + 1)
    kinetic = 0.5 * config.lateral_kinetic_weight * (offsets * spacing / dt) ** 2

    for k in range(1, steps):
        previous = cost[k - 1]
        best = np.full(cells, np.inf)
        best_parent = np.full(cells, -1, dtype=np.int32)

        # Slide the previous row by each offset. Cell ix reached from cell ix - d.
        for d, kinetic_cost in zip(offsets, kinetic, strict=True):
            candidate = np.full(cells, np.inf)
            if d >= 0:
                candidate[d:] = previous[: cells - d] if d > 0 else previous
            else:
                candidate[:d] = previous[-d:]
            candidate = candidate + (field[k] + kinetic_cost) * dt

            better = candidate < best
            best = np.where(better, candidate, best)
            best_parent = np.where(better, np.arange(cells) - d, best_parent)

        cost[k] = best
        parent[k] = best_parent

    terminal = int(np.argmin(cost[-1]))
    indices = np.empty(steps, dtype=np.int64)
    indices[-1] = terminal
    for k in range(steps - 1, 0, -1):
        indices[k - 1] = parent[k, indices[k]]

    return grid[indices], float(cost[-1, terminal])
