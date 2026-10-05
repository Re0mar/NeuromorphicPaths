"""
The minimum accumulated cost path through the surprise field.

Each step costs the field value at the position reached plus a lateral kinetic term, half a
weight times lateral speed squared, both integrated over the time step. A step may move at most
as many cells sideways as the lateral speed limit allows. Dynamic programming over (step, cell)
with parent pointers gives the cheapest path to every terminal cell, and the cheapest terminal
wins.

This is the professor's formulation, written from the slides. The vectorized inner loop below
gives the same result as the obvious double loop, which the test file keeps as the reference.

The same recurrence also runs backward, from the end of the horizon. Forward cost to a cell plus
backward cost from it is the cheapest whole path through that cell, which is what the planner's
information measure reads.
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


def start_cell_index(grid: np.ndarray, start_lateral_meters: float) -> int:
    """
    The grid cell a path starting at this lateral position starts in: the nearest one.

    :param grid: The lateral candidates.
    :param start_lateral_meters: Where the walker is now.
    :return: The index of the nearest cell.
    :rtype: int
    :raises ValueError: When the start is outside the grid.
    """
    if not grid[0] <= start_lateral_meters <= grid[-1]:
        raise ValueError(f"start {start_lateral_meters} m is outside the grid [{grid[0]}, {grid[-1]}]")
    return int(np.argmin(np.abs(grid - start_lateral_meters)))


def forward_costs(field: np.ndarray, start_cell: int, grid: np.ndarray, config: PlannerConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    The cheapest cost of arriving at every cell at every step, and the cell each arrival came from.

    :param field: (steps, cells) surprise, with any goal term already added.
    :param start_cell: The cell the path starts in, from start_cell_index.
    :param grid: The lateral candidates the field's columns correspond to.
    :param config: Time step, kinetic weight, lateral speed limit.
    :return: (cost, parent), both (steps, cells). Cost includes the arrival cell's own field cost
        for that step. A cell the start cannot reach in time costs inf and has parent -1.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: When the field does not match the grid, or the start cell is not on it.
    """
    _check_field(field, grid)
    steps, cells = field.shape
    if not 0 <= start_cell < cells:
        raise ValueError(f"start cell {start_cell} is not one of the grid's {cells} cells")

    dt = config.time_step_seconds
    offsets, kinetic = _lateral_moves(config, grid)

    cost = np.full((steps, cells), np.inf)
    parent = np.full((steps, cells), -1, dtype=np.int32)
    cost[0, start_cell] = field[0, start_cell] * dt

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

    return cost, parent


def backward_costs(field: np.ndarray, grid: np.ndarray, config: PlannerConfig) -> np.ndarray:
    """
    The cheapest cost of the rest of the path, from standing on every cell at every step.

    The cell's own field cost at that step is not included, because the forward cost already
    counts it. So forward plus backward at one (step, cell) is the cheapest whole path through it.

    :param field: (steps, cells) surprise, with any goal term already added.
    :param grid: The lateral candidates the field's columns correspond to.
    :param config: Time step, kinetic weight, lateral speed limit.
    :return: (steps, cells). The last row is zero, since nothing is left to pay for there.
    :rtype: np.ndarray
    :raises ValueError: When the field does not match the grid.
    """
    _check_field(field, grid)
    steps, cells = field.shape
    dt = config.time_step_seconds
    offsets, kinetic = _lateral_moves(config, grid)

    to_go = np.zeros((steps, cells))
    for k in range(steps - 2, -1, -1):
        # What the next step costs on arrival at each cell, plus everything after it.
        ahead = to_go[k + 1] + field[k + 1] * dt
        best = np.full(cells, np.inf)

        # Cell ix moves on to cell ix + d.
        for d, kinetic_cost in zip(offsets, kinetic, strict=True):
            candidate = np.full(cells, np.inf)
            if d >= 0:
                candidate[: cells - d] = ahead[d:] if d > 0 else ahead
            else:
                candidate[-d:] = ahead[:d]
            best = np.minimum(best, candidate + kinetic_cost * dt)

        to_go[k] = best

    return to_go


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
    # Checked before the start is resolved, so a field of the wrong shape is reported as that and
    # not as whatever the start check happens to trip on.
    _check_field(field, grid)
    cost, parent = forward_costs(field, start_cell_index(grid, start_lateral_meters), grid, config)
    steps = field.shape[0]

    terminal = int(np.argmin(cost[-1]))
    indices = np.empty(steps, dtype=np.int64)
    indices[-1] = terminal
    for k in range(steps - 1, 0, -1):
        indices[k - 1] = parent[k, indices[k]]

    return grid[indices], float(cost[-1, terminal])


def _check_field(field: np.ndarray, grid: np.ndarray) -> None:
    """Refuse a field the recurrence cannot run over. Shared, so every pass refuses the same things."""
    steps, cells = field.shape
    if cells != len(grid):
        raise ValueError(f"field has {cells} columns and the grid has {len(grid)} cells")
    if steps < 1:
        raise ValueError("field must have at least one step")
    if not np.all(np.isfinite(field)):
        raise ValueError("field contains a non-finite value")


def _lateral_moves(config: PlannerConfig, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Every sideways move one step may make, in cells, and its kinetic cost rate.

    One place for both, so the forward and backward passes cannot disagree on reach or price.

    :return: (offsets, kinetic), each (2 * reach + 1,). Kinetic is per second, before dt.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    dt = config.time_step_seconds
    max_offset = reachable_cell_offset(config, grid)
    spacing = float(grid[1] - grid[0]) if len(grid) > 1 else config.grid_spacing_meters
    # Each candidate offset d is a lateral speed of d spacing / dt, and the kinetic term for it is
    # the same for every cell, so it is one number per offset.
    offsets = np.arange(-max_offset, max_offset + 1)
    kinetic = 0.5 * config.lateral_kinetic_weight * (offsets * spacing / dt) ** 2
    return offsets, kinetic
