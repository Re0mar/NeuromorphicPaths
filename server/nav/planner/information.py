"""
How much the scene shaped the plan, in bits: the professor's Bayesian surprise of the plan.

At one step of the horizon, every reachable cell has a cheapest whole path through it. Read as a
distribution, p proportional to 2^(-cost), that is the planner's posterior over where the walker
will be then. The prior is the same thing for a planner that sees nothing, with only the cost of
moving sideways and the goal. The KL divergence of the posterior from the prior says how far what
the camera saw moved the plan.

It is not certainty. An empty corridor gives a straight plan the planner is completely sure of, and
0 bits, because nothing in view moved it anywhere.

The ceiling is not log2 of the cell count. That would hold against a uniform prior, and this prior
favors the center because every sideways cell costs effort. A plan pinned to one cell reads minus
log2 of that cell's prior probability, which is largest at the edge.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.dynamic_programming import backward_costs, forward_costs


def path_cost_through_cells(
    field: np.ndarray,
    start_cell: int,
    grid: np.ndarray,
    config: PlannerConfig,
    step_index: int,
) -> np.ndarray:
    """
    The cheapest whole path's cost through each cell at one step.

    :param field: (steps, cells), goal term included.
    :param start_cell: Where the path starts, from start_cell_index.
    :param grid: The lateral candidates.
    :param config: Time step, kinetic weight, lateral speed limit.
    :param step_index: The step to read, 0 to steps - 1.
    :return: (cells,) cost, in the units the dynamic program accumulates. inf where the start cannot
        reach the cell by then.
    :rtype: np.ndarray
    :raises ValueError: When the step is not one of the field's.
    """
    steps = field.shape[0]
    if not 0 <= step_index < steps:
        raise ValueError(f"step {step_index} is not one of the field's {steps} steps")
    # Arriving at the step only reads the rows up to it, and leaving it only reads the rows after,
    # so each pass runs over its half of the field. Together they cost one full pass, not two.
    forward, _ = forward_costs(field[: step_index + 1], start_cell, grid, config)
    to_go = backward_costs(field[step_index:], grid, config)
    return forward[-1] + to_go[0]


def softmin_distribution(costs: np.ndarray) -> np.ndarray:
    """
    p proportional to 2^(-cost) over the finite costs, zero where the cost is inf.

    The smallest cost is subtracted first, so large costs cannot underflow every weight to zero.

    :param costs: (cells,).
    :return: (cells,), summing to 1.
    :rtype: np.ndarray
    :raises ValueError: When no cost is finite, so there is nothing to spread the probability over.
    """
    finite = np.isfinite(costs)
    if not finite.any():
        raise ValueError("no cell is reachable, so the costs give no distribution")
    weights = np.zeros_like(costs, dtype=np.float64)
    weights[finite] = np.exp2(-(costs[finite] - costs[finite].min()))
    return weights / weights.sum()


def scene_information_bits(
    posterior_field: np.ndarray,
    prior_field: np.ndarray,
    start_cell: int,
    grid: np.ndarray,
    config: PlannerConfig,
    step_index: int,
) -> float:
    """
    KL divergence of the plan's posterior from its prior at one step, in bits.

    :param posterior_field: The field the planner planned through, goal term included.
    :param prior_field: The same with no obstacle terms: zero but for the goal term.
    :param start_cell: Where both paths start.
    :param grid: The lateral candidates.
    :param config: Time step, kinetic weight, lateral speed limit.
    :param step_index: The step to compare at, the arrow's lookahead.
    :return: Bits, zero or more.
    :rtype: float
    :raises ValueError: When the fields differ in shape, or a cell the posterior can reach is one
        the prior cannot, which would make the divergence infinite.
    """
    if posterior_field.shape != prior_field.shape:
        raise ValueError(f"posterior field {posterior_field.shape} and prior field {prior_field.shape} must match")
    posterior = softmin_distribution(path_cost_through_cells(posterior_field, start_cell, grid, config, step_index))
    prior = softmin_distribution(path_cost_through_cells(prior_field, start_cell, grid, config, step_index))

    # 0 log 0 is 0, so cells the posterior gives nothing take no part.
    carried = posterior > 0.0
    if np.any(prior[carried] == 0.0):
        raise ValueError("the posterior reaches a cell the prior gives no probability, so the divergence is infinite")
    divergence = float(np.sum(posterior[carried] * np.log2(posterior[carried] / prior[carried])))
    # A divergence is never negative. Identical distributions can round a hair below zero.
    return max(divergence, 0.0)
