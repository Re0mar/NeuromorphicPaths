"""
The field the dynamic program plans through: a cost rate for every future time slice and every
lateral position.

The geometry lives here once. The walker advances at walking speed, so at slice k it is k dt v
further forward and every obstacle is that much nearer. A group with a known velocity is slid by it
when motion prediction is on. Everything else is held where it is. Each cost term is handed the
center distance from every candidate position to every point, and the noise of every point, and
says what each point costs per second.

A term's points are combined the way the professor combines his: the most costly point within a
group, then the sum across groups. That happens per term, because one term's worst point need not
be another's, and each term is separate evidence about the same scene. Today the scene hands over
one point per group, its nearest, so the maximum within a group is that one point. It's written as a
maximum so a scene that sends several points per group needs no change here.
"""

# Standard library imports
from collections.abc import Sequence
from typing import Protocol

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig


class CostTerm(Protocol):
    """One kind of cost a point carries, as a rate per second, before the per-group combination."""

    def point_cost(
        self,
        center_distance_meters: np.ndarray,
        noise_meters: np.ndarray,
        config: PlannerConfig,
        walker: WalkerConfig,
    ) -> np.ndarray:
        """
        The cost rate of every point seen from every candidate position, at one time slice.

        :param center_distance_meters: (cells, points) distance from each candidate position to each point.
        :param noise_meters: (points,) each point's N, walls already multiplied.
        :param config: The planner's constants.
        :param walker: The footprint.
        :return: (cells, points) cost per second.
        :rtype: np.ndarray
        """
        ...


def effective_noise(point: ObstaclePoint, config: PlannerConfig) -> float:
    """
    The N the planner uses for a point. A wall's is multiplied, because brushing one costs more.

    It lives with the field rather than with either term because every term reads the same N, so
    a wall is treated one way whichever term is looking at it.

    :param point: The noise scale and wall flag.
    :param config: The wall multiplier.
    :return: N in meters.
    :rtype: float
    """
    if point.is_wall:
        return point.noise_scale_meters * config.wall_noise_multiplier
    return point.noise_scale_meters


def lateral_grid(config: PlannerConfig) -> np.ndarray:
    """The candidate lateral positions, centered on straight ahead, spacing from the config."""
    if config.grid_spacing_meters <= 0 or config.grid_half_width_meters <= 0:
        raise ValueError("grid spacing and half width must be positive")
    count = int(round(2 * config.grid_half_width_meters / config.grid_spacing_meters)) + 1
    return np.linspace(-config.grid_half_width_meters, config.grid_half_width_meters, count)


def step_count(config: PlannerConfig) -> int:
    """How many future time slices the horizon holds, including now."""
    if config.time_step_seconds <= 0 or config.horizon_seconds <= 0:
        raise ValueError("time step and horizon must be positive")
    return int(round(config.horizon_seconds / config.time_step_seconds)) + 1


def cost_field(
    obstacles: ObstacleSet,
    grid: np.ndarray,
    config: PlannerConfig,
    walker: WalkerConfig,
    terms: Sequence[CostTerm],
) -> np.ndarray:
    """
    Every term's cost at every candidate lateral position, for every future slice of the horizon.

    :param obstacles: This frame's groups, walker relative.
    :param grid: The lateral candidates, from lateral_grid.
    :param config: Horizon, speed, prediction flag, and whatever the terms read.
    :param walker: The footprint, handed to each term.
    :param terms: The costs to add up. At least one.
    :return: (steps, len(grid)) field. Zero everywhere when nothing is in view.
    :rtype: np.ndarray
    :raises ValueError: When terms is empty.
    """
    # A field with no terms prices every route at zero, which would let the plan walk through anything.
    if not terms:
        raise ValueError("cost_field needs at least one term, got none")

    steps = step_count(config)
    field = np.zeros((steps, len(grid)), dtype=np.float64)
    if not obstacles.points:
        return field

    # Points sorted by group, so each group is one contiguous run of columns and its maximum is one
    # reduceat. The sort is stable, so points keep their order within a group.
    group_ids = np.array([point.group_id for point in obstacles.points])
    order = np.argsort(group_ids, kind="stable")
    points = [obstacles.points[index] for index in order]
    group_starts = np.flatnonzero(np.r_[True, np.diff(group_ids[order]) != 0])

    lateral = np.array([point.lateral_meters for point in points])
    forward = np.array([point.forward_meters for point in points])
    noise = np.array([effective_noise(point, config) for point in points])

    velocity = np.zeros((len(points), 2))
    if config.predict_motion:
        for index, point in enumerate(points):
            if point.velocity_mps is not None:
                velocity[index] = point.velocity_mps

    for k in range(steps):
        elapsed = k * config.time_step_seconds
        walker_forward = config.walking_speed_mps * elapsed
        point_lateral = lateral + velocity[:, 0] * elapsed
        point_forward = forward + velocity[:, 1] * elapsed - walker_forward

        # (cells, points): distance from each candidate position to each point.
        center_distance = np.hypot(grid[:, None] - point_lateral[None, :], point_forward[None, :])

        for term in terms:
            per_point = term.point_cost(center_distance, noise, config, walker)
            # Max within each group, then the sum across groups. cumsum adds the groups strictly left
            # to right, the order a loop over them would, so the result is exact to the bit rather
            # than off by the rounding a pairwise sum would introduce.
            per_group = np.maximum.reduceat(per_point, group_starts, axis=1)
            field[k] += np.cumsum(per_group, axis=1)[:, -1]

    return field
