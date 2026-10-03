"""
Collision surprise, and the field of it over future time and lateral position.

A reading is surprising in proportion to how uncertain it is relative to how close it is. With
clearance S and noise scale N, both in meters, the surprise of one point is half of (N over S)
squared. A post two meters away whose distance wobbles by a centimeter is barely surprising. One
whose distance wobbles by half a meter is, and the planner steers around the wobble.

Each obstacle is represented by its most surprising point, and independent obstacles add. That
is the maximum within a group and the sum across groups, in that order. The geometry and that
combination live in field.py, which every cost term shares.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.field import cost_field
from nav.types import ObstacleSet
from nav.walker import WalkerConfig


def point_surprise(clearance_meters: np.ndarray | float, noise_scale_meters: np.ndarray | float, config: PlannerConfig) -> np.ndarray | float:
    """
    Half of (N over S) squared, with S floored at the epsilon and the result capped.

    Works on scalars and on arrays of any matching shape.

    :param clearance_meters: S, distance from the footprint edge.
    :param noise_scale_meters: N, how much S has been wobbling.
    :param config: The epsilon under S, the epsilon under N, and the cap.
    :return: Surprise, dimensionless.
    :rtype: np.ndarray | float
    """
    # The floor under S keeps the inverse finite at contact. The floor under N keeps a group that
    # has never moved from reading as exactly zero surprise through floating point noise.
    effective_clearance = np.maximum(clearance_meters, config.clearance_epsilon_meters)
    effective_noise = np.maximum(noise_scale_meters, config.noise_epsilon_meters)
    return np.minimum(0.5 * (effective_noise / effective_clearance) ** 2, config.surprise_cap)


class CollisionSurprise:
    """His term, as a cost term: clearance from the footprint edge, then point_surprise."""

    def point_cost(
        self,
        centre_distance_meters: np.ndarray,
        noise_meters: np.ndarray,
        config: PlannerConfig,
        walker: WalkerConfig,
    ) -> np.ndarray:
        """
        Half of (N over S) squared for every point from every candidate position.

        :param centre_distance_meters: (cells, points) distance from each candidate position to each point.
        :param noise_meters: (points,) each point's N, walls already multiplied.
        :param config: His epsilons and cap.
        :param walker: The footprint radius, subtracted to get S.
        :return: (cells, points) surprise per second.
        :rtype: np.ndarray
        """
        clearance = centre_distance_meters - walker.radius_meters
        return point_surprise(clearance, noise_meters[None, :], config)


def surprise_field(obstacles: ObstacleSet, grid: np.ndarray, config: PlannerConfig, walker: WalkerConfig) -> np.ndarray:
    """
    His surprise alone at every candidate lateral position, for every future step of the horizon.

    :param obstacles: This frame's groups, walker relative.
    :param grid: The lateral candidates, from lateral_grid.
    :param config: Horizon, speed, prediction flag, surprise constants.
    :param walker: The footprint radius, subtracted from centre distance to get clearance.
    :return: (steps, len(grid)) field. Zero everywhere when nothing is in view.
    :rtype: np.ndarray
    """
    return cost_field(obstacles, grid, config, walker, terms=(CollisionSurprise(),))
