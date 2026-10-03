"""
Collision surprise, and the field of it over future time and lateral position.

A reading is surprising in proportion to how uncertain it is relative to how close it is. With
clearance S and noise scale N, both in meters, the surprise of one point is half of (N over S)
squared. A post two meters away whose distance wobbles by a centimeter is barely surprising. One
whose distance wobbles by half a meter is, and the planner steers around the wobble.

Each obstacle is represented by its most surprising point, and independent obstacles add. That
is the maximum within a group and the sum across groups, in that order.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.types import ObstaclePoint, ObstacleSet
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


def effective_noise(point: ObstaclePoint, config: PlannerConfig) -> float:
    """
    The N the planner uses for a point. A wall's is multiplied, because brushing one costs more.

    :param point: The point.
    :param config: The wall multiplier.
    :return: N in meters.
    :rtype: float
    """
    if point.is_wall:
        return point.noise_scale_meters * config.wall_noise_multiplier
    return point.noise_scale_meters


def lateral_grid(config: PlannerConfig) -> np.ndarray:
    """The candidate lateral positions, centred on straight ahead, spacing from the config."""
    if config.grid_spacing_meters <= 0 or config.grid_half_width_meters <= 0:
        raise ValueError("grid spacing and half width must be positive")
    count = int(round(2 * config.grid_half_width_meters / config.grid_spacing_meters)) + 1
    return np.linspace(-config.grid_half_width_meters, config.grid_half_width_meters, count)


def step_count(config: PlannerConfig) -> int:
    """How many future time slices the horizon holds, including now."""
    if config.time_step_seconds <= 0 or config.horizon_seconds <= 0:
        raise ValueError("time step and horizon must be positive")
    return int(round(config.horizon_seconds / config.time_step_seconds)) + 1


def surprise_field(obstacles: ObstacleSet, grid: np.ndarray, config: PlannerConfig, walker: WalkerConfig) -> np.ndarray:
    """
    Surprise at every candidate lateral position, for every future step of the horizon.

    The walker advances at walking speed, so at step k it is k dt v further forward and every
    obstacle is that much nearer in the forward direction. A group with a known velocity is slid
    by it when motion prediction is on. Everything else is held where it is.

    :param obstacles: This frame's groups, walker relative.
    :param grid: The lateral candidates, from lateral_grid.
    :param config: Horizon, speed, prediction flag, surprise constants.
    :param walker: The footprint radius, subtracted from centre distance to get clearance.
    :return: (steps, len(grid)) field. Zero everywhere when nothing is in view.
    :rtype: np.ndarray
    """
    steps = step_count(config)
    field = np.zeros((steps, len(grid)), dtype=np.float64)
    if not obstacles.points:
        return field

    lateral = np.array([point.lateral_meters for point in obstacles.points])
    forward = np.array([point.forward_meters for point in obstacles.points])
    noise = np.array([effective_noise(point, config) for point in obstacles.points])
    group_ids = np.array([point.group_id for point in obstacles.points])

    velocity = np.zeros((len(obstacles.points), 2))
    if config.predict_motion:
        for index, point in enumerate(obstacles.points):
            if point.velocity_mps is not None:
                velocity[index] = point.velocity_mps

    unique_groups, group_index = np.unique(group_ids, return_inverse=True)

    for k in range(steps):
        elapsed = k * config.time_step_seconds
        walker_forward = config.walking_speed_mps * elapsed
        point_lateral = lateral + velocity[:, 0] * elapsed
        point_forward = forward + velocity[:, 1] * elapsed - walker_forward

        # (cells, points): clearance from each candidate position to each point.
        clearance = np.hypot(grid[:, None] - point_lateral[None, :], point_forward[None, :]) - walker.radius_meters
        per_point = point_surprise(clearance, noise[None, :], config)

        # Max within a group, then sum across groups. A loop over groups, because there are tens
        # of them and the array work inside each is what costs.
        total = np.zeros(len(grid))
        for group_position in range(len(unique_groups)):
            total += per_point[:, group_index == group_position].max(axis=1)
        field[k] = total

    return field
