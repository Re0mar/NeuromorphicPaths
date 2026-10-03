"""Covers the surprise formula and the field, against values worked out by hand."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.surprise import effective_noise, lateral_grid, point_surprise, step_count, surprise_field
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)


def _point(lateral: float, forward: float, group: int, noise: float = 0.1, is_wall: bool = False, closing: float | None = None, velocity=None) -> ObstaclePoint:
    clearance = max(0.0, float(np.hypot(lateral, forward)) - WALKER.radius_meters)
    # A point built by hand has no camera frame. The planner never reads camera_point.
    return ObstaclePoint(lateral, forward, group, clearance, noise, closing, velocity, is_wall, np.zeros(3))


def _set(*points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(timestamp_seconds=0.0, points=tuple(points), groups_in_view=len({p.group_id for p in points}))


def test_surprise_is_one_half_when_clearance_equals_noise() -> None:
    assert point_surprise(1.0, 1.0, CONFIG) == pytest.approx(0.5)
    assert point_surprise(0.2, 0.2, CONFIG) == pytest.approx(0.5)


def test_surprise_quadruples_when_clearance_halves() -> None:
    assert point_surprise(1.0, 0.5, CONFIG) * 4 == pytest.approx(point_surprise(0.5, 0.5, CONFIG))


def test_the_cap_binds_at_tiny_clearance() -> None:
    # At S floored to 0.06 m the cap of 2e4 binds once N over S passes 200, so N of 20 m. Below
    # that the epsilon under S is what keeps the value finite, and it is the uncapped number.
    assert point_surprise(0.0, 20.0, CONFIG) == CONFIG.surprise_cap
    assert point_surprise(0.0, 10.0, CONFIG) == pytest.approx(0.5 * (10.0 / CONFIG.clearance_epsilon_meters) ** 2)
    assert np.isfinite(point_surprise(0.0, 0.01, CONFIG))


def test_zero_noise_is_zero_surprise_not_nan() -> None:
    assert point_surprise(1.0, 0.0, CONFIG) == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(point_surprise(1.0, 0.0, CONFIG))


def test_surprise_vectorises_over_arrays() -> None:
    clearance = np.array([1.0, 0.5, 0.0])
    noise = np.array([1.0, 1.0, 20.0])

    result = point_surprise(clearance, noise, CONFIG)

    assert result.shape == (3,)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(2.0)
    assert result[2] == CONFIG.surprise_cap


def test_a_wall_point_has_its_noise_multiplied() -> None:
    post = _point(0.0, 2.0, group=1, noise=0.1)
    wall = _point(0.0, 2.0, group=2, noise=0.1, is_wall=True)

    assert effective_noise(post, CONFIG) == pytest.approx(0.1)
    assert effective_noise(wall, CONFIG) == pytest.approx(0.1 * CONFIG.wall_noise_multiplier)


def test_the_grid_is_symmetric_and_spaced_as_configured() -> None:
    grid = lateral_grid(CONFIG)

    assert grid[0] == pytest.approx(-CONFIG.grid_half_width_meters)
    assert grid[-1] == pytest.approx(CONFIG.grid_half_width_meters)
    assert np.diff(grid) == pytest.approx(CONFIG.grid_spacing_meters)
    assert len(grid) % 2 == 1, "an odd count puts a cell exactly at straight ahead"


def test_the_horizon_holds_now_plus_every_step() -> None:
    assert step_count(PlannerConfig(time_step_seconds=0.1, horizon_seconds=3.8)) == 39


def test_an_empty_scene_gives_a_zero_field_of_the_right_shape() -> None:
    field = surprise_field(_set(), lateral_grid(CONFIG), CONFIG, WALKER)

    assert field.shape == (step_count(CONFIG), len(lateral_grid(CONFIG)))
    assert np.all(field == 0.0)


def test_two_points_in_one_group_contribute_their_maximum() -> None:
    grid = lateral_grid(CONFIG)
    near = _point(0.0, 1.0, group=1, noise=0.2)
    far = _point(0.0, 2.0, group=1, noise=0.2)

    together = surprise_field(_set(near, far), grid, CONFIG, WALKER)[0]
    near_alone = surprise_field(_set(near), grid, CONFIG, WALKER)[0]

    # The far point is dominated everywhere by the near one at the same lateral position.
    assert together == pytest.approx(near_alone)


def test_two_groups_sum() -> None:
    grid = lateral_grid(CONFIG)
    left = _point(-1.0, 2.0, group=1, noise=0.2)
    right = _point(1.0, 2.0, group=2, noise=0.2)

    together = surprise_field(_set(left, right), grid, CONFIG, WALKER)[0]
    separately = surprise_field(_set(left), grid, CONFIG, WALKER)[0] + surprise_field(_set(right), grid, CONFIG, WALKER)[0]

    assert together == pytest.approx(separately)


def test_the_field_peaks_at_the_obstacles_lateral_position() -> None:
    grid = lateral_grid(CONFIG)
    field = surprise_field(_set(_point(1.0, 2.0, group=1, noise=0.3)), grid, CONFIG, WALKER)

    assert grid[int(np.argmax(field[0]))] == pytest.approx(1.0, abs=CONFIG.grid_spacing_meters)


def test_a_static_obstacle_gets_nearer_as_the_walker_advances() -> None:
    # The walker covers speed times dt per step, so the same post is more surprising later.
    grid = lateral_grid(CONFIG)
    field = surprise_field(_set(_point(0.0, 4.0, group=1, noise=0.3)), grid, CONFIG, WALKER)
    centre = len(grid) // 2

    assert field[5, centre] > field[0, centre]


def test_motion_prediction_slides_a_group_by_its_velocity() -> None:
    grid = lateral_grid(CONFIG)
    # Moving right at 1 m/s. After one second it should peak a meter to the right of where it
    # started, and only when prediction is on.
    moving = _point(0.0, 10.0, group=1, noise=0.3, velocity=np.array([1.0, CONFIG.walking_speed_mps]))
    k = int(round(1.0 / CONFIG.time_step_seconds))

    predicted = surprise_field(_set(moving), grid, PlannerConfig(predict_motion=True), WALKER)
    held = surprise_field(_set(moving), grid, PlannerConfig(predict_motion=False), WALKER)

    assert grid[int(np.argmax(predicted[k]))] == pytest.approx(1.0, abs=CONFIG.grid_spacing_meters)
    assert grid[int(np.argmax(held[k]))] == pytest.approx(0.0, abs=CONFIG.grid_spacing_meters)


def test_a_group_without_a_velocity_is_held_even_when_prediction_is_on() -> None:
    grid = lateral_grid(CONFIG)
    still = _point(0.0, 10.0, group=1, noise=0.3, velocity=None)
    k = int(round(1.0 / CONFIG.time_step_seconds))

    field = surprise_field(_set(still), grid, PlannerConfig(predict_motion=True, walking_speed_mps=0.0), WALKER)

    assert grid[int(np.argmax(field[k]))] == pytest.approx(0.0, abs=CONFIG.grid_spacing_meters)


@pytest.mark.parametrize("config", [PlannerConfig(grid_spacing_meters=0.0), PlannerConfig(time_step_seconds=0.0), PlannerConfig(horizon_seconds=-1.0)])
def test_degenerate_grid_or_horizon_is_refused(config: PlannerConfig) -> None:
    with pytest.raises(ValueError):
        lateral_grid(config)
        step_count(config)
