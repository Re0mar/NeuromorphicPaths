"""Covers the shared field builder: the geometry every term reads, and the per-term combination."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.field import cost_field, effective_noise, lateral_grid, step_count
from nav.planner.surprise import CollisionSurprise, point_surprise, surprise_field
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)


def _point(lateral: float, forward: float, group: int, noise: float = 0.1, is_wall: bool = False, velocity=None) -> ObstaclePoint:
    clearance = max(0.0, float(np.hypot(lateral, forward)) - WALKER.radius_meters)
    # A point built by hand has no camera frame. The planner never reads camera_point.
    return ObstaclePoint(lateral, forward, group, clearance, noise, None, velocity, is_wall, np.zeros(3))


def _set(*points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(timestamp_seconds=0.0, points=tuple(points), groups_in_view=len({p.group_id for p in points}))


def _reference_surprise_field(obstacles: ObstacleSet, grid: np.ndarray, config: PlannerConfig, walker: WalkerConfig) -> np.ndarray:
    """surprise_field as it stood before the geometry moved into field.py, kept verbatim as the reference."""
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

        clearance = np.hypot(grid[:, None] - point_lateral[None, :], point_forward[None, :]) - walker.radius_meters
        per_point = point_surprise(clearance, noise[None, :], config)

        total = np.zeros(len(grid))
        for group_position in range(len(unique_groups)):
            total += per_point[:, group_index == group_position].max(axis=1)
        field[k] = total

    return field


def _random_scene(seed: int) -> ObstacleSet:
    generator = np.random.default_rng(seed)
    points = []
    # Group ids scattered and points shuffled, as the scene can hand them over, so the field has to
    # gather each group's points itself rather than finding them side by side.
    group_ids = generator.choice(1000, size=int(generator.integers(1, 8)), replace=False)
    for group in (int(group_id) for group_id in group_ids):
        centre_lateral = float(generator.uniform(-3.0, 3.0))
        centre_forward = float(generator.uniform(0.3, 6.0))
        is_wall = bool(generator.random() < 0.3)
        velocity = generator.uniform(-1.0, 1.0, size=2) if generator.random() < 0.5 else None
        for _ in range(int(generator.integers(1, 6))):
            points.append(
                _point(
                    centre_lateral + float(generator.normal(0.0, 0.2)),
                    max(0.05, centre_forward + float(generator.normal(0.0, 0.2))),
                    group,
                    noise=float(generator.uniform(0.0, 0.5)),
                    is_wall=is_wall,
                    velocity=velocity,
                )
            )
    shuffled = [points[index] for index in generator.permutation(len(points))]
    return _set(*shuffled)


class _ConstantPerPoint:
    """A stub term that gives each point a fixed cost, whatever the geometry."""

    def __init__(self, costs: list[float]) -> None:
        self._costs = np.array(costs)

    def point_cost(self, centre_distance_meters, noise_meters, config, walker) -> np.ndarray:
        return np.broadcast_to(self._costs[None, :], centre_distance_meters.shape).copy()


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("predict_motion", [False, True])
def test_the_field_matches_the_old_surprise_field_exactly_on_random_scenes(seed: int, predict_motion: bool) -> None:
    config = PlannerConfig(predict_motion=predict_motion)
    grid = lateral_grid(config)
    scene = _random_scene(seed)

    np.testing.assert_array_equal(surprise_field(scene, grid, config, WALKER), _reference_surprise_field(scene, grid, config, WALKER))


def test_the_field_matches_the_old_surprise_field_exactly_with_many_groups() -> None:
    # Real frames carry about fifty groups. numpy sums eight or more numbers pairwise, which rounds
    # differently from the old loop's left-to-right sum, so only a scene this size tells the two apart.
    generator = np.random.default_rng(99)
    points = [
        _point(float(generator.uniform(-3.0, 3.0)), float(generator.uniform(0.3, 6.0)), group, noise=float(generator.uniform(0.01, 0.5)))
        for group in generator.permutation(60)
    ]
    scene = _set(*points)
    grid = lateral_grid(CONFIG)

    np.testing.assert_array_equal(surprise_field(scene, grid, CONFIG, WALKER), _reference_surprise_field(scene, grid, CONFIG, WALKER))


def test_two_terms_add() -> None:
    grid = lateral_grid(CONFIG)
    scene = _random_scene(3)
    first = _ConstantPerPoint([0.5] * len(scene.points))

    together = cost_field(scene, grid, CONFIG, WALKER, terms=(CollisionSurprise(), first))
    apart = cost_field(scene, grid, CONFIG, WALKER, terms=(CollisionSurprise(),)) + cost_field(scene, grid, CONFIG, WALKER, terms=(first,))

    np.testing.assert_allclose(together, apart, rtol=1e-12)


def test_each_term_takes_its_own_most_dangerous_point() -> None:
    # One group, two points. Term A's worst point is the first and term B's is the second, so the
    # group costs 3 + 5 = 8. Summing the terms per point before the group max would give max(4, 6) = 6.
    scene = _set(_point(0.0, 2.0, group=1), _point(1.0, 2.0, group=1))
    term_a = _ConstantPerPoint([3.0, 1.0])
    term_b = _ConstantPerPoint([1.0, 5.0])

    field = cost_field(scene, lateral_grid(CONFIG), CONFIG, WALKER, terms=(term_a, term_b))

    assert field == pytest.approx(np.full_like(field, 8.0))


def test_an_empty_term_list_is_refused() -> None:
    with pytest.raises(ValueError, match="term"):
        cost_field(_set(_point(0.0, 2.0, group=1)), lateral_grid(CONFIG), CONFIG, WALKER, terms=())


def test_no_points_gives_a_zero_field_of_the_right_shape() -> None:
    grid = lateral_grid(CONFIG)
    field = cost_field(_set(), grid, CONFIG, WALKER, terms=(CollisionSurprise(),))

    assert field.shape == (step_count(CONFIG), len(grid))
    assert not field.any()
