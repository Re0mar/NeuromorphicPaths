"""
Covers the height band, the ground projection, and grouping into cells, against known positions.

The box in the synthetic scene is at a lateral and forward position the fixture chose, so the
cell it must land in is a number computed from the config in the test, not read from the system.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.floor import height_above_floor
from nav.scene.grouping import (
    WORLD_GRID_INDEX_OFFSET,
    GroupSummary,
    assign_groups,
    assign_world_groups,
    clearance,
    filter_height_band,
    flatten_to_ground,
    summarize_groups,
    within_planning_window,
)
from nav.scene.unproject import unproject_depth
from nav.walker import WalkerConfig
from synthetic_depth import clean_scene

CONFIG = SceneConfig()
WALKER = WalkerConfig(radius_meters=0.35)


def _prepared(scene):
    """Cloud, heights against the analytic plane, and ground coordinates."""
    cloud = unproject_depth(scene.depth_meters, scene.intrinsics, stride=1, config=CONFIG)
    heights = height_above_floor(cloud, scene.floor_plane_camera)
    return cloud, heights


def _expected_group_id(lateral: float, forward: float, config: SceneConfig = CONFIG) -> int:
    columns = int(np.ceil(2 * config.grid_half_width_meters / config.cell_size_meters))
    column = int(np.floor((lateral + config.grid_half_width_meters) / config.cell_size_meters))
    row = int(np.floor(forward / config.cell_size_meters))
    return row * columns + column


def test_the_height_band_drops_the_floor_and_keeps_the_box() -> None:
    scene = clean_scene(box_height_meters=1.0)
    cloud, heights = _prepared(scene)

    kept, kept_heights = filter_height_band(cloud, heights, CONFIG)

    assert len(kept) > 0
    assert len(kept) < len(cloud) // 2, "most of the image is floor and must have gone"
    assert np.all(kept_heights > CONFIG.ankle_height_meters)
    assert np.all(kept_heights < CONFIG.head_height_meters)


def test_a_scene_with_only_floor_leaves_nothing_in_the_band() -> None:
    scene = clean_scene(box_lateral_meters=None, box_forward_meters=None, box_height_meters=None)
    cloud, heights = _prepared(scene)

    kept, _ = filter_height_band(cloud, heights, CONFIG)

    assert len(kept) == 0


def test_the_box_lands_in_the_cell_its_position_predicts() -> None:
    scene = clean_scene(box_lateral_meters=0.5, box_forward_meters=3.0, box_height_meters=1.0, box_half_width_meters=0.1)
    cloud, heights = _prepared(scene)
    kept, kept_heights = filter_height_band(cloud, heights, CONFIG)
    ground = flatten_to_ground(kept, scene.floor_plane_camera)

    ids = assign_groups(ground, CONFIG)
    summaries = summarize_groups(ground, kept_heights, ids, CONFIG)

    # The front face sits at forward 2.9 m, lateral 0.4 to 0.6 m. Expected cells are computed
    # here from the config, not read back from the result.
    expected = {_expected_group_id(0.45, 2.9), _expected_group_id(0.55, 2.9)}
    assert {summary.group_id for summary in summaries} <= expected
    assert len(summaries) >= 1


def test_a_tall_group_is_a_wall_and_a_short_one_is_not() -> None:
    tall = clean_scene(box_height_meters=1.8)
    short = clean_scene(box_height_meters=1.0)

    def walls(scene) -> list[bool]:
        cloud, heights = _prepared(scene)
        kept, kept_heights = filter_height_band(cloud, heights, CONFIG)
        ground = flatten_to_ground(kept, scene.floor_plane_camera)
        return [summary.is_wall for summary in summarize_groups(ground, kept_heights, assign_groups(ground, CONFIG), CONFIG)]

    assert any(walls(tall))
    assert not any(walls(short))


def test_a_group_with_too_few_points_is_dropped() -> None:
    ground = np.array([[0.0, 1.0], [2.0, 2.0]])  # two lone points in two different cells
    heights = np.array([1.0, 1.0])
    ids = assign_groups(ground, CONFIG)

    assert summarize_groups(ground, heights, ids, SceneConfig(min_points_per_cell=2)) == []
    assert len(summarize_groups(ground, heights, ids, SceneConfig(min_points_per_cell=1))) == 2


def test_points_outside_the_grid_get_no_group() -> None:
    ground = np.array(
        [
            [CONFIG.grid_half_width_meters + 0.1, 1.0],  # too far right
            [-CONFIG.grid_half_width_meters - 0.1, 1.0],  # too far left
            [0.0, -0.1],  # behind
            [0.0, CONFIG.grid_forward_meters + 0.1],  # too far ahead
            [0.0, 1.0],  # inside
        ]
    )

    ids = assign_groups(ground, CONFIG)

    assert list(ids[:4]) == [-1, -1, -1, -1]
    assert ids[4] == _expected_group_id(0.0, 1.0)


def test_the_exact_upper_edge_does_not_fall_off_the_grid() -> None:
    almost = np.nextafter(CONFIG.grid_half_width_meters, -np.inf)
    ground = np.array([[almost, np.nextafter(CONFIG.grid_forward_meters, -np.inf)]])

    assert assign_groups(ground, CONFIG)[0] >= 0


def test_group_ids_are_deterministic_for_the_same_position() -> None:
    ground = np.array([[0.3, 2.2]])

    assert assign_groups(ground, CONFIG)[0] == assign_groups(ground.copy(), CONFIG)[0] == _expected_group_id(0.3, 2.2)


def test_world_group_ids_do_not_change_when_the_walker_moves() -> None:
    # A post at world (1.0, 5.0). Body-frame ids would change as the walker approaches. World ids
    # are keyed on the post's own position and must not.
    post = np.array([[1.0, 5.0]])

    assert assign_world_groups(post, CONFIG)[0] == assign_world_groups(post, CONFIG)[0]
    assert assign_world_groups(post, CONFIG)[0] != assign_world_groups(post + np.array([[0.0, CONFIG.cell_size_meters]]), CONFIG)[0]


def test_world_group_ids_are_non_negative_for_points_behind_the_origin() -> None:
    assert np.all(assign_world_groups(np.array([[-3.0, -7.0], [0.0, 0.0]]), CONFIG) >= 0)


def test_a_point_beyond_the_world_grids_reach_is_refused_rather_than_aliased() -> None:
    # Past the index range the row arithmetic wraps and two different places share an id, which
    # the clearance history would then treat as one obstacle. Refusing is the only visible answer.
    too_far = np.array([[(WORLD_GRID_INDEX_OFFSET + 1) * CONFIG.cell_size_meters, 0.0]])

    with pytest.raises(ValueError, match="world grid"):
        assign_world_groups(too_far, CONFIG)


def test_the_planning_window_matches_the_body_grid_bounds() -> None:
    ground = np.array([[0.0, 1.0], [CONFIG.grid_half_width_meters + 0.1, 1.0], [0.0, -0.1]])

    assert list(within_planning_window(ground, CONFIG)) == [True, False, False]


def test_clearance_is_distance_minus_the_footprint_radius() -> None:
    group = GroupSummary(
        group_id=0,
        point_count=5,
        centroid_lateral_meters=0.0,
        centroid_forward_meters=2.0,
        nearest_lateral_meters=0.0,
        nearest_forward_meters=2.0,
        max_height_meters=1.0,
        is_wall=False,
    )

    assert clearance(group, WALKER) == pytest.approx(2.0 - 0.35)


def test_clearance_never_goes_negative() -> None:
    touching = GroupSummary(0, 5, 0.0, 0.1, 0.0, 0.1, 1.0, False)

    assert clearance(touching, WALKER) == 0.0


def test_the_radius_comes_from_the_walker_and_scene_config_has_none() -> None:
    assert not hasattr(CONFIG, "radius_meters")
    assert clearance(GroupSummary(0, 5, 0.0, 2.0, 0.0, 2.0, 1.0, False), WalkerConfig(radius_meters=0.5)) == pytest.approx(1.5)


def test_a_non_positive_cell_size_is_refused() -> None:
    with pytest.raises(ValueError, match="cell size"):
        assign_groups(np.zeros((1, 2)), SceneConfig(cell_size_meters=0.0))
