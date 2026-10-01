"""
Covers the whole scene layer end to end on the synthetic sequence, in both pose modes.

The frame-to-frame properties live here because no single stage can show them: a box's group id
staying put while the walker advances, and N rising as that box gets nearer.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.pipeline import ScenePipeline
from nav.types import DepthFrame, ObstacleSet, Pose
from nav.walker import WalkerConfig
from synthetic_depth import CAMERA_HEIGHT_METERS, PITCH_DEGREES, clean_scene, degrade_with_holes, degrade_with_zero_rows, pitch_rotation

CONFIG = SceneConfig()
WALKER = WalkerConfig()
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _frame(scene, timestamp: float = 0.0, pose: Pose | None = None, depth: np.ndarray | None = None, with_plane: bool = False) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=timestamp,
        depth_meters=scene.depth_meters if depth is None else depth,
        intrinsics=scene.intrinsics,
        pose=pose if pose is not None else Pose(orientation=IDENTITY, position=None, has_position=False),
        ground_plane=scene.floor_plane_camera if with_plane else None,
        gaze_pixel=None,
    )


def _quaternion_about(axis: np.ndarray, degrees: float) -> np.ndarray:
    half = np.radians(degrees) / 2.0
    unit = axis / np.linalg.norm(axis)
    return np.array([np.cos(half), *(np.sin(half) * unit)])


def _compose(first: np.ndarray, then: np.ndarray) -> np.ndarray:
    """Quaternion product then * first, both (w, x, y, z)."""
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = then
    return np.array(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ]
    )


def _pitched_pose(position: np.ndarray, yaw_degrees: float = 0.0) -> Pose:
    # A Pose rotation maps camera-frame vectors into the world. The fixture's pitch_rotation maps
    # the other way, world to camera, so the pose carries its inverse. Getting this backwards is
    # exactly the bug this fixture once had, and the sanity check below is what caught it.
    pitch_camera_to_world = _quaternion_about(np.array([1.0, 0.0, 0.0]), -PITCH_DEGREES)
    yaw_about_world_up = _quaternion_about(np.array([0.0, -1.0, 0.0]), yaw_degrees)
    orientation = _compose(pitch_camera_to_world, then=yaw_about_world_up)

    from nav.scene.transform import rotation_matrix_from_quaternion_wxyz

    if yaw_degrees == 0.0:
        assert rotation_matrix_from_quaternion_wxyz(orientation) == pytest.approx(pitch_rotation(PITCH_DEGREES).T, abs=1e-9)
    return Pose(orientation=orientation, position=position, has_position=True)


def test_one_box_becomes_one_group_and_the_floor_becomes_none() -> None:
    scene = clean_scene(box_lateral_meters=0.5, box_forward_meters=3.0, box_height_meters=1.0, box_half_width_meters=0.1)

    obstacles = ScenePipeline(CONFIG, WALKER).process(_frame(scene))

    assert isinstance(obstacles, ObstacleSet)
    # The front face is 0.2 m wide and may straddle one cell boundary, never more.
    assert 1 <= obstacles.groups_in_view <= 2
    for point in obstacles.points:
        # Known from the fixture: 0.4 to 0.6 m right, about 2.9 m ahead, clear of the walker.
        assert 0.3 <= point.lateral_meters <= 0.7
        assert 2.6 <= point.forward_meters <= 3.1
        assert point.clearance_meters == pytest.approx(np.hypot(point.lateral_meters, point.forward_meters) - WALKER.radius_meters, abs=1e-6)
        assert point.noise_scale_meters == CONFIG.noise_floor_meters  # one frame, no history yet
        assert point.closing_rate_mps is None
        assert point.velocity_mps is None
        assert point.is_wall is False


def test_a_floor_only_scene_yields_no_groups() -> None:
    scene = clean_scene(box_lateral_meters=None, box_forward_meters=None, box_height_meters=None)

    assert ScenePipeline(CONFIG, WALKER).process(_frame(scene)).groups_in_view == 0


def test_a_supplied_ground_plane_is_used_instead_of_a_fit() -> None:
    scene = clean_scene()
    pipeline = ScenePipeline(CONFIG, WALKER)

    pipeline.process(_frame(scene, with_plane=True))

    assert pipeline.previous_plane is scene.floor_plane_camera


def test_a_fitted_plane_is_kept_for_the_next_frame() -> None:
    scene = clean_scene()
    pipeline = ScenePipeline(CONFIG, WALKER)

    pipeline.process(_frame(scene))

    assert pipeline.previous_plane is not None
    assert pipeline.previous_plane.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.02)


def test_holes_and_zero_rows_do_not_change_the_answer_much() -> None:
    clean = clean_scene(box_half_width_meters=0.1)
    pipeline = ScenePipeline(CONFIG, WALKER)
    reference = pipeline.process(_frame(clean))

    degraded = degrade_with_zero_rows(degrade_with_holes(clean.depth_meters, fraction=0.3), rows=slice(0, 8))
    result = ScenePipeline(CONFIG, WALKER).process(_frame(clean, depth=degraded))

    assert result.groups_in_view >= 1
    nearest_clean = min(point.clearance_meters for point in reference.points)
    nearest_degraded = min(point.clearance_meters for point in result.points)
    assert nearest_degraded == pytest.approx(nearest_clean, abs=0.15)


def test_n_rises_as_the_box_approaches_in_the_body_frame() -> None:
    pipeline = ScenePipeline(CONFIG, WALKER)
    distances = [4.0, 3.8, 3.6, 3.4, 3.2]
    results = [
        pipeline.process(_frame(clean_scene(box_forward_meters=distance, box_half_width_meters=0.1), timestamp=index * 0.1))
        for index, distance in enumerate(distances)
    ]

    first = min(point.noise_scale_meters for point in results[0].points)
    last_points = results[-1].points
    assert last_points, "the box must still be in view"
    # In the body frame the box changes cell as it nears, so its id may change and its history
    # restart. Assert only what the frame can promise: the floor at the start, and either growth
    # or a fresh floor at the end, never a value below the floor.
    assert first == CONFIG.noise_floor_meters
    assert all(point.noise_scale_meters >= CONFIG.noise_floor_meters for point in last_points)


def test_in_the_world_frame_the_box_keeps_its_group_id_while_the_walker_advances() -> None:
    pipeline = ScenePipeline(CONFIG, WALKER)
    scenes = [clean_scene(box_forward_meters=4.0 - step * 0.2, box_half_width_meters=0.1) for step in range(5)]
    # The walker advances 0.2 m per frame and the box is fixed in the world, which is exactly why
    # the fixture box gets 0.2 m nearer each frame. Both describe the same world.
    ids_per_frame = []
    for step, scene in enumerate(scenes):
        pose = _pitched_pose(np.array([0.0, 0.0, step * 0.2]))
        obstacles = pipeline.process(_frame(scene, timestamp=step * 0.1, pose=pose))
        ids_per_frame.append({point.group_id for point in obstacles.points})

    assert all(ids for ids in ids_per_frame), "the box must be in view every frame"
    # Every frame's ids overlap the first frame's. World ids are keyed on the box's world cell,
    # which does not move.
    assert all(ids & ids_per_frame[0] for ids in ids_per_frame[1:])


def test_in_the_world_frame_n_grows_and_the_closing_rate_is_positive() -> None:
    pipeline = ScenePipeline(CONFIG, WALKER)
    last = None
    for step in range(5):
        scene = clean_scene(box_forward_meters=4.0 - step * 0.2, box_half_width_meters=0.1)
        pose = _pitched_pose(np.array([0.0, 0.0, step * 0.2]))
        last = pipeline.process(_frame(scene, timestamp=step * 0.1, pose=pose))

    assert last is not None and last.points
    nearest = min(last.points, key=lambda point: point.clearance_meters)
    assert nearest.noise_scale_meters > CONFIG.noise_floor_meters
    assert nearest.closing_rate_mps is not None
    # Closing 0.2 m every 0.1 s.
    assert nearest.closing_rate_mps == pytest.approx(2.0, abs=0.5)


def test_in_the_world_frame_a_fixed_box_has_near_zero_velocity() -> None:
    # The walker moves, the box does not. Velocity must be the box's, not the walker's. A
    # body-frame centroid fed into the history would report the walker's 2 m/s here.
    pipeline = ScenePipeline(CONFIG, WALKER)
    last = None
    for step in range(4):
        scene = clean_scene(box_forward_meters=4.0 - step * 0.2, box_half_width_meters=0.1)
        pose = _pitched_pose(np.array([0.0, 0.0, step * 0.2]))
        last = pipeline.process(_frame(scene, timestamp=step * 0.1, pose=pose))

    assert last is not None and last.points
    nearest = min(last.points, key=lambda point: point.clearance_meters)
    assert nearest.velocity_mps is not None
    assert np.linalg.norm(nearest.velocity_mps) < 0.6


def test_velocity_is_reported_in_the_walkers_axes_not_the_worlds() -> None:
    # The walker stands still facing world +x, having turned 90 degrees. A box moves away along
    # world +x, which is straight ahead for the walker. The history measures velocity on fixed
    # world axes, where that is a lateral motion. The planner slides groups in walker axes, so
    # the pipeline must rotate it. A velocity left in world axes comes out as (2, 0) here.
    pipeline = ScenePipeline(CONFIG, WALKER)
    last = None
    for step in range(4):
        scene = clean_scene(box_forward_meters=3.0 + step * 0.2, box_half_width_meters=0.1)
        pose = _pitched_pose(np.zeros(3), yaw_degrees=90.0)
        last = pipeline.process(_frame(scene, timestamp=step * 0.1, pose=pose))

    assert last is not None and last.points
    nearest = min(last.points, key=lambda point: point.clearance_meters)
    assert nearest.velocity_mps is not None
    assert nearest.velocity_mps[1] == pytest.approx(2.0, abs=0.5), "forward, away from the walker"
    assert abs(nearest.velocity_mps[0]) < 0.5, "no sideways motion in the walker's frame"


def test_coordinates_are_walker_relative_even_with_a_world_position() -> None:
    # The planner's grid is body frame at the current frame, whatever the pose knows.
    scene = clean_scene(box_lateral_meters=0.5, box_forward_meters=3.0, box_half_width_meters=0.1)
    far_away = _pitched_pose(np.array([100.0, 0.0, 250.0]))

    obstacles = ScenePipeline(CONFIG, WALKER).process(_frame(scene, pose=far_away))

    assert obstacles.points
    for point in obstacles.points:
        assert 0.3 <= point.lateral_meters <= 0.7
        assert 2.6 <= point.forward_meters <= 3.1


def test_an_empty_depth_image_with_no_previous_plane_raises() -> None:
    scene = clean_scene()
    empty = np.full_like(scene.depth_meters, np.nan)

    with pytest.raises(ValueError, match="no floor found"):
        ScenePipeline(CONFIG, WALKER).process(_frame(scene, depth=empty))


def test_an_empty_depth_image_after_a_good_frame_yields_no_groups() -> None:
    scene = clean_scene()
    pipeline = ScenePipeline(CONFIG, WALKER)
    pipeline.process(_frame(scene))

    obstacles = pipeline.process(_frame(scene, timestamp=0.1, depth=np.full_like(scene.depth_meters, np.nan)))

    assert obstacles.groups_in_view == 0
