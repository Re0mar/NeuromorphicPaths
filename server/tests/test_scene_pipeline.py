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
from nav.types import DepthFrame, FloorSource, ObstacleSet, Plane, Pose
from nav.walker import WalkerConfig
from synthetic_depth import (
    CAMERA_HEIGHT_METERS,
    PITCH_DEGREES,
    clean_scene,
    degrade_with_holes,
    degrade_with_zero_rows,
    degrade_without_floor,
    pitch_rotation,
)

CONFIG = SceneConfig()
WALKER = WalkerConfig()
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _frame(
    scene,
    timestamp: float = 0.0,
    pose: Pose | None = None,
    depth: np.ndarray | None = None,
    with_plane: bool = False,
    ground_plane: Plane | None = None,
) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=timestamp,
        depth_meters=scene.depth_meters if depth is None else depth,
        intrinsics=scene.intrinsics,
        pose=pose if pose is not None else Pose(orientation=IDENTITY, position=None, has_position=False),
        ground_plane=ground_plane if ground_plane is not None else (scene.floor_plane_camera if with_plane else None),
        gaze_pixel=None,
    )


def _a_meter_below(plane: Plane) -> Plane:
    """The first Pixel walk's defect: the same orientation, the camera 1.1 m further above it."""
    return Plane(normal=plane.normal, offset_meters=plane.offset_meters + 1.1)


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


# A level camera's axes (x right, y down, z forward) to the world's (x right, y up, z backward),
# which is ARCore's world and the one the wire format states. A half turn about x.
LEVEL_CAMERA_TO_WORLD = _quaternion_about(np.array([1.0, 0.0, 0.0]), 180.0)
HALF_TURN_ABOUT_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _pitched_pose(level_position: np.ndarray, yaw_degrees: float = 0.0, roll_degrees: float = 0.0) -> Pose:
    """
    The fixture camera's pose: pitched down, optionally yawed and rolled, placed in a y-up world.

    :param level_position: Where the camera is, in a level camera's axes (z forward). Converted to
        the world's axes here so the tests can keep saying "0.2 m forward".
    :param yaw_degrees: Turn about the vertical, positive to the right.
    :param roll_degrees: Turn about the optical axis. 90 puts image-right pointing at the floor,
        which is a phone held in portrait sending a landscape depth image.
    """
    # A Pose rotation maps camera-frame vectors into the world. The fixture's pitch_rotation maps
    # the other way, world to camera, so the pose carries its inverse. Getting this backwards is
    # exactly the bug this fixture once had, and the sanity check below is what caught it.
    roll_rolled_to_unrolled = _quaternion_about(np.array([0.0, 0.0, 1.0]), roll_degrees)
    pitch_camera_to_level = _quaternion_about(np.array([1.0, 0.0, 0.0]), -PITCH_DEGREES)
    # The same turn the fixture always made about the level camera's up, carried into the y-up
    # world by the half turn: a rotation about an axis conjugates to the rotation about the
    # mapped axis by the same angle.
    yaw_about_world_up = _quaternion_about(np.array([0.0, 1.0, 0.0]), yaw_degrees)
    orientation = _compose(roll_rolled_to_unrolled, then=pitch_camera_to_level)
    orientation = _compose(orientation, then=LEVEL_CAMERA_TO_WORLD)
    orientation = _compose(orientation, then=yaw_about_world_up)

    from nav.scene.transform import rotation_matrix_from_quaternion_wxyz

    if yaw_degrees == 0.0 and roll_degrees == 0.0:
        assert rotation_matrix_from_quaternion_wxyz(orientation) == pytest.approx(HALF_TURN_ABOUT_X @ pitch_rotation(PITCH_DEGREES).T, abs=1e-9)
    return Pose(orientation=orientation, position=HALF_TURN_ABOUT_X @ level_position, has_position=True)


def _rolled_image(scene) -> tuple[np.ndarray, np.ndarray]:
    """
    The fixture's depth image as a camera rolled 90 degrees would record it, with its intrinsics.

    Rotating the image a quarter turn counterclockwise makes the new image-right the old
    image-down, so the floor now runs down the right edge. Depth along the optical axis is the
    same number per pixel, which is why a rotation is all it takes.
    """
    depth = np.ascontiguousarray(np.rot90(scene.depth_meters))
    focal = scene.intrinsics[0, 0]
    height, width = scene.depth_meters.shape
    # out[i, j] = in[j, width - 1 - i], so the new principal point is (height / 2, width / 2 - 1).
    rolled_intrinsics = np.array([[focal, 0.0, height / 2.0], [0.0, focal, width / 2.0 - 1.0], [0.0, 0.0, 1.0]])
    return depth, rolled_intrinsics


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

    # Equality, not identity: the supplied plane is normalized on the way in. The fit would land
    # within a couple of centimeters of the truth, so the 1e-9 is what says this was not a fit.
    assert pipeline.previous_plane is not None
    assert pipeline.previous_plane.normal == pytest.approx(scene.floor_plane_camera.normal, abs=1e-9)
    assert pipeline.previous_plane.offset_meters == pytest.approx(scene.floor_plane_camera.offset_meters, abs=1e-9)
    assert pipeline.last_floor_source is FloorSource.SUPPLIED


def test_the_floor_source_is_none_before_the_first_frame() -> None:
    assert ScenePipeline(CONFIG, WALKER).last_floor_source is None


def test_a_supplied_plane_a_meter_below_the_floor_is_refused_and_the_fit_takes_over() -> None:
    # The first Pixel walk: the phone's plane put the camera 2.7 m up over a floor it was 1.6 m
    # above. Before the gate that plane was taken as given and the real floor became obstacles.
    scene = clean_scene()
    pipeline = ScenePipeline(CONFIG, WALKER)
    reference = ScenePipeline(CONFIG, WALKER).process(_frame(scene))

    obstacles = pipeline.process(_frame(scene, ground_plane=_a_meter_below(scene.floor_plane_camera)))

    assert pipeline.last_floor_source is FloorSource.FITTED
    assert pipeline.previous_plane is not None
    assert pipeline.previous_plane.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.02)
    # The same groups a frame with no plane at all produces: the box, not a floor's worth.
    assert obstacles.groups_in_view == reference.groups_in_view
    assert min(point.clearance_meters for point in obstacles.points) == pytest.approx(min(point.clearance_meters for point in reference.points), abs=0.05)


def test_a_refused_supplied_plane_falls_back_to_the_previous_plane_when_the_cloud_has_no_floor() -> None:
    scene = clean_scene()
    pipeline = ScenePipeline(CONFIG, WALKER)
    pipeline.process(_frame(scene))
    kept = pipeline.previous_plane
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)

    pipeline.process(_frame(scene, timestamp=0.1, depth=floorless, ground_plane=_a_meter_below(scene.floor_plane_camera)))

    assert pipeline.last_floor_source is FloorSource.PREVIOUS
    assert pipeline.previous_plane is kept


def test_a_refused_supplied_plane_with_no_cloud_floor_and_no_previous_raises() -> None:
    scene = clean_scene()
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)

    with pytest.raises(ValueError, match="no floor found"):
        ScenePipeline(CONFIG, WALKER).process(_frame(scene, depth=floorless, ground_plane=_a_meter_below(scene.floor_plane_camera)))


def test_in_the_world_frame_a_camera_rolled_ninety_degrees_still_finds_the_floor() -> None:
    # The Pixel held in portrait sends its depth image in the sensor's landscape orientation, so
    # image-up points sideways and the floor's normal is 90 degrees from it. The pose says which
    # way gravity points, and the fit has to use that rather than image-up. The box is still the
    # one group, at the same clearance it has from the unrolled camera.
    scene = clean_scene(box_lateral_meters=0.5, box_forward_meters=3.0, box_height_meters=1.0, box_half_width_meters=0.1)
    reference = ScenePipeline(CONFIG, WALKER).process(_frame(scene, pose=_pitched_pose(np.zeros(3))))
    depth, intrinsics = _rolled_image(scene)
    rolled = DepthFrame(0.0, depth, intrinsics, _pitched_pose(np.zeros(3), roll_degrees=90.0), None, None)
    pipeline = ScenePipeline(CONFIG, WALKER)

    obstacles = pipeline.process(rolled)

    assert pipeline.last_floor_source is FloorSource.FITTED
    assert pipeline.previous_plane is not None
    assert pipeline.previous_plane.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.03)
    # Against image-up the same floor leans 90 degrees.
    assert abs(pipeline.previous_plane.normal[1]) < 0.1
    assert obstacles.groups_in_view == reference.groups_in_view
    assert min(point.clearance_meters for point in obstacles.points) == pytest.approx(min(point.clearance_meters for point in reference.points), abs=0.1)


def test_a_supplied_plane_from_a_rolled_camera_is_judged_against_gravity() -> None:
    # The plane the app sends is right in the world and sideways in the image. Judged against
    # image-up it would be refused on tilt, and the first walk's log showed exactly that number.
    scene = clean_scene()
    depth, intrinsics = _rolled_image(scene)
    truth = scene.floor_plane_camera
    rolled_plane = Plane(normal=np.array([truth.normal[1], -truth.normal[0], truth.normal[2]]), offset_meters=truth.offset_meters)
    pipeline = ScenePipeline(CONFIG, WALKER)

    pipeline.process(DepthFrame(0.0, depth, intrinsics, _pitched_pose(np.zeros(3), roll_degrees=90.0), rolled_plane, None))

    assert pipeline.last_floor_source is FloorSource.SUPPLIED
    assert pipeline.previous_plane is not None
    assert pipeline.previous_plane.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=1e-9)


def test_a_supplied_plane_with_its_normal_pointing_down_is_read_the_right_way_up() -> None:
    # A source that writes its normal pointing at the floor describes the same plane. The scene
    # must not read it as the camera 1.6 m below the floor and refuse it.
    scene = clean_scene()
    truth = scene.floor_plane_camera
    pipeline = ScenePipeline(CONFIG, WALKER)

    pipeline.process(_frame(scene, ground_plane=Plane(normal=-truth.normal, offset_meters=-truth.offset_meters)))

    assert pipeline.last_floor_source is FloorSource.SUPPLIED
    assert pipeline.previous_plane is not None
    assert pipeline.previous_plane.offset_meters == pytest.approx(truth.offset_meters, abs=1e-9)


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
