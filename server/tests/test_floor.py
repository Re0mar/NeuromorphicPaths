"""
Covers finding the floor against a plane known analytically, not against the fit's own output.

The synthetic camera is pitched down twenty degrees over a floor 1.6 m below it, so the floor
plane has a tilted normal and a known offset, and a fit that was wrong by a constant everywhere
would fail here rather than agree with itself.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.floor import CAMERA_UP, fit_floor, ground_axes, height_above_floor, normalize_plane, plane_is_a_floor
from nav.scene.unproject import unproject_depth
from nav.types import Plane
from synthetic_depth import CAMERA_HEIGHT_METERS, clean_scene, degrade_without_floor

CONFIG = SceneConfig()


def _cloud(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return unproject_depth(depth, intrinsics, stride=1, config=CONFIG)


def _angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def _points_on_plane(normal: np.ndarray, offset: float, lateral: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Camera-frame points (x, y, z) on normal . p + offset == 0, for given x and z, solving y."""
    normal = normal / np.linalg.norm(normal)
    x, z = np.meshgrid(lateral, forward)
    y = (-offset - normal[0] * x - normal[2] * z) / normal[1]
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_a_wall_ahead_with_more_points_than_the_floor_does_not_win_the_vote() -> None:
    # Only points clearly below the camera vote. A wall two meters ahead has more points than
    # the sparse floor here, and without the candidate cut RANSAC fits the wall, the tilt gate
    # rejects it, and there is no floor at all.
    # 2025 floor points against 2400 on the wall, of which about 1300 sit below the candidate
    # line. The cut leaves the floor in the majority; without it the wall is.
    generator = np.random.default_rng(1)
    floor = _points_on_plane(np.array([0.0, -1.0, 0.0]), CAMERA_HEIGHT_METERS, np.linspace(-2.0, 2.0, 45), np.linspace(1.5, 6.0, 45))
    wall_x = generator.uniform(-2.0, 2.0, 2400)
    wall_y = generator.uniform(-0.4, 1.55, 2400)  # from head height down to just above the floor
    wall = np.column_stack((wall_x, wall_y, np.full(2400, 2.0)))

    fitted = fit_floor(np.vstack((floor, wall)), previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert _angle_between_degrees(fitted.normal, CAMERA_UP) < 2.0
    assert fitted.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.05)


def test_a_level_enough_plane_that_sits_too_close_to_the_camera_is_rejected() -> None:
    # Thirty degrees of tilt passes the tilt gate. Passing 0.25 m from the camera does not pass
    # the offset gate, which is what keeps a table top or a held bag from becoming the floor.
    # The plane slopes down and away, so the points ahead are well below the camera and vote.
    tilt = np.radians(30.0)
    normal = np.array([0.0, -np.cos(tilt), np.sin(tilt)])
    close_plane = _points_on_plane(normal, 0.25, np.linspace(-2.0, 2.0, 25), np.linspace(2.0, 6.0, 40))
    assert (close_plane[:, 1] > CONFIG.floor_candidate_min_below_camera_meters).sum() >= CONFIG.floor_min_candidate_points
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.6)

    assert fit_floor(close_plane, previous=previous, config=CONFIG, up_camera=CAMERA_UP) is previous


def test_the_analytic_floor_is_recovered() -> None:
    scene = clean_scene()

    fitted = fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=None, config=CONFIG, up_camera=CAMERA_UP)

    truth = scene.floor_plane_camera
    assert _angle_between_degrees(fitted.normal, truth.normal) < 1.0
    assert fitted.offset_meters == pytest.approx(truth.offset_meters, abs=0.02)


def test_the_fitted_normal_points_up() -> None:
    scene = clean_scene()

    fitted = fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert fitted.normal @ CAMERA_UP > 0


def test_the_camera_sits_at_its_height_above_the_fitted_floor() -> None:
    scene = clean_scene()
    fitted = fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=None, config=CONFIG, up_camera=CAMERA_UP)

    # The camera is the origin. Its height above the floor is a number the fixture chose.
    assert height_above_floor(np.zeros((1, 3)), fitted)[0] == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.02)


def test_floor_points_measure_as_zero_height_and_the_box_top_as_its_height() -> None:
    scene = clean_scene(box_height_meters=1.0)
    cloud = _cloud(scene.depth_meters, scene.intrinsics)
    fitted = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    heights = height_above_floor(cloud, fitted)

    # Most of the image is floor, so the median height is the floor's.
    assert np.median(heights) == pytest.approx(0.0, abs=0.03)
    # The tallest thing in view is the box top. Allow the fit's tolerance plus pixel quantisation.
    assert heights.max() == pytest.approx(1.0, abs=0.08)


def test_a_scene_with_no_floor_and_no_previous_plane_raises() -> None:
    scene = clean_scene()
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)

    with pytest.raises(ValueError, match="no floor found"):
        fit_floor(_cloud(floorless, scene.intrinsics), previous=None, config=CONFIG, up_camera=CAMERA_UP)


def test_a_scene_with_no_floor_keeps_the_previous_plane() -> None:
    scene = clean_scene()
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(_cloud(floorless, scene.intrinsics), previous=previous, config=CONFIG, up_camera=CAMERA_UP) is previous


def test_a_plane_tilted_past_the_threshold_is_rejected() -> None:
    # A camera pitched 50 degrees down sees a floor whose normal is 50 degrees from camera up,
    # past the 35 degree sanity check. The old file treated that as "not floor-like".
    scene = clean_scene(pitch_degrees=50.0)
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=previous, config=CONFIG, up_camera=CAMERA_UP) is previous


def test_too_few_candidate_points_keeps_the_previous_plane() -> None:
    scene = clean_scene()
    cloud = _cloud(scene.depth_meters, scene.intrinsics)[:50]
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(cloud, previous=previous, config=CONFIG, up_camera=CAMERA_UP) is previous


def test_a_fitted_plane_over_the_maximum_height_is_refused() -> None:
    # A level floor three meters down fits perfectly and is still not where a held or worn camera's
    # floor can be. The first Pixel walk's false plane was 2.3 m down, and only the fit's own
    # ceiling can refuse one that arrives through the fit.
    far_floor = _points_on_plane(np.array([0.0, -1.0, 0.0]), 3.0, np.linspace(-2.0, 2.0, 45), np.linspace(1.5, 6.0, 45))
    assert (far_floor[:, 1] > CONFIG.floor_candidate_min_below_camera_meters).sum() >= CONFIG.floor_min_candidate_points
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(far_floor, previous=previous, config=CONFIG, up_camera=CAMERA_UP) is previous


def test_a_level_floor_at_eye_height_passes_the_gate() -> None:
    assert plane_is_a_floor(clean_scene().floor_plane_camera, CONFIG, CAMERA_UP) is None


def test_a_supplied_plane_leaning_past_the_tilt_gate_is_refused_naming_the_limit() -> None:
    # Forty degrees of tilt and a plausible height. Only the tilt rule is broken.
    tilt = np.radians(40.0)
    leaning = normalize_plane(Plane(normal=np.array([0.0, -np.cos(tilt), np.sin(tilt)]), offset_meters=1.5), CAMERA_UP)

    refusal = plane_is_a_floor(leaning, CONFIG, CAMERA_UP)

    assert refusal is not None
    assert "leans 40.0 deg" in refusal and f"limit {CONFIG.floor_max_tilt_degrees:.1f}" in refusal


def test_a_supplied_plane_too_close_to_the_camera_is_refused_naming_the_minimum() -> None:
    close = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=0.25)

    refusal = plane_is_a_floor(close, CONFIG, CAMERA_UP)

    assert refusal is not None
    assert "0.25 m above it" in refusal and f"under the minimum {CONFIG.floor_min_offset_meters:.2f}" in refusal


def test_a_supplied_plane_over_the_maximum_height_is_refused_naming_the_maximum() -> None:
    # The first Pixel walk's number. Level, so only the ceiling rule is broken.
    far = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=2.3)

    refusal = plane_is_a_floor(far, CONFIG, CAMERA_UP)

    assert refusal is not None
    assert "2.30 m above it" in refusal and f"over the maximum {CONFIG.floor_max_offset_meters:.2f}" in refusal


def test_normalize_plane_flips_a_downward_normal_and_scales_the_offset() -> None:
    # Normal pointing down with twice unit length, offset to match: the same plane, written the
    # way the scene reads it, is normal up and the camera 1.6 m above it.
    upside_down = Plane(normal=np.array([0.0, 2.0, 0.0]), offset_meters=-3.2)

    normalized = normalize_plane(upside_down, CAMERA_UP)

    assert normalized.normal == pytest.approx(np.array([0.0, -1.0, 0.0]))
    assert normalized.offset_meters == pytest.approx(1.6)
    assert height_above_floor(np.zeros((1, 3)), normalized)[0] == pytest.approx(1.6)


def test_normalize_plane_refuses_a_zero_normal() -> None:
    with pytest.raises(ValueError, match="zero length"):
        normalize_plane(Plane(normal=np.zeros(3), offset_meters=1.0), CAMERA_UP)


def _points_on_plane_in_its_own_axes(normal: np.ndarray, offset: float) -> np.ndarray:
    """Camera-frame points on normal . p + offset == 0 for any normal, gridded in the plane's own axes ahead of the camera."""
    normal = normal / np.linalg.norm(normal)
    foot = -offset * normal
    camera_forward = np.array([0.0, 0.0, 1.0])
    forward = camera_forward - (camera_forward @ normal) * normal
    forward = forward / np.linalg.norm(forward)
    lateral = np.cross(forward, normal)
    across, along = np.meshgrid(np.linspace(-2.0, 2.0, 45), np.linspace(1.5, 6.0, 45))
    return foot + across.ravel()[:, None] * lateral + along.ravel()[:, None] * forward


def test_a_camera_rolled_ninety_degrees_finds_the_floor_against_gravity_and_not_against_image_up() -> None:
    # The Pixel held in portrait: the depth image is in the sensor's landscape orientation, so
    # image-up points sideways and the floor's normal sits 90 degrees from it. The same cloud is
    # a level floor 1.6 m down once up means gravity, which the pose supplies.
    pitch = np.radians(20.0)
    unrolled_normal = np.array([0.0, -np.cos(pitch), np.sin(pitch)])
    rolled_normal = np.array([unrolled_normal[1], -unrolled_normal[0], unrolled_normal[2]])  # image-right now points down
    cloud = _points_on_plane_in_its_own_axes(rolled_normal, CAMERA_HEIGHT_METERS)
    gravity_up = rolled_normal / np.linalg.norm(rolled_normal)

    fitted = fit_floor(cloud, previous=None, config=CONFIG, up_camera=gravity_up)

    assert _angle_between_degrees(fitted.normal, gravity_up) < 1.0
    assert fitted.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.02)
    with pytest.raises(ValueError, match="no floor found"):
        fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)


def test_ground_axes_are_unit_orthogonal_and_on_the_plane() -> None:
    plane = clean_scene().floor_plane_camera

    lateral, forward = ground_axes(plane)

    assert np.linalg.norm(lateral) == pytest.approx(1.0)
    assert np.linalg.norm(forward) == pytest.approx(1.0)
    assert lateral @ forward == pytest.approx(0.0, abs=1e-9)
    assert lateral @ plane.normal == pytest.approx(0.0, abs=1e-9)
    assert forward @ plane.normal == pytest.approx(0.0, abs=1e-9)


def test_lateral_is_the_cameras_right() -> None:
    plane = clean_scene().floor_plane_camera

    lateral, forward = ground_axes(plane)

    # Camera x is right. Lateral must agree with it, and forward with camera z.
    assert lateral @ np.array([1.0, 0.0, 0.0]) > 0.9
    assert forward @ np.array([0.0, 0.0, 1.0]) > 0.9


def test_ground_axes_survive_looking_straight_down() -> None:
    straight_down = Plane(normal=np.array([0.0, 0.0, -1.0]), offset_meters=1.6)

    lateral, forward = ground_axes(straight_down)

    assert np.linalg.norm(lateral) == pytest.approx(1.0)
    assert np.linalg.norm(forward) == pytest.approx(1.0)
    assert np.all(np.isfinite(lateral)) and np.all(np.isfinite(forward))
