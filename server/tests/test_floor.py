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
from nav.scene.floor import CAMERA_UP, fit_floor, ground_axes, height_above_floor
from nav.scene.unproject import unproject_depth
from nav.types import Plane
from synthetic_depth import CAMERA_HEIGHT_METERS, clean_scene, degrade_without_floor

CONFIG = SceneConfig()


def _cloud(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return unproject_depth(depth, intrinsics, stride=1, config=CONFIG)


def _angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def test_the_analytic_floor_is_recovered() -> None:
    scene = clean_scene()

    fitted = fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=None, config=CONFIG)

    truth = scene.floor_plane_camera
    assert _angle_between_degrees(fitted.normal, truth.normal) < 1.0
    assert fitted.offset_meters == pytest.approx(truth.offset_meters, abs=0.02)


def test_the_fitted_normal_points_up() -> None:
    scene = clean_scene()

    fitted = fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=None, config=CONFIG)

    assert fitted.normal @ CAMERA_UP > 0


def test_the_camera_sits_at_its_height_above_the_fitted_floor() -> None:
    scene = clean_scene()
    fitted = fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=None, config=CONFIG)

    # The camera is the origin. Its height above the floor is a number the fixture chose.
    assert height_above_floor(np.zeros((1, 3)), fitted)[0] == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.02)


def test_floor_points_measure_as_zero_height_and_the_box_top_as_its_height() -> None:
    scene = clean_scene(box_height_meters=1.0)
    cloud = _cloud(scene.depth_meters, scene.intrinsics)
    fitted = fit_floor(cloud, previous=None, config=CONFIG)

    heights = height_above_floor(cloud, fitted)

    # Most of the image is floor, so the median height is the floor's.
    assert np.median(heights) == pytest.approx(0.0, abs=0.03)
    # The tallest thing in view is the box top. Allow the fit's tolerance plus pixel quantisation.
    assert heights.max() == pytest.approx(1.0, abs=0.08)


def test_a_scene_with_no_floor_and_no_previous_plane_raises() -> None:
    scene = clean_scene()
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)

    with pytest.raises(ValueError, match="no floor found"):
        fit_floor(_cloud(floorless, scene.intrinsics), previous=None, config=CONFIG)


def test_a_scene_with_no_floor_keeps_the_previous_plane() -> None:
    scene = clean_scene()
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(_cloud(floorless, scene.intrinsics), previous=previous, config=CONFIG) is previous


def test_a_plane_tilted_past_the_threshold_is_rejected() -> None:
    # A camera pitched 50 degrees down sees a floor whose normal is 50 degrees from camera up,
    # past the 35 degree sanity check. The old file treated that as "not floor-like".
    scene = clean_scene(pitch_degrees=50.0)
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(_cloud(scene.depth_meters, scene.intrinsics), previous=previous, config=CONFIG) is previous


def test_too_few_candidate_points_keeps_the_previous_plane() -> None:
    scene = clean_scene()
    cloud = _cloud(scene.depth_meters, scene.intrinsics)[:50]
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(cloud, previous=previous, config=CONFIG) is previous


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
