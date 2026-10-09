"""
Covers finding the floor against a plane known analytically, not against the fit's own output.

The synthetic camera is pitched down twenty degrees over a floor 1.6 m below it, so the floor
plane has a tilted normal and a known offset, and a fit that was wrong by a constant everywhere
would fail here rather than agree with itself.
"""

# Standard library imports
import dataclasses
import math

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene import floor as floor_module
from nav.scene.config import SceneConfig
from nav.scene.floor import (
    CAMERA_UP,
    FloorRefusal,
    FloorRefusalCause,
    LevelFloorChoice,
    LevelFloorHistory,
    fit_floor,
    fit_floor_with_refusal,
    fit_level_floor,
    ground_axes,
    height_above_floor,
    normalize_plane,
    plane_is_a_floor,
)
from nav.scene.unproject import unproject_depth
from nav.types import Plane
from synthetic_depth import CAMERA_HEIGHT_METERS, PITCH_DEGREES, clean_scene, degrade_without_floor, floor_plane_in_camera, pitch_rotation

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
    assert refusal.cause is FloorRefusalCause.LEANS
    assert refusal.measured == pytest.approx(40.0) and refusal.limit == CONFIG.floor_max_tilt_degrees
    assert str(refusal) == f"leans 40.0 deg from up, limit {CONFIG.floor_max_tilt_degrees:.1f}"


def test_a_supplied_plane_too_close_to_the_camera_is_refused_naming_the_minimum() -> None:
    close = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=0.25)

    refusal = plane_is_a_floor(close, CONFIG, CAMERA_UP)

    assert refusal is not None
    assert refusal.cause is FloorRefusalCause.TOO_CLOSE
    assert refusal.measured == pytest.approx(0.25) and refusal.limit == CONFIG.floor_min_offset_meters
    assert str(refusal) == f"camera 0.25 m above it, under the minimum {CONFIG.floor_min_offset_meters:.2f}"


def test_a_supplied_plane_over_the_maximum_height_is_refused_naming_the_maximum() -> None:
    # The first Pixel walk's number. Level, so only the ceiling rule is broken.
    far = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=2.3)

    refusal = plane_is_a_floor(far, CONFIG, CAMERA_UP)

    assert refusal is not None
    assert refusal.cause is FloorRefusalCause.TOO_FAR
    assert refusal.measured == pytest.approx(2.3) and refusal.limit == CONFIG.floor_max_offset_meters
    assert str(refusal) == f"camera 2.30 m above it, over the maximum {CONFIG.floor_max_offset_meters:.2f}"


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


def _noisy_floor_with_clutter(plane: Plane, sigma_meters: float, clutter_count: int, seed: int) -> np.ndarray:
    """A floor with Gaussian noise along its normal, plus clutter standing 0.25 to 1.0 m above it."""
    generator = np.random.default_rng(seed)
    normal = plane.normal / np.linalg.norm(plane.normal)
    floor = _points_on_plane_in_its_own_axes(normal, plane.offset_meters)
    floor = floor + generator.normal(0.0, sigma_meters, len(floor))[:, None] * normal
    # Clutter sits on random floor points, lifted along the normal, so it is ahead of the camera
    # and mostly low enough to vote. It is what a RANSAC has to refuse.
    clutter_bases = floor[generator.integers(0, len(floor), clutter_count)]
    clutter = clutter_bases + generator.uniform(0.25, 1.0, clutter_count)[:, None] * normal
    return np.vstack((floor, clutter))


def _identical(first: Plane, second: Plane) -> bool:
    """Bit for bit, not close. An unseeded fit is close every time and still moves a replay's numbers."""
    return bool(np.array_equal(first.normal, second.normal)) and first.offset_meters == second.offset_meters


def test_the_same_noisy_cloud_fitted_twice_gives_the_identical_plane() -> None:
    cloud = _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.02, clutter_count=700, seed=4)

    first = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)
    second = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert _identical(first, second), f"{first} against {second}"


def test_a_fresh_fit_does_not_depend_on_an_earlier_fit_of_another_cloud() -> None:
    # A generator that carried state from one call to the next would draw different hypotheses
    # for the third fit than for the first, and a replay would depend on every frame before it.
    cloud = _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.02, clutter_count=700, seed=5)
    other_cloud = _noisy_floor_with_clutter(floor_plane_in_camera(pitch_degrees=10.0), sigma_meters=0.02, clutter_count=300, seed=6)

    before = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)
    fit_floor(other_cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)
    after = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert _identical(before, after), f"{before} against {after}"


def test_a_noisy_floor_with_outliers_is_recovered_to_its_known_plane() -> None:
    # Repeating is not the same as being right. A fit that returned one wrong plane every time
    # would pass both tests above, so this one checks against the plane the cloud was built from.
    truth = floor_plane_in_camera()
    cloud = _noisy_floor_with_clutter(truth, sigma_meters=0.02, clutter_count=1000, seed=7)

    fitted = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert _angle_between_degrees(fitted.normal, truth.normal) < 1.0
    assert fitted.offset_meters == pytest.approx(truth.offset_meters, abs=0.02)


def test_a_different_seed_draws_different_hypotheses() -> None:
    # A fit that ignored its seed would pass every repeatability test above. On a noisy floor the
    # winning hypothesis decides which edge points count as inliers, so a different draw refits
    # to a different plane in the last bits, while each seed still repeats itself.
    cloud = _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.03, clutter_count=700, seed=8)
    other_seed = SceneConfig(floor_ransac_seed=1)

    seed_zero = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)
    seed_one = fit_floor(cloud, previous=None, config=other_seed, up_camera=CAMERA_UP)

    assert not _identical(seed_zero, seed_one)
    assert _identical(seed_one, fit_floor(cloud, previous=None, config=other_seed, up_camera=CAMERA_UP))


@pytest.mark.parametrize("planes_per_chunk", [1, 7, 300])
def test_the_chunk_size_does_not_change_the_fitted_plane(planes_per_chunk: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # Chunking exists to bound memory. If it ever changed which plane wins, tuning it would
    # quietly change every replay's numbers.
    cloud = _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.02, clutter_count=700, seed=10)
    default = fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    monkeypatch.setattr(floor_module, "_PLANES_PER_CHUNK", planes_per_chunk)

    assert _identical(fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP), default)


def _count_planes_scored(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap the scorer so a test can see how many planes the search tried before it stopped."""
    scored: list[int] = []
    real_score_planes = floor_module._score_planes

    def counting_score_planes(*arguments, **keyword_arguments):
        counts, rmse = real_score_planes(*arguments, **keyword_arguments)
        scored.append(len(counts))
        return counts, rmse

    monkeypatch.setattr(floor_module, "_score_planes", counting_score_planes)
    # One plane per chunk, so the count is the number of planes tried and not a multiple of a chunk.
    monkeypatch.setattr(floor_module, "_PLANES_PER_CHUNK", 1)
    return scored


def test_on_a_clear_floor_the_search_stops_long_before_its_iteration_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Open3D's stopping rule: with nearly every point on the floor, the first good plane makes a
    # better one unlikely enough after a handful of draws. Without the stop, every frame pays for
    # all 300, which measured at about 30 ms a frame on the Pixel walk against 0.6 ms for Open3D.
    cloud = _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.01, clutter_count=0, seed=12)
    scored = _count_planes_scored(monkeypatch)

    fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert 0 < sum(scored) < 20


def test_a_success_probability_of_one_tries_every_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    cloud = _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.01, clutter_count=0, seed=12)
    scored = _count_planes_scored(monkeypatch)

    fit_floor(cloud, previous=None, config=SceneConfig(floor_ransac_success_probability=1.0), up_camera=CAMERA_UP)

    # Every draw is tried, less the few that pick the same point twice. Those have no plane and
    # are dropped before scoring. The early stop above tries under 20.
    assert CONFIG.floor_ransac_iterations - 10 <= sum(scored) <= CONFIG.floor_ransac_iterations


def test_at_half_inliers_the_search_does_not_stop_early(monkeypatch: pytest.MonkeyPatch) -> None:
    # The other side of the stop. With half the candidates on the floor, Open3D's rule asks for
    # log(1 - 0.99999999) / log(1 - 0.5 ** 3), about 138 draws. A rule that dropped the cube would
    # stop near 27, and one that stopped at the first plane would try 1, and both fit a clear floor
    # well enough to pass every other test here.
    # A level camera, so every floor and clutter point clears the candidate cut. Pitched down, the
    # far floor rises above it and the cloud that reaches the search is mostly floor again.
    level_floor = floor_plane_in_camera(pitch_degrees=0.0)
    floor_count = len(_noisy_floor_with_clutter(level_floor, sigma_meters=0.01, clutter_count=0, seed=14))
    cloud = _noisy_floor_with_clutter(level_floor, sigma_meters=0.01, clutter_count=floor_count, seed=14)
    candidates = cloud[-(cloud @ CAMERA_UP) > CONFIG.floor_candidate_min_below_camera_meters]
    assert len(candidates) == 2 * floor_count, "the fixture must reach the search as half floor, half clutter"
    scored = _count_planes_scored(monkeypatch)

    fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert sum(scored) > 60


def _one_point_many_times(count: int) -> np.ndarray:
    """One point repeated: every triple is three copies of it, so no draw has a plane."""
    return np.tile(np.array([0.0, CAMERA_HEIGHT_METERS, 3.0]), (count, 1))


# A config check that only ran once the search started would pass the first cloud and miss the
# other two, which is where a bad value used to go unnoticed.
CLOUDS_THE_CHECK_MUST_NOT_DEPEND_ON = {
    "fittable": lambda: _noisy_floor_with_clutter(floor_plane_in_camera(), sigma_meters=0.02, clutter_count=0, seed=13),
    "every draw degenerate": lambda: _one_point_many_times(CONFIG.floor_min_candidate_points + 50),
    "too few candidates": lambda: _one_point_many_times(CONFIG.floor_min_candidate_points - 1),
}


@pytest.mark.parametrize("cloud_kind", CLOUDS_THE_CHECK_MUST_NOT_DEPEND_ON)
@pytest.mark.parametrize("probability", [0.0, -0.5, 1.5])
def test_a_success_probability_outside_zero_to_one_is_refused(probability: float, cloud_kind: str) -> None:
    cloud = CLOUDS_THE_CHECK_MUST_NOT_DEPEND_ON[cloud_kind]()
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)
    config = SceneConfig(floor_ransac_success_probability=probability)

    with pytest.raises(ValueError, match=f"floor_ransac_success_probability must be above 0 and at most 1, got {probability}"):
        fit_floor(cloud, previous=previous, config=config, up_camera=CAMERA_UP)


@pytest.mark.parametrize("cloud_kind", CLOUDS_THE_CHECK_MUST_NOT_DEPEND_ON)
def test_a_negative_seed_is_refused_naming_the_field(cloud_kind: str) -> None:
    cloud = CLOUDS_THE_CHECK_MUST_NOT_DEPEND_ON[cloud_kind]()
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    with pytest.raises(ValueError, match="floor_ransac_seed must be non-negative, got -1"):
        fit_floor(cloud, previous=previous, config=SceneConfig(floor_ransac_seed=-1), up_camera=CAMERA_UP)


def test_a_cloud_with_only_degenerate_triples_keeps_the_previous_plane() -> None:
    cloud = _one_point_many_times(CONFIG.floor_min_candidate_points + 50)
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    assert fit_floor(cloud, previous=previous, config=CONFIG, up_camera=CAMERA_UP) is previous


# *******************************************
# Why a fit gave nothing
# *******************************************


def test_a_wall_alone_keeps_the_previous_plane_and_says_it_leans() -> None:
    # A wall 2 m ahead, reaching from above the camera to well below it, so plenty of its points
    # are candidates. The best plane through them is the wall, which stands 90 degrees from up.
    rows, columns = np.mgrid[0:40, 0:40]
    wall = np.column_stack((columns.ravel() * 0.05 - 1.0, rows.ravel() * 0.05 - 0.5, np.full(1600, 2.0)))
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    floor, refusal = fit_floor_with_refusal(wall, previous=previous, config=CONFIG, up_camera=CAMERA_UP)

    assert floor is previous
    assert refusal.cause is FloorRefusalCause.LEANS
    assert refusal.measured == pytest.approx(90.0, abs=1.0)


def test_too_few_candidates_keeps_the_previous_plane_and_gives_the_count() -> None:
    scene = clean_scene()
    few = _cloud(scene.depth_meters, scene.intrinsics)[: CONFIG.floor_min_candidate_points - 1]
    few = few[-(few @ CAMERA_UP) > CONFIG.floor_candidate_min_below_camera_meters]
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    floor, refusal = fit_floor_with_refusal(few, previous=previous, config=CONFIG, up_camera=CAMERA_UP)

    assert floor is previous
    assert refusal == FloorRefusal(FloorRefusalCause.TOO_FEW_CANDIDATES, float(len(few)), float(CONFIG.floor_min_candidate_points))


def test_only_degenerate_triples_say_there_was_no_plane() -> None:
    previous = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=1.5)

    floor, refusal = fit_floor_with_refusal(_one_point_many_times(CONFIG.floor_min_candidate_points + 50), previous, CONFIG, CAMERA_UP)

    assert floor is previous
    assert refusal == FloorRefusal(FloorRefusalCause.NO_PLANE, None, None)
    assert str(refusal) == "every drawn triple was degenerate, so there was no plane to judge"


def test_a_fitted_floor_comes_with_no_refusal_and_matches_fit_floor_bit_for_bit() -> None:
    scene = clean_scene()
    cloud = _cloud(scene.depth_meters, scene.intrinsics)

    floor, refusal = fit_floor_with_refusal(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP)

    assert refusal is None
    assert _identical(floor, fit_floor(cloud, previous=None, config=CONFIG, up_camera=CAMERA_UP))


def test_every_refusal_cause_has_a_message() -> None:
    for cause in FloorRefusalCause:
        assert str(FloorRefusal(cause, 1.0, 2.0))


# *******************************************
# The level-surface floor
# *******************************************

# A level camera, so up is image-up and a surface's drop below the camera is its points' y.
LEVEL_UP = CAMERA_UP
FLOOR_DROP_METERS = 1.6
PREVIOUS = Plane(normal=LEVEL_UP, offset_meters=1.5)


def _level_patch(drop_meters: float, forward_from: float, forward_to: float, spacing: float = 0.1) -> np.ndarray:
    """A level surface drop_meters below a level camera, 4 m wide, from forward_from up to forward_to."""
    lateral = np.arange(-2.0, 2.0 + spacing / 2, spacing)
    forward = np.arange(forward_from, forward_to - spacing / 2, spacing)
    return _points_on_plane(np.array([0.0, -1.0, 0.0]), drop_meters, lateral, forward)


def _ramp(lean_degrees: float) -> np.ndarray:
    """A plane rising away from the camera at the given lean, 1.6 m below it at 1 m ahead, out to 6 m."""
    rise = np.tan(np.radians(lean_degrees))
    x, z = np.meshgrid(np.linspace(-2.0, 2.0, 41), np.linspace(1.0, 6.0, 101))
    y = FLOOR_DROP_METERS - (z - 1.0) * rise
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _floor_and_lower_tier() -> np.ndarray:
    # The walker's tier 1.6 m down out to 3 m, and a lower tier 2.0 m down from 3.5 m, holding more points.
    return np.vstack((_level_patch(FLOOR_DROP_METERS, 1.0, 3.0), _level_patch(2.0, 3.5, 6.5)))


def _fresh_history() -> LevelFloorHistory:
    return LevelFloorHistory(CONFIG.floor_level_history)


def _history_at(height_meters: float, frames: int = 3) -> LevelFloorHistory:
    history = _fresh_history()
    for _ in range(frames):
        history.record_supplied(Plane(normal=LEVEL_UP, offset_meters=height_meters))
    return history


def test_the_floor_is_the_deepest_level_surface_not_a_raised_tier() -> None:
    # The raised tier holds 1640 points against the floor's 615, so one plane fit takes the tier.
    # Tiers stand on the floor, so the deepest level surface is the floor whatever its size.
    cloud = np.vstack((_level_patch(FLOOR_DROP_METERS, 1.5, 3.0), _level_patch(1.2, 3.0, 7.0)))

    choice = fit_level_floor(cloud, None, _fresh_history(), CONFIG, LEVEL_UP)
    ransac, _ = fit_floor_with_refusal(cloud, None, CONFIG, LEVEL_UP)

    assert choice.refusal is None
    assert choice.floor.offset_meters == pytest.approx(FLOOR_DROP_METERS, abs=1e-6)
    assert _angle_between_degrees(choice.floor.normal, LEVEL_UP) < 0.01
    assert ransac.offset_meters == pytest.approx(1.2, abs=0.01)


def test_a_staircase_slope_is_refused_and_the_level_floor_kept() -> None:
    # Ten treads 0.3 m deep rising 0.1 m each from 1.5 m ahead. Every tread lies within the 5 cm
    # inlier distance of one plane through the step edges, leaning atan(0.1 / 0.3) = 18.4 degrees,
    # and holding 4860 points to the floor's 1620, so one plane fit takes that slope. Spread over a
    # meter of height, it never makes a level surface. Each tread is under the 8 % share on its own,
    # so what this guards is the floor's refit staying on the floor rather than reaching the first
    # tread 0.1 m up. The tilt limit itself is the ramp test's.
    floor = _level_patch(FLOOR_DROP_METERS, 0.5, 1.5, spacing=0.05)
    treads = [_level_patch(FLOOR_DROP_METERS - 0.1 * step, 1.5 + 0.3 * (step - 1), 1.5 + 0.3 * step, spacing=0.05) for step in range(1, 11)]
    cloud = np.vstack([floor, *treads])

    choice = fit_level_floor(cloud, None, _fresh_history(), CONFIG, LEVEL_UP)
    ransac, _ = fit_floor_with_refusal(cloud, None, CONFIG, LEVEL_UP)

    assert _angle_between_degrees(ransac.normal, LEVEL_UP) == pytest.approx(18.4, abs=1.5)
    assert choice.refusal is None
    assert choice.floor.offset_meters == pytest.approx(FLOOR_DROP_METERS, abs=1e-6)
    assert _angle_between_degrees(choice.floor.normal, LEVEL_UP) < 0.01


@pytest.mark.parametrize(("lean_degrees", "refused"), [(7.0, False), (9.0, True), (12.0, True)])
def test_a_surface_is_level_up_to_the_tilt_limit_and_a_slope_past_it(lean_degrees: float, refused: bool) -> None:
    # A plain ramp. Each height's slice of it refits to the ramp's own lean exactly, so the 8 degree
    # limit is all that decides. Past it, the previous floor stands and the refusal names the lean.
    choice = fit_level_floor(_ramp(lean_degrees), PREVIOUS, _fresh_history(), CONFIG, LEVEL_UP)

    if refused:
        assert choice.floor is PREVIOUS
        assert choice.refusal.cause is FloorRefusalCause.LEANS
        assert choice.refusal.measured == pytest.approx(lean_degrees, abs=0.01)
        assert choice.refusal.limit == CONFIG.floor_level_max_tilt_degrees
    else:
        assert choice.refusal is None
        assert _angle_between_degrees(choice.floor.normal, LEVEL_UP) == pytest.approx(lean_degrees, abs=0.01)


def test_a_surface_near_the_recent_floor_wins_over_a_deeper_one() -> None:
    # From a higher tier, the lower tiers are deeper. With the walker's floor recently at 1.6 m, the
    # surface there is taken over the lower tier 0.4 m further down. Without that history, deepest wins.
    cloud = _floor_and_lower_tier()

    with_history = fit_level_floor(cloud, None, _history_at(FLOOR_DROP_METERS), CONFIG, LEVEL_UP)
    without_history = fit_level_floor(cloud, None, _fresh_history(), CONFIG, LEVEL_UP)

    assert with_history.floor.offset_meters == pytest.approx(FLOOR_DROP_METERS, abs=1e-6)
    assert with_history.passed_over_deeper and not with_history.reset
    assert without_history.floor.offset_meters == pytest.approx(2.0, abs=1e-6)
    assert not without_history.passed_over_deeper


def test_after_enough_frames_the_deeper_surface_is_taken_and_the_history_resets() -> None:
    # A deeper surface passed over frame after frame means the recent floor settled on the wrong
    # surface, a seat row say. It's passed over reset_frames - 1 times, taken on the next, and the
    # history then starts from it.
    cloud = _floor_and_lower_tier()
    history = _history_at(FLOOR_DROP_METERS)
    offsets = []
    for _ in range(CONFIG.floor_level_reset_frames):
        choice = fit_level_floor(cloud, None, history, CONFIG, LEVEL_UP)
        history.record(choice)
        offsets.append(round(choice.floor.offset_meters, 3))

    assert offsets == [FLOOR_DROP_METERS] * (CONFIG.floor_level_reset_frames - 1) + [2.0]
    assert choice.reset
    assert history.reference_meters == pytest.approx(2.0, abs=1e-6)
    assert history.passed_over_frames == 0
    # And it stays on the new level rather than flipping back.
    assert fit_level_floor(cloud, None, history, CONFIG, LEVEL_UP).floor.offset_meters == pytest.approx(2.0, abs=1e-6)


def test_with_nothing_near_the_recent_floor_the_deepest_is_taken_and_the_history_kept() -> None:
    # The walker's own level out of view, a lower tier in it. The lower tier is the only floor on
    # offer, but one such frame mustn't move the reference, or the next frame would lock onto it.
    history = _history_at(FLOOR_DROP_METERS)

    choice = fit_level_floor(_level_patch(2.0, 3.5, 6.5), None, history, CONFIG, LEVEL_UP)
    history.record(choice)

    assert choice.floor.offset_meters == pytest.approx(2.0, abs=1e-6)
    assert not choice.passed_over_deeper and not choice.reset
    assert history.reference_meters == pytest.approx(FLOOR_DROP_METERS, abs=1e-6)


def test_with_no_level_surface_the_previous_floor_stands_with_its_refusal() -> None:
    # Points spread evenly from 0.6 to 2.2 m down: no 10 cm of height holds the 8 % a surface needs.
    generator = np.random.default_rng(3)
    count = 4000
    cloud = np.column_stack((generator.uniform(-2.0, 2.0, count), generator.uniform(0.6, 2.2, count), generator.uniform(1.0, 6.0, count)))

    choice = fit_level_floor(cloud, PREVIOUS, _fresh_history(), CONFIG, LEVEL_UP)

    assert choice.floor is PREVIOUS
    assert choice.refusal.cause is FloorRefusalCause.NO_LEVEL_SURFACE
    assert choice.refusal.limit == math.ceil(CONFIG.floor_level_min_share * count)
    assert 0 < choice.refusal.measured < choice.refusal.limit


def test_a_level_surface_below_the_height_limit_is_refused_as_too_far() -> None:
    choice = fit_level_floor(_level_patch(2.9, 1.0, 5.0), PREVIOUS, _fresh_history(), CONFIG, LEVEL_UP)

    assert choice.floor is PREVIOUS
    assert choice.refusal.cause is FloorRefusalCause.TOO_FAR
    assert choice.refusal.measured == pytest.approx(2.9, abs=1e-6)


def test_with_no_level_surface_and_no_previous_floor_it_raises() -> None:
    with pytest.raises(ValueError, match="no previous plane"):
        fit_level_floor(_ramp(12.0), None, _fresh_history(), CONFIG, LEVEL_UP)


def test_too_few_candidates_is_refused_as_before() -> None:
    # The same cut and the same refusal as the RANSAC route, so a report counts them as one cause.
    few = _level_patch(FLOOR_DROP_METERS, 1.0, 1.4)
    assert len(few) < CONFIG.floor_min_candidate_points

    choice = fit_level_floor(few, PREVIOUS, _fresh_history(), CONFIG, LEVEL_UP)
    _, ransac_refusal = fit_floor_with_refusal(few, PREVIOUS, CONFIG, LEVEL_UP)

    assert choice.floor is PREVIOUS
    expected = FloorRefusal(FloorRefusalCause.TOO_FEW_CANDIDATES, float(len(few)), float(CONFIG.floor_min_candidate_points))
    assert choice.refusal == ransac_refusal == expected


def test_the_same_frame_always_picks_the_same_floor() -> None:
    generator = np.random.default_rng(5)
    cloud = _floor_and_lower_tier()
    cloud = cloud + generator.normal(0.0, 0.01, cloud.shape)

    first = fit_level_floor(cloud, None, _history_at(FLOOR_DROP_METERS), CONFIG, LEVEL_UP)
    second = fit_level_floor(cloud.copy(), None, _history_at(FLOOR_DROP_METERS), CONFIG, LEVEL_UP)

    assert _identical(first.floor, second.floor)
    assert (first.refusal, first.passed_over_deeper, first.reset) == (second.refusal, second.passed_over_deeper, second.reset)


def test_a_pitched_camera_finds_the_analytic_floor_along_gravity() -> None:
    # Up from gravity, not from the image, which is what the scene hands over on a worn camera.
    scene = clean_scene(box_lateral_meters=None)
    up = scene.floor_plane_camera.normal

    choice = fit_level_floor(_cloud(scene.depth_meters, scene.intrinsics), None, _fresh_history(), CONFIG, up)

    assert choice.refusal is None
    assert choice.floor.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=1e-3)
    assert _angle_between_degrees(choice.floor.normal, floor_plane_in_camera().normal) < 0.05


def test_a_previous_floor_adds_nothing_to_the_history() -> None:
    history = _history_at(FLOOR_DROP_METERS)
    history.record(LevelFloorChoice(Plane(normal=LEVEL_UP, offset_meters=1.6), None, passed_over_deeper=True))

    history.record(LevelFloorChoice(PREVIOUS, FloorRefusal(FloorRefusalCause.NO_LEVEL_SURFACE, 10.0, 200.0)))

    assert history.passed_over_frames == 1
    assert history.reference_meters == pytest.approx(FLOOR_DROP_METERS, abs=1e-6)


def test_the_recent_floor_is_a_median_over_the_last_few() -> None:
    # A median, so one stray height can't drag the reference toward it. Bounded, so an old level is
    # forgotten. Over the last 3, the median is 1.6. Unbounded it would be 1.3, and the mean 4.07.
    history = LevelFloorHistory(3)
    for height in (1.0, 1.0, 1.0, 1.6, 1.6, 9.0):
        history.record(LevelFloorChoice(Plane(normal=LEVEL_UP, offset_meters=height), None))

    assert history.reference_meters == pytest.approx(1.6)


def test_of_the_surfaces_near_the_recent_floor_the_deepest_is_taken() -> None:
    # A stair tread 0.15 m above the walker's floor is inside the 0.25 m tolerance too. Deepest-wins
    # holds among the near surfaces as well, so the tread loses to the floor.
    cloud = np.vstack((_level_patch(1.45, 1.0, 2.5), _level_patch(FLOOR_DROP_METERS, 2.5, 4.0), _level_patch(2.0, 4.0, 6.0)))

    choice = fit_level_floor(cloud, None, _history_at(1.5), CONFIG, LEVEL_UP)

    assert choice.floor.offset_meters == pytest.approx(FLOOR_DROP_METERS, abs=1e-6)
    assert choice.passed_over_deeper


def test_on_a_small_frame_a_surface_still_needs_two_hundred_points() -> None:
    # 8 % of 1000 candidates is 80 points, so the 200-point floor is what decides here. A 150-point
    # patch is no surface, as on the Pixel's sparse frames before ARCore's first floor.
    small_patch = _level_patch(FLOOR_DROP_METERS, 1.0, 1.4)
    generator = np.random.default_rng(7)
    spread = np.column_stack((generator.uniform(-2.0, 2.0, 850), generator.uniform(0.6, 1.4, 850), generator.uniform(1.0, 6.0, 850)))
    cloud = np.vstack((small_patch, spread))
    # The patch beats 8 % and misses 200, so only the 200 can refuse it.
    assert math.ceil(CONFIG.floor_level_min_share * len(cloud)) < len(small_patch) < CONFIG.floor_min_candidate_points

    choice = fit_level_floor(cloud, PREVIOUS, _fresh_history(), CONFIG, LEVEL_UP)

    assert choice.floor is PREVIOUS
    assert choice.refusal.cause is FloorRefusalCause.NO_LEVEL_SURFACE
    assert choice.refusal.limit == CONFIG.floor_min_candidate_points


def test_the_passed_over_count_restarts_when_a_frame_takes_the_deepest() -> None:
    # Glimpses of a lower tier between frames that see only the walker's floor never add up to a
    # reset. Only reset_frames in a row do.
    both = _floor_and_lower_tier()
    floor_only = _level_patch(FLOOR_DROP_METERS, 1.0, 3.0)
    history = _history_at(FLOOR_DROP_METERS)
    offsets = []
    for cloud in [both] * (CONFIG.floor_level_reset_frames - 2) + [floor_only] + [both] * CONFIG.floor_level_reset_frames:
        choice = fit_level_floor(cloud, None, history, CONFIG, LEVEL_UP)
        history.record(choice)
        offsets.append(round(choice.floor.offset_meters, 3))

    expected = [FLOOR_DROP_METERS] * (2 * CONFIG.floor_level_reset_frames - 2) + [2.0]
    assert offsets == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("floor_level_bin_meters", 0.0),
        ("floor_level_min_share", -0.1),
        ("floor_level_min_share", 1.5),
        ("floor_level_tolerance_meters", -0.1),
        ("floor_level_reset_frames", 0),
    ],
)
def test_a_level_setting_out_of_range_is_refused_naming_it(field: str, value: float) -> None:
    config = dataclasses.replace(CONFIG, **{field: value})

    with pytest.raises(ValueError, match=field):
        fit_level_floor(_floor_and_lower_tier(), PREVIOUS, _fresh_history(), config, LEVEL_UP)


def test_a_history_of_no_frames_is_refused() -> None:
    with pytest.raises(ValueError, match="floor_level_history"):
        LevelFloorHistory(0)
