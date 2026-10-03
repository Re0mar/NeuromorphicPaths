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
from nav.scene import floor as floor_module
from nav.scene.config import SceneConfig
from nav.scene.floor import CAMERA_UP, fit_floor, ground_axes, height_above_floor, normalize_plane, plane_is_a_floor
from nav.scene.unproject import unproject_depth
from nav.types import Plane
from synthetic_depth import CAMERA_HEIGHT_METERS, clean_scene, degrade_without_floor, floor_plane_in_camera

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
