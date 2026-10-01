"""Covers depth pixels becoming camera-frame points, and which pixels never do."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.unproject import downsample, unproject_depth
from synthetic_depth import intrinsics

CONFIG = SceneConfig()
CAMERA = intrinsics(height=8, width=8, focal=4.0)  # principal point at (4, 4)


def _depth(value: float = 2.0) -> np.ndarray:
    return np.full((8, 8), value, dtype=np.float32)


def test_the_principal_point_unprojects_straight_ahead() -> None:
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    depth[4, 4] = 2.0

    points = unproject_depth(depth, CAMERA, stride=1, config=CONFIG)

    # Known from the pinhole model, not read back out of the function.
    assert points == pytest.approx(np.array([[0.0, 0.0, 2.0]]))


def test_an_off_centre_pixel_lands_where_the_pinhole_model_says() -> None:
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    depth[6, 2] = 4.0  # row 6, column 2: two pixels right of... two left, two down

    points = unproject_depth(depth, CAMERA, stride=1, config=CONFIG)

    # x = (2 - 4) * 4 / 4 = -2, y = (6 - 4) * 4 / 4 = 2, z = 4.
    assert points == pytest.approx(np.array([[-2.0, 2.0, 4.0]]))


@pytest.mark.parametrize("invalid", [np.nan, 0.0, -1.0, np.inf, 0.05, 31.0])
def test_invalid_depth_produces_no_point(invalid: float) -> None:
    depth = _depth(2.0)
    depth[3, 3] = invalid

    points = unproject_depth(depth, CAMERA, stride=1, config=CONFIG)

    assert len(points) == 63
    assert np.all(np.isfinite(points))


def test_depth_exactly_at_the_range_bounds_is_kept() -> None:
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    depth[0, 0] = CONFIG.min_depth_meters
    depth[7, 7] = CONFIG.max_depth_meters

    assert len(unproject_depth(depth, CAMERA, stride=1, config=CONFIG)) == 2


def test_stride_two_keeps_a_quarter_of_the_pixels() -> None:
    assert len(unproject_depth(_depth(), CAMERA, stride=2, config=CONFIG)) == 16


def test_stride_two_keeps_the_pixel_coordinates_not_the_subsampled_ones() -> None:
    # The bug this catches: computing u, v on the strided image and then unprojecting with the
    # full-resolution intrinsics, which puts every point at half its true x and y.
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    depth[6, 6] = 4.0

    points = unproject_depth(depth, CAMERA, stride=2, config=CONFIG)

    assert points == pytest.approx(np.array([[2.0, 2.0, 4.0]]))


def test_a_stride_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="stride"):
        unproject_depth(_depth(), CAMERA, stride=0, config=CONFIG)


def test_an_all_invalid_image_yields_an_empty_cloud_of_the_right_shape() -> None:
    points = unproject_depth(np.zeros((8, 8), dtype=np.float32), CAMERA, stride=1, config=CONFIG)

    assert points.shape == (0, 3)


def test_downsampling_a_dense_cloud_leaves_fewer_points() -> None:
    points = unproject_depth(_depth(), CAMERA, stride=1, config=CONFIG)

    thinned = downsample(points, voxel_size_meters=10.0)

    assert 1 <= len(thinned) < len(points)
    assert thinned.shape[1] == 3


def test_downsampling_an_empty_cloud_is_fine() -> None:
    assert downsample(np.empty((0, 3)), voxel_size_meters=0.05).shape == (0, 3)


def test_downsampling_with_a_non_positive_voxel_is_refused() -> None:
    with pytest.raises(ValueError, match="voxel"):
        downsample(np.zeros((3, 3)), voxel_size_meters=0.0)
