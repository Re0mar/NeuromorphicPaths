"""
Covers calibrations and undistortion.

The reference for where a pixel should land is OpenCV's own forward projection, cv2.projectPoints,
not this module's undistortion run backwards. A round trip through one module's own pair of
functions passes whatever mistake both share.
"""

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.sources.camera_model import (
    CameraCalibration,
    CameraModelError,
    Undistorter,
    horizontal_field_of_view_degrees,
    scale_intrinsics,
)

HEIGHT, WIDTH = 480, 640
CAMERA_MATRIX = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
# Barrel distortion in OpenCV's eight-coefficient rational model, the length the Neon reports.
BARREL = np.array([-0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
NO_DISTORTION = np.zeros(8)

SCENE_CAMERA_MATRIX = np.array([[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]])
SCENE_SIZE = (1200, 1600)
DEPTH_SIZE = (300, 400)

# The Neon's own calibration, as read off the device.
NEON_CAMERA_MATRIX = np.array([[890.9483, 0.0, 807.2718], [0.0, 890.5604, 608.4522], [0.0, 0.0, 1.0]])
NEON_DISTORTION = np.array([-0.1307, 0.1092, -0.0003, -0.0005, 0.0, 0.1702, 0.0519, 0.0255])
NEON_SIZE = (1200, 1600)


def _calibration(distortion: np.ndarray = BARREL, matrix: np.ndarray = CAMERA_MATRIX) -> CameraCalibration:
    return CameraCalibration(camera_matrix=matrix, distortion_coefficients=distortion, image_size=(HEIGHT, WIDTH))


def _neon_calibration() -> CameraCalibration:
    return CameraCalibration(camera_matrix=NEON_CAMERA_MATRIX, distortion_coefficients=NEON_DISTORTION, image_size=NEON_SIZE)


def test_scale_intrinsics_refuses_a_zero_source_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        scale_intrinsics(SCENE_CAMERA_MATRIX, (0, 1600), (300, 400))


def test_scale_intrinsics_leaves_the_homogeneous_row_alone() -> None:
    scaled = scale_intrinsics(SCENE_CAMERA_MATRIX, SCENE_SIZE, DEPTH_SIZE)

    assert scaled[2] == pytest.approx([0.0, 0.0, 1.0])
    assert scaled[0, 0] == pytest.approx(225.0)
    assert scaled[1, 2] == pytest.approx(150.0)


def test_scale_intrinsics_scales_x_by_the_width_ratio_and_y_by_the_height_ratio() -> None:
    """A model's process resolution need not keep the aspect ratio, so each axis has its own ratio."""
    # (100, 200) to (50, 400): the width doubles and the height halves. Every other scaling in the
    # suite is uniform, where swapping the two ratios changes nothing.
    camera_matrix = np.array([[80.0, 0.0, 100.0], [0.0, 60.0, 50.0], [0.0, 0.0, 1.0]])

    scaled = scale_intrinsics(camera_matrix, (100, 200), (50, 400))

    assert scaled == pytest.approx(np.array([[160.0, 0.0, 200.0], [0.0, 30.0, 25.0], [0.0, 0.0, 1.0]]))


def test_horizontal_field_of_view_of_a_centered_camera() -> None:
    # Half the width over the focal length is tan of half the angle: 320 / 400 gives 38.66 a side.
    assert horizontal_field_of_view_degrees(CAMERA_MATRIX, WIDTH) == pytest.approx(2 * np.degrees(np.arctan(0.8)))


def test_every_undistorted_neon_pixel_samples_inside_the_delivered_image() -> None:
    """The estimator invents depth for any border pixel with no source, so the crop must leave none."""
    undistorter = Undistorter(_neon_calibration())
    height, width = NEON_SIZE

    # The maps the undistorter actually remaps with, turned back into float sample positions.
    # Half a pixel past the edge is what linear interpolation on the outermost row reaches.
    sample_x, sample_y = cv2.convertMaps(undistorter._map_x, undistorter._map_y, cv2.CV_32FC1)

    assert sample_x.min() >= -0.5
    assert sample_x.max() <= width - 0.5
    assert sample_y.min() >= -0.5
    assert sample_y.max() <= height - 0.5


def test_the_undistortion_maps_are_built_once_not_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """The maps depend only on the calibration, so building them per frame repeats the same work every frame."""
    built = []
    real_build = cv2.initUndistortRectifyMap

    def counting_build(*arguments):
        built.append(arguments)
        return real_build(*arguments)

    monkeypatch.setattr(cv2, "initUndistortRectifyMap", counting_build)
    undistorter = Undistorter(_calibration())
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for _ in range(3):
        undistorter.undistort_image(image)

    assert len(built) == 1


def test_an_undistorted_pixel_lands_where_the_pinhole_model_puts_the_scene_point() -> None:
    undistorter = Undistorter(_calibration())
    scene_point = np.array([[0.5, -0.3, 2.0]])
    no_rotation = np.zeros(3)
    no_translation = np.zeros(3)

    distorted, _ = cv2.projectPoints(scene_point, no_rotation, no_translation, CAMERA_MATRIX, BARREL)
    pinhole, _ = cv2.projectPoints(scene_point, no_rotation, no_translation, undistorter.camera_matrix, NO_DISTORTION)

    landed = undistorter.undistort_pixel(distorted.reshape(2))

    assert landed == pytest.approx(pinhole.reshape(2), abs=0.1)


def test_zero_distortion_leaves_the_matrix_and_the_image_unchanged() -> None:
    undistorter = Undistorter(_calibration(distortion=NO_DISTORTION))
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)

    assert undistorter.camera_matrix == pytest.approx(CAMERA_MATRIX, abs=1e-6)
    assert np.array_equal(undistorter.undistort_image(image), image)


def test_undistorted_pixels_are_square() -> None:
    # OpenCV's alpha 0 matrix for the Neon has fx 636 and fy 754, which squashes the picture the
    # depth model sees.
    matrix = Undistorter(_neon_calibration()).camera_matrix

    assert matrix[0, 0] == pytest.approx(matrix[1, 1])
    # The larger of OpenCV's two, so the straightened view keeps the full 77 degrees vertically.
    assert matrix[1, 1] == pytest.approx(754.4, abs=0.5)


def test_barrel_undistortion_narrows_the_field_of_view() -> None:
    # The crop that keeps the border clean takes some of the edges off. This is the number the
    # hardware check reports, so it has to move the right way.
    undistorter = Undistorter(_calibration())

    assert undistorter.field_of_view_after_degrees < undistorter.field_of_view_before_degrees


def test_the_neon_keeps_ninety_three_degrees_across_after_undistortion() -> None:
    """The angle the depth model sees is the straightened matrix's, not the lens's own matrix's."""
    # 2 atan(800 / 754.4) is about 93.4 degrees, the figure recorded when square pixels were chosen.
    # Taken from the distorted matrix instead, it reads about 84.
    undistorter = Undistorter(_neon_calibration())

    assert undistorter.field_of_view_after_degrees == pytest.approx(93.4, abs=0.1)


def test_a_point_near_the_neon_corner_undistorts_to_where_the_pinhole_model_puts_it() -> None:
    """Gaze near the edge of the lens has to land on the same spot as the straightened image."""
    # This direction projects about 807 px from the principal point, where a solver that stops
    # after a fixed few rounds was 6 px off. The straightened point is still inside the crop.
    undistorter = Undistorter(_neon_calibration())
    scene_point = np.array([[-2.0, -1.6, 2.0]])
    no_rotation = np.zeros(3)
    no_translation = np.zeros(3)

    distorted, _ = cv2.projectPoints(scene_point, no_rotation, no_translation, NEON_CAMERA_MATRIX, NEON_DISTORTION)
    pinhole, _ = cv2.projectPoints(scene_point, no_rotation, no_translation, undistorter.camera_matrix, np.zeros(8))
    distance_from_center = np.linalg.norm(distorted.reshape(2) - NEON_CAMERA_MATRIX[:2, 2])

    landed = undistorter.undistort_pixel(distorted.reshape(2))

    assert distance_from_center > 800.0
    assert landed is not None
    assert landed == pytest.approx(pinhole.reshape(2), abs=0.05)


def test_the_neon_field_of_view_before_undistortion_is_the_lens_edge_to_edge_angle() -> None:
    """The before figure reports what the lens really covers, from the edge pixels' true directions."""
    # Each edge pixel on the center row maps to a direction whose forward projection lands back on
    # that pixel. Their angles from the axis, added, are the true width. Found by search against
    # projectPoints rather than by this module's own undistortion.
    principal_x, principal_y = NEON_CAMERA_MATRIX[0, 2], NEON_CAMERA_MATRIX[1, 2]

    def edge_angle(edge_x: float) -> float:
        low, high = 0.0, 1.5
        for _ in range(100):
            tangent = (low + high) / 2.0
            direction = np.array([[np.sign(edge_x - principal_x) * tangent, 0.0, 1.0]])
            projected, _ = cv2.projectPoints(direction, np.zeros(3), np.zeros(3), NEON_CAMERA_MATRIX, NEON_DISTORTION)
            if abs(projected.reshape(2)[0] - principal_x) < abs(edge_x - principal_x):
                low = tangent
            else:
                high = tangent
        return float(np.degrees(np.arctan((low + high) / 2.0)))

    expected = edge_angle(0.0) + edge_angle(float(NEON_SIZE[1]))

    assert Undistorter(_neon_calibration()).field_of_view_before_degrees == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.eye(2), r"\(3, 3\)"),
        (np.array([[np.nan, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]), "finite"),
        (np.array([[0.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]), "focal lengths"),
    ],
)
def test_calibration_refuses_a_non_finite_or_misshapen_matrix(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(CameraModelError, match=message):
        _calibration(matrix=matrix)


@pytest.mark.parametrize("length", [3, 7])
def test_calibration_refuses_a_distortion_vector_of_a_length_opencv_rejects(length: int) -> None:
    with pytest.raises(CameraModelError, match="distortion_coefficients"):
        _calibration(distortion=np.zeros(length))


def test_calibration_refuses_a_non_finite_distortion_coefficient() -> None:
    distortion = BARREL.copy()
    distortion[1] = np.inf

    with pytest.raises(CameraModelError, match="finite"):
        _calibration(distortion=distortion)


def test_calibration_refuses_a_principal_point_outside_the_image() -> None:
    # The usual cause is a matrix for another resolution, such as the scene camera's native size
    # against a downscaled frame.
    with pytest.raises(CameraModelError, match="outside"):
        _calibration(matrix=SCENE_CAMERA_MATRIX)


def test_undistorting_an_image_of_another_size_is_refused() -> None:
    undistorter = Undistorter(_calibration())

    with pytest.raises(CameraModelError, match="calibration describes"):
        undistorter.undistort_image(np.zeros((HEIGHT // 2, WIDTH // 2, 3), dtype=np.uint8))


def test_a_gaze_point_cropped_off_by_undistortion_becomes_none() -> None:
    # A barrel lens's corner is pulled outward when straightened, past the cropped edge.
    undistorter = Undistorter(_calibration())

    assert undistorter.undistort_pixel(np.array([0.0, 0.0])) is None


def test_the_calibration_cannot_be_changed_through_the_callers_array() -> None:
    callers_matrix = CAMERA_MATRIX.copy()
    calibration = _calibration(matrix=callers_matrix)

    callers_matrix[0, 0] = 1.0

    assert calibration.camera_matrix[0, 0] == pytest.approx(400.0)
