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


def _calibration(distortion: np.ndarray = BARREL, matrix: np.ndarray = CAMERA_MATRIX) -> CameraCalibration:
    return CameraCalibration(camera_matrix=matrix, distortion_coefficients=distortion, image_size=(HEIGHT, WIDTH))


def test_scale_intrinsics_refuses_a_zero_source_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        scale_intrinsics(SCENE_CAMERA_MATRIX, (0, 1600), (300, 400))


def test_scale_intrinsics_leaves_the_homogeneous_row_alone() -> None:
    scaled = scale_intrinsics(SCENE_CAMERA_MATRIX, SCENE_SIZE, DEPTH_SIZE)

    assert scaled[2] == pytest.approx([0.0, 0.0, 1.0])
    assert scaled[0, 0] == pytest.approx(225.0)
    assert scaled[1, 2] == pytest.approx(150.0)


def test_horizontal_field_of_view_of_a_centred_camera() -> None:
    # Half the width over the focal length is tan of half the angle: 320 / 400 gives 38.66 a side.
    assert horizontal_field_of_view_degrees(CAMERA_MATRIX, WIDTH) == pytest.approx(2 * np.degrees(np.arctan(0.8)))


def test_undistortion_keeps_the_image_size_and_leaves_no_black_border() -> None:
    undistorter = Undistorter(_calibration())
    gray = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)

    straightened = undistorter.undistort_image(gray)

    assert straightened.shape == gray.shape
    assert straightened.dtype == np.uint8
    assert int(straightened.min()) > 0


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


def test_barrel_undistortion_narrows_the_field_of_view() -> None:
    # The crop that keeps the border clean costs some of the edges. This is the number the
    # hardware check reports, so it has to move the right way.
    undistorter = Undistorter(_calibration())

    assert undistorter.field_of_view_after_degrees < undistorter.field_of_view_before_degrees


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
