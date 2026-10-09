"""
A camera's own calibration, and undistorting what it sees.

The depth estimator assumes a pinhole camera, and the planner reads lateral position. A wide lens
bends lateral position most at the edges of the image, which is exactly where an obstacle beside
the walker sits. So a camera that knows its own distortion gets its frames straightened here before
the estimator sees them, and the straightened camera matrix becomes the frame's intrinsics.

OpenCV and numpy only. No device client is imported, so the suite can exercise all of it.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import cv2
import numpy as np

# The lengths OpenCV's distortion models accept. The Neon reports eight, the rational model.
ACCEPTED_DISTORTION_LENGTHS = frozenset({4, 5, 8, 12, 14})

# Plain undistortPoints stops after a fixed few iterations, which left the Neon's corners up to
# 7.7 px off. Iterate until the point reprojects within a millionth of a pixel, or 50 rounds.
UNDISTORT_POINT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6)


class CameraModelError(ValueError):
    """A calibration that cannot describe a real camera, or an image it does not fit."""


def scale_intrinsics(camera_matrix: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """
    Rescale a camera matrix from one image size to another.

    :param camera_matrix: (3, 3) at from_size.
    :param from_size: (height, width) the matrix describes.
    :param to_size: (height, width) wanted.
    :return: (3, 3) at to_size.
    :rtype: np.ndarray
    """
    from_height, from_width = from_size
    to_height, to_width = to_size
    if from_height <= 0 or from_width <= 0:
        raise ValueError(f"source image size must be positive, got {from_size}")
    scaled = np.array(camera_matrix, dtype=np.float64)
    scaled[0, :] *= to_width / from_width
    scaled[1, :] *= to_height / from_height
    return scaled


def undistorter_for(
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    calibrated_size: tuple[int, int],
    image_size: tuple[int, int],
) -> "Undistorter":
    """
    The straightener for images of one size, from a calibration made at another.

    Every camera that straightens builds it this way, so two routes reading the same camera can't
    straighten it differently.

    :param camera_matrix: (3, 3) at calibrated_size.
    :param distortion_coefficients: The calibration's coefficients. Unitless, so they don't scale.
    :param calibrated_size: (height, width) the matrix describes.
    :param image_size: (height, width) of the images to straighten.
    :return: The straightener, its maps built at image_size.
    :rtype: Undistorter
    :raises CameraModelError: When the calibration can't describe a camera of that size.
    """
    if image_size != calibrated_size:
        camera_matrix = scale_intrinsics(camera_matrix, calibrated_size, image_size)
    calibration = CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_size=image_size,
    )
    return Undistorter(calibration)


def horizontal_field_of_view_degrees(camera_matrix: np.ndarray, image_width: int) -> float:
    """
    The horizontal angle a pinhole camera matrix covers across an image of the given width.

    :param camera_matrix: (3, 3).
    :param image_width: Pixels.
    :return: Degrees, left edge to right edge.
    :rtype: float
    """
    focal_x, principal_x = camera_matrix[0, 0], camera_matrix[0, 2]
    left = np.arctan(principal_x / focal_x)
    right = np.arctan((image_width - principal_x) / focal_x)
    return float(np.degrees(left + right))


@dataclass(frozen=True)
class CameraCalibration:
    """A camera matrix, its distortion coefficients, and the image size both describe."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]  # (height, width)

    def __post_init__(self) -> None:
        matrix = np.array(self.camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise CameraModelError(f"camera_matrix must be (3, 3), got shape {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise CameraModelError(f"camera_matrix must be finite, got {matrix.tolist()}")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise CameraModelError(f"camera_matrix focal lengths must be positive, got {matrix[0, 0]} and {matrix[1, 1]}")

        distortion = np.array(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if distortion.size not in ACCEPTED_DISTORTION_LENGTHS:
            raise CameraModelError(
                f"distortion_coefficients must have one of {sorted(ACCEPTED_DISTORTION_LENGTHS)} entries, got {distortion.size}"
            )
        if not np.all(np.isfinite(distortion)):
            raise CameraModelError(f"distortion_coefficients must be finite, got {distortion.tolist()}")

        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise CameraModelError(f"image_size must be positive, got {self.image_size}")
        principal_x, principal_y = matrix[0, 2], matrix[1, 2]
        # A principal point off the image means the matrix belongs to some other resolution.
        if not (0.0 <= principal_x <= width and 0.0 <= principal_y <= height):
            raise CameraModelError(
                f"camera_matrix principal point ({principal_x}, {principal_y}) is outside the {width}x{height} image"
            )

        # Copies, so a caller's array cannot change the calibration afterwards.
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "image_size", (int(height), int(width)))


class Undistorter:
    """Straightens one camera's images and pixel positions. The maps are built once."""

    def __init__(self, calibration: CameraCalibration) -> None:
        height, width = calibration.image_size
        self._calibration = calibration
        # alpha 0 crops to the region every output pixel has a source for. Same size out as in,
        # and no black border for the estimator to invent depth in.
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            (width, height),
            0,
            (width, height),
        )
        new_matrix = np.asarray(new_matrix, dtype=np.float64)
        # OpenCV fills the frame by scaling x and y separately, and on the Neon that came out as
        # fx 636 against fy 754, a picture squashed 16 percent sideways. The depth model learned on
        # square pixels, so both take the larger focal. That crops a little more off the wider axis.
        square_focal = max(new_matrix[0, 0], new_matrix[1, 1])
        new_matrix[0, 0] = new_matrix[1, 1] = square_focal
        self._camera_matrix = new_matrix
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            self._camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        """The pinhole matrix that describes undistorted images, at the calibration's size."""
        return self._camera_matrix.copy()

    @property
    def field_of_view_before_degrees(self) -> float:
        """The horizontal angle the delivered, distorted image covers, edge to edge through its center row."""
        # Not the pinhole angle of the distorted matrix. A barrel lens puts more angle into its edge
        # pixels than the matrix alone says, so the true angle comes from undistorting those pixels.
        height, width = self._calibration.image_size
        principal_y = self._calibration.camera_matrix[1, 2]
        edges = np.array([[[0.0, principal_y]], [[float(width), principal_y]]])
        # No rectification and no new matrix, so the result is in normalized coordinates.
        normalized = cv2.undistortPointsIter(
            edges,
            self._calibration.camera_matrix,
            self._calibration.distortion_coefficients,
            None,
            None,
            UNDISTORT_POINT_CRITERIA,
        ).reshape(2, 2)
        return float(np.degrees(np.arctan(-normalized[0, 0]) + np.arctan(normalized[1, 0])))

    @property
    def field_of_view_after_degrees(self) -> float:
        return horizontal_field_of_view_degrees(self._camera_matrix, self._calibration.image_size[1])

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Straighten one image.

        :param image: (height, width, channels) at the calibration's size.
        :return: The undistorted image, same size and dtype.
        :rtype: np.ndarray
        :raises CameraModelError: When the image is not the calibration's size.
        """
        if image.shape[:2] != self._calibration.image_size:
            raise CameraModelError(
                f"image is {image.shape[1]}x{image.shape[0]}, the calibration describes "
                f"{self._calibration.image_size[1]}x{self._calibration.image_size[0]}"
            )
        # Replicate rather than the default black, because linear interpolation on the outermost
        # row can still reach half a pixel past the source, and black there is depth the estimator invents.
        return cv2.remap(image, self._map_x, self._map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def undistort_pixel(self, pixel_xy: np.ndarray) -> np.ndarray | None:
        """
        Where a pixel of the distorted image lands in the undistorted one.

        :param pixel_xy: (2,) x and y in the distorted image.
        :return: (2,) in the undistorted image, or None when the crop cut it off.
        :rtype: np.ndarray | None
        """
        points = np.asarray(pixel_xy, dtype=np.float64).reshape(1, 1, 2)
        undistorted = cv2.undistortPointsIter(
            points,
            self._calibration.camera_matrix,
            self._calibration.distortion_coefficients,
            None,
            self._camera_matrix,
            UNDISTORT_POINT_CRITERIA,
        ).reshape(2)
        height, width = self._calibration.image_size
        if not (0.0 <= undistorted[0] < width and 0.0 <= undistorted[1] < height):
            return None
        return undistorted
