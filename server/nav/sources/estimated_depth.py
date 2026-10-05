"""
An RGB source plus an estimator, composed into a source of depth frames.

The estimator is composed in here rather than run as a stage of its own, because a DepthFrame
without depth is not a frame. If the estimator were a stage, the scene's input type would depend
on which source was plugged in, which is the thing the layering exists to prevent.
"""

# Standard library imports
import dataclasses
import logging
from collections.abc import Iterator

# Third party imports
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.pose.imu_orientation import identity_pose
from nav.sources.camera_model import scale_intrinsics
from nav.sources.config import EstimatorConfig
from nav.sources.estimator import DepthEstimate, DepthEstimatorProtocol, apply_confidence_filter, fallback_intrinsics
from nav.sources.rgb import RgbFrame, RgbSource
from nav.types import DepthFrame

log = logging.getLogger(__name__)


class EstimatedDepthSource:
    """Turns a camera into a depth source by running the estimator on every frame."""

    def __init__(
        self,
        rgb_source: RgbSource,
        estimator: DepthEstimatorProtocol,
        config: EstimatorConfig,
    ) -> None:
        self._rgb_source = rgb_source
        self._estimator = estimator
        self._config = config
        self._warned_about_fallback_intrinsics = False

    def frames(self) -> Iterator[DepthFrame]:
        # Before the first real frame is even asked for, so the walker's first arrow and the first
        # latency figure do not carry the model's start-up cost.
        self._estimator.warm_up()

        for rgb_frame in self._rgb_source.frames():
            estimate = self._estimator.estimate(rgb_frame.image_rgb)
            depth = apply_confidence_filter(estimate, self._config.confidence_drop_percentile)
            intrinsics = self._intrinsics_for(rgb_frame, estimate, depth.shape)
            depth_meters = to_meters(depth, intrinsics, estimate.canonical_focal_pixels)

            timing = None
            if rgb_frame.timing is not None:
                timing = dataclasses.replace(rgb_frame.timing, depth_ready_seconds=laptop_time_seconds())

            # A camera with no orientation gets the camera's own axes, and the scene falls back to
            # image-up for it.
            pose = rgb_frame.pose if rgb_frame.pose is not None else identity_pose()

            yield DepthFrame(
                timestamp_seconds=rgb_frame.timestamp_seconds,
                depth_meters=depth_meters,
                intrinsics=intrinsics,
                pose=pose,
                # No source that goes through an estimator knows where the ground is. The scene
                # fits a plane for these.
                ground_plane=None,
                gaze_pixel=_gaze_in_depth_pixels(
                    rgb_frame.gaze_pixel,
                    scene_shape=rgb_frame.image_rgb.shape[:2],
                    depth_shape=depth_meters.shape,
                ),
                timing=timing,
            )

    def close(self) -> None:
        self._rgb_source.close()

    def _intrinsics_for(self, rgb_frame: RgbFrame, estimate: DepthEstimate, depth_shape: tuple[int, int]) -> np.ndarray:
        """The camera's own calibration first, then the model's, then a guessed field of view."""
        if rgb_frame.camera_matrix is not None:
            # The calibration describes the image the estimator was handed, so it scales by the
            # same ratio the depth image does.
            return scale_intrinsics(rgb_frame.camera_matrix, rgb_frame.image_rgb.shape[:2], depth_shape)
        if estimate.intrinsics is not None:
            return estimate.intrinsics

        if not self._warned_about_fallback_intrinsics:
            log.warning(
                "no camera calibration and the model returned no intrinsics, assuming a %.0f degree half field of view",
                self._config.fallback_half_field_of_view_degrees,
            )
            self._warned_about_fallback_intrinsics = True
        return fallback_intrinsics(*depth_shape, self._config.fallback_half_field_of_view_degrees)


def to_meters(depth: np.ndarray, intrinsics: np.ndarray, canonical_focal_pixels: float | None) -> np.ndarray:
    """
    Convert depth estimated for a canonical focal length into meters for this camera.

    Same rule as Depth Anything 3's own apply_metric_scaling: the mean of the two focal lengths,
    in pixels at the depth image's resolution, over the canonical one. Skipping it put the glasses'
    floor 2.9 m below a walker whose eyes were 1.5 m up, and the floor check refused it.

    :param depth: Depth as the estimator returned it, NaN where dropped.
    :param intrinsics: Camera matrix at the depth image's resolution.
    :param canonical_focal_pixels: The focal the depth assumes, or None when it is already meters.
    :return: Depth in meters. The input itself when there was nothing to convert.
    :rtype: np.ndarray
    """
    if canonical_focal_pixels is None:
        return depth
    focal_pixels = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
    return (depth * np.float32(focal_pixels / canonical_focal_pixels)).astype(np.float32)


def _gaze_in_depth_pixels(
    gaze_pixel: np.ndarray | None,
    scene_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> np.ndarray | None:
    """Rescale a gaze point from scene camera pixels to depth image pixels.

    The model works at its own process resolution, so the depth image is rarely the same size as
    the frame it came from. Skipping this puts the gaze goal in the wrong place by the ratio
    between them, which looks like a planner that steers slightly off what the wearer is watching.
    """
    if gaze_pixel is None:
        return None

    scene_height, scene_width = scene_shape
    depth_height, depth_width = depth_shape
    if scene_height == 0 or scene_width == 0:
        return None

    return np.array(
        [
            gaze_pixel[0] * (depth_width / scene_width),
            gaze_pixel[1] * (depth_height / scene_height),
        ]
    )
