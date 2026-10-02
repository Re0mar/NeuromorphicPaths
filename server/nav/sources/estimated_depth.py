"""
An RGB source plus an estimator, composed into a source of depth frames.

The estimator is composed in here rather than run as a stage of its own, because a DepthFrame
without depth is not a frame. If the estimator were a stage, the scene's input type would depend
on which source was plugged in, which is the thing the layering exists to prevent.
"""

# Standard library imports
import logging
from collections.abc import Iterator

# Third party imports
import numpy as np

# Local package imports
from nav.pose.imu_orientation import identity_pose, pose_from_imu
from nav.sources.config import EstimatorConfig
from nav.sources.estimator import DepthEstimatorProtocol, apply_confidence_filter
from nav.sources.rgb import RgbSource
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

    def frames(self) -> Iterator[DepthFrame]:
        for rgb_frame in self._rgb_source.frames():
            estimate = self._estimator.estimate(rgb_frame.image_rgb)
            depth_meters = apply_confidence_filter(estimate, self._config.confidence_drop_percentile)

            if rgb_frame.imu_orientation_wxyz is not None:
                pose = pose_from_imu(rgb_frame.imu_orientation_wxyz)
            else:
                pose = identity_pose()

            yield DepthFrame(
                timestamp_seconds=rgb_frame.timestamp_seconds,
                depth_meters=depth_meters,
                intrinsics=estimate.intrinsics,
                pose=pose,
                # No source that goes through an estimator knows where the ground is. The scene
                # fits a plane for these.
                ground_plane=None,
                gaze_pixel=_gaze_in_depth_pixels(
                    rgb_frame.gaze_pixel,
                    scene_shape=rgb_frame.image_rgb.shape[:2],
                    depth_shape=depth_meters.shape,
                ),
            )

    def close(self) -> None:
        self._rgb_source.close()


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
