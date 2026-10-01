"""
Monocular depth estimation, and the only file in the package allowed to import torch.

Everything else works on the DepthEstimate this produces, so a different estimator is a change
here and nowhere else. A guard test enforces that the torch import does not spread.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from typing import Protocol

# Third party imports
import numpy as np

# Local package imports
from nav.sources.config import EstimatorConfig

log = logging.getLogger(__name__)

# The old script's fallback when the model returns no intrinsics: roughly a 100 degree horizontal
# field of view, which is about what the Neon's scene camera has.
FALLBACK_HALF_FIELD_OF_VIEW_DEGREES = 50.0


@dataclass(frozen=True)
class DepthEstimate:
    """One frame's depth, the camera it implies, and how sure the model was."""

    depth_meters: np.ndarray
    intrinsics: np.ndarray
    confidence: np.ndarray | None


class DepthEstimatorProtocol(Protocol):
    """What the composed source needs from an estimator.

    Narrow on purpose, so the test stub satisfies it without inheriting anything and without the
    test suite ever importing torch.
    """

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate: ...


def fallback_intrinsics(height: int, width: int) -> np.ndarray:
    """
    Build a pinhole camera matrix from an assumed field of view.

    Used only when the model returns no intrinsics of its own. The depth is still metric, but the
    unprojected points will be wrong in x and y by whatever the real field of view differs by.

    :param height: Depth image height in pixels.
    :param width: Depth image width in pixels.
    :return: A (3, 3) camera matrix.
    :rtype: np.ndarray
    """
    focal_length = (width / 2.0) / np.tan(np.radians(FALLBACK_HALF_FIELD_OF_VIEW_DEGREES))
    return np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def apply_confidence_filter(estimate: DepthEstimate, drop_percentile: float) -> np.ndarray:
    """
    Return the depth image with the least confident pixels replaced by NaN.

    NaN is what the scene already treats as invalid, so dropping a pixel here needs no second
    channel travelling alongside the depth.

    :param estimate: The estimate to filter. Unchanged, a copy is returned.
    :param drop_percentile: Percentage of least confident pixels to drop, 0 to below 100.
    :return: A copy of depth_meters with dropped pixels set to NaN.
    :rtype: np.ndarray
    """
    if estimate.confidence is None or drop_percentile <= 0.0:
        return estimate.depth_meters

    # The old file took this percentile over the strided subsample, because it filtered inside the
    # unprojection loop. Here it runs over the whole map, before the scene ever strides. On a
    # megapixel frame the two agree closely, and the whole map is the more defensible of the two.
    usable = np.isfinite(estimate.confidence)
    if not np.any(usable):
        log.warning("confidence map has no finite values, dropping nothing")
        return estimate.depth_meters

    threshold = np.percentile(estimate.confidence[usable], drop_percentile)
    filtered = estimate.depth_meters.copy()
    # Keeps >= threshold, matching the old file's `ok &= c >= np.percentile(...)`.
    filtered[~usable | (estimate.confidence < threshold)] = np.nan
    return filtered


class DepthEstimator:
    """Depth Anything 3, loaded once and run per frame."""

    def __init__(self, config: EstimatorConfig) -> None:
        # Heavy C extension. Deferred so that importing nav.sources costs nothing for a run that
        # never touches an RGB camera, and so the shared layers can be proved to import without it.
        import os

        # The one environment write in the package, and it is a write rather than a read. The
        # accelerated downloader is not always present and falls over on partial downloads.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        import torch

        from depth_anything_3.api import DepthAnything3

        self._torch = torch
        self._config = config
        self._warned_about_fallback_intrinsics = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading %s on %s", config.model_name, device)
        self._model = DepthAnything3.from_pretrained(config.model_name).to(device)
        log.info("loaded %s", config.model_name)

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate:
        """
        Run the model on one frame.

        :param image_rgb: (H, W, 3) uint8 RGB image.
        :return: Metric depth, intrinsics at the depth resolution, and confidence when offered.
        :rtype: DepthEstimate
        """
        with self._torch.inference_mode():
            prediction = self._model.inference([image_rgb], process_res=self._config.process_resolution)

        depth_meters = np.asarray(prediction.depth[0], dtype=np.float32)
        confidence = None if prediction.conf is None else np.asarray(prediction.conf[0], dtype=np.float32)

        if prediction.intrinsics is not None:
            intrinsics = np.asarray(prediction.intrinsics[0], dtype=np.float64)
        else:
            intrinsics = fallback_intrinsics(*depth_meters.shape)
            if not self._warned_about_fallback_intrinsics:
                log.warning(
                    "%s returned no intrinsics, assuming a %.0f degree half field of view",
                    self._config.model_name,
                    FALLBACK_HALF_FIELD_OF_VIEW_DEGREES,
                )
                self._warned_about_fallback_intrinsics = True

        return DepthEstimate(depth_meters=depth_meters, intrinsics=intrinsics, confidence=confidence)
