"""
Monocular depth estimation, and the only file in the package allowed to import torch.

Everything else works on the DepthEstimate this produces, so a different estimator is a change
here and nowhere else. A guard test enforces that the torch import does not spread.
"""

# Standard library imports
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

# Third party imports
import numpy as np

# Local package imports
from nav.sources.config import DepthCheckpoint, DepthScale, EstimatorConfig

log = logging.getLogger(__name__)

# The divisor in the library's own apply_metric_scaling, and in the metric model's usage notes.
METRIC_MODEL_CANONICAL_FOCAL_PIXELS = 300.0


@dataclass(frozen=True)
class DepthEstimate:
    """One frame's depth, the camera it implies, and how sure the model was."""

    # Meters when canonical_focal_pixels is None. Otherwise meters for a camera with that focal
    # length, and the composed source converts once it has picked the real one.
    depth: np.ndarray
    # None when the model offers none, which the metric model does for a plain image. Choosing what
    # to use instead belongs to the composed source, because only it knows whether the camera
    # brought a calibration of its own.
    intrinsics: np.ndarray | None
    confidence: np.ndarray | None
    canonical_focal_pixels: float | None = None


def canonical_focal_for(checkpoint: DepthCheckpoint) -> float | None:
    """
    The focal length a checkpoint's depth assumes, or None when its depth is already meters.

    Depth Anything 3's metric model answers as if every camera had a 300 px focal at the resolution
    it ran at. Meters are output times the real focal over 300. The library only converts inside
    its nested model, so a standalone metric checkpoint hands back the raw value.
    Measured on one glasses frame: 1.91, 2.91 and 3.55 m raw at 504, 336 and 280 px, 1.51 to 1.56 m converted.

    :param checkpoint: The checkpoint the estimator runs.
    :return: 300 for the standalone metric checkpoint, None for one that already gives meters.
    :rtype: float | None
    :raises ValueError: For a relative checkpoint, whose depth no focal turns into meters.
    """
    match checkpoint.depth_scale:
        case DepthScale.METERS_AT_CANONICAL_FOCAL:
            return METRIC_MODEL_CANONICAL_FOCAL_PIXELS
        case DepthScale.METERS:
            return None
        case DepthScale.RELATIVE:
            # EstimatorConfig already refuses these, so reaching here means a config was bypassed.
            raise ValueError(f"{checkpoint.value} gives relative depth, and no focal turns that into meters")


class DepthEstimatorProtocol(Protocol):
    """What the composed source needs from an estimator.

    Narrow on purpose, so the test stub satisfies it without inheriting anything and without the
    test suite ever importing torch.
    """

    # "cuda" or "cpu", where the model runs. Read once, to warn about a live run on the CPU.
    device: str

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate: ...

    def warm_up(self) -> None:
        """Run once before the first real frame, so start-up cost does not land on it."""
        ...


def fallback_intrinsics(height: int, width: int, half_field_of_view_degrees: float = 50.0) -> np.ndarray:
    """
    Build a pinhole camera matrix from an assumed field of view.

    Used only when the camera brings no calibration and the model returns no intrinsics.
    The metric model's depth is converted with this focal, which cancels it out of x and y.
    So a wrong field of view scales only the distance along the camera's axis, by assumed over real focal.
    A level camera's height lies in y and comes out right. A camera pitched down mixes that axis
    into its height, so the floor reads too low or too high and leans. At 40 degrees down, assuming
    100 degrees for a real 75 reads a 1.5 m camera at 1.19 m, with the floor leaning 12.5 degrees.

    :param height: Depth image height in pixels.
    :param width: Depth image width in pixels.
    :param half_field_of_view_degrees: Half the horizontal field of view to assume.
    :return: A (3, 3) camera matrix.
    :rtype: np.ndarray
    """
    focal_length = (width / 2.0) / np.tan(np.radians(half_field_of_view_degrees))
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

    :param estimate: The estimate to filter. Never written to.
    :param drop_percentile: Percentage of least confident pixels to drop, 0 to below 100.
    :return: A copy of the depth with dropped pixels set to NaN. When nothing is dropped, because
        the percentile is 0 or the model gave no usable confidence, the estimate's own depth array.
    :rtype: np.ndarray
    """
    if estimate.confidence is None or drop_percentile <= 0.0:
        return estimate.depth

    # The old file took this percentile over the strided subsample, because it filtered inside the
    # unprojection loop. Here it runs over the whole map, before the scene ever strides. On a
    # megapixel frame the two agree closely, and the whole map is the more defensible of the two.
    usable = np.isfinite(estimate.confidence)
    if not np.any(usable):
        log.warning("confidence map has no finite values, dropping nothing")
        return estimate.depth

    threshold = np.percentile(estimate.confidence[usable], drop_percentile)
    filtered = estimate.depth.copy()
    # Keeps >= threshold, matching the old file's `ok &= c >= np.percentile(...)`.
    filtered[~usable | (estimate.confidence < threshold)] = np.nan
    return filtered


class DepthEstimator:
    """Depth Anything 3, loaded once and run per frame."""

    def __init__(self, config: EstimatorConfig) -> None:
        # The one environment write in the package, and it is a write rather than a read. The
        # accelerated downloader is not always present and falls over on partial downloads. It has
        # to happen before the Hugging Face client is imported, which the model import does.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        # Heavy C extensions. Deferred so that importing nav.sources costs nothing for a run that
        # never touches an RGB camera, and so the shared layers can be proved to import without them.
        import torch

        from depth_anything_3.api import DepthAnything3

        self._torch = torch
        self._config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading %s on %s", config.model_name.value, self.device)
        self._model = DepthAnything3.from_pretrained(config.model_name.value).to(self.device)
        self._preprocess_on_the_calling_thread()
        log.info("loaded %s", config.model_name.value)

    def _preprocess_on_the_calling_thread(self) -> None:
        """
        Stop Depth Anything 3 from preprocessing each frame on a fresh pool of 8 threads.

        It builds that pool on every call, even for one image, and every pool leaves about 7 native
        threads behind. Measured: 48 threads after loading, 259 after 30 calls. On a live walk that
        is 20 a second, and by the time the run had 3000 the laptop was spending four cores on them
        and the glasses' video decoder fell seconds behind.
        """
        processor = getattr(self._model, "input_processor", None)
        if not callable(processor):
            # A later Depth Anything 3 that renamed it. The estimator still works, it just leaks again.
            log.warning("Depth Anything 3 has no input_processor to make sequential, so each frame may leave threads behind")
            return
        self._model.input_processor = functools.partial(processor, sequential=True)

    def warm_up(self) -> None:
        """
        Run one inference on a blank image before any real frame.

        The first inference on CUDA also creates the context and selects kernels. Without this,
        that start-up time lands on the walker's first frame and on the first latency figure.
        """
        side = self._config.process_resolution
        started = time.perf_counter()
        self.estimate(np.zeros((side, side, 3), dtype=np.uint8))
        log.info("warmed up %s on %s in %.1f s", self._config.model_name.value, self.device, time.perf_counter() - started)

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate:
        """
        Run the model on one frame.

        :param image_rgb: (H, W, 3) uint8 RGB image.
        :return: Depth at the canonical focal the checkpoint assumes, the model's intrinsics at the
            depth resolution when it offers them, and confidence when offered.
        :rtype: DepthEstimate
        """
        with self._torch.inference_mode():
            prediction = self._model.inference([image_rgb], process_res=self._config.process_resolution)

        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        confidence = None if prediction.conf is None else np.asarray(prediction.conf[0], dtype=np.float32)
        intrinsics = None if prediction.intrinsics is None else np.asarray(prediction.intrinsics[0], dtype=np.float64)

        return DepthEstimate(
            depth=depth,
            intrinsics=intrinsics,
            confidence=confidence,
            canonical_focal_pixels=canonical_focal_for(self._config.model_name),
        )
