"""
The RGB boundary: what a camera hands over before any depth exists.

A source that already has depth, such as the Pixel over TCP, never produces one of these. It goes
straight to DepthFrame. This type exists so that a plain camera and the Neon can share the
estimator without the estimator knowing which one it is reading from.
"""

# Standard library imports
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

# Third party imports
import numpy as np


@dataclass(frozen=True)
class RgbFrame:
    """One camera frame, plus whatever else the device knew at the same moment."""

    timestamp_seconds: float
    image_rgb: np.ndarray
    gaze_pixel: np.ndarray | None
    imu_orientation_wxyz: np.ndarray | None

    def __post_init__(self) -> None:
        # Same reasoning as DepthFrame. Two unrelated sources build these, and a wrongly shaped
        # array from either surfaces much later as a broadcasting error inside the estimator.
        if self.image_rgb.ndim != 3 or self.image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must be (height, width, 3), got shape {self.image_rgb.shape}")
        if self.gaze_pixel is not None and self.gaze_pixel.shape != (2,):
            raise ValueError(f"gaze_pixel must be (2,) when present, got shape {self.gaze_pixel.shape}")
        if self.imu_orientation_wxyz is not None and self.imu_orientation_wxyz.shape != (4,):
            raise ValueError(
                f"imu_orientation_wxyz must be (4,) when present, got shape {self.imu_orientation_wxyz.shape}"
            )


class RgbSource(Protocol):
    """What every camera is, seen from the estimator."""

    def frames(self) -> Iterator[RgbFrame]: ...

    def close(self) -> None: ...
