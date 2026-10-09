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

# Local package imports
from nav.types import FrameTiming, Pose


@dataclass(frozen=True)
class RgbFrame:
    """One camera frame, plus whatever else the device knew at the same moment."""

    timestamp_seconds: float
    image_rgb: np.ndarray
    gaze_pixel: np.ndarray | None
    # Built by the source, which is the only thing that knows how its IMU sits on its camera.
    # None means the device knows nothing about orientation.
    pose: Pose | None
    # (3, 3) for image_rgb as delivered, after any undistortion. None means the camera has no
    # calibration and the intrinsics come from the model or a guess.
    camera_matrix: np.ndarray | None
    # Capture and arrival on the laptop clock. depth_ready_seconds is still None at this point,
    # because no depth exists yet.
    timing: FrameTiming | None
    # Which unbroken stretch of video this frame belongs to. A source that can switch, or start a
    # recording over, counts up, and the loop clears what the scene and planner remember when it
    # changes. Every other source leaves it at 0.
    source_generation: int = 0

    def __post_init__(self) -> None:
        # Same reasoning as DepthFrame. Two unrelated sources build these, and a wrongly shaped
        # array from either surfaces much later as a broadcasting error inside the estimator.
        if self.image_rgb.ndim != 3 or self.image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must be (height, width, 3), got shape {self.image_rgb.shape}")
        if self.gaze_pixel is not None and self.gaze_pixel.shape != (2,):
            raise ValueError(f"gaze_pixel must be (2,) when present, got shape {self.gaze_pixel.shape}")
        if self.pose is not None and not isinstance(self.pose, Pose):
            raise ValueError(f"pose must be a Pose when present, got {type(self.pose).__name__}")
        if self.camera_matrix is not None:
            if self.camera_matrix.shape != (3, 3):
                raise ValueError(f"camera_matrix must be (3, 3) when present, got shape {self.camera_matrix.shape}")
            if not np.all(np.isfinite(self.camera_matrix)):
                raise ValueError("camera_matrix must be finite when present")


class RgbSource(Protocol):
    """What every camera is, seen from the estimator."""

    def frames(self) -> Iterator[RgbFrame]: ...

    def close(self) -> None: ...
