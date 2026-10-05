"""
Test doubles, kept out of nav so nothing shippable can reach for one.

The estimator stub is the reason the test suite never installs torch. It satisfies
DepthEstimatorProtocol structurally, so it needs no base class and no import from the real thing
beyond the DepthEstimate it returns.
"""

# Standard library imports
from dataclasses import dataclass, field

# Third party imports
import numpy as np

# Local package imports
from nav.sources.estimator import DepthEstimate, fallback_intrinsics


@dataclass(frozen=True)
class StubDepthEstimator:
    """Returns a flat floor seen by a level camera, optionally with one box standing on it.

    The floor is computed from the pinhole model rather than painted as a gradient, so a plane fit
    recovers a real plane from it and the height band means what it says. A stub that returned an
    arbitrary ramp would let a broken floor fit pass.

    One deviation from the real estimator, named here because it is where it is configured: this
    stub returns intrinsics by default. The metric model returns none for a plain video. Set
    returns_intrinsics False to behave like it, which is how the composed source's fallback path is
    tested.
    """

    height: int = 48
    width: int = 64
    camera_height_meters: float = 1.6
    box_distance_meters: float | None = 3.0
    box_rows: tuple[int, int] = (18, 30)
    box_columns: tuple[int, int] = (26, 38)
    confidence_value: float | None = 1.0
    returns_intrinsics: bool = True
    # The real estimator reports where it runs. A test asking about the CPU warning sets this.
    device: str = "cuda"
    # Mutable through a frozen dataclass on purpose: the one thing a test reads back from the stub.
    calls: list = field(default_factory=list, compare=False)

    def warm_up(self) -> None:
        self.calls.append("warm_up")

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate:
        self.calls.append("estimate")
        intrinsics = fallback_intrinsics(self.height, self.width)
        focal_y = intrinsics[1, 1]
        principal_y = intrinsics[1, 2]

        rows = np.arange(self.height, dtype=np.float64)[:, None]
        below_horizon = rows - principal_y

        # z = camera height * fy / (v - cy) for a level camera over a flat floor. Above the
        # horizon there is no floor to hit, so those pixels are NaN, which is also what the scene
        # treats as invalid.
        with np.errstate(divide="ignore", invalid="ignore"):
            floor_distance = np.where(
                below_horizon > 0,
                self.camera_height_meters * focal_y / below_horizon,
                np.nan,
            )

        depth_meters = np.broadcast_to(floor_distance, (self.height, self.width)).astype(np.float32).copy()

        if self.box_distance_meters is not None:
            row_start, row_end = self.box_rows
            column_start, column_end = self.box_columns
            depth_meters[row_start:row_end, column_start:column_end] = self.box_distance_meters

        confidence = None
        if self.confidence_value is not None:
            confidence = np.full((self.height, self.width), self.confidence_value, dtype=np.float32)

        return DepthEstimate(
            depth=depth_meters,
            intrinsics=intrinsics if self.returns_intrinsics else None,
            confidence=confidence,
        )


@dataclass(frozen=True)
class WrongShapeDepthEstimator:
    """Returns depth with the wrong number of dimensions, to prove DepthFrame refuses it."""

    device: str = "cuda"

    def warm_up(self) -> None:
        """Nothing to warm."""

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate:
        return DepthEstimate(
            depth=np.ones((4, 4, 4), dtype=np.float32),
            intrinsics=fallback_intrinsics(4, 4),
            confidence=None,
        )
