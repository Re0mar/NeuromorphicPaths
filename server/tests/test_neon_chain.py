"""
Drives the whole Neon path in one chain: a fake device, the real source, the real composition with
the estimator, and the real scene.

The mount, the calibration and the timing are each tested on their own elsewhere. This is the test
that catches them disagreeing with each other, which separately tested halves never do. The only
stand-ins are the device and the estimator, and the estimator hands back the exact depth a level
pair of glasses would see over a flat floor.
"""

# Standard library imports
import time
from dataclasses import dataclass

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.pipeline import ScenePipeline
from nav.sources.config import EstimatorConfig, NeonConfig
from nav.sources.estimated_depth import EstimatedDepthSource
from nav.sources.estimator import DepthEstimate
from nav.sources.neon_live import NEON_SCENE_SIZE, NeonLiveRgbSource
from nav.types import FloorSource
from nav.walker import WalkerConfig
from synthetic_depth import CAMERA_HEIGHT_METERS, HEIGHT, WIDTH, clean_scene
from test_neon_live import FakeCalibration, FakeDevice, FakeDeviceError, FakeImuDatum, FakeMatched, FakeQuaternion, FakeScene

# Level glasses: the scene camera looks 12 degrees down, which is what the documented mount says.
LEVEL_GLASSES_SCENE = clean_scene(box_lateral_meters=None, pitch_degrees=12.0)
# The device's calibration at its native size, chosen so that scaled to the fake frame's size it
# is exactly the synthetic scene's camera. No distortion, so undistortion leaves it alone.
NATIVE_SCALE = NEON_SCENE_SIZE[1] / WIDTH
NATIVE_CAMERA_MATRIX = LEVEL_GLASSES_SCENE.intrinsics * np.array([[NATIVE_SCALE], [NATIVE_SCALE], [1.0]])


@dataclass
class LevelFloorEstimator:
    """Returns the synthetic floor's depth, and no intrinsics of its own, as the metric model does."""

    device: str = "cuda"

    def warm_up(self) -> None:
        """Nothing to warm."""

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate:
        return DepthEstimate(depth=LEVEL_GLASSES_SCENE.depth_meters.copy(), intrinsics=None, confidence=None)


def test_a_fake_neon_frame_reaches_the_scene_with_gravity_up_device_intrinsics_and_timing() -> None:
    captured_on_the_neon = time.time() - 0.3
    device = FakeDevice(
        matched=[FakeMatched(FakeScene(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), captured_on_the_neon), None)],
        imu=[FakeImuDatum(FakeQuaternion(w=1.0, x=0.0, y=0.0, z=0.0))],
        calibration=FakeCalibration(NATIVE_CAMERA_MATRIX, np.zeros(8)),
        offsets_ms=[0.0, 0.0],
    )
    camera = NeonLiveRgbSource(NeonConfig())
    camera._device = device
    camera._device_failures = (FakeDeviceError, OSError, ValueError)
    source = EstimatedDepthSource(camera, LevelFloorEstimator(), EstimatorConfig())
    scene = ScenePipeline(SceneConfig(), WalkerConfig())

    try:
        frame = next(iter(source.frames()))
        scene.process(frame)
    finally:
        source.close()

    # Gravity from the mounted IMU, so the floor under level glasses passes at the default tilt.
    assert frame.pose.orientation_is_gravity_aligned is True
    assert scene.last_floor_source is FloorSource.FITTED
    assert scene.previous_plane.offset_meters == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.05)
    # The device's own calibration, scaled to the depth image, not the estimator's fallback guess.
    assert frame.intrinsics == pytest.approx(LEVEL_GLASSES_SCENE.intrinsics, abs=1e-6)
    # And the frame's trip, in the order it happened.
    timing = frame.timing
    assert timing.capture_seconds == pytest.approx(captured_on_the_neon)
    assert timing.capture_seconds <= timing.arrival_seconds <= timing.depth_ready_seconds
