"""
Covers the composed source: a camera plus an estimator becoming a source of depth frames.

The real estimator never appears here. It needs 1.3 GB of weights and minutes on CPU, and the
point of the stub is that the whole suite runs in an environment with no torch in it at all.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from conftest import SYNTHETIC_VIDEO_FRAME_COUNT
from nav.sources.config import EstimatorConfig
from nav.sources.estimated_depth import EstimatedDepthSource
from nav.sources.estimator import DepthEstimate, apply_confidence_filter
from nav.sources.rgb import RgbFrame
from nav.sources.video_file import VideoFileRgbSource
from stubs import StubDepthEstimator, WrongShapeDepthEstimator


class ListRgbSource:
    """An RGB source over frames the test built, for cases a real video cannot express."""

    def __init__(self, frames: list[RgbFrame]) -> None:
        self._frames = frames
        self.closed = False

    def frames(self):
        yield from self._frames

    def close(self) -> None:
        self.closed = True


def _rgb_frame(**overrides) -> RgbFrame:
    fields = {
        "timestamp_seconds": 0.0,
        "image_rgb": np.zeros((48, 64, 3), dtype=np.uint8),
        "gaze_pixel": None,
        "imu_orientation_wxyz": None,
    }
    fields.update(overrides)
    return RgbFrame(**fields)


def test_every_camera_frame_becomes_a_depth_frame(synthetic_video: Path) -> None:
    stub = StubDepthEstimator()
    source = EstimatedDepthSource(
        VideoFileRgbSource(str(synthetic_video)),
        stub,
        EstimatorConfig(),
    )
    try:
        frames = list(source.frames())
    finally:
        source.close()

    assert len(frames) == SYNTHETIC_VIDEO_FRAME_COUNT
    first = frames[0]
    assert first.depth_meters.shape == (stub.height, stub.width)
    # A video has no orientation, so the scene must rebuild in the body frame every frame.
    assert first.pose.has_position is False
    # And no source behind an estimator knows where the ground is. The scene fits it.
    assert first.ground_plane is None


def test_an_imu_quaternion_becomes_the_frame_pose() -> None:
    rgb_source = ListRgbSource([_rgb_frame(imu_orientation_wxyz=np.array([0.0, 0.0, 0.0, 2.0]))])
    source = EstimatedDepthSource(rgb_source, StubDepthEstimator(), EstimatorConfig())

    frame = next(iter(source.frames()))

    assert np.linalg.norm(frame.pose.orientation) == pytest.approx(1.0)
    assert frame.pose.has_position is False


def test_gaze_is_rescaled_from_scene_pixels_to_depth_pixels() -> None:
    # The model works at its own resolution, so the depth image is rarely the size of the frame it
    # came from. Skipping the rescale steers slightly off whatever the wearer is actually watching.
    stub = StubDepthEstimator(height=24, width=32)
    rgb_source = ListRgbSource([_rgb_frame(gaze_pixel=np.array([32.0, 24.0]))])
    source = EstimatedDepthSource(rgb_source, stub, EstimatorConfig())

    frame = next(iter(source.frames()))

    # Scene is 64 by 48, depth is 32 by 24, so the centre of one is the centre of the other.
    assert frame.gaze_pixel == pytest.approx(np.array([16.0, 12.0]))


def test_no_gaze_stays_no_gaze() -> None:
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame()]), StubDepthEstimator(), EstimatorConfig())

    assert next(iter(source.frames())).gaze_pixel is None


def test_closing_closes_the_camera_underneath() -> None:
    rgb_source = ListRgbSource([])
    EstimatedDepthSource(rgb_source, StubDepthEstimator(), EstimatorConfig()).close()

    assert rgb_source.closed is True


def test_an_estimator_returning_the_wrong_shape_is_refused() -> None:
    source = EstimatedDepthSource(
        ListRgbSource([_rgb_frame()]),
        WrongShapeDepthEstimator(),
        EstimatorConfig(),
    )

    # DepthFrame's own validation is what catches this, one layer before the scene would have hit
    # it as a broadcasting error inside unprojection.
    with pytest.raises(ValueError, match="depth_meters"):
        next(iter(source.frames()))


def test_confidence_filter_drops_exactly_the_requested_percentile() -> None:
    confidence = np.arange(100, dtype=np.float32).reshape(10, 10)
    estimate = DepthEstimate(
        depth_meters=np.ones((10, 10), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=confidence,
    )

    filtered = apply_confidence_filter(estimate, drop_percentile=30.0)

    assert np.count_nonzero(np.isnan(filtered)) == 30
    # The dropped ones are the least confident, not an arbitrary 30.
    assert np.all(np.isnan(filtered[confidence < 30]))
    assert not np.any(np.isnan(filtered[confidence >= 30]))


def test_confidence_filter_leaves_depth_alone_when_asked_for_nothing() -> None:
    estimate = DepthEstimate(
        depth_meters=np.ones((10, 10), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=np.arange(100, dtype=np.float32).reshape(10, 10),
    )

    assert not np.any(np.isnan(apply_confidence_filter(estimate, drop_percentile=0.0)))


def test_confidence_filter_leaves_depth_alone_when_the_model_offers_no_confidence() -> None:
    estimate = DepthEstimate(
        depth_meters=np.ones((10, 10), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=None,
    )

    assert not np.any(np.isnan(apply_confidence_filter(estimate, drop_percentile=30.0)))


def test_confidence_filter_survives_a_map_with_no_finite_values() -> None:
    # A model that returned all-NaN confidence would otherwise make np.percentile return NaN and
    # the comparison would silently drop nothing, which looks identical to a working filter.
    estimate = DepthEstimate(
        depth_meters=np.ones((4, 4), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=np.full((4, 4), np.nan, dtype=np.float32),
    )

    assert not np.any(np.isnan(apply_confidence_filter(estimate, drop_percentile=30.0)))
