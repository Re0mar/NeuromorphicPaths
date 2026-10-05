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
from nav.clock import laptop_time_seconds
from nav.pose.imu_orientation import pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.config import EstimatorConfig
from nav.sources.estimated_depth import EstimatedDepthSource, to_meters
from nav.sources.estimator import (
    METRIC_MODEL_CANONICAL_FOCAL_PIXELS,
    DepthEstimate,
    apply_confidence_filter,
    canonical_focal_for,
    fallback_intrinsics,
)
from nav.sources.rgb import RgbFrame
from nav.sources.video_file import VideoFileRgbSource
from nav.types import FrameTiming
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
        "pose": None,
        "camera_matrix": None,
        "timing": None,
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


def test_the_cameras_own_pose_becomes_the_frame_pose() -> None:
    camera_pose = pose_from_imu(np.array([0.0, 0.0, 0.0, 2.0]), NEON_IMU_MOUNT)
    rgb_source = ListRgbSource([_rgb_frame(pose=camera_pose)])
    source = EstimatedDepthSource(rgb_source, StubDepthEstimator(), EstimatorConfig())

    frame = next(iter(source.frames()))

    assert frame.pose is camera_pose
    assert frame.pose.orientation_is_gravity_aligned is True


def test_a_frame_without_a_pose_gets_identity_and_does_not_claim_gravity() -> None:
    # A camera with no IMU reading yet must not tell the scene it knows where up is.
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame()]), StubDepthEstimator(), EstimatorConfig())

    frame = next(iter(source.frames()))

    assert frame.pose.orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert frame.pose.orientation_is_gravity_aligned is False


def test_an_rgb_frame_refuses_a_pose_that_is_not_a_pose() -> None:
    with pytest.raises(ValueError, match="pose must be a Pose"):
        _rgb_frame(pose=np.array([1.0, 0.0, 0.0, 0.0]))


def test_a_camera_matrix_from_the_source_overrides_the_estimators_and_is_scaled_to_the_depth_size() -> None:
    # The frame is 64 by 48 and the stub's depth is 32 by 24, so the camera's matrix halves.
    camera_matrix = np.array([[50.0, 0.0, 30.0], [0.0, 50.0, 22.0], [0.0, 0.0, 1.0]])
    stub = StubDepthEstimator(height=24, width=32)
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame(camera_matrix=camera_matrix)]), stub, EstimatorConfig())

    frame = next(iter(source.frames()))

    assert frame.intrinsics == pytest.approx(np.array([[25.0, 0.0, 15.0], [0.0, 25.0, 11.0], [0.0, 0.0, 1.0]]))


def test_without_a_source_matrix_the_estimators_intrinsics_are_used() -> None:
    stub = StubDepthEstimator(height=24, width=32)
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame()]), stub, EstimatorConfig())

    frame = next(iter(source.frames()))

    assert frame.intrinsics == pytest.approx(stub.estimate(np.zeros((1, 1, 3))).intrinsics)


def test_only_the_fallback_path_warns_that_the_field_of_view_was_assumed(caplog: pytest.LogCaptureFixture) -> None:
    config = EstimatorConfig(fallback_half_field_of_view_degrees=40.0)
    with_intrinsics = EstimatedDepthSource(ListRgbSource([_rgb_frame()]), StubDepthEstimator(), config)
    without_any = EstimatedDepthSource(
        ListRgbSource([_rgb_frame(), _rgb_frame()]),
        StubDepthEstimator(returns_intrinsics=False),
        config,
    )

    with caplog.at_level("WARNING", logger="nav.sources.estimated_depth"):
        list(with_intrinsics.frames())
        assert not [record for record in caplog.records if "assuming" in record.message]
        frames = list(without_any.frames())

    assumed = [record for record in caplog.records if "assuming" in record.message]
    assert len(assumed) == 1, "warned once, not once per frame"
    stub = StubDepthEstimator()
    assert frames[0].intrinsics == pytest.approx(fallback_intrinsics(stub.height, stub.width, 40.0))


@pytest.mark.parametrize("camera_matrix", [np.eye(2), np.full((3, 3), np.nan)])
def test_an_rgb_frame_with_a_misshapen_camera_matrix_is_refused(camera_matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="camera_matrix"):
        _rgb_frame(camera_matrix=camera_matrix)


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


def test_fallback_intrinsics_put_the_principal_point_at_the_centre_with_the_focal_from_the_field_of_view() -> None:
    # Ninety degrees of horizontal field of view means the image edge is at 45 degrees, so the
    # focal length in pixels equals half the width. This is the camera every stub frame and every
    # estimator frame without model intrinsics is unprojected through.
    camera = fallback_intrinsics(480, 640, half_field_of_view_degrees=45.0)

    assert camera[0, 0] == pytest.approx(320.0)
    assert camera[1, 1] == pytest.approx(320.0)
    assert camera[0, 2] == pytest.approx(320.0)
    assert camera[1, 2] == pytest.approx(240.0)
    assert camera[2] == pytest.approx([0.0, 0.0, 1.0])
    # A narrower view is a longer focal length, which is the whole reason the flag exists.
    assert fallback_intrinsics(480, 640, half_field_of_view_degrees=37.5)[0, 0] > camera[0, 0]


def test_confidence_filter_drops_exactly_the_requested_percentile() -> None:
    confidence = np.arange(100, dtype=np.float32).reshape(10, 10)
    estimate = DepthEstimate(
        depth=np.ones((10, 10), dtype=np.float32),
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
        depth=np.ones((10, 10), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=np.arange(100, dtype=np.float32).reshape(10, 10),
    )

    assert not np.any(np.isnan(apply_confidence_filter(estimate, drop_percentile=0.0)))


def test_confidence_filter_leaves_depth_alone_when_the_model_offers_no_confidence() -> None:
    estimate = DepthEstimate(
        depth=np.ones((10, 10), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=None,
    )

    assert not np.any(np.isnan(apply_confidence_filter(estimate, drop_percentile=30.0)))


def test_confidence_filter_survives_a_map_with_no_finite_values() -> None:
    # A model that returned all-NaN confidence would otherwise make np.percentile return NaN and
    # the comparison would silently drop nothing, which looks identical to a working filter.
    estimate = DepthEstimate(
        depth=np.ones((4, 4), dtype=np.float32),
        intrinsics=np.eye(3),
        confidence=np.full((4, 4), np.nan, dtype=np.float32),
    )

    assert not np.any(np.isnan(apply_confidence_filter(estimate, drop_percentile=30.0)))


def test_depth_ready_is_stamped_after_estimation_and_after_arrival() -> None:
    arrived = FrameTiming(capture_seconds=None, arrival_seconds=laptop_time_seconds(), depth_ready_seconds=None)
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame(timing=arrived)]), StubDepthEstimator(), EstimatorConfig())

    frame = next(iter(source.frames()))

    assert frame.timing.arrival_seconds == arrived.arrival_seconds
    assert frame.timing.depth_ready_seconds >= arrived.arrival_seconds


def test_a_frame_without_timing_stays_without_timing() -> None:
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame()]), StubDepthEstimator(), EstimatorConfig())

    assert next(iter(source.frames())).timing is None


def test_the_estimator_is_warmed_once_before_the_first_frame() -> None:
    stub = StubDepthEstimator()
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame(), _rgb_frame()]), stub, EstimatorConfig())

    list(source.frames())

    assert stub.calls == ["warm_up", "estimate", "estimate"]


class CanonicalDepthEstimator:
    """Answers like the metric checkpoint: one raw value everywhere, for a 300 px focal."""

    device = "cuda"

    def __init__(self, raw_depth: float, height: int, width: int) -> None:
        self._raw_depth = raw_depth
        self._shape = (height, width)

    def warm_up(self) -> None:
        """Nothing to warm."""

    def estimate(self, image_rgb: np.ndarray) -> DepthEstimate:
        return DepthEstimate(
            depth=np.full(self._shape, self._raw_depth, dtype=np.float32),
            intrinsics=None,
            confidence=None,
            canonical_focal_pixels=METRIC_MODEL_CANONICAL_FOCAL_PIXELS,
        )


def test_canonical_depth_is_converted_with_the_cameras_focal_at_the_depth_resolution() -> None:
    # Camera focal 600 px on a 64 wide frame is 150 px on the 16 wide depth image, half of 300.
    camera_matrix = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 24.0], [0.0, 0.0, 1.0]])
    estimator = CanonicalDepthEstimator(raw_depth=3.0, height=12, width=16)
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame(camera_matrix=camera_matrix)]), estimator, EstimatorConfig())

    frame = next(iter(source.frames()))

    assert frame.depth_meters == pytest.approx(np.full((12, 16), 1.5))


def test_canonical_depth_without_a_calibration_converts_with_the_fallback_focal() -> None:
    estimator = CanonicalDepthEstimator(raw_depth=3.0, height=12, width=16)
    config = EstimatorConfig()
    source = EstimatedDepthSource(ListRgbSource([_rgb_frame()]), estimator, config)

    frame = next(iter(source.frames()))

    fallback_focal = fallback_intrinsics(12, 16, config.fallback_half_field_of_view_degrees)[0, 0]
    assert frame.depth_meters == pytest.approx(np.full((12, 16), 3.0 * fallback_focal / METRIC_MODEL_CANONICAL_FOCAL_PIXELS))


def test_depth_already_in_meters_passes_through_unconverted() -> None:
    depth = np.array([[1.0, np.nan]], dtype=np.float32)

    assert to_meters(depth, np.diag([600.0, 600.0, 1.0]), canonical_focal_pixels=None) is depth


def test_conversion_averages_the_two_focal_lengths_and_keeps_dropped_pixels_dropped() -> None:
    converted = to_meters(np.array([[2.0, np.nan]], dtype=np.float32), np.diag([200.0, 400.0, 1.0]), canonical_focal_pixels=300.0)

    assert converted[0, 0] == pytest.approx(2.0)
    assert np.isnan(converted[0, 1])
    assert converted.dtype == np.float32


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("depth-anything/DA3METRIC-LARGE", METRIC_MODEL_CANONICAL_FOCAL_PIXELS),
        # The nested model converts inside the library, and the relative ones are not meters at all.
        ("depth-anything/DA3NESTED-GIANT-LARGE", None),
        ("depth-anything/DA3-LARGE", None),
    ],
)
def test_only_the_standalone_metric_checkpoint_needs_converting(model_name: str, expected: float | None) -> None:
    assert canonical_focal_for(model_name) == expected
