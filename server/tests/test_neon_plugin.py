"""
Covers reading the Neon Player depth plugin's cache back as depth frames.

The recording reader is faked. A native Neon recording is binary and undocumented and the test
cannot build one. The plugin's cache array it can build exactly, down to the directory the player
puts it in, and that is the half this source actually parses.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.pose.imu_orientation import pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.config import NeonPluginConfig, NeonPluginModel
from nav.sources.neon_plugin import (
    PLUGIN_CACHE_RELATIVE_DIR,
    NeonPluginDepthFrameSource,
    xyzw_to_wxyz,
)

FRAME_COUNT = 6
SCENE_SIZE = (1200, 1600)  # height, width, the Neon scene camera
DEPTH_SIZE = (300, 400)  # a quarter of each, which is what the plugin caches
SCENE_CAMERA_MATRIX = np.array([[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]])
FIRST_TIME_NS = 1_700_000_000_000_000_000
FRAME_GAP_NS = 33_333_333


class FakeReader:
    """Stands in for the native recording. Returns whatever the test told it to."""

    def __init__(
        self,
        frame_count: int = FRAME_COUNT,
        quaternions_wxyz: np.ndarray | None = "default",
        gaze: np.ndarray | None = "default",
    ) -> None:
        self.times = FIRST_TIME_NS + np.arange(frame_count, dtype=np.int64) * FRAME_GAP_NS
        if isinstance(quaternions_wxyz, str):
            # A small pitch, unnormalized on purpose so the test can see normalization happen.
            quaternions_wxyz = np.tile(np.array([2.0, 0.2, 0.0, 0.0]), (frame_count, 1))
        self.quaternions = quaternions_wxyz
        if isinstance(gaze, str):
            gaze = np.tile(np.array([800.0, 600.0]), (frame_count, 1))
        self.gaze = gaze
        self.asked_tolerance_ns: int | None = None

    def scene_times_ns(self) -> np.ndarray:
        return self.times

    def scene_size(self) -> tuple[int, int]:
        return SCENE_SIZE

    def scene_camera_matrix(self) -> np.ndarray:
        return SCENE_CAMERA_MATRIX

    def imu_quaternions_wxyz_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None:
        self.asked_tolerance_ns = tolerance_ns
        return None if self.quaternions is None else self.quaternions[: len(times_ns)]

    def gaze_points_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None:
        return None if self.gaze is None else self.gaze[: len(times_ns)]


def _write_cache(recording_dir: Path, model: NeonPluginModel = NeonPluginModel.METRIC_LARGE, frame_count: int = FRAME_COUNT, shape=DEPTH_SIZE) -> np.ndarray:
    cache_dir = recording_dir / PLUGIN_CACHE_RELATIVE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Each frame a different constant, so a frame mix-up is visible.
    stack = np.zeros((frame_count, *shape), dtype=np.float32)
    for index in range(frame_count):
        stack[index] = 1.0 + index
    np.save(cache_dir / f"depth_values_{model.cache_stem}.npy", stack)
    return stack


def _source(recording_dir: Path, reader: FakeReader | None = None, **config_overrides) -> NeonPluginDepthFrameSource:
    reader = reader if reader is not None else FakeReader()
    config = NeonPluginConfig(recording_dir=str(recording_dir), **config_overrides)
    return NeonPluginDepthFrameSource(config, reader_factory=lambda path: reader)


def test_the_cache_path_is_where_neon_player_puts_it() -> None:
    # Read from neon-player's plugin base class: rec_dir / ".neon_player" / "cache" / class name.
    assert PLUGIN_CACHE_RELATIVE_DIR == Path(".neon_player") / "cache" / "DepthEstimationPlugin"
    assert NeonPluginModel.METRIC_LARGE.cache_stem == "DA3Metric-Large"


def test_every_cached_map_becomes_a_depth_frame(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    stack = _write_cache(recording)

    frames = list(_source(recording).frames())

    assert len(frames) == FRAME_COUNT
    for index, frame in enumerate(frames):
        assert frame.depth_meters == pytest.approx(stack[index])
        assert frame.depth_meters.dtype == np.float32
        assert frame.pose.has_position is False
        assert frame.ground_plane is None
    timestamps = [frame.timestamp_seconds for frame in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[1] - timestamps[0] == pytest.approx(FRAME_GAP_NS / 1e9, rel=1e-6)


def test_intrinsics_are_scaled_to_the_quarter_resolution_map(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording)

    frame = next(iter(_source(recording).frames()))

    # Scene is 1600 by 1200, the cache is 400 by 300, so everything scales by a quarter.
    assert frame.intrinsics == pytest.approx(SCENE_CAMERA_MATRIX * np.array([[0.25], [0.25], [1.0]]))


def test_imu_orientation_becomes_the_mounted_camera_pose(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording)

    frame = next(iter(_source(recording).frames()))

    # The recorded IMU is the same IMU the live stream reads, so it goes through the same mount.
    expected = pose_from_imu(np.array([2.0, 0.2, 0.0, 0.0]), NEON_IMU_MOUNT)
    assert frame.pose.orientation == pytest.approx(expected.orientation)
    assert frame.pose.orientation_is_gravity_aligned is True


def test_gaze_is_rescaled_to_depth_pixels(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording)

    frame = next(iter(_source(recording).frames()))

    assert frame.gaze_pixel == pytest.approx(np.array([200.0, 150.0]))


def test_a_recording_with_no_imu_gets_identity_poses(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording)

    with caplog.at_level("WARNING"):
        frames = list(_source(recording, FakeReader(quaternions_wxyz=None)).frames())

    assert all(frame.pose.orientation == pytest.approx([1.0, 0.0, 0.0, 0.0]) for frame in frames)
    assert all(frame.pose.orientation_is_gravity_aligned is False for frame in frames)
    assert any("no IMU data" in record.message for record in caplog.records)


def test_a_recording_with_no_gaze_yields_no_gaze(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording)

    frames = list(_source(recording, FakeReader(gaze=None)).frames())

    assert all(frame.gaze_pixel is None for frame in frames)


@pytest.mark.parametrize("empty_sample", [np.zeros(4), np.full(4, np.nan)], ids=["zero", "nan"])
def test_an_empty_imu_sample_mid_recording_carries_the_last_real_orientation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    empty_sample: np.ndarray,
) -> None:
    """An empty reading is a missing sample, so the replay goes on with the last real one. Live no longer does, it poses from the reading at capture or not at all."""
    # Each frame a different pitch, so carrying the wrong frame's orientation shows.
    recording = tmp_path / "rec"
    _write_cache(recording)
    quaternions = np.array([[1.0, 0.05 * index, 0.0, 0.0] for index in range(FRAME_COUNT)])
    quaternions[2] = empty_sample
    quaternions[4] = empty_sample

    with caplog.at_level("WARNING"):
        frames = list(_source(recording, FakeReader(quaternions_wxyz=quaternions)).frames())

    assert len(frames) == FRAME_COUNT
    assert frames[2].pose.orientation == pytest.approx(pose_from_imu(quaternions[1], NEON_IMU_MOUNT).orientation)
    assert frames[2].pose.orientation_is_gravity_aligned is True
    assert frames[3].pose.orientation == pytest.approx(pose_from_imu(quaternions[3], NEON_IMU_MOUNT).orientation)
    assert frames[4].pose.orientation == pytest.approx(pose_from_imu(quaternions[3], NEON_IMU_MOUNT).orientation)
    empty_warnings = [record for record in caplog.records if "empty IMU orientations" in record.message]
    assert len(empty_warnings) == 1, "warned once per replay, not once per empty sample"


def test_empty_imu_samples_before_any_real_one_give_identity_without_gravity(tmp_path: Path) -> None:
    """With no real reading yet, nothing knows where up is, so the scene must not be told it does."""
    recording = tmp_path / "rec"
    _write_cache(recording)
    quaternions = np.tile(np.array([1.0, 0.1, 0.0, 0.0]), (FRAME_COUNT, 1))
    quaternions[:2] = 0.0

    frames = list(_source(recording, FakeReader(quaternions_wxyz=quaternions)).frames())

    for frame in frames[:2]:
        assert frame.pose.orientation == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert frame.pose.orientation_is_gravity_aligned is False
    assert frames[2].pose.orientation_is_gravity_aligned is True


def test_the_sample_tolerance_reaches_the_reader(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording)
    reader = FakeReader()

    list(_source(recording, reader, sample_tolerance_seconds=0.02).frames())

    assert reader.asked_tolerance_ns == 20_000_000


def test_a_missing_cache_says_to_run_the_plugin(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    recording.mkdir()

    with pytest.raises(FileNotFoundError, match="Depth Estimation plugin"):
        next(iter(_source(recording).frames()))


def test_a_missing_recording_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        next(iter(_source(tmp_path / "nowhere").frames()))


@pytest.mark.parametrize("model", [NeonPluginModel.SMALL, NeonPluginModel.BASE])
def test_a_relative_depth_model_is_refused_by_name(tmp_path: Path, model: NeonPluginModel) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording, model=model)

    # The planner's clearance is meters. Relative depth would make every surprise value wrong by
    # an unknown scale, which is the same defect the old script's default model had.
    with pytest.raises(ValueError, match=model.value):
        next(iter(_source(recording, model=model).frames()))


def test_only_the_metric_model_is_metric() -> None:
    assert [model for model in NeonPluginModel if model.is_metric] == [NeonPluginModel.METRIC_LARGE]


def test_a_cache_with_the_wrong_rank_is_refused(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    cache_dir = recording / PLUGIN_CACHE_RELATIVE_DIR
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / f"depth_values_{NeonPluginModel.METRIC_LARGE.cache_stem}.npy", np.ones((300, 400), dtype=np.float32))

    with pytest.raises(ValueError, match="frames, rows, columns"):
        next(iter(_source(recording).frames()))


def test_an_empty_cache_is_refused(tmp_path: Path) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording, frame_count=0)

    with pytest.raises(ValueError, match="no frames"):
        next(iter(_source(recording, FakeReader(frame_count=0)).frames()))


def test_a_frame_count_mismatch_uses_the_shorter_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    recording = tmp_path / "rec"
    _write_cache(recording, frame_count=FRAME_COUNT)

    # The plugin zips without strict, so one more scene frame than depth map is possible. Many
    # more means the cache came from another recording, and the warning carries both numbers.
    with caplog.at_level("WARNING"):
        frames = list(_source(recording, FakeReader(frame_count=FRAME_COUNT + 1)).frames())

    assert len(frames) == FRAME_COUNT
    assert any(f"{FRAME_COUNT} depth maps but {FRAME_COUNT + 1} scene frames" in record.message for record in caplog.records)


def test_quaternion_columns_are_reordered_from_the_recordings_xyzw() -> None:
    # The recording stores x, y, z, w. Getting this wrong flips pitch and breaks the floor fit
    # with no error anywhere.
    xyzw = np.array([[0.1, 0.2, 0.3, 0.9], [0.0, 0.0, 0.0, 1.0]])

    assert xyzw_to_wxyz(xyzw) == pytest.approx(np.array([[0.9, 0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]]))


def test_quaternion_reorder_refuses_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        xyzw_to_wxyz(np.zeros((3, 3)))
