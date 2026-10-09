"""
Covers replaying a native Neon recording through the straightened camera.

The recording reader is faked, because the native format is binary and undocumented and a test can't
build one. Everything after the reader is real: the straightener the live glasses use, the IMU rule
the live glasses use, and the composed estimator source. One test at the end checks the native
reader's attribute names against the installed library, where it is installed.
"""

# Standard library imports
import dataclasses
import inspect
import logging
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.config import build_run_config, build_source
from nav.pose.imu_orientation import IMU_MATCH_TOLERANCE_SECONDS, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import scale_intrinsics, undistorter_for
from nav.sources.config import NeonRecordingConfig
from nav.sources.neon_camera import NEON_SCENE_SIZE
from nav.sources.neon_recording import (
    GAZE_SAMPLE_TOLERANCE_NANOSECONDS,
    NeonRecordingCalibrationError,
    NeonRecordingRgbSource,
    sample_nearest,
    thinned_indices,
    xyzw_to_wxyz,
)
from stubs import CanonicalDepthEstimator

# The Neon's own calibration, as read off the device. The same numbers test_camera_model.py pins its
# 754.4 px square focal to.
NEON_CAMERA_MATRIX = np.array([[890.9483, 0.0, 807.2718], [0.0, 890.5604, 608.4522], [0.0, 0.0, 1.0]])
NEON_DISTORTION = np.array([-0.1307, 0.1092, -0.0003, -0.0005, 0.0, 0.1702, 0.0519, 0.0255])
FIRST_TIME_NS = 1_700_000_000_000_000_000
FRAME_GAP_NS = 33_333_333
QUARTER_SIZE = (300, 400)
# A small pitch, unnormalized on purpose, as the device reports it.
PITCHED = np.array([2.0, 0.2, 0.0, 0.0])
EMPTY = np.zeros(4)


class FakeRecordingReader:
    """Stands in for the native recording. Every stream is whatever the test hands it."""

    def __init__(
        self,
        frame_count: int = 6,
        size: tuple[int, int] = QUARTER_SIZE,
        camera_matrix: np.ndarray | None = NEON_CAMERA_MATRIX,
        distortion: np.ndarray = NEON_DISTORTION,
        imu: tuple[np.ndarray, np.ndarray] | None = None,
        gaze: tuple[np.ndarray, np.ndarray] | None = None,
        decoded_size: tuple[int, int] | None = None,
        frame_gap_ns: int = FRAME_GAP_NS,
    ) -> None:
        self.times = FIRST_TIME_NS + np.arange(frame_count, dtype=np.int64) * frame_gap_ns
        self.size = size
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        if imu is None:
            imu = (self.times.copy(), np.tile(PITCHED, (frame_count, 1)))
        self.imu = imu
        self.gaze = gaze
        self.decoded_size = decoded_size if decoded_size is not None else size
        self.converted: list[int] = []
        self.closed = False

    def scene_times_ns(self) -> np.ndarray:
        return self.times

    def scene_size(self) -> tuple[int, int]:
        return self.size

    def scene_camera_matrix(self) -> np.ndarray:
        if self.camera_matrix is None:
            raise FileNotFoundError("recording has no calibration.bin, so the scene camera intrinsics are unknown")
        return self.camera_matrix

    def scene_distortion_coefficients(self) -> np.ndarray:
        return self.distortion

    def scene_frame_rgb(self, index: int) -> np.ndarray:
        self.converted.append(index)
        height, width = self.decoded_size
        # A different value per frame and a gradient across it, so a mix-up or a missed straightening shows.
        columns = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
        return np.stack([columns, columns, np.full((height, width), index, dtype=np.uint8)], axis=-1)

    def imu_samples(self) -> tuple[np.ndarray, np.ndarray]:
        return self.imu

    def gaze_points_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None:
        if self.gaze is None:
            return None
        return sample_nearest(self.gaze[0], self.gaze[1], times_ns, tolerance_ns)

    def close(self) -> None:
        self.closed = True


def _source(tmp_path: Path, reader: FakeRecordingReader, frames_per_second: float = 1000.0) -> NeonRecordingRgbSource:
    # A rate far above the recording's own keeps every frame, unless a test is about thinning.
    config = NeonRecordingConfig(recording_dir=str(tmp_path), frames_per_second=frames_per_second)
    return NeonRecordingRgbSource(config, reader_factory=lambda path: reader)


# *******************************************
# Thinning
# *******************************************


def test_thinning_keeps_the_first_frame_at_or_after_each_tick() -> None:
    # Exact 30 fps stamps. Frame 15 sits 5 ns before the 500 ms tick, so the property is checked
    # rather than a fixed spacing, which would depend on how the stamps were generated.
    times = FIRST_TIME_NS + np.arange(300, dtype=np.int64) * FRAME_GAP_NS
    kept = thinned_indices(times, 2.0)

    assert kept.size == 20
    for tick, index in enumerate(kept.tolist()):
        tick_time = FIRST_TIME_NS + tick * 500_000_000
        assert times[index] >= tick_time
        assert index == 0 or times[index - 1] < tick_time


def test_a_rate_above_the_recordings_keeps_every_frame() -> None:
    times = FIRST_TIME_NS + np.arange(30, dtype=np.int64) * FRAME_GAP_NS
    assert thinned_indices(times, 120.0).tolist() == list(range(30))


def test_a_gap_gives_one_frame_after_it_not_a_burst() -> None:
    before = np.arange(0, 1_000_000_000, FRAME_GAP_NS, dtype=np.int64)
    after = np.arange(4_000_000_000, 5_000_000_000, FRAME_GAP_NS, dtype=np.int64)
    times = FIRST_TIME_NS + np.concatenate([before, after])

    kept_times = times[thinned_indices(times, 2.0)] - FIRST_TIME_NS

    # Ticks at 0 and 0.5 s, then the 3 s gap, then one frame at the 4.0 s tick and the grid carries
    # on from there. Each kept frame is the first at or after its tick, which at 30 fps is 0.533 s
    # for the 0.5 s tick, because the frame before it lands 5 ns short.
    assert [round(time / 1e9, 3) for time in kept_times.tolist()] == [0.0, 0.533, 4.0, 4.533]


def test_thinning_is_exact_at_epoch_nanoseconds() -> None:
    # A float64 holds epoch seconds to about 240 ns, so a frame 1 ns before the tick would round onto
    # it. In integer nanoseconds it stays before, and the frame on the tick is the one kept.
    period_ns = 500_000_000
    times = FIRST_TIME_NS + np.array([0, period_ns - 1, period_ns], dtype=np.int64)

    assert thinned_indices(times, 2.0).tolist() == [0, 2]


def test_a_rate_too_high_for_a_whole_nanosecond_keeps_every_frame() -> None:
    # 1e9 / 3e9 rounds to a 0 ns period. The period is held at 1 ns, so this keeps every frame
    # rather than dividing by zero.
    times = FIRST_TIME_NS + np.arange(5, dtype=np.int64) * FRAME_GAP_NS

    assert thinned_indices(times, 3e9).tolist() == [0, 1, 2, 3, 4]


def test_a_recording_of_one_frame_keeps_it() -> None:
    assert thinned_indices(np.array([FIRST_TIME_NS], dtype=np.int64), 2.0).tolist() == [0]


@pytest.mark.parametrize("rate", [0.0, -2.0, float("nan")])
def test_thinning_refuses_a_rate_that_is_not_a_finite_positive_number(rate: float) -> None:
    with pytest.raises(ValueError, match="frames_per_second"):
        thinned_indices(np.arange(5, dtype=np.int64), rate)


@pytest.mark.parametrize("times", [[0, 10, 10, 20], [0, 20, 10]])
def test_thinning_refuses_times_that_repeat_or_go_backwards(times: list[int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        thinned_indices(np.array(times, dtype=np.int64), 2.0)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan")])
def test_the_config_refuses_a_rate_that_is_not_a_finite_positive_number(rate: float) -> None:
    with pytest.raises(ValueError, match="frames_per_second"):
        NeonRecordingConfig(recording_dir="a_recording", frames_per_second=rate)


# *******************************************
# Sampling a stream at frame times
# *******************************************


def test_a_gap_in_a_stream_costs_only_the_frames_inside_it() -> None:
    # The library's own sampling refuses the whole request on one miss, which would end a replay on
    # the first IMU gap.
    sample_times = FIRST_TIME_NS + np.concatenate([np.arange(0, 1_000_000_000, 9_000_000), np.arange(2_000_000_000, 3_000_000_000, 9_000_000)])
    values = np.arange(sample_times.size, dtype=np.float64).reshape(-1, 1)
    targets = FIRST_TIME_NS + np.array([500_000_000, 1_500_000_000, 2_500_000_000], dtype=np.int64)

    sampled = sample_nearest(sample_times, values, targets, 50_000_000)

    assert np.isfinite(sampled[0, 0]) and np.isnan(sampled[1, 0]) and np.isfinite(sampled[2, 0])


def test_sampling_picks_the_nearest_sample_and_none_from_an_empty_stream() -> None:
    sample_times = np.array([0, 100, 200], dtype=np.int64)
    values = np.array([[0.0], [1.0], [2.0]])

    assert sample_nearest(sample_times, values, np.array([140, 160]), 50)[:, 0].tolist() == [1.0, 2.0]
    assert np.isnan(sample_nearest(np.empty(0, dtype=np.int64), np.empty((0, 1)), np.array([5]), 50)).all()


def test_sampling_refuses_sample_times_out_of_order() -> None:
    with pytest.raises(ValueError, match="ascending"):
        sample_nearest(np.array([0, 200, 100]), np.zeros((3, 1)), np.array([50]), 10)


# *******************************************
# The straightened camera
# *******************************************


def test_frames_carry_the_straightened_matrix_and_the_straightened_image(tmp_path: Path) -> None:
    reader = FakeRecordingReader()
    expected = undistorter_for(NEON_CAMERA_MATRIX, NEON_DISTORTION, NEON_SCENE_SIZE, QUARTER_SIZE)

    frames = list(_source(tmp_path, reader).frames())

    assert len(frames) == 6
    for index, frame in enumerate(frames):
        assert np.array_equal(frame.camera_matrix, expected.camera_matrix)
        assert np.array_equal(frame.image_rgb, expected.undistort_image(reader.scene_frame_rgb(index)))


def test_at_the_native_size_the_square_focal_is_the_one_measured_on_the_glasses(tmp_path: Path) -> None:
    # Pinned to a number from outside this code, so a wrong coefficient order can't pass by being
    # applied the same wrong way to everything a test compares.
    reader = FakeRecordingReader(frame_count=1, size=NEON_SCENE_SIZE)

    (frame,) = list(_source(tmp_path, reader).frames())

    assert frame.camera_matrix[0, 0] == pytest.approx(754.4, abs=0.5)
    assert frame.camera_matrix[1, 1] == frame.camera_matrix[0, 0]


def test_only_the_kept_frames_are_converted_to_color(tmp_path: Path) -> None:
    # The library still decodes the frames in between. Converting a frame to color is the per-frame
    # cost the source controls, so that's what is limited to the kept frames.
    reader = FakeRecordingReader(frame_count=60)

    frames = list(_source(tmp_path, reader, frames_per_second=2.0).frames())

    assert reader.converted[len(frames):] == []
    assert reader.converted == thinned_indices(reader.times, 2.0).tolist()


def test_gaze_is_straightened_with_the_picture(tmp_path: Path) -> None:
    times = FIRST_TIME_NS + np.arange(3, dtype=np.int64) * FRAME_GAP_NS
    # In native pixels, as the recording stores gaze: the optical center, a NaN, and a corner the crop removes.
    points = np.array([[NEON_CAMERA_MATRIX[0, 2], NEON_CAMERA_MATRIX[1, 2]], [np.nan, np.nan], [0.0, 0.0]])
    reader = FakeRecordingReader(frame_count=3, size=NEON_SCENE_SIZE, gaze=(times, points))
    expected = undistorter_for(NEON_CAMERA_MATRIX, NEON_DISTORTION, NEON_SCENE_SIZE, NEON_SCENE_SIZE)

    center, missing, cropped = (frame.gaze_pixel for frame in _source(tmp_path, reader).frames())

    assert center == pytest.approx(expected.camera_matrix[:2, 2], abs=0.01)
    assert missing is None
    assert cropped is None


def test_gaze_in_native_pixels_is_scaled_to_a_smaller_decoded_frame_before_straightening(tmp_path: Path) -> None:
    times = FIRST_TIME_NS + np.arange(1, dtype=np.int64)
    native_point = np.array([[1000.0, 700.0]])
    reader = FakeRecordingReader(frame_count=1, size=QUARTER_SIZE, gaze=(times, native_point))
    expected = undistorter_for(NEON_CAMERA_MATRIX, NEON_DISTORTION, NEON_SCENE_SIZE, QUARTER_SIZE)

    (frame,) = list(_source(tmp_path, reader).frames())

    assert frame.gaze_pixel == pytest.approx(expected.undistort_pixel(native_point[0] / 4.0))


# *******************************************
# Pose, by the live glasses' rule
# *******************************************


def test_a_frame_near_a_usable_reading_is_posed_from_it(tmp_path: Path) -> None:
    reader = FakeRecordingReader(frame_count=2)

    frames = list(_source(tmp_path, reader).frames())

    expected = pose_from_imu(PITCHED, NEON_IMU_MOUNT)
    assert all(np.allclose(frame.pose.orientation, expected.orientation) for frame in frames)
    assert all(frame.pose.orientation_is_gravity_aligned for frame in frames)


def test_an_empty_reading_is_passed_over_for_the_nearest_usable_one(tmp_path: Path) -> None:
    frame_times = FIRST_TIME_NS + np.arange(1, dtype=np.int64)
    imu_times = frame_times[0] + np.array([0, 20_000_000], dtype=np.int64)
    reader = FakeRecordingReader(frame_count=1, imu=(imu_times, np.array([EMPTY, PITCHED])))

    (frame,) = list(_source(tmp_path, reader).frames())

    assert np.allclose(frame.pose.orientation, pose_from_imu(PITCHED, NEON_IMU_MOUNT).orientation)


def test_a_frame_with_nothing_usable_near_it_gets_no_pose_and_nothing_is_carried(tmp_path: Path) -> None:
    # Frames 200 ms apart, so no frame's reading is within the 50 ms tolerance of its neighbor's.
    reader = FakeRecordingReader(frame_count=3, frame_gap_ns=200_000_000)
    # A real reading on frame 0, an empty one on frame 1 and nothing else near it, a real one on frame 2.
    reader.imu = (reader.times.copy(), np.array([PITCHED, EMPTY, PITCHED]))

    first, second, third = list(_source(tmp_path, reader).frames())

    assert first.pose is not None
    # Frame 0's orientation is not carried onto frame 1, and the empty reading at frame 1 is not one.
    assert second.pose is None
    assert third.pose is not None


def test_a_recording_with_no_imu_gives_no_pose_and_says_how_many(tmp_path: Path, caplog) -> None:
    reader = FakeRecordingReader(frame_count=4, imu=(np.empty(0, dtype=np.int64), np.empty((0, 4))))

    with caplog.at_level(logging.INFO):
        frames = list(_source(tmp_path, reader).frames())

    assert all(frame.pose is None for frame in frames)
    assert "4 of 4 frames" in caplog.text


# *******************************************
# What can't be replayed
# *******************************************


def test_a_recording_with_no_calibration_is_refused_by_name(tmp_path: Path) -> None:
    reader = FakeRecordingReader(camera_matrix=None)

    with pytest.raises(NeonRecordingCalibrationError, match=tmp_path.name):
        list(_source(tmp_path, reader).frames())


def test_seven_distortion_coefficients_are_refused_as_the_recordings_calibration(tmp_path: Path) -> None:
    reader = FakeRecordingReader(distortion=NEON_DISTORTION[:7])

    with pytest.raises(NeonRecordingCalibrationError, match="cannot describe"):
        list(_source(tmp_path, reader).frames())


def test_a_decoded_frame_of_another_size_is_refused(tmp_path: Path) -> None:
    reader = FakeRecordingReader(decoded_size=(150, 200))

    with pytest.raises(NeonRecordingCalibrationError, match="200x150"):
        list(_source(tmp_path, reader).frames())


def test_a_missing_folder_is_refused_before_any_reader_is_made(tmp_path: Path) -> None:
    made: list[Path] = []
    config = NeonRecordingConfig(recording_dir=str(tmp_path / "nowhere"))
    source = NeonRecordingRgbSource(config, reader_factory=lambda path: made.append(path) or FakeRecordingReader())

    with pytest.raises(FileNotFoundError):
        list(source.frames())
    assert made == []


def test_closing_the_source_closes_the_recording(tmp_path: Path) -> None:
    reader = FakeRecordingReader(frame_count=1)
    source = _source(tmp_path, reader)
    list(source.frames())

    source.close()

    assert reader.closed


# *******************************************
# Through the production entry point
# *******************************************


def test_the_command_line_source_converts_depth_with_the_straightened_focal(tmp_path: Path) -> None:
    reader = FakeRecordingReader(frame_count=3)
    depth_shape = (12, 16)
    # The parser takes a folder for a recording only if it has the Companion's info.json.
    (tmp_path / "info.json").write_text("{}", encoding="utf-8")
    config = build_run_config(["--source", "neon_recording", "--recording-dir", str(tmp_path), "--recording-rate", "1000", "--sink", "none"])
    config = dataclasses.replace(
        config,
        estimator_factory=lambda estimator_config: CanonicalDepthEstimator(raw_depth=3.0, height=depth_shape[0], width=depth_shape[1]),
        recording_reader_factory=lambda path: reader,
    )

    frames = list(build_source(config).frames())

    straightened = undistorter_for(NEON_CAMERA_MATRIX, NEON_DISTORTION, NEON_SCENE_SIZE, QUARTER_SIZE).camera_matrix
    at_depth = scale_intrinsics(straightened, QUARTER_SIZE, depth_shape)
    assert len(frames) == 3
    for frame in frames:
        assert frame.intrinsics == pytest.approx(at_depth)
        mean_focal = (at_depth[0, 0] + at_depth[1, 1]) / 2.0
        assert frame.depth_meters == pytest.approx(np.full(depth_shape, 3.0 * mean_focal / 300.0))


# *******************************************
# The real library, where it is installed
# *******************************************


def test_the_native_reader_uses_names_the_installed_library_has() -> None:
    # The fake above proves nothing about these. The suite's dev venv leaves the glasses extra out on
    # purpose, so this runs in the glasses venv, which is where the native reader runs too.
    recording_module = pytest.importorskip("pupil_labs.neon_recording", reason="the recording library comes with the glasses extra")
    from pupil_labs.neon_recording.calib import Calibration
    from pupil_labs.neon_recording.timeseries.gaze import GazeTimeseries
    from pupil_labs.neon_recording.timeseries.av.video import SceneVideoTimeseries
    from pupil_labs.neon_recording.timeseries.imu.imu_timeseries import IMUTimeseries
    from pupil_labs.video.frame import VideoFrame

    recording = recording_module.NeonRecording
    assert issubclass(recording.SensorError, Exception)
    for name in ("scene", "imu", "gaze", "calibration", "close"):
        assert hasattr(recording, name), name
    # The library's fields are descriptors that read a record, so asking one on the class would fail.
    # getattr_static finds the name without calling it.
    for owner, names in (
        (Calibration, ("scene_camera_matrix", "scene_distortion_coefficients")),
        (SceneVideoTimeseries, ("time", "width", "height", "__getitem__")),
        (IMUTimeseries, ("time", "rotation")),
        (GazeTimeseries, ("time", "point")),
    ):
        for name in names:
            assert inspect.getattr_static(owner, name, None) is not None, f"{owner.__name__}.{name}"
    assert hasattr(VideoFrame, "rgb")
    assert GAZE_SAMPLE_TOLERANCE_NANOSECONDS == int(IMU_MATCH_TOLERANCE_SECONDS * 1e9)


def test_sampling_refuses_times_and_values_that_do_not_pair() -> None:
    with pytest.raises(ValueError, match="3 sample times for 2 values"):
        sample_nearest(np.array([0, 100, 200]), np.zeros((2, 1)), np.array([50]), 10)


def test_quaternion_columns_are_reordered_from_the_recordings_xyzw() -> None:
    # The recording stores x, y, z, w. Getting this wrong flips pitch and breaks the floor fit
    # with no error anywhere.
    xyzw = np.array([[0.1, 0.2, 0.3, 0.9], [0.0, 0.0, 0.0, 1.0]])

    assert xyzw_to_wxyz(xyzw) == pytest.approx(np.array([[0.9, 0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]]))


def test_quaternion_reorder_refuses_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        xyzw_to_wxyz(np.zeros((3, 3)))
