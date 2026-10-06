"""
Covers what the Neon source does with what the device hands it, mostly against a fake device.

What is covered is the part that goes wrong silently: which quaternion field lands where, what
happens between IMU readings, a frame with no gaze, whether the image and the gaze are
straightened together, and what a stalled stream, a missing calibration or a replay does.

One test at the end runs the production path instead: the source's own connect, a spawned device
process, the real receiver and the real decoder, on a capture it encodes itself. That one skips
without the glasses extra. Discovery and the live glasses are left to the by-hand check script.
"""

# Standard library imports
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.clock import laptop_time_seconds
from nav.pose.imu_orientation import pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import scale_intrinsics
from nav.sources.config import NeonConfig
from nav.sources.neon_device import NeonDeviceError, NeonStreamEnded, NeonUnexpectedFailure
from nav.sources.neon_live import NEON_SCENE_SIZE, RECEIVE_POLL_SECONDS, NeonCalibrationError, NeonLiveRgbSource
from neon_captures import encode_h264, write_capture

# The device describes its native 1600 by 1200 scene camera. The fake frames are a fifth of that,
# which also exercises the scaling the source does when the stream is not at native size.
NATIVE_CAMERA_MATRIX = np.array([[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]])
FRAME_SIZE = (240, 320)  # height, width
FRAME_CAMERA_MATRIX = scale_intrinsics(NATIVE_CAMERA_MATRIX, NEON_SCENE_SIZE, FRAME_SIZE)
NO_DISTORTION = np.zeros(8)
BARREL = np.array([-0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class FakeDeviceError(Exception):
    """Stands in for the client's DeviceError. Imported by test_neon_chain.py.

    The source never sees the real one. The device process turns it into NeonDeviceError first,
    which test_neon_device.py checks.
    """


class FakeOutOfFrames(Exception):
    """Raised by the fake when its script runs out, so a test that took too few frames can tell."""


@dataclass
class FakeCalibration:
    scene_camera_matrix: np.ndarray
    scene_distortion_coefficients: np.ndarray


@dataclass
class FakeEstimate:
    median: float


@dataclass
class FakeTimeEcho:
    time_offset_ms: FakeEstimate
    roundtrip_duration_ms: FakeEstimate


@dataclass
class FakeQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass
class FakeImuDatum:
    quaternion: FakeQuaternion | None


@dataclass
class FakeGaze:
    x: float
    y: float


@dataclass
class FakeScene:
    bgr_pixels: np.ndarray
    timestamp_unix_seconds: float


@dataclass
class FakeMatched:
    """Shaped like the client's `MatchedItem`, whose scene frame is `frame`.

    The eye-video variant names it `scene`. A fake that copied that name passed every test while
    the real stream failed on its first frame.
    """

    frame: FakeScene
    gaze: FakeGaze | None


def test_the_fake_matched_item_has_the_real_clients_field_names() -> None:
    # Checked against the installed client when it is present. The suite is meant to run without
    # it, so this skips there, and the lane venv that has the glasses extra is where it bites.
    models = pytest.importorskip("pupil_labs.realtime_api.simple.models")

    assert tuple(FakeMatched.__dataclass_fields__) == models.MatchedItem._fields


@dataclass
class FakeDevice:
    """Hands out scripted matched frames and IMU readings, the way the simple client does.

    It has no eye video method on purpose. A source that asks for eye video fails on it.
    """

    matched: list
    imu: list
    calibration: FakeCalibration | None = None
    calibration_error: BaseException | None = None
    silent_polls: int = 0
    receive_timeouts: list = field(default_factory=list)
    closed: bool = False
    # Medians, in milliseconds, one per call. None for a Companion app without Time Echo.
    offsets_ms: list = field(default_factory=list)
    offset_calls: list = field(default_factory=list)
    # How long a receive takes before it hands its frame over.
    receive_delay_seconds: float = 0.0

    def estimate_time_offset(self, number_of_measurements: int = 100):
        self.offset_calls.append(number_of_measurements)
        if not self.offsets_ms:
            return None
        median = self.offsets_ms.pop(0)
        return FakeTimeEcho(FakeEstimate(median), FakeEstimate(4.0))

    def get_calibration(self) -> FakeCalibration:
        if self.calibration_error is not None:
            raise self.calibration_error
        if self.calibration is not None:
            return self.calibration
        return FakeCalibration(NATIVE_CAMERA_MATRIX, NO_DISTORTION)

    def receive_matched_scene_video_frame_and_gaze(self, timeout_seconds: float | None = None):
        self.receive_timeouts.append(timeout_seconds)
        if self.silent_polls > 0:
            self.silent_polls -= 1
            return None
        if not self.matched:
            raise FakeOutOfFrames
        time.sleep(self.receive_delay_seconds)
        return self.matched.pop(0)

    def receive_imu_datum(self, timeout_seconds: float):
        return self.imu.pop(0) if self.imu else None

    def close(self) -> None:
        self.closed = True


def _scene(stamp: float, pixels: np.ndarray | None = None) -> FakeScene:
    if pixels is None:
        pixels = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        pixels[..., 0] = 255  # blue in BGR, so the conversion to RGB can be checked
    return FakeScene(bgr_pixels=pixels, timestamp_unix_seconds=stamp)


def _source_with(device: FakeDevice, config: NeonConfig | None = None) -> NeonLiveRgbSource:
    source = NeonLiveRgbSource(config if config is not None else NeonConfig())
    # Standing in for _connect(). Everything after it, the failure types included, is production.
    source._device = device
    return source


def _take(source: NeonLiveRgbSource, count: int) -> list:
    """Up to `count` frames, fewer only when the fake's script runs out or the frames end."""
    frames = source.frames()
    taken = []
    for _ in range(count):
        try:
            taken.append(next(frames))
        except (FakeOutOfFrames, StopIteration):
            break
    return taken


def test_the_imu_quaternion_is_read_by_field_name_and_mounted_into_the_pose() -> None:
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), FakeGaze(100.0, 200.0))],
        imu=[FakeImuDatum(FakeQuaternion(w=0.9, x=0.1, y=0.2, z=0.3))],
    )

    frame = _take(_source_with(device), 1)[0]

    expected = pose_from_imu(np.array([0.9, 0.1, 0.2, 0.3]), NEON_IMU_MOUNT)
    assert frame.pose.orientation == pytest.approx(expected.orientation)
    assert frame.pose.orientation_is_gravity_aligned is True
    assert frame.gaze_pixel == pytest.approx([100.0, 200.0])
    assert frame.timestamp_seconds == pytest.approx(1.0)
    assert frame.image_rgb[0, 0].tolist() == [0, 0, 255], "BGR in, RGB out"


def test_a_frame_without_a_new_imu_reading_keeps_the_previous_orientation() -> None:
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), None), FakeMatched(_scene(2.0), None), FakeMatched(_scene(3.0), None)],
        imu=[FakeImuDatum(FakeQuaternion(1.0, 0.0, 0.0, 0.0)), None, FakeImuDatum(None)],
    )

    frames = _take(_source_with(device), 3)

    assert len(frames) == 3
    level = pose_from_imu(np.array([1.0, 0.0, 0.0, 0.0]), NEON_IMU_MOUNT).orientation
    assert frames[0].pose.orientation == pytest.approx(level)
    # No datum, then a datum with no quaternion. Both carry the last orientation forward rather
    # than dropping to None, which would flip the scene into fitting the floor from scratch.
    assert frames[1].pose.orientation == pytest.approx(level)
    assert frames[2].pose.orientation == pytest.approx(level)


def test_before_the_first_imu_reading_the_frame_has_no_pose_and_no_gaze_is_none() -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])

    frame = _take(_source_with(device), 1)[0]

    # No pose rather than an identity one, so nothing downstream reads a gravity it was never told.
    assert frame.pose is None
    assert frame.gaze_pixel is None


def test_closing_closes_the_device() -> None:
    device = FakeDevice(matched=[], imu=[])
    source = _source_with(device)

    source.close()

    assert device.closed is True


def test_frames_carry_the_undistorted_camera_matrix_and_an_undistorted_image() -> None:
    # A barrel lens. The matrix that comes out describes the straightened image, so it is not the
    # device's matrix scaled down, and the image is not the one that went in.
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 255, size=(*FRAME_SIZE, 3), dtype=np.uint8)
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0, pixels), None)],
        imu=[],
        calibration=FakeCalibration(NATIVE_CAMERA_MATRIX, BARREL),
    )

    frame = _take(_source_with(device), 1)[0]

    assert frame.camera_matrix is not None
    assert frame.camera_matrix.shape == (3, 3)
    assert not np.allclose(frame.camera_matrix, FRAME_CAMERA_MATRIX)
    assert frame.image_rgb.shape == (*FRAME_SIZE, 3)
    assert not np.array_equal(frame.image_rgb, cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))


def test_with_no_distortion_the_matrix_is_the_devices_scaled_to_the_frame() -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])

    frame = _take(_source_with(device), 1)[0]

    # The calibration describes the native 1600 by 1200 camera, and the frames are a fifth of that.
    assert frame.camera_matrix == pytest.approx(FRAME_CAMERA_MATRIX, abs=1e-6)


def test_the_undistortion_maps_are_built_once_and_not_per_frame() -> None:
    """Building the maps takes milliseconds the estimator is waiting on, at every frame."""
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), None), FakeMatched(_scene(2.0), None), FakeMatched(_scene(3.0), None)],
        imu=[],
        calibration=FakeCalibration(NATIVE_CAMERA_MATRIX, BARREL),
    )
    source = _source_with(device)
    frames = source.frames()

    next(frames)
    built_on_the_first_frame = source.undistorter
    next(frames)
    next(frames)

    assert built_on_the_first_frame is not None
    assert source.undistorter is built_on_the_first_frame


def test_the_gaze_point_is_undistorted_with_the_image() -> None:
    # A white dot drawn where a barrel lens would put a scene point, and the gaze reported at the
    # same distorted pixel. After the source straightens both, the gaze must still sit on the dot.
    scene_point = np.array([[0.4, 0.25, 1.0]])
    distorted, _ = cv2.projectPoints(scene_point, np.zeros(3), np.zeros(3), FRAME_CAMERA_MATRIX, BARREL)
    gaze_x, gaze_y = distorted.reshape(2)
    pixels = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    cv2.circle(pixels, (int(round(gaze_x)), int(round(gaze_y))), 2, (255, 255, 255), -1)
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0, pixels), FakeGaze(float(gaze_x), float(gaze_y)))],
        imu=[],
        calibration=FakeCalibration(NATIVE_CAMERA_MATRIX, BARREL),
    )

    frame = _take(_source_with(device), 1)[0]

    bright_rows, bright_columns = np.nonzero(frame.image_rgb[..., 0] > 128)
    dot_center = np.array([bright_columns.mean(), bright_rows.mean()])
    assert frame.gaze_pixel == pytest.approx(dot_center, abs=1.0)
    # And the dot moved, so the test is not passing on an image nobody straightened.
    assert np.linalg.norm(dot_center - np.array([gaze_x, gaze_y])) > 2.0


def test_the_source_asks_for_scene_and_gaze_without_eye_video() -> None:
    # The fake has no eye video method. A source still asking for the eyes stream fails here.
    assert not hasattr(FakeDevice, "receive_matched_scene_and_eyes_video_frames_and_gaze")
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])

    frames = _take(_source_with(device), 1)

    assert len(frames) == 1


def test_a_silent_stream_polls_with_a_short_timeout_and_warns_after_the_stall_interval(caplog: pytest.LogCaptureFixture) -> None:
    # 25 empty polls of a quarter second each is 6.25 s of silence, so one warning at 5 s and no
    # second one until 10.
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[], silent_polls=25)

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_live"):
        frames = _take(_source_with(device, NeonConfig(stall_warning_seconds=5.0)), 1)

    assert len(frames) == 1
    assert device.receive_timeouts == [RECEIVE_POLL_SECONDS] * 26
    warnings = [record for record in caplog.records if "no scene frame" in record.message]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "failure",
    # What the device process raises: the client's own failure, a network one, and a value one.
    [NeonDeviceError("DeviceError: (500, 'Failed to fetch calibration')"), ConnectionRefusedError("refused"), ValueError("bad buffer")],
)
def test_a_device_that_cannot_give_its_calibration_ends_the_run_with_a_message(failure: BaseException) -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[], calibration_error=failure)

    with pytest.raises(NeonCalibrationError, match="did not provide its camera calibration") as refused:
        next(_source_with(device).frames())
    assert type(failure).__name__ in str(refused.value)


def test_a_camera_matrix_that_is_not_nine_numbers_ends_the_run_with_a_message() -> None:
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), None)],
        imu=[],
        calibration=FakeCalibration(np.arange(8, dtype=np.float64), NO_DISTORTION),
    )

    with pytest.raises(NeonCalibrationError, match=r"did not provide its camera calibration \(caught ValueError\)"):
        next(_source_with(device).frames())


def test_a_calibration_that_cannot_describe_a_camera_ends_the_run_with_a_message() -> None:
    # Seven coefficients is a length no OpenCV distortion model has.
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), None)],
        imu=[],
        calibration=FakeCalibration(NATIVE_CAMERA_MATRIX, np.zeros(7)),
    )

    with pytest.raises(NeonCalibrationError, match="cannot describe a camera"):
        next(_source_with(device).frames())


def test_an_unexpected_calibration_failure_is_not_disguised_as_a_known_one() -> None:
    # Anything the source cannot name goes to the loop's unexpected branch with its own type.
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[], calibration_error=KeyError("surprise"))

    with pytest.raises(KeyError):
        next(_source_with(device).frames())


def test_capture_is_moved_onto_the_laptop_clock_by_the_measured_offset() -> None:
    # The laptop's clock runs 250 ms ahead of the Neon's, so a Neon stamp of 100.0 is 100.25 here.
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[], offsets_ms=[250.0])

    frame = _take(_source_with(device), 1)[0]

    assert frame.timing.capture_seconds == pytest.approx(100.25)
    assert device.offset_calls == [100]


def test_arrival_is_stamped_on_receipt_on_the_laptop_clock() -> None:
    """Arrival is when the frame reached the laptop, so it comes after the wait for it and not before."""
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[], receive_delay_seconds=0.3)
    frames = _source_with(device).frames()

    before = laptop_time_seconds()
    frame = next(frames)
    after = laptop_time_seconds()

    assert before + 0.3 <= frame.timing.arrival_seconds <= after
    assert frame.timing.depth_ready_seconds is None, "no depth exists yet at the camera"


def test_the_offset_is_measured_again_on_close_and_the_drift_logged(caplog: pytest.LogCaptureFixture) -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[], offsets_ms=[250.0, 253.0])
    source = _source_with(device)
    _take(source, 1)

    with caplog.at_level(logging.INFO, logger="nav.sources.neon_live"):
        source.close()

    assert device.offset_calls == [100, 20]
    drift = [record.getMessage() for record in caplog.records if "drifted" in record.getMessage()]
    assert drift == ["clock offset drifted 3.0 ms over the run, 250.0 ms at connect and 253.0 ms at close"]
    assert device.closed is True


def test_close_does_not_measure_again_when_connect_measured_nothing() -> None:
    """With no offset at connect there is nothing to compare a closing one with."""
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[], offsets_ms=[])
    source = _source_with(device)
    _take(source, 1)

    source.close()

    assert device.offset_calls == [100]
    assert device.closed is True


def test_an_unexpected_failure_measuring_at_close_still_closes_the_device(caplog: pytest.LogCaptureFixture) -> None:
    """Close runs in the loop's finally. Raising there skipped closing the device and the end-of-run report."""

    class GoneByCloseDevice(FakeDevice):
        def estimate_time_offset(self, number_of_measurements: int = 100):
            self.offset_calls.append(number_of_measurements)
            if len(self.offset_calls) > 1:
                raise NeonUnexpectedFailure("UNEXPECTED failure in the Neon process: Traceback ... ServerDisconnectedError")
            return FakeTimeEcho(FakeEstimate(250.0), FakeEstimate(4.0))

    device = GoneByCloseDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])
    source = _source_with(device)
    _take(source, 1)

    with caplog.at_level(logging.ERROR, logger="nav.sources.neon_live"):
        source.close()

    assert device.closed is True
    assert any("UNEXPECTED" in record.getMessage() and record.exc_info for record in caplog.records)


def test_a_replay_logs_the_offset_recorded_with_the_capture_and_checks_no_drift(caplog: pytest.LogCaptureFixture) -> None:
    """A replay measures nothing. Reading the stored offset twice and calling it 0 ms of drift is a made-up result."""
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[], offsets_ms=[1300.0, 1300.0])
    source = _source_with(device, NeonConfig(replay_dir="captures/walk_1"))

    with caplog.at_level(logging.INFO, logger="nav.sources.neon_live"):
        frame = _take(source, 1)[0]
        source.close()

    messages = [record.getMessage() for record in caplog.records]
    assert frame.timing.capture_seconds == pytest.approx(101.3)
    assert any("1300.0 ms" in message and "recorded with the capture" in message for message in messages)
    assert not any("measurements" in message for message in messages)
    assert not any("drifted" in message for message in messages)
    assert device.offset_calls == [100]
    assert device.closed is True


def test_a_replay_recorded_without_an_offset_says_so_and_gives_no_capture_time(caplog: pytest.LogCaptureFixture) -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[], offsets_ms=[])
    source = _source_with(device, NeonConfig(replay_dir="captures/walk_1"))

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_live"):
        frame = _take(source, 1)[0]

    assert frame.timing.capture_seconds is None
    assert any("recorded without a clock offset" in record.getMessage() for record in caplog.records)


def test_a_companion_without_time_echo_gives_no_capture_time_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None), FakeMatched(_scene(2.0), None)], imu=[])

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_live"):
        frames = _take(_source_with(device), 2)

    assert [frame.timing.capture_seconds for frame in frames] == [None, None]
    assert len([record for record in caplog.records if "Time Echo" in record.message]) == 1


def test_a_failing_time_echo_is_logged_and_the_stream_goes_on(caplog: pytest.LogCaptureFixture) -> None:
    class RefusingDevice(FakeDevice):
        def estimate_time_offset(self, number_of_measurements: int = 100):
            raise ConnectionRefusedError("time echo port closed")

    device = RefusingDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_live"):
        frames = _take(_source_with(device), 1)

    assert frames[0].timing.capture_seconds is None
    assert any("ConnectionRefusedError" in record.message for record in caplog.records)


@pytest.mark.parametrize("empty", [0.0, float("nan")], ids=["zero", "nan"])
def test_an_empty_quaternion_from_the_imu_is_skipped_and_the_last_real_orientation_carries_on(
    empty: float, caplog: pytest.LogCaptureFixture
) -> None:
    # The glasses sent only zeros for minutes on 2026-10-05. Before this, the first one ended the
    # run. A NaN is no orientation either, and would end it the same way.
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), None), FakeMatched(_scene(2.0), None), FakeMatched(_scene(3.0), None)],
        imu=[
            FakeImuDatum(FakeQuaternion(empty, empty, empty, empty)),
            FakeImuDatum(FakeQuaternion(1.0, 0.0, 0.0, 0.0)),
            FakeImuDatum(FakeQuaternion(empty, empty, empty, empty)),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_live"):
        frames = _take(_source_with(device), 3)

    level = pose_from_imu(np.array([1.0, 0.0, 0.0, 0.0]), NEON_IMU_MOUNT).orientation
    assert frames[0].pose is None, "no real reading yet, so no gravity"
    assert frames[1].pose.orientation == pytest.approx(level)
    assert frames[2].pose.orientation == pytest.approx(level)
    assert len([record for record in caplog.records if "empty orientations" in record.message]) == 1


def test_a_played_back_capture_that_ends_ends_the_frames_normally() -> None:
    class EndingDevice(FakeDevice):
        def receive_matched_scene_video_frame_and_gaze(self, timeout_seconds: float | None = None):
            if self.matched:
                return self.matched.pop(0)
            raise NeonStreamEnded("the capture has been played to the end")

    device = EndingDevice(matched=[FakeMatched(_scene(1.0), None), FakeMatched(_scene(2.0), None)], imu=[])

    frames = list(_source_with(device).frames())

    assert [frame.timestamp_seconds for frame in frames] == [1.0, 2.0]


def test_a_real_capture_plays_through_the_real_device_process_into_frames(tmp_path: Path) -> None:
    """
    The production path, end to end: the source's own connect, a spawned device process, the real
    receiver and the real H.264 decoder. Every other source test hands the source a fake device.
    """
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    pytest.importorskip("pupil_labs.realtime_api.streaming.nal_unit", reason="the client comes with the glasses extra")
    stream = encode_h264(frame_count=12)
    capture = write_capture(
        tmp_path / "capture",
        [(500.0 + index / 30, picture) for index, picture in enumerate(stream.pictures)],
        offset_ms=1300.0,
        parameter_sets=stream.parameter_sets,
    )
    source = NeonLiveRgbSource(NeonConfig(replay_dir=str(capture)))
    try:
        frames = list(source.frames())
        process = source._device._process
    finally:
        source.close()

    assert frames, "the capture ended without a frame"
    for frame in frames:
        assert frame.camera_matrix.shape == (3, 3)
        assert np.all(np.isfinite(frame.camera_matrix))
        # Stamped as if it were happening now, so capture and arrival are both this moment.
        assert frame.timing.capture_seconds == pytest.approx(frame.timing.arrival_seconds, abs=1.0)
    height, width = frames[-1].image_rgb.shape[:2]
    # The last picture, which only the decoder's flush lets out, is the brightest one.
    assert float(frames[-1].image_rgb[height // 2, width // 2].mean()) == pytest.approx(stream.brightness[-1], abs=6)
    assert process.exitcode == 0, "the device process was ended by force rather than closed"
