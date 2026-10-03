"""
Covers what the Neon source does with what the client hands it, against a fake device.

The Pupil Labs client itself is kept out of the suite on purpose, so connecting and discovery are
not covered here. What is covered is the part that goes wrong silently: which quaternion field
lands where, what happens between IMU readings, a frame with no gaze, whether the image and the
gaze are straightened together, and what a stalled stream or a missing calibration does.
"""

# Standard library imports
import logging
import threading
import time
from dataclasses import dataclass, field

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.pose.imu_orientation import pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import scale_intrinsics
from nav.sources.config import NeonConfig
from nav.sources.neon_live import NEON_SCENE_SIZE, RECEIVE_POLL_SECONDS, NeonCalibrationError, NeonLiveRgbSource

# The device describes its native 1600 by 1200 scene camera. The fake frames are a fifth of that,
# which also exercises the scaling the source does when the stream is not at native size.
NATIVE_CAMERA_MATRIX = np.array([[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]])
FRAME_SIZE = (240, 320)  # height, width
FRAME_CAMERA_MATRIX = scale_intrinsics(NATIVE_CAMERA_MATRIX, NEON_SCENE_SIZE, FRAME_SIZE)
NO_DISTORTION = np.zeros(8)
BARREL = np.array([-0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class FakeDeviceError(Exception):
    """Stands in for the client's DeviceError, which the suite does not import."""


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
    scene: FakeScene
    gaze: FakeGaze | None


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
    offset_blocks_until: threading.Event | None = None
    offset_calls: list = field(default_factory=list)

    def estimate_time_offset(self, number_of_measurements: int = 100):
        self.offset_calls.append(number_of_measurements)
        if self.offset_blocks_until is not None:
            # Bounded so a source with no guard leaves this thread to die with the test process.
            self.offset_blocks_until.wait(timeout=30.0)
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
            raise StopIteration
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
    # Standing in for _connect(): the connected device, and the failure types it would have
    # imported from the client.
    source._device = device
    source._device_failures = (FakeDeviceError, OSError, ValueError)
    return source


def _take(source: NeonLiveRgbSource, count: int) -> list:
    frames = source.frames()
    taken = []
    for _ in range(count):
        try:
            taken.append(next(frames))
        except (StopIteration, RuntimeError):
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
    dot_centre = np.array([bright_columns.mean(), bright_rows.mean()])
    assert frame.gaze_pixel == pytest.approx(dot_centre, abs=1.0)
    # And the dot moved, so the test is not passing on an image nobody straightened.
    assert np.linalg.norm(dot_centre - np.array([gaze_x, gaze_y])) > 2.0


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
    [FakeDeviceError(500, "Failed to fetch calibration"), ConnectionRefusedError("refused"), ValueError("bad buffer")],
)
def test_a_device_that_cannot_give_its_calibration_ends_the_run_with_a_message(failure: BaseException) -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[], calibration_error=failure)

    with pytest.raises(NeonCalibrationError, match="did not provide its camera calibration"):
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
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[])
    before = time.time()

    frame = _take(_source_with(device), 1)[0]

    assert frame.timing.arrival_seconds == pytest.approx(before, abs=1.0)
    assert frame.timing.depth_ready_seconds is None, "no depth exists yet at the camera"


def test_the_offset_is_measured_again_on_close_and_the_drift_logged(caplog: pytest.LogCaptureFixture) -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(100.0), None)], imu=[], offsets_ms=[250.0, 253.0])
    source = _source_with(device)
    _take(source, 1)

    with caplog.at_level(logging.INFO, logger="nav.sources.neon_live"):
        source.close()

    assert device.offset_calls == [100, 20]
    drift = [record.message for record in caplog.records if "drifted" in record.message]
    assert drift and "3.0 ms" in drift[0]
    assert device.closed is True


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


def test_a_time_echo_that_never_returns_is_abandoned_after_the_timeout() -> None:
    never = threading.Event()
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[], offsets_ms=[250.0, 250.0], offset_blocks_until=never)
    source = _source_with(device, NeonConfig(time_echo_timeout_seconds=0.2))
    try:
        started = time.monotonic()
        frames = _take(source, 1)
        connect_seconds = time.monotonic() - started

        assert frames[0].timing.capture_seconds is None
        assert connect_seconds < 2.0, "the source waited out the stalled call instead of abandoning it"

        # Nothing was measured at connect, so close has no offset to check and must not try.
        started = time.monotonic()
        source.close()
        assert time.monotonic() - started < 2.0
    finally:
        never.set()


def test_a_close_time_echo_that_never_returns_does_not_hold_up_the_shutdown() -> None:
    # The phone answered at connect and then went away. Close is where Ctrl+C lands, so this is
    # the wait that would otherwise leave the person staring at a run that will not end.
    never = threading.Event()

    class GoneByCloseDevice(FakeDevice):
        def estimate_time_offset(self, number_of_measurements: int = 100):
            self.offset_calls.append(number_of_measurements)
            if len(self.offset_calls) > 1:
                never.wait(timeout=30.0)
                return None
            return FakeTimeEcho(FakeEstimate(250.0), FakeEstimate(4.0))

    device = GoneByCloseDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])
    source = _source_with(device, NeonConfig(time_echo_timeout_seconds=0.2))
    try:
        _take(source, 1)
        started = time.monotonic()
        source.close()

        assert time.monotonic() - started < 2.0
        assert device.closed is True
    finally:
        never.set()
