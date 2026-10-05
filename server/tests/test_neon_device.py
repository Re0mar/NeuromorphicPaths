"""
Covers the child process that holds the Neon connection, against fake devices.

Two kinds of test. The protocol tests run the child's request loop on a thread over a real pipe in
this process, so every request, reply and failure can be checked quickly. The process tests start
a real spawned child with a fake device, because starting, connecting, failing to connect and
closing are the parts a thread cannot stand in for.

The fake device class and its factories live at module level, because a spawned child finds them
by importing this module.
"""

# Standard library imports
import ctypes
import ctypes.wintypes
import dataclasses
import multiprocessing
import sys
import threading
import time
from multiprocessing import shared_memory

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources import neon_device
from nav.sources.config import NeonConfig
from nav.sources.neon_device import NeonDeviceError, NeonDeviceProcess, NeonStreamEnded, SharedFrameBuffer, answer, serve

FRAME_SHAPE = (12, 16, 3)


@dataclasses.dataclass
class _Quaternion:
    w: float
    x: float
    y: float
    z: float


@dataclasses.dataclass
class _Imu:
    quaternion: _Quaternion | None


@dataclasses.dataclass
class _Frame:
    bgr_pixels: np.ndarray
    timestamp_unix_seconds: float


@dataclasses.dataclass
class _Gaze:
    x: float
    y: float


@dataclasses.dataclass
class _Matched:
    frame: _Frame
    gaze: _Gaze | None


@dataclasses.dataclass
class _Estimate:
    median: float


@dataclasses.dataclass
class _TimeEcho:
    time_offset_ms: _Estimate
    roundtrip_duration_ms: _Estimate


@dataclasses.dataclass
class _Calibration:
    scene_camera_matrix: np.ndarray
    scene_distortion_coefficients: np.ndarray


class ChildFakeDevice:
    """Answers like the client's simple Device. Scripted by its constructor arguments."""

    def __init__(self, gaze: bool = True, time_echo: bool = True, offset_delay_seconds: float = 0.0) -> None:
        self._gaze = gaze
        self._time_echo = time_echo
        self._offset_delay_seconds = offset_delay_seconds
        self._frame_index = 0
        self._imu = [None, _Imu(None), _Imu(_Quaternion(0.9, 0.1, 0.2, 0.3))]
        self.closed = False

    def get_calibration(self) -> _Calibration:
        return _Calibration(np.eye(3) * 2.0, np.arange(8, dtype=np.float64))

    def estimate_time_offset(self, number_of_measurements: int = 100) -> _TimeEcho | None:
        time.sleep(self._offset_delay_seconds)
        if not self._time_echo:
            return None
        return _TimeEcho(_Estimate(1326.0 + number_of_measurements), _Estimate(7.0))

    def receive_matched_scene_video_frame_and_gaze(self, timeout_seconds: float | None = None) -> _Matched:
        self._frame_index += 1
        pixels = np.full(FRAME_SHAPE, self._frame_index, dtype=np.uint8)
        gaze = _Gaze(3.0, 4.0) if self._gaze else None
        return _Matched(_Frame(pixels, 100.0 + self._frame_index), gaze)

    def receive_imu_datum(self, timeout_seconds: float | None = None) -> _Imu | None:
        return self._imu.pop(0) if self._imu else None

    def close(self) -> None:
        self.closed = True


class RefusingDevice(ChildFakeDevice):
    """Fails the way the network and the client fail, one request at a time."""

    def get_calibration(self):
        raise ConnectionRefusedError("calibration port closed")

    def estimate_time_offset(self, number_of_measurements: int = 100):
        raise ValueError("not enough valid samples")

    def receive_imu_datum(self, timeout_seconds: float | None = None):
        raise KeyError("a failure nobody named")


def make_fake_device(config: NeonConfig) -> ChildFakeDevice:
    return ChildFakeDevice()


def make_unreachable_device(config: NeonConfig) -> ChildFakeDevice:
    raise ConnectionError("no Neon found within 10 s")


@pytest.fixture
def served() -> NeonDeviceProcess:
    """A proxy wired to the child's request loop running on a thread, over a real pipe."""
    return _serve_on_thread(ChildFakeDevice())


def _serve_on_thread(device: ChildFakeDevice) -> NeonDeviceProcess:
    parent_end, child_end = multiprocessing.Pipe()
    thread = threading.Thread(target=serve, args=(child_end, device, (OSError, ValueError)), daemon=True)
    thread.start()
    proxy = NeonDeviceProcess(NeonConfig(time_echo_timeout_seconds=0.2))
    proxy._connection = parent_end
    return proxy


def test_calibration_crosses_the_pipe_in_the_clients_shape(served: NeonDeviceProcess) -> None:
    calibration = served.get_calibration()

    assert calibration.scene_camera_matrix == pytest.approx(np.eye(3) * 2.0)
    assert calibration.scene_distortion_coefficients == pytest.approx(np.arange(8))


def test_a_matched_frame_crosses_the_pipe_with_its_pixels_stamp_and_gaze(served: NeonDeviceProcess) -> None:
    matched = served.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert matched.frame.bgr_pixels.shape == FRAME_SHAPE
    assert np.all(matched.frame.bgr_pixels == 1)
    assert matched.frame.timestamp_unix_seconds == pytest.approx(101.0)
    assert (matched.gaze.x, matched.gaze.y) == (3.0, 4.0)


def test_a_frame_without_gaze_crosses_as_no_gaze() -> None:
    proxy = _serve_on_thread(ChildFakeDevice(gaze=False))

    assert proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25).gaze is None


def test_imu_answers_keep_no_datum_and_a_datum_without_a_quaternion_apart(served: NeonDeviceProcess) -> None:
    # The source carries the last orientation forward on both, but they are different answers.
    assert served.receive_imu_datum(timeout_seconds=0.0) is None
    assert served.receive_imu_datum(timeout_seconds=0.0).quaternion is None
    quaternion = served.receive_imu_datum(timeout_seconds=0.0).quaternion
    assert (quaternion.w, quaternion.x, quaternion.y, quaternion.z) == (0.9, 0.1, 0.2, 0.3)


def test_the_time_offset_crosses_with_its_medians(served: NeonDeviceProcess) -> None:
    estimates = served.estimate_time_offset(number_of_measurements=20)

    assert estimates.time_offset_ms.median == pytest.approx(1346.0)
    assert estimates.roundtrip_duration_ms.median == pytest.approx(7.0)


def test_a_companion_without_time_echo_crosses_as_none() -> None:
    proxy = _serve_on_thread(ChildFakeDevice(time_echo=False))

    assert proxy.estimate_time_offset() is None


def test_known_failures_cross_as_their_own_kinds() -> None:
    proxy = _serve_on_thread(RefusingDevice())

    with pytest.raises(ConnectionError, match="calibration port closed"):
        proxy.get_calibration()
    with pytest.raises(ValueError, match="not enough valid samples"):
        proxy.estimate_time_offset()


def test_an_unknown_failure_is_raised_as_unexpected_and_not_as_a_known_kind() -> None:
    proxy = _serve_on_thread(RefusingDevice())

    with pytest.raises(RuntimeError, match="UNEXPECTED") as raised:
        proxy.receive_imu_datum(timeout_seconds=0.0)
    assert "KeyError" in str(raised.value)
    assert not isinstance(raised.value, (OSError, ValueError))


def test_a_request_without_a_timeout_is_refused_before_it_can_hold_the_pipe(served: NeonDeviceProcess) -> None:
    with pytest.raises(ValueError, match="timeout is required"):
        served.receive_matched_scene_video_frame_and_gaze()


def test_a_reply_that_arrives_after_its_request_gave_up_is_not_read_as_the_next_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The time measurement takes 0.6 s and the proxy waits 0.2 s for it. Its late reply is already
    # in the pipe when the frame request goes out, and must not be taken as that request's answer.
    monkeypatch.setattr(neon_device, "RESPONSE_GRACE_SECONDS", 0.0)
    proxy = _serve_on_thread(ChildFakeDevice(offset_delay_seconds=0.6))

    with pytest.raises(NeonDeviceError, match="did not answer"):
        proxy.estimate_time_offset()
    time.sleep(0.6)
    matched = proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert matched.frame.timestamp_unix_seconds == pytest.approx(101.0)


def test_a_child_that_has_stopped_is_reported_as_stopped() -> None:
    parent_end, child_end = multiprocessing.Pipe()
    child_end.close()
    proxy = NeonDeviceProcess(NeonConfig())
    proxy._connection = parent_end

    with pytest.raises(NeonDeviceError, match="stopped|did not answer"):
        proxy.get_calibration()


def test_a_spawned_child_serves_frames_and_exits_on_close() -> None:
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_fake_device)
    proxy.start()
    process = proxy._process
    try:
        first = proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)
        second = proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)
    finally:
        proxy.close()

    assert first.frame.timestamp_unix_seconds == pytest.approx(101.0)
    assert second.frame.timestamp_unix_seconds == pytest.approx(102.0)
    assert not process.is_alive()
    assert process.exitcode == 0


def test_a_child_that_cannot_reach_the_device_fails_start_with_its_reason_and_exits() -> None:
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_unreachable_device)

    with pytest.raises(ConnectionError, match="no Neon found"):
        proxy.start()
    process = proxy._process
    proxy.close()

    assert not process.is_alive()


def test_a_child_that_dies_is_reported_on_the_next_request() -> None:
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_fake_device)
    proxy.start()
    try:
        proxy._process.terminate()
        proxy._process.join(5.0)

        with pytest.raises(NeonDeviceError):
            proxy.get_calibration()
    finally:
        proxy.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the priority is only raised on Windows, where the pipeline runs")
def test_the_spawned_child_runs_above_normal_priority() -> None:
    # Below this the decoder fell seconds behind with the head moving while the estimator ran.
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_fake_device)
    proxy.start()
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
        kernel32.GetPriorityClass.argtypes = (ctypes.wintypes.HANDLE,)
        kernel32.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, proxy._process.pid)
        try:
            priority_class = kernel32.GetPriorityClass(handle)
        finally:
            kernel32.CloseHandle(handle)
    finally:
        proxy.close()

    assert priority_class == neon_device.ABOVE_NORMAL_PRIORITY_CLASS


def test_a_frame_reply_carries_where_the_pixels_are_and_not_the_pixels() -> None:
    # 5.8 MB through the pipe needed the pipeline's GIL to receive, and put frames 1.9 s behind
    # while the planner held it. Only the block's name and the frame's shape may cross now.
    frames = SharedFrameBuffer()
    try:
        payload = answer(ChildFakeDevice(), "matched", (0.25,), frames)
    finally:
        frames.close()

    assert not any(isinstance(part, np.ndarray) for part in payload)
    block_name, shape, dtype, timestamp, gaze = payload
    assert isinstance(block_name, str)
    assert tuple(shape) == FRAME_SHAPE
    assert np.dtype(dtype) == np.uint8


def test_consecutive_frames_come_out_with_their_own_pixels(served: NeonDeviceProcess) -> None:
    # One block, written once per request. The first frame's copy must not change when the second
    # frame is written into the same block.
    first = served.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)
    second = served.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert np.all(first.frame.bgr_pixels == 1)
    assert np.all(second.frame.bgr_pixels == 2)


def test_a_larger_frame_gets_a_larger_block_and_comes_out_whole() -> None:
    frames = SharedFrameBuffer()
    try:
        small_name, _, _ = frames.put(np.zeros((4, 4, 3), dtype=np.uint8))
        large = np.arange(64 * 48 * 3, dtype=np.uint32).reshape(64, 48, 3).astype(np.uint8)
        large_name, shape, dtype = frames.put(large)
        reader = shared_memory.SharedMemory(name=large_name)
        try:
            copied = np.ndarray(shape, dtype=np.dtype(dtype), buffer=reader.buf).copy()
        finally:
            reader.close()
    finally:
        frames.close()

    assert large_name != small_name
    assert np.array_equal(copied, large)


class EndedDevice(ChildFakeDevice):
    def receive_matched_scene_video_frame_and_gaze(self, timeout_seconds: float | None = None):
        raise EOFError("the capture has been played to the end")


def test_the_end_of_a_capture_crosses_as_its_own_kind_and_not_as_a_failure() -> None:
    parent_end, child_end = multiprocessing.Pipe()
    thread = threading.Thread(target=serve, args=(child_end, EndedDevice(), (OSError, ValueError, EOFError)), daemon=True)
    thread.start()
    proxy = NeonDeviceProcess(NeonConfig())
    proxy._connection = parent_end

    with pytest.raises(NeonStreamEnded) as ended:
        proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert not isinstance(ended.value, (OSError, ValueError))
