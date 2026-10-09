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
import types
from multiprocessing import shared_memory

# Third party imports
import numpy as np
import pytest

# Local package imports
from fake_neon_client import FakeNeonScript, install
from nav.sources import neon_device
from nav.sources.config import NeonConfig
from nav.sources.neon_device import (
    DeviceRequest,
    NeonDeviceError,
    NeonDeviceProcess,
    NeonStreamEnded,
    NeonUnexpectedFailure,
    SharedFrameBuffer,
    _PipeVideoListener,
    answer,
    known_child_failures,
    serve,
)
from nav.sources.neon_stream import NeonStreamDevice
from nav.sources.scene_video import AccessUnit, VideoDescription, pack_unit, unpack_unit

FRAME_SHAPE = (12, 16, 3)


@dataclasses.dataclass
class _Quaternion:
    w: float
    x: float
    y: float
    z: float


@dataclasses.dataclass
class _ImuStatus:
    readings: int
    empty_readings: int
    unstamped_empty_readings: int
    frames_without_orientation: int


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
    orientation: _Quaternion | None


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
    """Answers like the receiver in the device process. Scripted by its constructor arguments."""

    def __init__(self, gaze: bool = True, time_echo: bool = True, offset_delay_seconds: float = 0.0) -> None:
        self._gaze = gaze
        self._time_echo = time_echo
        self._offset_delay_seconds = offset_delay_seconds
        self._frame_index = 0
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
        # Odd frames had an IMU reading near their capture, even ones did not.
        orientation = _Quaternion(0.9, 0.1, 0.2, 0.3) if self._frame_index % 2 == 1 else None
        return _Matched(_Frame(pixels, 100.0 + self._frame_index), gaze, orientation)

    def imu_status(self) -> _ImuStatus:
        return _ImuStatus(readings=40, empty_readings=3, unstamped_empty_readings=2, frames_without_orientation=1)

    def close(self) -> None:
        self.closed = True


class RefusingDevice(ChildFakeDevice):
    """Fails the way the network and the client fail, one request at a time."""

    def get_calibration(self):
        raise ConnectionRefusedError("calibration port closed")

    def estimate_time_offset(self, number_of_measurements: int = 100):
        raise ValueError("not enough valid samples")

    def imu_status(self):
        raise KeyError("a failure nobody named")


def make_fake_device(config: NeonConfig) -> ChildFakeDevice:
    return ChildFakeDevice()


def make_unreachable_device(config: NeonConfig) -> ChildFakeDevice:
    raise ConnectionError("no Neon found within 10 s")


def make_surprising_device(config: NeonConfig) -> ChildFakeDevice:
    raise KeyError("a connect failure nobody named")


def make_live_device_whose_time_echo_never_answers(config: NeonConfig) -> NeonStreamDevice:
    # The real receiver, with the client's classes swapped for fakes in this child only. Not
    # started, because a Time Echo measurement needs nothing but the address.
    install(FakeNeonScript(time_echo_never_answers=True), setattr)
    return NeonStreamDevice(config)


class ClientFailingDevice(ChildFakeDevice):
    """Fails its calibration the way the Pupil Labs client does when the glasses or the network go."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__()
        self._failure = failure

    def get_calibration(self):
        raise self._failure


@pytest.fixture
def served() -> NeonDeviceProcess:
    """A proxy wired to the child's request loop running on a thread, over a real pipe."""
    return _serve_on_thread(ChildFakeDevice())


def _serve_on_thread(device: ChildFakeDevice) -> NeonDeviceProcess:
    # The child's own list of known failures, so trimming it fails the tests that rely on a member.
    parent_end, child_end = multiprocessing.Pipe()
    thread = threading.Thread(target=serve, args=(child_end, device, known_child_failures()), daemon=True)
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


def test_a_frames_orientation_crosses_the_pipe_with_it_field_for_field_and_none_as_none(served: NeonDeviceProcess) -> None:
    with_reading = served.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)
    without_reading = served.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    quaternion = with_reading.orientation
    assert (quaternion.w, quaternion.x, quaternion.y, quaternion.z) == (0.9, 0.1, 0.2, 0.3)
    assert without_reading.orientation is None


def test_the_imu_counts_cross_the_pipe_as_the_device_counted_them(served: NeonDeviceProcess) -> None:
    status = served.imu_status()

    assert (status.readings, status.empty_readings, status.unstamped_empty_readings, status.frames_without_orientation) == (40, 3, 2, 1)


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

    with pytest.raises(NeonUnexpectedFailure, match="UNEXPECTED") as raised:
        proxy.imu_status()
    assert "KeyError" in str(raised.value)
    assert not isinstance(raised.value, (OSError, ValueError))


@pytest.mark.parametrize("client_failure", ["device_error", "server_disconnected"])
def test_the_clients_own_failures_cross_as_device_failures(client_failure: str) -> None:
    """A dropped HTTP connection is not an OSError, and used to reach the source as unexpected."""
    pytest.importorskip("pupil_labs.realtime_api", reason="the client comes with the glasses extra")
    if client_failure == "device_error":
        from pupil_labs.realtime_api.device import DeviceError

        failure, message = DeviceError(500, "Failed to fetch calibration"), "Failed to fetch calibration"
    else:
        aiohttp = pytest.importorskip("aiohttp", reason="the client's HTTP library comes with the glasses extra")
        failure, message = aiohttp.ServerDisconnectedError("Server disconnected"), "Server disconnected"
    proxy = _serve_on_thread(ClientFailingDevice(failure))

    with pytest.raises(NeonDeviceError, match=message) as raised:
        proxy.get_calibration()
    assert type(failure).__name__ in str(raised.value)


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

    with pytest.raises(NeonDeviceError, match="did not answer 'time_offset'"):
        proxy.estimate_time_offset()
    time.sleep(0.6)
    matched = proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert matched.frame.timestamp_unix_seconds == pytest.approx(101.0)


def test_requests_from_other_threads_wait_for_the_one_holding_the_pipe_and_get_their_own_replies() -> None:
    """The source polls frames while a clock measurement may still be out. Each reply must reach its own request."""
    proxy = _serve_on_thread(ChildFakeDevice(offset_delay_seconds=0.6))
    results: dict[str, object] = {}

    def measure() -> None:
        results["offset"] = proxy.estimate_time_offset(number_of_measurements=20)

    def receive(name: str) -> None:
        results[name] = proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    # Daemons, so two threads stuck reading one pipe fail this test rather than hang the suite.
    measuring = threading.Thread(target=measure, daemon=True)
    measuring.start()
    time.sleep(0.1)
    receivers = [threading.Thread(target=receive, args=(f"frame {index}",), daemon=True) for index in range(3)]
    for receiver in receivers:
        receiver.start()
    for thread in (measuring, *receivers):
        thread.join(15.0)

    assert results["offset"].time_offset_ms.median == pytest.approx(1346.0)
    stamps = sorted(results[f"frame {index}"].frame.timestamp_unix_seconds for index in range(3))
    assert stamps == pytest.approx([101.0, 102.0, 103.0])


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


def test_a_child_that_fails_to_connect_in_an_unnamed_way_raises_it_as_unexpected_with_its_traceback() -> None:
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_surprising_device)

    with pytest.raises(NeonUnexpectedFailure, match="UNEXPECTED failure in the Neon process") as raised:
        proxy.start()
    process = proxy._process
    proxy.close()

    message = str(raised.value)
    assert "Traceback" in message
    assert "KeyError: 'a connect failure nobody named'" in message
    assert not isinstance(raised.value, (OSError, ValueError))
    assert not process.is_alive()


def test_a_time_echo_that_never_answers_comes_back_from_the_child_as_none_within_its_timeout() -> None:
    """
    The real receiver's measurement in a real child, with a phone that never answers.

    Unbounded in the child, the parent waited out its own 5.5 s and blamed the next request, and
    the child was ended by force at close.
    """
    pytest.importorskip("pupil_labs.realtime_api", reason="the client comes with the glasses extra")
    proxy = NeonDeviceProcess(
        NeonConfig(address="192.0.2.7", time_echo_timeout_seconds=0.5),
        device_factory=make_live_device_whose_time_echo_never_answers,
    )
    proxy.start()
    process = proxy._process
    try:
        started = time.monotonic()
        offset = proxy.estimate_time_offset(number_of_measurements=20)
        measuring_seconds = time.monotonic() - started
    finally:
        closing_started = time.monotonic()
        proxy.close()
        closing_seconds = time.monotonic() - closing_started

    assert offset is None
    # The parent's own wait is the timeout plus 5 s. This has to come in well inside it.
    assert measuring_seconds < 2.5
    assert closing_seconds < 3.0
    assert process.exitcode == 0, "the child was ended by force rather than closed"


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
        payload = answer(ChildFakeDevice(), DeviceRequest.MATCHED, (0.25,), frames)
    finally:
        frames.close()

    assert not any(isinstance(part, np.ndarray) for part in payload)
    block_name, shape, dtype, timestamp, gaze, orientation = payload
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
    proxy = _serve_on_thread(EndedDevice())

    with pytest.raises(NeonStreamEnded) as ended:
        proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert not isinstance(ended.value, (OSError, ValueError))


# *******************************************
# The video tee
# *******************************************

_DESCRIPTION = VideoDescription(codec="avc1.42801f", parameter_sets=(b"\x00\x00\x00\x01\x67sps", b"\x00\x00\x00\x01\x68pps"))
_KEYFRAME = AccessUnit(timestamp_seconds=100.0, data=b"\x00\x00\x00\x01\x65key", keyframe=True)
_DELTA = AccessUnit(timestamp_seconds=100.033, data=b"\x00\x00\x00\x01\x61delta", keyframe=False)


class ProvidingFakeDevice(ChildFakeDevice):
    """A device with video to offer. Hands the units it was built with to whoever subscribes."""

    def __init__(self, units: tuple[AccessUnit, ...] = (), stream_until_closed: bool = False) -> None:
        super().__init__()
        self._units = units
        self._stream_until_closed = stream_until_closed
        self.listener = None
        self._streamer: threading.Thread | None = None

    def subscribe_video(self, listener) -> None:
        self.listener = listener
        listener.describe(_DESCRIPTION)
        for unit in self._units:
            listener.offer(unit)
        if self._stream_until_closed:
            self._streamer = threading.Thread(target=self._stream, daemon=True)
            self._streamer.start()

    def _stream(self) -> None:
        while not self.closed:
            self.listener.offer(AccessUnit(timestamp_seconds=time.time(), data=b"\x00\x00\x00\x01\x65" + b"x" * 200, keyframe=True))
            time.sleep(0.005)

    def close(self) -> None:
        super().close()
        if self._streamer is not None:
            self._streamer.join(2.0)


def make_providing_device(config: NeonConfig) -> ProvidingFakeDevice:
    return ProvidingFakeDevice(units=(_KEYFRAME, _DELTA))


def make_streaming_device(config: NeonConfig) -> ProvidingFakeDevice:
    return ProvidingFakeDevice(stream_until_closed=True)


class _UnitCollector:
    def __init__(self) -> None:
        self.descriptions: list[VideoDescription] = []
        self.units: list[AccessUnit] = []

    def describe(self, description: VideoDescription) -> None:
        self.descriptions.append(description)

    def offer(self, unit: AccessUnit) -> None:
        self.units.append(unit)


def _serve_on_thread_with_video(device: ChildFakeDevice) -> NeonDeviceProcess:
    parent_end, child_end = multiprocessing.Pipe()
    video_parent_end, video_child_end = multiprocessing.Pipe(duplex=False)
    thread = threading.Thread(target=serve, args=(child_end, device, known_child_failures(), video_child_end), daemon=True)
    thread.start()
    proxy = NeonDeviceProcess(NeonConfig(time_echo_timeout_seconds=0.2))
    proxy._connection = parent_end
    proxy._video_connection = video_parent_end
    return proxy


def _wait_for_units(collector: _UnitCollector, count: int, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while len(collector.units) < count and time.monotonic() < deadline:
        time.sleep(0.02)


def test_units_cross_the_video_pipe_in_order_with_the_description_first() -> None:
    device = ProvidingFakeDevice()
    proxy = _serve_on_thread_with_video(device)
    collector = _UnitCollector()

    proxy.subscribe_video(collector)
    device.listener.offer(_KEYFRAME)
    device.listener.offer(_DELTA)
    _wait_for_units(collector, 2)

    assert collector.descriptions == [_DESCRIPTION]
    assert collector.units == [_KEYFRAME, _DELTA]


def test_the_child_sends_no_video_until_the_parent_asks() -> None:
    proxy = _serve_on_thread_with_video(ProvidingFakeDevice(units=(_KEYFRAME,)))

    proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)
    proxy.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)

    assert not proxy._video_connection.poll(0.3), "something crossed the video pipe before any VIDEO request"


def test_a_video_request_on_a_device_that_is_not_a_provider_is_refused_as_a_value_error() -> None:
    proxy = _serve_on_thread_with_video(ChildFakeDevice())

    with pytest.raises(ValueError, match="ChildFakeDevice provides no scene video"):
        proxy.subscribe_video(_UnitCollector())


def test_a_video_request_without_a_video_pipe_is_refused_by_name(served: NeonDeviceProcess) -> None:
    # The thread-served fixture gives the child no video pipe at all.
    served._video_connection = multiprocessing.Pipe(duplex=False)[0]

    with pytest.raises(ValueError, match="without a video pipe"):
        served.subscribe_video(_UnitCollector())


def test_the_spawned_child_forwards_units_and_closes_the_video_pipe_on_exit() -> None:
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_providing_device)
    proxy.start()
    collector = _UnitCollector()
    try:
        proxy.subscribe_video(collector)
        reader = proxy._video_thread
        _wait_for_units(collector, 2)
    finally:
        proxy.close()

    assert collector.descriptions == [_DESCRIPTION]
    assert collector.units == [_KEYFRAME, _DELTA]
    assert reader is not None and not reader.is_alive(), "the reader thread outlived close"


def test_a_parent_that_closes_while_units_are_crossing_ends_the_reader_without_an_error(caplog: pytest.LogCaptureFixture) -> None:
    proxy = NeonDeviceProcess(NeonConfig(), device_factory=make_streaming_device)
    proxy.start()
    collector = _UnitCollector()
    with caplog.at_level("ERROR", logger="nav.sources.neon_device"):
        try:
            proxy.subscribe_video(collector)
            reader = proxy._video_thread
            _wait_for_units(collector, 5)
        finally:
            started = time.monotonic()
            proxy.close()
            closing_seconds = time.monotonic() - started

    assert len(collector.units) >= 5
    assert not reader.is_alive()
    assert closing_seconds < 6.0
    assert [record for record in caplog.records if record.levelname == "ERROR"] == []


def _unit(index: int, keyframe: bool) -> AccessUnit:
    # Bigger than the pipe's buffer, so a sender whose parent reads nothing blocks on the first one.
    return AccessUnit(timestamp_seconds=100.0 + index / 30, data=b"\x00\x00\x00\x01" + bytes([0x65 if keyframe else 0x61]) + bytes([index]) * 30_000, keyframe=keyframe)


def test_a_slow_parent_never_holds_the_decoders_offer_and_the_units_after_a_drop_start_at_a_keyframe() -> None:
    # Nothing reads the parent end until every offer is in, so the sender blocks on the first unit
    # and the queue fills behind it, which is a parent that fell a whole keyframe gap behind. The
    # offers come from the decode thread in the child, so each has to return at once regardless.
    parent_end, child_end = multiprocessing.Pipe(duplex=False)
    listener = _PipeVideoListener(child_end, queue_limit=4)
    units = [_unit(index, keyframe=index % 5 == 0) for index in range(20)]
    try:
        listener.describe(_DESCRIPTION)
        offer_seconds: list[float] = []

        def offer_all() -> None:
            for unit in units:
                started = time.perf_counter()
                listener.offer(unit)
                offer_seconds.append(time.perf_counter() - started)
                time.sleep(0.005)  # The decode thread's pace, so the sender gets to take the first unit.

        # On a thread, so an offer that blocks fails this test instead of hanging it.
        decode_thread = threading.Thread(target=offer_all, daemon=True)
        decode_thread.start()
        decode_thread.join(3.0)
        time.sleep(0.2)

        assert not decode_thread.is_alive(), f"an offer is still holding the decode thread after {len(offer_seconds)} offers"
        assert max(offer_seconds) < 0.05, f"an offer held the decode thread for {max(offer_seconds) * 1000:.0f} ms"
        assert listener.dropped > 0
        assert parent_end.recv() == _DESCRIPTION
        # The pipe is in message mode, and a message the sender is still blocked on shows
        # nothing to poll, so a thread reads until the pipe ends. The end comes once the queue
        # has drained and the listener and the pipe are closed, so a wrong drop count fails
        # the count below rather than hanging a read.
        crossed: list[AccessUnit] = []

        def read_until_the_end() -> None:
            try:
                while True:
                    crossed.append(unpack_unit(parent_end.recv_bytes()))
            except EOFError:
                return

        reader = threading.Thread(target=read_until_the_end, daemon=True)
        reader.start()
        deadline = time.monotonic() + 3.0
        while not listener._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.1)
    finally:
        listener.close()
        child_end.close()
    reader.join(3.0)
    parent_end.close()
    assert not reader.is_alive(), "the pipe never ended for the reader"

    assert crossed, "nothing crossed after the description"
    assert crossed[0] == units[0]
    indices = [units.index(unit) for unit in crossed]
    for earlier, later in zip(indices, indices[1:]):
        if later != earlier + 1:
            assert units[later].keyframe, f"after a gap the next unit was {later}, a delta"
    assert len(crossed) + listener.dropped == len(units)


def test_a_parent_that_keeps_up_gets_every_unit_in_order_and_nothing_is_dropped() -> None:
    parent_end, child_end = multiprocessing.Pipe(duplex=False)
    listener = _PipeVideoListener(child_end)
    units = [_unit(index, keyframe=index == 0) for index in range(10)]
    try:
        listener.describe(_DESCRIPTION)
        assert parent_end.recv() == _DESCRIPTION
        crossed = []
        for unit in units:
            listener.offer(unit)
            crossed.append(unpack_unit(parent_end.recv_bytes()))
    finally:
        listener.close()
        child_end.close()
        parent_end.close()

    assert crossed == units
    assert listener.dropped == 0


def test_closing_the_listener_with_units_still_queued_counts_them_as_dropped_and_ends_its_thread() -> None:
    # Units bigger than the pipe's buffer, so the sender blocks on the first one and the rest wait
    # in the queue when close comes. Nothing reads the parent end until after the close.
    parent_end, child_end = multiprocessing.Pipe(duplex=False)
    listener = _PipeVideoListener(child_end, queue_limit=8)
    big = [AccessUnit(timestamp_seconds=100.0 + index / 30, data=b"\x00\x00\x00\x01\x65" + bytes([index]) * 30_000, keyframe=True) for index in range(6)]
    listener.describe(_DESCRIPTION)
    for unit in big:
        listener.offer(unit)
    time.sleep(0.2)

    listener.close()
    child_end.close()
    listener._thread.join(2.0)

    assert not listener._thread.is_alive(), "the sender outlived the pipe"
    assert listener.dropped >= 1
    assert parent_end.recv() == _DESCRIPTION
    crossed = 0
    try:
        while True:
            unpack_unit(parent_end.recv_bytes())
            crossed += 1
    except EOFError:
        pass
    parent_end.close()
    assert crossed + listener.dropped == len(big)


def _reader_on_a_thread(parent_end, collector: _UnitCollector) -> tuple[NeonDeviceProcess, threading.Thread]:
    """The parent's video reader on its own end of a pipe, as subscribe_video starts it."""
    proxy = NeonDeviceProcess(NeonConfig())
    proxy._video_connection = parent_end
    thread = threading.Thread(target=proxy._read_video, args=(collector,), daemon=True)
    thread.start()
    proxy._video_thread = thread
    return proxy, thread


def test_a_malformed_unit_on_the_video_pipe_is_skipped_with_a_warning_and_the_next_unit_still_arrives(caplog: pytest.LogCaptureFixture) -> None:
    parent_end, child_end = multiprocessing.Pipe(duplex=False)
    collector = _UnitCollector()
    _, reader = _reader_on_a_thread(parent_end, collector)
    with caplog.at_level("WARNING", logger="nav.sources.neon_device"):
        child_end.send(_DESCRIPTION)
        child_end.send_bytes(b"not a unit")
        child_end.send_bytes(pack_unit(_KEYFRAME))
        _wait_for_units(collector, 1)
        child_end.close()
        reader.join(3.0)
    parent_end.close()

    assert collector.units == [_KEYFRAME]
    assert any("malformed unit" in record.getMessage() for record in caplog.records)
    assert not reader.is_alive()


def test_closing_the_parent_ends_its_reader_even_when_the_child_keeps_its_end_open() -> None:
    # A child that is stuck, or killed before its own close, never closes its end of the video
    # pipe. The parent's close() has to end the reader itself by closing the end it holds.
    parent_end, child_end = multiprocessing.Pipe(duplex=False)
    request_parent_end, request_child_end = multiprocessing.Pipe()
    threading.Thread(target=serve, args=(request_child_end, ChildFakeDevice(), known_child_failures()), daemon=True).start()
    collector = _UnitCollector()
    proxy, reader = _reader_on_a_thread(parent_end, collector)
    proxy._connection = request_parent_end
    # A stand-in for the child process, so close() takes its full path without spawning one.
    proxy._process = types.SimpleNamespace(join=lambda timeout: None, is_alive=lambda: False, terminate=lambda: None)
    child_end.send(_DESCRIPTION)
    _wait_for(lambda: collector.descriptions == [_DESCRIPTION])

    proxy.close()

    assert not reader.is_alive(), "the reader outlived close() while the child's end stayed open"
    child_end.close()


def test_the_child_closes_its_video_end_when_it_stops_serving_so_a_waiting_parent_reader_ends() -> None:
    parent_end, child_end = multiprocessing.Pipe(duplex=False)
    request_parent_end, request_child_end = multiprocessing.Pipe()
    device = ProvidingFakeDevice()
    threading.Thread(target=serve, args=(request_child_end, device, known_child_failures(), child_end), daemon=True).start()
    proxy = NeonDeviceProcess(NeonConfig(time_echo_timeout_seconds=0.2))
    proxy._connection = request_parent_end
    proxy._video_connection = parent_end
    collector = _UnitCollector()
    proxy.subscribe_video(collector)
    reader = proxy._video_thread
    _wait_for(lambda: collector.descriptions == [_DESCRIPTION])
    # This test holds the child's end, so only serve() closing it can end the reader.
    assert reader is not None and reader.is_alive()

    proxy._request(DeviceRequest.CLOSE, wait_seconds=2.0)

    reader.join(3.0)
    assert not reader.is_alive(), "serve() returned without closing the child's video end"
    parent_end.close()
    child_end.close()


def _wait_for(condition, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
