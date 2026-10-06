"""
The Neon's network client, run in a process of its own.

The Pupil Labs client decodes the scene video on a background thread of whatever process it runs
in. In the pipeline's process that thread shares the GIL with the estimator, the scene, the planner
and the web page, and loses. Measured on the glasses: frames 100 ms old with nothing else running,
19 s old with two busy Python threads beside the reader. So the client lives here, in a child
process that does nothing but decode and keep the newest frame, and the pipeline asks it for that
frame when it wants one.

NeonDeviceProcess answers the same calls the client's simple Device does, with the same shapes,
so NeonLiveRgbSource neither knows nor cares which one it holds. Requests and replies cross a pipe
as tuples of plain values and this file's enums, because the client's own objects are not promised
to pickle.

This file does not import the Pupil Labs client. neon_stream.py does, inside the child, and
test_import_boundaries.py keeps the import to that file and neon_plugin.py.
"""

# Standard library imports
import ctypes
import ctypes.wintypes
import itertools
import logging
import multiprocessing
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.sources.config import NeonConfig
from nav.sources.neon_stream import NeonStreamDevice, client_failure_types

log = logging.getLogger(__name__)

# How long past a request's own timeout the parent waits before deciding the child has stopped
# answering. A reply is a few small values, since the frame goes through shared memory, so this is
# mostly room for a child the estimator has starved of CPU.
RESPONSE_GRACE_SECONDS = 5.0
# Starting the child means a fresh interpreter importing numpy, OpenCV and the client. A few
# seconds on this laptop, so this is generous on purpose.
START_GRACE_SECONDS = 30.0
CLOSE_JOIN_SECONDS = 5.0
# Windows' priority class one step above normal. Not high or realtime, which can starve the desktop.
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


class DeviceRequest(Enum):
    """What the parent asks the child for. Every request gets exactly one reply."""

    CONNECT = "connect"
    CALIBRATION = "calibration"
    TIME_OFFSET = "time_offset"
    MATCHED = "matched"
    IMU = "imu"
    CLOSE = "close"


class ReplyStatus(Enum):
    """Whether the child answered a request or failed it. A failure carries a FailureKind."""

    OK = "ok"
    ERROR = "error"


class FailureKind(Enum):
    """How a failure in the child crosses the pipe. The parent raises each one as its own type."""

    DEVICE = "device"
    OS = "os"
    VALUE = "value"
    # Not a failure. A played-back capture has reached its end.
    ENDED = "ended"
    UNEXPECTED = "unexpected"


class NeonDeviceError(ConnectionError):
    """The device refused a request, or the process holding the connection stopped answering."""


class NeonUnexpectedFailure(RuntimeError):
    """The child failed in a way nobody named. Carries the child's traceback in its message."""


class NeonStreamEnded(Exception):
    """A played-back capture has handed over its last frame. The normal end of a replay run.

    Not an OSError or a ValueError, so nothing that handles a failure catches it by accident.
    """


# Shaped like the client's own types, field for field, so the source reads either the same way.
@dataclass(frozen=True)
class DeviceCalibration:
    scene_camera_matrix: np.ndarray
    scene_distortion_coefficients: np.ndarray


@dataclass(frozen=True)
class DeviceEstimate:
    median: float


@dataclass(frozen=True)
class DeviceTimeEcho:
    time_offset_ms: DeviceEstimate
    roundtrip_duration_ms: DeviceEstimate


@dataclass(frozen=True)
class DeviceFrame:
    bgr_pixels: np.ndarray
    timestamp_unix_seconds: float


@dataclass(frozen=True)
class DeviceGaze:
    x: float
    y: float


@dataclass(frozen=True)
class DeviceMatched:
    frame: DeviceFrame
    gaze: DeviceGaze | None


@dataclass(frozen=True)
class DeviceQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DeviceImuDatum:
    quaternion: DeviceQuaternion | None


def apply_opencv_pyav_import_workaround() -> None:
    """
    Open and close a one-pixel OpenCV window before the Pupil Labs client is imported.

    Works around an import-order crash between OpenCV and PyAV on Windows. The old script did this
    at module scope with no explanation. Kept because a rewrite that drops a workaround on the
    grounds that the new design makes it unnecessary rediscovers the bug a week later.
    """
    cv2.imshow("opencv pyav import order", np.zeros(1))
    cv2.destroyAllWindows()


class NeonDeviceProcess:
    """A Neon connection held by a child process, with the calls the client's simple Device has."""

    def __init__(
        self,
        config: NeonConfig,
        device_factory: Callable[[NeonConfig], object] | None = None,
    ) -> None:
        """
        :param config: Where the device is and how long to look for it.
        :param device_factory: Builds the device inside the child. None means the real client.
            Tests pass a module-level function, because it has to reach the child by import.
        """
        self._config = config
        self._device_factory = device_factory
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None
        self._request_ids = itertools.count()
        # One request at a time on the pipe. A time measurement the source gave up on can still be
        # waiting for its reply, and the next request must not read that reply as its own.
        # Re-entrant, because a frame is copied out of shared memory while the lock is still held.
        self._lock = threading.RLock()
        # The child's frame block, opened on the first frame and reopened if the child replaces it.
        self._frame_memory: shared_memory.SharedMemory | None = None

    def start(self) -> None:
        """
        Start the child and wait for it to connect to the device.

        :raises ConnectionError: When the child cannot reach the device, with the child's reason.
        """
        context = multiprocessing.get_context("spawn")
        parent_end, child_end = context.Pipe()
        self._process = context.Process(
            target=run_device_process,
            args=(child_end, self._config, self._device_factory, logging.getLogger("nav").getEffectiveLevel()),
            name="neon-device",
            # Ends with the pipeline even if the pipeline dies without closing it.
            daemon=True,
        )
        self._process.start()
        child_end.close()
        self._connection = parent_end
        self._request(DeviceRequest.CONNECT, wait_seconds=self._config.discovery_timeout_seconds + START_GRACE_SECONDS)

    def get_calibration(self) -> DeviceCalibration:
        camera_matrix, distortion = self._request(DeviceRequest.CALIBRATION, wait_seconds=RESPONSE_GRACE_SECONDS)
        return DeviceCalibration(scene_camera_matrix=camera_matrix, scene_distortion_coefficients=distortion)

    def estimate_time_offset(self, number_of_measurements: int = 100) -> DeviceTimeEcho | None:
        """
        Laptop clock minus Neon clock, measured in the child.

        The child gives up after `time_echo_timeout_seconds` and answers None. This waits that long
        plus the grace, so the child's bound is always the one that fires first.

        :return: The medians, or None when the Companion app cannot answer or did not in time.
        """
        answer = self._request(
            DeviceRequest.TIME_OFFSET,
            number_of_measurements,
            wait_seconds=self._config.time_echo_timeout_seconds + RESPONSE_GRACE_SECONDS,
        )
        if answer is None:
            return None
        offset_median_ms, round_trip_median_ms = answer
        return DeviceTimeEcho(DeviceEstimate(offset_median_ms), DeviceEstimate(round_trip_median_ms))

    def receive_matched_scene_video_frame_and_gaze(self, timeout_seconds: float | None = None) -> DeviceMatched | None:
        if timeout_seconds is None:
            raise ValueError("a timeout is required, because a request that never returns holds the pipe for good")
        with self._lock:
            answer = self._request(DeviceRequest.MATCHED, timeout_seconds, wait_seconds=timeout_seconds + RESPONSE_GRACE_SECONDS)
            if answer is None:
                return None
            block_name, shape, dtype, timestamp_unix_seconds, gaze = answer
            # Copied while the lock is held, so no other request can reach the child, and the
            # child cannot write the next frame over this one before it is out.
            bgr_pixels = self._copy_frame(block_name, shape, dtype)
        return DeviceMatched(
            frame=DeviceFrame(bgr_pixels=bgr_pixels, timestamp_unix_seconds=timestamp_unix_seconds),
            gaze=None if gaze is None else DeviceGaze(*gaze),
        )

    def receive_imu_datum(self, timeout_seconds: float | None = None) -> DeviceImuDatum | None:
        if timeout_seconds is None:
            raise ValueError("a timeout is required, because a request that never returns holds the pipe for good")
        answer = self._request(DeviceRequest.IMU, timeout_seconds, wait_seconds=timeout_seconds + RESPONSE_GRACE_SECONDS)
        if answer is None:
            return None
        return DeviceImuDatum(quaternion=None if answer == () else DeviceQuaternion(*answer))

    def close(self) -> None:
        """Ask the child to close the device and exit. Ends it by force if it does not."""
        if self._process is None:
            return
        try:
            self._request(DeviceRequest.CLOSE, wait_seconds=CLOSE_JOIN_SECONDS)
        except NeonDeviceError as unanswered:
            # Already gone or stuck. Either way the join and terminate below finish the job.
            log.warning("the Neon process did not close cleanly (caught %s, expected): %s", type(unanswered).__name__, unanswered)
        self._process.join(CLOSE_JOIN_SECONDS)
        if self._process.is_alive():
            log.warning("the Neon process did not exit within %.0f s, ending it", CLOSE_JOIN_SECONDS)
            self._process.terminate()
            self._process.join(CLOSE_JOIN_SECONDS)
        if self._connection is not None:
            self._connection.close()
        self._close_frame_memory()
        self._process = None
        self._connection = None

    def _copy_frame(self, block_name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
        if self._frame_memory is None or self._frame_memory.name != block_name:
            self._close_frame_memory()
            self._frame_memory = shared_memory.SharedMemory(name=block_name)
        return np.ndarray(shape, dtype=np.dtype(dtype), buffer=self._frame_memory.buf).copy()

    def _close_frame_memory(self) -> None:
        if self._frame_memory is not None:
            # Only this process's handle. The block itself belongs to the child.
            self._frame_memory.close()
            self._frame_memory = None

    def _request(self, name: DeviceRequest, *arguments: object, wait_seconds: float) -> object:
        if self._connection is None:
            raise NeonDeviceError("the Neon process is not running")
        with self._lock:
            request_id = next(self._request_ids)
            try:
                self._connection.send((request_id, name, arguments))
                while True:
                    if not self._connection.poll(wait_seconds):
                        raise NeonDeviceError(f"the Neon process did not answer {name.value!r} within {wait_seconds:.1f} s")
                    reply_id, status, payload = self._connection.recv()
                    # A reply to a request someone stopped waiting for. Its answer is no use now.
                    if reply_id == request_id:
                        break
            except (EOFError, BrokenPipeError, ConnectionResetError) as gone:
                raise NeonDeviceError(f"the Neon process has stopped (caught {type(gone).__name__})") from gone

        if status is ReplyStatus.OK:
            return payload
        kind, message = payload
        match kind:
            case FailureKind.DEVICE:
                raise NeonDeviceError(message)
            case FailureKind.OS:
                raise ConnectionError(message)
            case FailureKind.VALUE:
                raise ValueError(message)
            case FailureKind.ENDED:
                raise NeonStreamEnded(message)
            case FailureKind.UNEXPECTED:
                # Not one of the failures this file knows about, so not dressed up as one.
                raise NeonUnexpectedFailure(f"UNEXPECTED failure in the Neon process: {message}")
            case _:
                # Unreachable while every member above is handled. Here so a new member fails loudly.
                raise ValueError(f"no handling for the failure kind {kind}")


def known_child_failures() -> tuple[type[BaseException], ...]:
    """
    The failures the child reports as their own kinds. Anything else crosses as unexpected.

    EOFError is not a failure. It is a played-back capture reaching its end.
    """
    return (*client_failure_types(), OSError, ValueError, EOFError)


def run_device_process(
    connection: Connection,
    config: NeonConfig,
    device_factory: Callable[[NeonConfig], object] | None,
    log_level: int,
) -> None:
    """
    The child's whole life: connect, answer requests until told to close, then close the device.

    Module level, because a spawned process finds its target by import.
    """
    # Ctrl+C reaches every process in the console. The parent decides when this one ends, through
    # close, so the device is shut down in order rather than torn out from under a read.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("nav").setLevel(log_level)
    _raise_own_priority()

    known_failures = known_child_failures()
    build = device_factory if device_factory is not None else _connect_client
    first_request_id, _, _ = connection.recv()
    try:
        device = build(config)
    except known_failures as failure:
        connection.send((first_request_id, ReplyStatus.ERROR, _describe_failure(failure)))
        return
    except Exception as unexpected:  # noqa: BLE001, sent to the parent as unexpected, which raises it there
        connection.send((first_request_id, ReplyStatus.ERROR, (FailureKind.UNEXPECTED, traceback.format_exc())))
        log.error("UNEXPECTED %s connecting to the Neon, may need a handler", type(unexpected).__name__, exc_info=True)
        return
    connection.send((first_request_id, ReplyStatus.OK, None))

    try:
        serve(connection, device, known_failures)
    finally:
        device.close()


class SharedFrameBuffer:
    """One block of shared memory the child writes each frame into, owned by the child.

    A frame through the pipe is 5.8 MB that the pipeline process has to receive and unpickle, and
    that needs the GIL the scene and the planner are holding. Measured with the head moving, that
    put frames up to 1.9 s behind. Through shared memory only the block's name and the frame's
    shape cross the pipe, and the parent copies the pixels straight out.

    Requests are answered one at a time, so the child writes the block only when asked for a frame,
    and the parent has copied the last one out before it asks again. One block is enough.
    """

    def __init__(self) -> None:
        self._memory: shared_memory.SharedMemory | None = None

    def put(self, pixels: np.ndarray) -> tuple[str, tuple[int, ...], str]:
        """
        Copy a frame into the block, growing it if the frame no longer fits.

        :return: The block's name, the frame's shape and its dtype, which is all the parent needs.
        :rtype: tuple[str, tuple[int, ...], str]
        """
        pixels = np.ascontiguousarray(pixels)
        if self._memory is None or self._memory.size < pixels.nbytes:
            self.close()
            self._memory = shared_memory.SharedMemory(create=True, size=pixels.nbytes)
        np.ndarray(pixels.shape, dtype=pixels.dtype, buffer=self._memory.buf)[...] = pixels
        return self._memory.name, pixels.shape, pixels.dtype.str

    def close(self) -> None:
        if self._memory is None:
            return
        self._memory.close()
        # A no-op on Windows, where the block goes when its last handle closes. Needed elsewhere.
        self._memory.unlink()
        self._memory = None


def serve(connection: Connection, device: object, known_failures: tuple[type[BaseException], ...]) -> None:
    """
    Answer requests on the pipe until a close request or until the parent goes away.

    Split out from the process function so the protocol can be tested over a pipe in one process.
    """
    frames = SharedFrameBuffer()
    try:
        _serve_until_closed(connection, device, known_failures, frames)
    finally:
        frames.close()


def _serve_until_closed(
    connection: Connection,
    device: object,
    known_failures: tuple[type[BaseException], ...],
    frames: SharedFrameBuffer,
) -> None:
    while True:
        try:
            request_id, name, arguments = connection.recv()
        except EOFError:
            log.info("the pipeline went away, closing the Neon")
            return
        if name is DeviceRequest.CLOSE:
            connection.send((request_id, ReplyStatus.OK, None))
            return
        try:
            payload = answer(device, name, arguments, frames)
        except known_failures as failure:
            connection.send((request_id, ReplyStatus.ERROR, _describe_failure(failure)))
            continue
        except Exception as unexpected:  # noqa: BLE001, sent to the parent as unexpected, which raises it there
            log.error("UNEXPECTED %s answering %s, may need a handler", type(unexpected).__name__, name.value, exc_info=True)
            connection.send((request_id, ReplyStatus.ERROR, (FailureKind.UNEXPECTED, traceback.format_exc())))
            continue
        connection.send((request_id, ReplyStatus.OK, payload))


def answer(device: object, name: DeviceRequest, arguments: tuple, frames: SharedFrameBuffer) -> object:
    """
    One request, answered from the device, as plain values that cross a pipe.

    :raises ValueError: For a request this function does not answer, such as a close.
    """
    match name:
        case DeviceRequest.CALIBRATION:
            calibration = device.get_calibration()
            return (
                np.asarray(calibration.scene_camera_matrix, dtype=np.float64),
                np.asarray(calibration.scene_distortion_coefficients, dtype=np.float64),
            )
        case DeviceRequest.TIME_OFFSET:
            (number_of_measurements,) = arguments
            estimates = device.estimate_time_offset(number_of_measurements=number_of_measurements)
            if estimates is None:
                return None
            return (float(estimates.time_offset_ms.median), float(estimates.roundtrip_duration_ms.median))
        case DeviceRequest.MATCHED:
            (timeout_seconds,) = arguments
            matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=timeout_seconds)
            if matched is None:
                return None
            gaze = None if matched.gaze is None else (float(matched.gaze.x), float(matched.gaze.y))
            # The pixels go into shared memory. Only where to find them crosses the pipe.
            block_name, shape, dtype = frames.put(matched.frame.bgr_pixels)
            return (block_name, shape, dtype, float(matched.frame.timestamp_unix_seconds), gaze)
        case DeviceRequest.IMU:
            (timeout_seconds,) = arguments
            datum = device.receive_imu_datum(timeout_seconds=timeout_seconds)
            if datum is None:
                return None
            quaternion = datum.quaternion
            # An empty tuple is a datum that came without a quaternion, which the source treats
            # differently from no datum at all.
            if quaternion is None:
                return ()
            return (float(quaternion.w), float(quaternion.x), float(quaternion.y), float(quaternion.z))
        case _:
            raise ValueError(f"the Neon process does not answer {name} here")


def _raise_own_priority() -> bool:
    """
    Ask the OS to run this process ahead of normal ones. Windows only, and best effort.

    The decoder keeps up easily on an idle laptop, about 100 ms behind with the head moving, and
    falls seconds behind once the estimator and the scene take every core. Ahead of them it keeps up.

    :return: Whether the priority was raised.
    :rtype: bool
    """
    if sys.platform != "win32":
        # Raising priority on Linux or macOS needs privileges a user run does not have.
        log.info("not raising the Neon process's priority on %s", sys.platform)
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declared, because ctypes otherwise passes the handle as a 32-bit int, which truncates it on
    # 64-bit Windows, and the call fails with an invalid handle.
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = (ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD)
    kernel32.SetPriorityClass.restype = ctypes.wintypes.BOOL
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS):
        log.warning("could not raise the Neon process's priority, Windows error %d", ctypes.get_last_error())
        return False
    return True


def _describe_failure(failure: BaseException) -> tuple[FailureKind, str]:
    message = f"{type(failure).__name__}: {failure}"
    if isinstance(failure, EOFError):
        return (FailureKind.ENDED, message)
    if isinstance(failure, client_failure_types()):
        return (FailureKind.DEVICE, message)
    if isinstance(failure, OSError):
        return (FailureKind.OS, message)
    return (FailureKind.VALUE, message)


def _connect_client(config: NeonConfig) -> object:
    """
    Our own receiver, on the glasses or on a capture folder, started and ready to answer.

    Not the client's simple Device. That one converts all 30 frames a second on its decode thread,
    and fell seconds behind with the head moving while the pipeline ran.
    """
    # Before the receiver starts, because that is where PyAV and the client are first imported.
    apply_opencv_pyav_import_workaround()

    if config.replay_dir is not None:
        log.info("playing back the capture in %s in place of the glasses", config.replay_dir)
    elif config.address is not None:
        log.info("connecting to the Neon at %s:%s", config.address, config.port)
    device = NeonStreamDevice(config)
    device.start()
    return device
