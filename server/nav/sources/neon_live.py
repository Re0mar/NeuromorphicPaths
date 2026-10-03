"""
A Pupil Labs Neon streaming over the network, as an RGB source.

The only file allowed to import the Pupil Labs client, so that the rest of the package installs
and imports without it. A guard test enforces that.

Scene frames drive the loop. The IMU runs at its own, faster rate on a separate stream, so it is
polled without blocking and the most recent orientation is carried forward onto whichever scene
frame arrives next.

Every frame is undistorted with the device's own calibration before it leaves here, and carries the
undistorted camera matrix. The scene camera's lens is wide enough that guessing a field of view
instead puts obstacles at the edges of the image in the wrong place sideways.
"""

# Standard library imports
import logging
import threading
from collections.abc import Callable, Iterator

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.pose.imu_orientation import pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import CameraCalibration, CameraModelError, Undistorter, scale_intrinsics
from nav.sources.config import NeonConfig
from nav.sources.rgb import RgbFrame
from nav.types import FrameTiming

log = logging.getLogger(__name__)

# Time Echo round trips at connect, where the offset every capture time depends on is set, and at
# close, where it is only being checked for drift and the person is waiting for the run to end.
CONNECT_OFFSET_MEASUREMENTS = 100  # The client's own default.
CLOSE_OFFSET_MEASUREMENTS = 20

# Long enough that a dropped IMU packet does not stall the scene stream, short enough that the
# orientation never lags a frame behind. The IMU runs far faster than the scene camera.
IMU_POLL_TIMEOUT_SECONDS = 0.0
# How long one receive may block. Short, so Ctrl+C lands within a quarter of a second even when
# the stream has stopped, which a receive with no timeout does not allow.
RECEIVE_POLL_SECONDS = 0.25
# The scene camera's native size, which the device's calibration describes. The calibration buffer
# does not carry a size of its own.
NEON_SCENE_SIZE = (1200, 1600)  # height, width


class NeonCalibrationError(ValueError):
    """The device could not give us its camera calibration, so the run cannot place anything."""


def apply_opencv_pyav_import_workaround() -> None:
    """
    Open and close a one-pixel OpenCV window before the Pupil Labs client is imported.

    Works around an import-order crash between OpenCV and PyAV on Windows. The old script did this
    at module scope with no explanation. Kept because a rewrite that drops a workaround on the
    grounds that the new design makes it unnecessary rediscovers the bug a week later.
    """
    cv2.imshow("opencv pyav import order", np.zeros(1))
    cv2.destroyAllWindows()


class NeonLiveRgbSource:
    """Yields undistorted RGB frames, gaze and a mounted pose from a Neon on the network."""

    def __init__(self, config: NeonConfig) -> None:
        self._config = config
        self._device = None
        self._latest_orientation_wxyz: np.ndarray | None = None
        self._calibration_matrix: np.ndarray | None = None
        self._distortion_coefficients: np.ndarray | None = None
        self._undistorter: Undistorter | None = None
        # Laptop clock minus Neon clock, in seconds. None until measured, and for good when the
        # Companion app cannot answer, in which case capture times are left out.
        self._clock_offset_seconds: float | None = None
        self._clock_offset_measured = False
        # The failures a request to the device can end in. The client's own DeviceError joins these
        # once the client is imported in _connect, so this file stays importable without it.
        self._device_failures: tuple[type[BaseException], ...] = (OSError, ValueError)

    @property
    def undistorter(self) -> Undistorter | None:
        """The straightening built on the first frame, for a hardware check to report. None before it."""
        return self._undistorter

    @property
    def clock_offset_seconds(self) -> float | None:
        """Laptop clock minus Neon clock as measured at connect, or None when it could not be."""
        return self._clock_offset_seconds

    def _connect(self):
        if self._device is not None:
            return self._device

        apply_opencv_pyav_import_workaround()

        # Optional dependency. Absent in any environment installed without the glasses extra.
        from pupil_labs.realtime_api.device import DeviceError
        from pupil_labs.realtime_api.simple import Device, discover_one_device

        self._device_failures = (DeviceError, OSError, ValueError)

        if self._config.address is None:
            log.info("discovering a Neon, up to %.0f s", self._config.discovery_timeout_seconds)
            device = discover_one_device(max_search_duration_seconds=self._config.discovery_timeout_seconds)
            if device is None:
                # Discovery uses mDNS, which university networks routinely block between subnets.
                # That is what the old script's hard-coded address was working around.
                raise ConnectionError(
                    f"no Neon found within {self._config.discovery_timeout_seconds:.0f} s. "
                    "If the network blocks mDNS, read the address off the Companion app's "
                    "streaming screen and pass --neon-address"
                )
            log.info("discovered a Neon at %s:%s", device.address, device.port)
        else:
            log.info("connecting to the Neon at %s:%s", self._config.address, self._config.port)
            device = Device(address=self._config.address, port=self._config.port)

        self._device = device
        return device

    def _read_calibration(self, device) -> None:
        """Ask the device for its scene camera's matrix and distortion, once per connection."""
        try:
            calibration = device.get_calibration()
            camera_matrix = np.asarray(calibration.scene_camera_matrix, dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(calibration.scene_distortion_coefficients, dtype=np.float64).reshape(-1)
        except self._device_failures as failure:
            # Without it every frame would fall back to a guessed field of view, which is the
            # silent error this source exists to avoid. The loop ends the run with this message.
            raise NeonCalibrationError(
                f"the Neon did not provide its camera calibration (caught {type(failure).__name__}): {failure}"
            ) from failure
        self._calibration_matrix = camera_matrix
        self._distortion_coefficients = distortion

    def _undistorter_for(self, image_shape: tuple[int, int]) -> Undistorter:
        """Build the undistortion maps on the first frame, at that frame's size."""
        if self._undistorter is not None:
            return self._undistorter
        if self._calibration_matrix is None or self._distortion_coefficients is None:
            raise NeonCalibrationError("no calibration was read before the first frame")

        camera_matrix = self._calibration_matrix
        if image_shape != NEON_SCENE_SIZE:
            # A different streaming resolution. Distortion coefficients are unitless and stay.
            log.info("scene frames are %dx%d, scaling the %dx%d calibration to match", image_shape[1], image_shape[0], NEON_SCENE_SIZE[1], NEON_SCENE_SIZE[0])
            camera_matrix = scale_intrinsics(camera_matrix, NEON_SCENE_SIZE, image_shape)
        try:
            calibration = CameraCalibration(
                camera_matrix=camera_matrix,
                distortion_coefficients=self._distortion_coefficients,
                image_size=image_shape,
            )
        except CameraModelError as unusable:
            raise NeonCalibrationError(f"the Neon's calibration cannot describe a camera: {unusable}") from unusable

        self._undistorter = Undistorter(calibration)
        log.info(
            "undistorting with the device calibration: %.1f deg wide before, %.1f deg after the crop",
            self._undistorter.field_of_view_before_degrees,
            self._undistorter.field_of_view_after_degrees,
        )
        return self._undistorter

    def _measure_clock_offset(self, device, measurement_count: int) -> float | None:
        """
        The laptop's clock minus the Neon's, by the client's Time Echo protocol, or None.

        Bounded, because the client runs it in an asyncio.run of its own with no timeout we can
        pass, and a phone that has gone away would otherwise hold the process until it gave up.
        """
        try:
            outcome = _call_with_timeout(
                lambda: device.estimate_time_offset(number_of_measurements=measurement_count),
                self._config.time_echo_timeout_seconds,
            )
        except self._device_failures as failure:
            log.warning("the clock offset could not be measured (caught %s, expected): %s", type(failure).__name__, failure)
            return None
        if outcome is _TIMED_OUT:
            log.warning("the clock offset measurement did not return within %.0f s, abandoned", self._config.time_echo_timeout_seconds)
            return None
        if outcome is None:
            # The client returns None when the Companion app predates Time Echo.
            log.warning("the Companion app does not answer Time Echo, so latency is measured from arrival only")
            return None
        offset_seconds = outcome.time_offset_ms.median / 1000.0
        log.info(
            "clock offset %.1f ms (laptop minus Neon), round trip %.1f ms, over %d measurements",
            outcome.time_offset_ms.median,
            outcome.roundtrip_duration_ms.median,
            measurement_count,
        )
        return offset_seconds

    def frames(self) -> Iterator[RgbFrame]:
        device = self._connect()
        if self._calibration_matrix is None:
            self._read_calibration(device)
        if not self._clock_offset_measured:
            self._clock_offset_seconds = self._measure_clock_offset(device, CONNECT_OFFSET_MEASUREMENTS)
            self._clock_offset_measured = True

        while True:
            matched = self._receive_matched(device)
            arrival_seconds = laptop_time_seconds()
            self._poll_imu(device)

            image_rgb = cv2.cvtColor(matched.scene.bgr_pixels, cv2.COLOR_BGR2RGB)
            undistorter = self._undistorter_for(image_rgb.shape[:2])

            gaze_pixel = None
            if matched.gaze is not None:
                # Undistorted with the image, or it points at a different scene point. None when
                # the crop cut it off.
                gaze_pixel = undistorter.undistort_pixel(np.array([matched.gaze.x, matched.gaze.y]))

            pose = None
            if self._latest_orientation_wxyz is not None:
                pose = pose_from_imu(self._latest_orientation_wxyz, NEON_IMU_MOUNT)

            capture_seconds = None
            if self._clock_offset_seconds is not None:
                capture_seconds = matched.scene.timestamp_unix_seconds + self._clock_offset_seconds

            yield RgbFrame(
                timestamp_seconds=matched.scene.timestamp_unix_seconds,
                image_rgb=undistorter.undistort_image(image_rgb),
                gaze_pixel=gaze_pixel,
                pose=pose,
                camera_matrix=undistorter.camera_matrix,
                timing=FrameTiming(capture_seconds=capture_seconds, arrival_seconds=arrival_seconds, depth_ready_seconds=None),
            )

    def _receive_matched(self, device):
        """
        The next scene frame with its gaze, waiting in short slices.

        Scene and gaze only. The eye video is never read here, and asking for it costs bandwidth
        on the same wifi the scene video is on.
        """
        polls_without_frame = 0
        next_warning_seconds = self._config.stall_warning_seconds
        while True:
            matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=RECEIVE_POLL_SECONDS)
            if matched is not None:
                return matched
            polls_without_frame += 1
            silent_seconds = polls_without_frame * RECEIVE_POLL_SECONDS
            if silent_seconds >= next_warning_seconds:
                # Waiting rather than giving up, because a wifi hiccup in the middle of a walk
                # should not end the walk. Ctrl+C still works within one poll.
                log.warning("no scene frame from the Neon for %.0f s, still waiting", silent_seconds)
                next_warning_seconds += self._config.stall_warning_seconds

    def _poll_imu(self, device) -> None:
        """Take the newest IMU reading if one is waiting, otherwise keep the previous one."""
        datum = device.receive_imu_datum(timeout_seconds=IMU_POLL_TIMEOUT_SECONDS)
        if datum is None or datum.quaternion is None:
            return

        # Read by field name rather than by position. The client exposes w, x, y and z explicitly,
        # so there is no order to guess at, and guessing wrong would flip pitch and quietly break
        # the floor fit.
        quaternion = datum.quaternion
        self._latest_orientation_wxyz = np.array(
            [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
        )

    def close(self) -> None:
        if self._device is None:
            return
        if self._clock_offset_seconds is not None:
            # The end-of-walk check. The offset was measured once at connect and every capture time
            # leans on it, so how far it moved over the walk is how far those times can be off.
            closing_offset = self._measure_clock_offset(self._device, CLOSE_OFFSET_MEASUREMENTS)
            if closing_offset is not None:
                log.info(
                    "clock offset drifted %.1f ms over the run, %.1f ms at connect and %.1f ms at close",
                    (closing_offset - self._clock_offset_seconds) * 1000.0,
                    self._clock_offset_seconds * 1000.0,
                    closing_offset * 1000.0,
                )
        self._device.close()
        self._device = None


# Returned by _call_with_timeout when the call did not finish, so a None result stays distinguishable.
_TIMED_OUT = object()


def _call_with_timeout(function: Callable[[], object], timeout_seconds: float) -> object:
    """
    Run a blocking call on a daemon thread and wait at most timeout_seconds for it.

    A call still running at the deadline is abandoned, not cancelled. The thread is a daemon, so it
    cannot keep the process alive after the run ends.

    :return: The call's result, or _TIMED_OUT.
    :raises: Whatever the call raised, re-raised here on the caller's thread.
    """
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["value"] = function()
        except BaseException as raised:  # noqa: BLE001, handed to the caller's thread unchanged and re-raised there
            outcome["error"] = raised

    worker = threading.Thread(target=run, name="neon-time-echo", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        return _TIMED_OUT
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")
