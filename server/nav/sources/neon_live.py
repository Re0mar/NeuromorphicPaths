"""
A Pupil Labs Neon streaming over the network, as an RGB source.

The connection itself lives in a child process, in neon_device.py. Here, the source asks that
process for the newest frame when it wants one. Run in this process, the client's video decoder
lost the GIL to the estimator and the scene, and frames reached the planner 4 to 12 s old.

Scene frames drive the loop. The IMU runs at its own, faster rate on a separate stream, so it is
polled without blocking and the most recent orientation is carried forward onto whichever scene
frame arrives next.

Every frame is undistorted with the device's own calibration before it leaves here, and carries the
undistorted camera matrix. The scene camera's lens is wide enough that guessing a field of view
instead puts obstacles at the edges of the image in the wrong place sideways.
"""

# Standard library imports
import logging
from collections.abc import Iterator

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.pose.imu_orientation import is_usable_orientation, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import CameraModelError, Undistorter, undistorter_for
from nav.sources.config import NeonConfig
from nav.sources.neon_camera import NEON_SCENE_SIZE
from nav.sources.neon_device import (
    DeviceMatched,
    NeonDeviceError,
    NeonDeviceProcess,
    NeonStreamEnded,
    NeonUnexpectedFailure,
)
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
# The failures a request to the device can end in. NeonDeviceError is a ConnectionError, named
# anyway so the tuple says what it means.
DEVICE_FAILURES: tuple[type[BaseException], ...] = (NeonDeviceError, OSError, ValueError)


class NeonCalibrationError(ValueError):
    """The device could not give us its camera calibration, so the run cannot place anything."""


class NeonLiveRgbSource:
    """Yields undistorted RGB frames, gaze and a mounted pose from a Neon on the network."""

    def __init__(self, config: NeonConfig) -> None:
        self._config = config
        self._device: NeonDeviceProcess | None = None
        self._latest_orientation_wxyz: np.ndarray | None = None
        self._calibration_matrix: np.ndarray | None = None
        self._distortion_coefficients: np.ndarray | None = None
        self._undistorter: Undistorter | None = None
        # Laptop clock minus Neon clock, in seconds. None until measured, and for good when the
        # Companion app cannot answer, in which case capture times are left out.
        self._clock_offset_seconds: float | None = None
        self._clock_offset_measured = False
        self._warned_about_empty_imu = False

    @property
    def undistorter(self) -> Undistorter | None:
        """The straightening built on the first frame, for a hardware check to report. None before it."""
        return self._undistorter

    @property
    def clock_offset_seconds(self) -> float | None:
        """Laptop clock minus Neon clock as measured at connect, or None when it could not be."""
        return self._clock_offset_seconds

    def _connect(self) -> NeonDeviceProcess:
        if self._device is not None:
            return self._device

        device = NeonDeviceProcess(self._config)
        try:
            device.start()
        except BaseException:
            # A child that started and then failed to connect is still a process. Leave none behind.
            device.close()
            raise
        self._device = device
        return device

    def _read_calibration(self, device: NeonDeviceProcess) -> None:
        """Ask the device for its scene camera's matrix and distortion, once per connection."""
        try:
            calibration = device.get_calibration()
            camera_matrix = np.asarray(calibration.scene_camera_matrix, dtype=np.float64).reshape(3, 3)
            distortion = np.asarray(calibration.scene_distortion_coefficients, dtype=np.float64).reshape(-1)
        except DEVICE_FAILURES as failure:
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

        if image_shape != NEON_SCENE_SIZE:
            # A different streaming resolution. Distortion coefficients are unitless and stay.
            log.info("scene frames are %dx%d, scaling the %dx%d calibration to match", image_shape[1], image_shape[0], NEON_SCENE_SIZE[1], NEON_SCENE_SIZE[0])
        try:
            self._undistorter = undistorter_for(self._calibration_matrix, self._distortion_coefficients, NEON_SCENE_SIZE, image_shape)
        except CameraModelError as unusable:
            raise NeonCalibrationError(f"the Neon's calibration cannot describe a camera: {unusable}") from unusable

        log.info(
            "undistorting with the device calibration: %.1f deg wide before, %.1f deg after the crop",
            self._undistorter.field_of_view_before_degrees,
            self._undistorter.field_of_view_after_degrees,
        )
        return self._undistorter

    def _measure_clock_offset(self, device: NeonDeviceProcess, measurement_count: int) -> float | None:
        """
        The laptop's clock minus the Neon's, by the client's Time Echo protocol, or None.

        Bounded in the device process, which gives up after `time_echo_timeout_seconds` and answers
        None. On a replay nothing is measured, and the answer is the offset stored with the capture.
        """
        try:
            outcome = device.estimate_time_offset(number_of_measurements=measurement_count)
        except DEVICE_FAILURES as failure:
            log.warning("the clock offset could not be measured (caught %s, expected): %s", type(failure).__name__, failure)
            return None
        replaying = self._config.replay_dir is not None
        if outcome is None:
            if replaying:
                log.warning("the capture was recorded without a clock offset, so latency is measured from arrival only")
            else:
                # The client returns None when the Companion app predates Time Echo, and the device
                # process does when the phone did not answer in time.
                log.warning("the Companion app did not answer Time Echo, so latency is measured from arrival only")
            return None
        offset_seconds = outcome.time_offset_ms.median / 1000.0
        if replaying:
            log.info(
                "clock offset %.1f ms (laptop minus Neon), the one recorded with the capture, not measured now",
                outcome.time_offset_ms.median,
            )
        else:
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
            try:
                matched = self._receive_matched(device)
            except NeonStreamEnded:
                # Only a played-back capture ends. The run finishes the way a recording does.
                log.info("the capture has been played to the end")
                return
            arrival_seconds = laptop_time_seconds()
            self._poll_imu(device)

            image_rgb = cv2.cvtColor(matched.frame.bgr_pixels, cv2.COLOR_BGR2RGB)
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
                capture_seconds = matched.frame.timestamp_unix_seconds + self._clock_offset_seconds

            yield RgbFrame(
                timestamp_seconds=matched.frame.timestamp_unix_seconds,
                image_rgb=undistorter.undistort_image(image_rgb),
                gaze_pixel=gaze_pixel,
                pose=pose,
                camera_matrix=undistorter.camera_matrix,
                timing=FrameTiming(capture_seconds=capture_seconds, arrival_seconds=arrival_seconds, depth_ready_seconds=None),
            )

    def _receive_matched(self, device: NeonDeviceProcess) -> DeviceMatched:
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

    def _poll_imu(self, device: NeonDeviceProcess) -> None:
        """Take the newest IMU reading if one is waiting, otherwise keep the previous one."""
        datum = device.receive_imu_datum(timeout_seconds=IMU_POLL_TIMEOUT_SECONDS)
        if datum is None or datum.quaternion is None:
            return

        # Read by field name rather than by position. The client exposes w, x, y and z explicitly,
        # so there is no order to guess at, and guessing wrong would flip pitch and quietly break
        # the floor fit.
        quaternion = datum.quaternion
        orientation = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
        if not is_usable_orientation(orientation):
            # The glasses sent nothing but zero quaternions for minutes at a time on 2026-10-05.
            # A zero is no orientation at all, so it is skipped like a reading without one, and the
            # last real orientation carries on. Letting it through ended the run on its first frame.
            if not self._warned_about_empty_imu:
                log.warning("the Neon's IMU is sending empty orientations, frames carry the last real one or none")
                self._warned_about_empty_imu = True
            return
        self._latest_orientation_wxyz = orientation

    def close(self) -> None:
        if self._device is None:
            return
        try:
            # A replay reads back the offset stored with the capture, so a drift would be that
            # value compared with itself.
            if self._clock_offset_seconds is not None and self._config.replay_dir is None:
                self._log_clock_drift(self._device)
        except NeonUnexpectedFailure:
            # Logged and not raised. Close runs in the loop's finally, and raising there would skip
            # the end-of-run report for a check that only informs.
            log.error("UNEXPECTED failure measuring the clock offset at close, may need a handler", exc_info=True)
        finally:
            self._device.close()
            self._device = None

    def _log_clock_drift(self, device: NeonDeviceProcess) -> None:
        """
        The end-of-walk check. The offset was measured once at connect and every capture time leans
        on it, so how far it moved over the walk is how far those times can be off.
        """
        closing_offset = self._measure_clock_offset(device, CLOSE_OFFSET_MEASUREMENTS)
        if closing_offset is None:
            return
        log.info(
            "clock offset drifted %.1f ms over the run, %.1f ms at connect and %.1f ms at close",
            (closing_offset - self._clock_offset_seconds) * 1000.0,
            self._clock_offset_seconds * 1000.0,
            closing_offset * 1000.0,
        )
