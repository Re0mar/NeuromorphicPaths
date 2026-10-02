"""
A Pupil Labs Neon streaming over the network, as an RGB source.

The only file allowed to import the Pupil Labs client, so that the rest of the package installs
and imports without it. A guard test enforces that.

Scene frames drive the loop. The IMU runs at its own, faster rate on a separate stream, so it is
polled without blocking and the most recent orientation is carried forward onto whichever scene
frame arrives next.
"""

# Standard library imports
import logging
from collections.abc import Iterator

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.sources.config import NeonConfig
from nav.sources.rgb import RgbFrame

log = logging.getLogger(__name__)

# Long enough that a dropped IMU packet does not stall the scene stream, short enough that the
# orientation never lags a frame behind. The IMU runs far faster than the scene camera.
IMU_POLL_TIMEOUT_SECONDS = 0.0


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
    """Yields RGB frames, gaze and IMU orientation from a Neon on the network."""

    def __init__(self, config: NeonConfig) -> None:
        self._config = config
        self._device = None
        self._latest_orientation_wxyz: np.ndarray | None = None

    def _connect(self):
        if self._device is not None:
            return self._device

        apply_opencv_pyav_import_workaround()

        # Optional dependency. Absent in any environment installed without the glasses extra.
        from pupil_labs.realtime_api.simple import Device, discover_one_device

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

    def frames(self) -> Iterator[RgbFrame]:
        device = self._connect()

        while True:
            matched = device.receive_matched_scene_and_eyes_video_frames_and_gaze()
            if matched is None:
                # The stream yielded nothing this round. Expected on a congested network, costs
                # one frame, and the loop keeps the device open rather than tearing it down.
                log.debug("no matched scene frame this round")
                continue

            self._poll_imu(device)

            gaze_pixel = None
            if matched.gaze is not None:
                gaze_pixel = np.array([matched.gaze.x, matched.gaze.y])

            yield RgbFrame(
                timestamp_seconds=matched.scene.timestamp_unix_seconds,
                image_rgb=cv2.cvtColor(matched.scene.bgr_pixels, cv2.COLOR_BGR2RGB),
                gaze_pixel=gaze_pixel,
                imu_orientation_wxyz=self._latest_orientation_wxyz,
            )

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
        if self._device is not None:
            self._device.close()
            self._device = None
