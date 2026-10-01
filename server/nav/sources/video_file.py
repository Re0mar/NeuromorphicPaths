"""
A recording on disk or an IP camera's stream URL, read through OpenCV.

One class covers both, because OpenCV opens them the same way. That is why the design has no
separate phone_camera source: pointing this at an IP camera app's URL is the phone camera.
"""

# Standard library imports
import logging
import time
from collections.abc import Iterator
from pathlib import Path

# Third party imports
import cv2

# Local package imports
from nav.sources.rgb import RgbFrame

log = logging.getLogger(__name__)

# A URL has a scheme. Anything else is treated as a path on disk and must exist before we try.
URL_MARKER = "://"


class VideoFileRgbSource:
    """Yields RGB frames from a recording or a stream URL."""

    def __init__(self, path_or_url: str) -> None:
        self._path_or_url = path_or_url
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        if self._capture is not None:
            return self._capture

        is_url = URL_MARKER in self._path_or_url
        # A missing file and an unreachable camera are different problems with different fixes,
        # so they get different exceptions rather than one "could not open" for both.
        if not is_url and not Path(self._path_or_url).is_file():
            raise FileNotFoundError(f"no video file at {self._path_or_url}")

        capture = cv2.VideoCapture(self._path_or_url)
        if not capture.isOpened():
            capture.release()
            if is_url:
                raise ConnectionError(f"could not open the stream at {self._path_or_url}")
            raise ConnectionError(f"OpenCV could not decode {self._path_or_url}, the file exists but did not open")

        self._capture = capture
        return capture

    def frames(self) -> Iterator[RgbFrame]:
        capture = self._open()
        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
        wall_clock_start = time.monotonic()
        announced_fallback = False
        index = 0

        while True:
            received, image_bgr = capture.read()
            if not received:
                return

            position_milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
            if position_milliseconds > 0:
                timestamp_seconds = position_milliseconds / 1000.0
            elif frames_per_second > 0:
                timestamp_seconds = index / frames_per_second
            else:
                # A live stream that reports neither position nor frame rate. Wall clock is the
                # only honest answer left, and the scene's noise window is measured in seconds, so
                # a made-up frame rate here would quietly scale N.
                timestamp_seconds = time.monotonic() - wall_clock_start
                if not announced_fallback:
                    log.warning("%s reports no timestamp and no frame rate, using wall clock", self._path_or_url)
                    announced_fallback = True

            yield RgbFrame(
                timestamp_seconds=timestamp_seconds,
                image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                gaze_pixel=None,
                imu_orientation_wxyz=None,
            )
            index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
