"""
A depth source whose frames gain a measured position, for a camera that only knows its orientation.

Composed behind the depth estimator, so the tracker sees exactly the depth the scene will. The scene
already places a positioned frame in a world that stays put, so nothing past here changes.

The tracker runs on a thread of its own. Open3D's tracking hung on some runs during the evaluation,
and a hung call can't be interrupted, so a call that takes too long is abandoned. A tracker that
fails in a way nobody predicted is treated the same. Either way that frame and every later one go
on without a position, and the walker keeps the arrow.
"""

# Standard library imports
import dataclasses
import logging
import queue
import threading
import time
from collections.abc import Iterator

# Third party imports
import numpy as np

# Local package imports
from nav.pose.device import pose_with_position
from nav.pose.tracking import PositionTracker, TrackingOutcome
from nav.types import DepthFrame, DepthFrameSource

log = logging.getLogger(__name__)

# How often the running figures are logged, in seconds of wall time.
SUMMARY_INTERVAL_SECONDS = 10.0


class PositionedDepthSource:
    """Passes frames through, each with the tracker's position added when it has one."""

    def __init__(self, inner: DepthFrameSource, tracker: PositionTracker, stall_seconds: float) -> None:
        self._inner = inner
        self._tracker = tracker
        self._stall_seconds = stall_seconds
        self._requests: queue.Queue[DepthFrame | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[TrackingOutcome | BaseException] = queue.Queue(maxsize=1)
        # Set when the tracker is no longer asked: it stalled, it failed, or the source is closing.
        self._stalled = False
        self._failed = False
        self._closing = False
        self._thread = threading.Thread(target=self._serve, name="position-tracker", daemon=True)
        self._thread.start()
        self._summary = _RunningSummary()

    @property
    def inner(self) -> DepthFrameSource:
        """The source underneath, for a caller that needs what only it knows."""
        return self._inner

    def frames(self) -> Iterator[DepthFrame]:
        for frame in self._inner.frames():
            yield self._positioned(frame)

    def close(self) -> None:
        # The source first, so the frame it may still hand over while closing isn't sent to a
        # tracker that's being stopped. The live and recording switch closes from another thread.
        self._closing = True
        self._inner.close()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            # A frame is waiting, so the thread is alive and will meet the stop after it.
            log.debug("a frame was still queued for the tracker at close (caught queue.Full, expected)")
        self._thread.join(self._stall_seconds)
        if self._thread.is_alive():
            # Stuck inside a call. Closing from here stops a tracker that runs in its own process,
            # which is the one that hangs.
            log.warning("the position tracker's thread didn't stop at close, closing the tracker from here")
            self._tracker.close()
        self._summary.log_now()

    def _positioned(self, frame: DepthFrame) -> DepthFrame:
        # Without gravity the frame's orientation is the camera's own axes, not a world's, so there's
        # nothing to place a position in. It goes on as it came.
        if self._closing or self._stalled or self._failed or not frame.pose.orientation_is_gravity_aligned or frame.pose.has_position:
            return frame
        began = time.perf_counter()
        self._requests.put(frame)
        try:
            outcome = self._results.get(timeout=self._stall_seconds)
        except queue.Empty:
            self._stalled = True
            log.error(
                "the position tracker took over %.1f s on one frame and is abandoned. Frames go on without a position for the rest of the run",
                self._stall_seconds,
            )
            return frame
        if isinstance(outcome, BaseException):
            # Logged with its traceback on the tracker's thread. A position is optional, so losing the
            # tracker costs the rest of the run its position and nothing else.
            self._failed = True
            return frame
        self._summary.add(outcome, time.perf_counter() - began)
        if outcome.position is None:
            return frame
        return dataclasses.replace(frame, pose=pose_with_position(frame.pose, outcome.position))

    def _serve(self) -> None:
        try:
            while True:
                frame = self._requests.get()
                if frame is None:
                    return
                try:
                    outcome = self._tracker.track(frame)
                except Exception as unexpected:  # noqa: BLE001, reported to the pipeline's thread, which goes on without a position
                    log.error(
                        "UNEXPECTED %s in the position tracker, may need a handler. Frames go on without a position for the rest of the run",
                        type(unexpected).__name__,
                        exc_info=True,
                    )
                    self._results.put(unexpected)
                    return
                self._results.put(outcome)
        finally:
            # On this thread, not close()'s, so GPU memory is freed by the thread that allocated it.
            self._tracker.close()


class _RunningSummary:
    """Positioned share, fitness and cost, logged every interval so a run says how the tracker is doing."""

    def __init__(self) -> None:
        self._fitness: list[float] = []
        self._positioned: list[bool] = []
        self._cost_seconds: list[float] = []
        self._last_logged = time.monotonic()

    def add(self, outcome: TrackingOutcome, cost_seconds: float) -> None:
        self._fitness.append(outcome.fitness)
        self._positioned.append(outcome.position is not None)
        self._cost_seconds.append(cost_seconds)
        if time.monotonic() - self._last_logged >= SUMMARY_INTERVAL_SECONDS:
            self.log_now()

    def log_now(self) -> None:
        self._last_logged = time.monotonic()
        if not self._fitness:
            return
        log.info(
            "position tracker over %d frames: positioned %.0f %%, fitness median %.2f, %.0f ms median a frame",
            len(self._fitness),
            100.0 * float(np.mean(self._positioned)),
            float(np.median(self._fitness)),
            1000.0 * float(np.median(self._cost_seconds)),
        )
