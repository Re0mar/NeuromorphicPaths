"""
Runs the slow part on the newest frame only, so the source loop never blocks.

Depth, scene and planner together take longer than a frame arrives, so processing every frame
would fall further behind forever. The old monolith's Worker held one pending frame and let a
newer one replace it. This does the same, counts what it dropped, and refuses to let the main
loop keep running on a thread that has died.
"""

# Standard library imports
import logging
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

# Local package imports
from nav.types import DepthFrame

log = logging.getLogger(__name__)

Result = TypeVar("Result")

IDLE_SLEEP_SECONDS = 0.002
# A camera pointed at the ceiling for a minute is thirty seconds of "no floor" at every frame. One
# warning per ten says so without filling the log with a line per frame.
CONSECUTIVE_SKIPS_PER_WARNING = 10


class NewestFrameWorker(threading.Thread, Generic[Result]):
    """A thread that always works on the most recently submitted frame."""

    def __init__(self, process: Callable[[DepthFrame], Result], on_dropped: Callable[[DepthFrame], None] | None = None) -> None:
        """
        :param process: Runs on this thread for each frame taken.
        :param on_dropped: Called on the submitting thread with each frame replaced before it was
            taken. Must not block, because that thread is the one reading frames.
        """
        super().__init__(name="pipeline-worker", daemon=True)
        self._process = process
        self._on_dropped = on_dropped
        self._lock = threading.Lock()
        self._pending: DepthFrame | None = None
        self._latest: Result | None = None
        self._failure: BaseException | None = None
        self._busy = False
        self._stopping = threading.Event()
        self.submitted = 0
        self.processed = 0
        self.dropped = 0
        self.skipped = 0
        self._consecutive_skips = 0

    def submit(self, frame: DepthFrame) -> None:
        """Hand over a frame. If the previous one was never picked up, it is dropped and counted."""
        with self._lock:
            replaced = self._pending
            if replaced is not None:
                self.dropped += 1
            self._pending = frame
            self.submitted += 1
        # Outside the lock, so the callback can never hold up the worker taking the new frame.
        if replaced is not None and self._on_dropped is not None:
            self._on_dropped(replaced)

    def latest_result(self) -> Result | None:
        """
        The newest result, or None before the first one.

        :raises: Whatever unexpected exception ended the thread, so the loop cannot carry on
            publishing a stale path from a worker that is no longer working.
        """
        if self._failure is not None:
            raise self._failure
        return self._latest

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stopping.set()
        if self.is_alive():
            self.join(timeout_seconds)

    def wait_until_idle(self, timeout_seconds: float = 5.0) -> bool:
        """
        Block until no frame is pending and none is being processed, or the timeout passes.

        The loop calls this when a source ends, so the last frame the source produced is planned
        and published before the loop waits for the next connection. Stopping the worker instead
        would end the thread, and a reconnect would need a new one with fresh counters.

        :param timeout_seconds: How long to wait at most.
        :return: True once idle, or once the thread has ended. False when the timeout passed first.
        :rtype: bool
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._lock:
                idle = self._pending is None and not self._busy
            if idle or not self.is_alive():
                return True
            time.sleep(IDLE_SLEEP_SECONDS)
        return False

    def _set_idle(self) -> None:
        with self._lock:
            self._busy = False

    def run(self) -> None:
        while True:
            with self._lock:
                frame, self._pending = self._pending, None
                self._busy = frame is not None
            if frame is None:
                # Stopping drains: a frame handed over just before stop() is still processed,
                # so the loop can publish the last thing the source produced. Only an empty
                # queue ends the thread.
                if self._stopping.is_set():
                    return
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            try:
                result = self._process(frame)
            except ValueError as degenerate_error:
                # A frame the scene could not use, such as one with no floor and no previous plane.
                # Expected at the start of a run and on a camera pointed at nothing. Costs this
                # frame and the loop continues.
                self.skipped += 1
                self._consecutive_skips += 1
                if self._consecutive_skips % CONSECUTIVE_SKIPS_PER_WARNING == 0:
                    log.warning("%d consecutive frames skipped (caught ValueError, expected): %s", self._consecutive_skips, degenerate_error)
                else:
                    log.debug("frame skipped (caught ValueError, expected): %s", degenerate_error)
                self._set_idle()
                continue
            except Exception as unexpected_error:
                log.error("UNEXPECTED %s in the worker, may need a handler", type(unexpected_error).__name__, exc_info=True)
                self._failure = unexpected_error
                self._set_idle()
                return

            self._consecutive_skips = 0
            self.processed += 1
            self._latest = result
            # Idle is declared after the result is in place, so a waiter that wakes up finds it.
            self._set_idle()
