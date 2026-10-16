"""
A position tracker run in a process of its own, so a call that hangs can be killed.

Open3D's dense SLAM hung inside its own calls on some runs, on both its CPU and CUDA builds. On the
GPU a hung call also held up the depth model, which shares the card, and the whole pipeline froze.
A thread can't be stopped, but a process can, and killing it frees its GPU work. A call past the
time limit kills the process and starts a fresh one in the background. Frames go on without a
position until it is ready.

The new process starts its world at the camera, where the old one had it at the last position it
reported. Both worlds share the IMU's axes, so that last position is added to everything the new
one reports, and the scene's world-fixed cells stay on the same places. After `max_restarts` the
tracker is given up on for the run.
"""

# Standard library imports
import logging
import multiprocessing
import threading
from collections.abc import Callable
from enum import Enum
from multiprocessing.connection import Connection

# Third party imports
import numpy as np

# Local package imports
from nav.pose.config import PositionConfig
from nav.pose.tracking import PositionTracker, TrackingOutcome
from nav.types import DepthFrame, Pose

log = logging.getLogger(__name__)

# How long close() waits for the process to finish on its own before it's killed.
CLOSE_WAIT_SECONDS = 3.0


class TrackerMessage(Enum):
    """What crosses the pipe, first in every message."""

    READY = "ready"
    TRACK = "track"
    CLOSE = "close"
    OUTCOME = "outcome"


def _serve_tracker(connection: Connection, build: Callable[[PositionConfig], PositionTracker], config: PositionConfig) -> None:
    """The process's whole life: build the tracker, then answer one frame at a time until told to stop."""
    tracker = build(config)
    connection.send((TrackerMessage.READY,))
    while True:
        message = connection.recv()
        match message[0]:
            case TrackerMessage.TRACK:
                outcome = tracker.track(_frame_from_payload(message[1]))
                connection.send((TrackerMessage.OUTCOME, outcome.position, outcome.fitness))
            case TrackerMessage.CLOSE:
                tracker.close()
                return
            case _:
                raise ValueError(f"the tracker process got a message it doesn't know: {message[0]}")


def _payload(frame: DepthFrame) -> dict:
    # Only what a tracker reads. The rest of the frame stays with the pipeline.
    return {
        "timestamp_seconds": frame.timestamp_seconds,
        "depth_meters": frame.depth_meters,
        "intrinsics": frame.intrinsics,
        "orientation": frame.pose.orientation,
        "gravity_aligned": frame.pose.orientation_is_gravity_aligned,
    }


def _frame_from_payload(payload: dict) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=payload["timestamp_seconds"],
        depth_meters=payload["depth_meters"],
        intrinsics=payload["intrinsics"],
        pose=Pose(orientation=payload["orientation"], position=None, has_position=False, orientation_is_gravity_aligned=payload["gravity_aligned"]),
        ground_plane=None,
        gaze_pixel=None,
    )


class IsolatedTracker:
    """A PositionTracker whose work happens in a child process that is killed and replaced when it hangs."""

    def __init__(self, build: Callable[[PositionConfig], PositionTracker], config: PositionConfig) -> None:
        """
        :param build: Makes the real tracker inside the child. Must be a module-level function, since
            the child is spawned and gets it by import.
        :param config: Handed to build. Its stall_seconds is the time limit per frame, and its
            max_restarts the number of replacements before the tracker is given up on.
        """
        self._build = build
        self._config = config
        self._context = multiprocessing.get_context("spawn")
        # close() can come from the pipeline's thread while a frame is being tracked on another.
        self._lock = threading.Lock()
        self._process = None
        self._connection: Connection | None = None
        self._ready = False
        self._given_up = False
        # Added to every position the current process reports, so its world lines up with the last.
        self._offset = np.zeros(3)
        self._last_position = np.zeros(3)
        self.restarts = 0
        self._start()

    @property
    def given_up(self) -> bool:
        """Whether the restarts ran out, so no frame of this run will get a position from it again."""
        return self._given_up

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        with self._lock:
            if self._given_up or self._process is None:
                return TrackingOutcome(None, 0.0)
            if not self._ready and not self._became_ready():
                # Still starting. Waiting here would hold up the frame for the seconds a CUDA start takes.
                return TrackingOutcome(None, 0.0)
            try:
                self._connection.send((TrackerMessage.TRACK, _payload(frame)))
                if not self._connection.poll(self._config.stall_seconds):
                    self._replace(f"hung for over {self._config.stall_seconds:.1f} s")
                    return TrackingOutcome(None, 0.0)
                _, position, fitness = self._connection.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError) as gone:
                # The process died on its own, a crash inside the library. Same treatment as a hang.
                self._replace(f"ended (caught {type(gone).__name__}, expected)")
                return TrackingOutcome(None, 0.0)
            if position is None:
                return TrackingOutcome(None, fitness)
            self._last_position = position + self._offset
            return TrackingOutcome(self._last_position.copy(), fitness)

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            try:
                if self._ready:
                    self._connection.send((TrackerMessage.CLOSE,))
            except (BrokenPipeError, ConnectionResetError, OSError) as gone:
                log.debug("the position tracker's process was already gone at close (caught %s, expected)", type(gone).__name__)
            self._process.join(CLOSE_WAIT_SECONDS)
            self._stop()

    def _start(self) -> None:
        parent_end, child_end = self._context.Pipe()
        self._process = self._context.Process(
            target=_serve_tracker, args=(child_end, self._build, self._config), name="position-tracker", daemon=True,
        )
        self._process.start()
        child_end.close()
        self._connection = parent_end
        self._ready = False

    def _stop(self) -> None:
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        self._connection.close()
        self._process = None

    def _became_ready(self) -> bool:
        try:
            if self._connection.poll(0):
                message = self._connection.recv()
                self._ready = message[0] == TrackerMessage.READY
        except (EOFError, BrokenPipeError, ConnectionResetError) as gone:
            self._replace(f"ended while starting (caught {type(gone).__name__}, expected)")
        return self._ready

    def _replace(self, what_happened: str) -> None:
        """Kill the process, and start another unless the run has used up its replacements."""
        self._stop()
        if self.restarts >= self._config.max_restarts:
            self._given_up = True
            log.error(
                "the position tracker's process %s after %d restarts and is given up on. Frames go on without a position for the rest of the run",
                what_happened,
                self.restarts,
            )
            return
        self.restarts += 1
        self._offset = self._last_position.copy()
        log.warning("the position tracker's process %s and was restarted, carrying its last position", what_happened)
        self._start()
