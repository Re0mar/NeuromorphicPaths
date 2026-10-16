"""
A position tracker run in a process of its own, so a call that hangs can be killed.

Open3D's dense SLAM hung inside its own calls on some runs, on both its CPU and CUDA builds. On the
GPU a hung call also held up the depth model, which shares the card, and the whole pipeline froze.
A thread can't be stopped, but a process can, and killing it frees its GPU work. So a call past the
time limit kills the process and starts a fresh one in the background. The world starts over in the
new one, which the scene meets as new cells, and frames go on without a position until it's ready.
"""

# Standard library imports
import logging
import multiprocessing
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
    RESET = "reset"
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
            case TrackerMessage.RESET:
                tracker.reset()
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
        :param config: Handed to build, and its stall_seconds is the time limit per frame.
        """
        self._build = build
        self._config = config
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._connection: Connection | None = None
        self._ready = False
        self.restarts = 0
        self._start()

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        if not self._ready and not self._became_ready():
            # Still starting. Waiting here would hold up the frame for the seconds a CUDA start takes.
            return TrackingOutcome(None, 0.0)
        try:
            self._connection.send((TrackerMessage.TRACK, _payload(frame)))
            if not self._connection.poll(self._config.stall_seconds):
                log.warning(
                    "the position tracker hung for over %.1f s and was restarted, its world starts over",
                    self._config.stall_seconds,
                )
                self._restart()
                return TrackingOutcome(None, 0.0)
            _, position, fitness = self._connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError) as gone:
            # The process died on its own, a crash inside the library. Same treatment as a hang.
            log.warning("the position tracker's process ended (caught %s, expected) and was restarted", type(gone).__name__)
            self._restart()
            return TrackingOutcome(None, 0.0)
        return TrackingOutcome(position, fitness)

    def reset(self) -> None:
        if self._ready:
            self._connection.send((TrackerMessage.RESET,))

    def close(self) -> None:
        if self._process is None:
            return
        try:
            if self._ready:
                self._connection.send((TrackerMessage.CLOSE,))
        except (BrokenPipeError, ConnectionResetError, OSError) as gone:
            log.debug("the position tracker's process was already gone at close (caught %s, expected)", type(gone).__name__)
        self._process.join(CLOSE_WAIT_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        self._connection.close()
        self._process = None

    def _start(self) -> None:
        parent_end, child_end = self._context.Pipe()
        self._process = self._context.Process(
            target=_serve_tracker, args=(child_end, self._build, self._config), name="position-tracker", daemon=True,
        )
        self._process.start()
        child_end.close()
        self._connection = parent_end
        self._ready = False

    def _became_ready(self) -> bool:
        try:
            if self._connection.poll(0):
                message = self._connection.recv()
                self._ready = message[0] == TrackerMessage.READY
        except (EOFError, BrokenPipeError, ConnectionResetError) as gone:
            log.warning("the position tracker's process ended while starting (caught %s, expected) and was restarted", type(gone).__name__)
            self._restart()
        return self._ready

    def _restart(self) -> None:
        self._process.kill()
        self._process.join()
        self._connection.close()
        self.restarts += 1
        self._start()
