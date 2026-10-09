"""
The glasses and a demo capture behind one picture source, and a switch between them while a run goes on.

A demo should look live, and the glasses can fail in the room. So one run holds both: the glasses
link and a capture to replay, with the page choosing which one the laptop plans on. The depth model,
the scene, the planner and the displays are built once, and only the pictures change.

Each side runs on a thread of its own, because a source only gives control back when it has a
frame, and glasses that have dropped out never do. Only the chosen side's frames reach the planner,
newest first, and only its video reaches the page.

- The glasses stay connected while the demo plays, so switching back is immediate. Their frames are
  read and dropped meanwhile, and their IMU stream stays claimed. A link that won't come up, or
  drops, is retried every few seconds, and the state says why.
- The demo starts from the beginning every time it is chosen, and loops when it ends.

Every switch, and every time the demo starts over, begins a new generation. Frames carry it, and the
loop clears what the scene and the planner remember from the last one, since a floor and a previous
plan from another room would steer the first frames of this one.
"""

# Standard library imports
import dataclasses
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum

# Local package imports
from nav.sources.config import SourceMode
from nav.sources.rgb import RgbFrame, RgbSource
from nav.sources.scene_video import AccessUnit, SceneVideoFeed, VideoDescription
from nav.types import DepthFrame, DepthFrameSource

log = logging.getLogger(__name__)

# How long the frames generator waits for a frame before looking again. Short, so a stop lands quickly.
FRAME_POLL_SECONDS = 0.25
# How long the glasses side waits after a failed connect before trying again.
GLASSES_RETRY_SECONDS = 5.0
# How long a closing side may take to finish.
CLOSE_JOIN_SECONDS = 5.0
# What a side's source can fail with: no glasses on the network, a calibration they won't give, a
# capture that won't read. The neon_live source raises these as ConnectionError, OSError and
# ValueError, which covers its own NeonDeviceError and NeonCalibrationError.
SOURCE_FAILURES: tuple[type[BaseException], ...] = (ConnectionError, OSError, ValueError, EOFError)


class GlassesLink(Enum):
    """Where the glasses side is. Values are the strings the page shows."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class SwitchState:
    """What every page shows: the side in use, and how the glasses are doing whichever side that is."""

    mode: SourceMode
    glasses: GlassesLink
    # Why the glasses are unreachable, for the page to say. None otherwise.
    glasses_detail: str | None = None


class _VideoGate:
    """One side's video, passed on to the page's feed only while that side is chosen."""

    def __init__(self, switch: "SwitchableRgbSource", mode: SourceMode) -> None:
        self._switch = switch
        self._mode = mode
        self.description: VideoDescription | None = None

    def describe(self, description: VideoDescription) -> None:
        self.description = description
        if self._switch.mode is self._mode:
            self._switch.video_feed_describe(description)

    def offer(self, unit: AccessUnit) -> None:
        if self._switch.mode is self._mode:
            self._switch.video_feed_offer(unit)


class SwitchableRgbSource:
    """Yields the chosen side's frames, newest first, until closed. Switched by `request`, from any thread."""

    def __init__(
        self,
        glasses_factory: Callable[[SceneVideoFeed | None], RgbSource],
        demo_factory: Callable[[SceneVideoFeed | None], RgbSource],
        initial: SourceMode,
        video_feed: SceneVideoFeed | None = None,
        glasses_retry_seconds: float = GLASSES_RETRY_SECONDS,
    ) -> None:
        """
        :param glasses_factory: Builds the glasses source, given the feed its video should go to.
        :param demo_factory: Builds a fresh demo source, given the same. Called on every start over.
        :param initial: The side planned on from the start.
        :param video_feed: The page's video feed, or None when no display shows video.
        :param glasses_retry_seconds: How long to wait between attempts to reach the glasses.
        """
        self._factories = {SourceMode.GLASSES: glasses_factory, SourceMode.DEMO: demo_factory}
        self._video_feed = video_feed
        self._retry_seconds = glasses_retry_seconds
        self._lock = threading.Condition()
        self._mode = initial
        self._generation = 0
        self._glasses = GlassesLink.CONNECTING
        self._glasses_detail: str | None = None
        self._newest: RgbFrame | None = None
        self._stopping = False
        self._listeners: list[Callable[[SwitchState], None]] = []
        self._sources: dict[SourceMode, RgbSource | None] = {SourceMode.GLASSES: None, SourceMode.DEMO: None}
        # Each side's feed carries its video into a gate, which forwards it only while that side is chosen.
        self._gates = {mode: _VideoGate(self, mode) for mode in SourceMode}
        self._threads: list[threading.Thread] = []

    @property
    def mode(self) -> SourceMode:
        with self._lock:
            return self._mode

    @property
    def generation(self) -> int:
        """The stretch of video now being handed over. Frames from an earlier one are stale."""
        with self._lock:
            return self._generation

    @property
    def video_feed(self) -> SceneVideoFeed | None:
        """The page's video feed, which only the chosen side's video reaches."""
        return self._video_feed

    def state(self) -> SwitchState:
        with self._lock:
            return SwitchState(self._mode, self._glasses, self._glasses_detail)

    def subscribe(self, listener: Callable[[SwitchState], None]) -> Callable[[], None]:
        """
        Call the listener with every new state, now and on each change, and return the call that stops it.

        Listeners are called on whichever thread changed the state, so one must hand off rather than block.
        """
        with self._lock:
            self._listeners.append(listener)
        listener(self.state())

        def unsubscribe() -> None:
            with self._lock:
                self._listeners = [existing for existing in self._listeners if existing is not listener]

        return unsubscribe

    def request(self, mode: SourceMode) -> None:
        """Plan on the given side from the next frame on. Choosing the side already in use does nothing."""
        with self._lock:
            if mode is self._mode:
                return
            self._mode = mode
            self._generation += 1
            self._newest = None
            self._lock.notify_all()
        log.info("switched to the %s", mode.value)
        # The page's decoder starts over on the new stream's description. A side that hasn't
        # described its stream yet does so when it starts.
        description = self._gates[mode].description
        if description is not None and mode is SourceMode.GLASSES:
            self.video_feed_describe(description)
        self._announce()

    def frames(self) -> Iterator[RgbFrame]:
        self._start_threads()
        while True:
            with self._lock:
                while self._newest is None and not self._stopping:
                    self._lock.wait(FRAME_POLL_SECONDS)
                if self._stopping:
                    return
                frame, self._newest = self._newest, None
            yield frame

    def close(self) -> None:
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        # Closing a source ends its side's blocked receive, so the threads can finish. Each source is
        # closed by whoever takes it first, this or its own thread, never by both.
        for mode in SourceMode:
            source = self._take_source(mode, None)
            if source is not None:
                self._close_quietly(source)
        for thread in self._threads:
            thread.join(CLOSE_JOIN_SECONDS)
            if thread.is_alive():
                log.warning("the %s side did not stop within %.0f s", thread.name, CLOSE_JOIN_SECONDS)
        self._threads = []

    def video_feed_describe(self, description: VideoDescription) -> None:
        if self._video_feed is not None:
            self._video_feed.describe(description)

    def video_feed_offer(self, unit: AccessUnit) -> None:
        if self._video_feed is not None:
            self._video_feed.offer(unit)

    def _start_threads(self) -> None:
        if self._threads:
            return
        for mode, run in ((SourceMode.GLASSES, self._run_glasses), (SourceMode.DEMO, self._run_demo)):
            thread = threading.Thread(target=run, name=f"{mode.value}-source", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _side_feed(self, mode: SourceMode) -> SceneVideoFeed | None:
        """A feed for one side, wired to that side's gate. None when no display wants video."""
        if self._video_feed is None:
            return None
        feed = SceneVideoFeed()
        feed.subscribe(self._gates[mode])
        return feed

    def _run_glasses(self) -> None:
        """The glasses side: connect, stream for as long as the link holds, retry when it doesn't."""
        while not self._is_stopping():
            source = self._factories[SourceMode.GLASSES](self._side_feed(SourceMode.GLASSES))
            self._put_source(SourceMode.GLASSES, source)
            self._set_glasses(GlassesLink.CONNECTING, None)
            first_frame = True
            try:
                for frame in source.frames():
                    if self._is_stopping():
                        return
                    if first_frame:
                        first_frame = False
                        self._set_glasses(GlassesLink.CONNECTED, None)
                        # A link that came back is a new stretch of video for whoever is planning on it.
                        self._next_generation_if(SourceMode.GLASSES)
                    self._deposit(SourceMode.GLASSES, frame)
                reason = "the glasses stream ended"
            except SOURCE_FAILURES as failure:
                if self._is_stopping():
                    return
                # Expected in a room where the glasses are off, or on another network. The run goes on
                # with the demo, and the page says why the glasses aren't there.
                reason = f"{type(failure).__name__}: {failure}"
                log.warning("the glasses are unreachable, trying again in %.0f s (caught %s, expected): %s", self._retry_seconds, type(failure).__name__, failure)
            except Exception as unexpected_error:
                if self._is_stopping():
                    # A source closed under its own thread at shutdown can fail any way at all.
                    return
                # Nobody named this one. Kept as unexpected with its traceback, and the glasses are
                # retried rather than the demo taken down with them.
                reason = f"{type(unexpected_error).__name__}: {unexpected_error}"
                log.error("UNEXPECTED %s on the glasses side, may need a handler, trying again", type(unexpected_error).__name__, exc_info=True)
            finally:
                if self._take_source(SourceMode.GLASSES, source) is not None:
                    self._close_quietly(source)
            self._set_glasses(GlassesLink.UNREACHABLE, reason)
            with self._lock:
                self._lock.wait_for(lambda: self._stopping, self._retry_seconds)

    def _run_demo(self) -> None:
        """The demo side: idle until chosen, then play from the start, looping, until another side is chosen."""
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._stopping or self._mode is SourceMode.DEMO)
                if self._stopping:
                    return
            source = self._factories[SourceMode.DEMO](self._side_feed(SourceMode.DEMO))
            self._put_source(SourceMode.DEMO, source)
            # A fresh start is a new stretch of video, even when the demo was already chosen.
            self._next_generation_if(SourceMode.DEMO)
            log.info("playing the demo from the start")
            try:
                for frame in source.frames():
                    if self._is_stopping() or self.mode is not SourceMode.DEMO:
                        break
                    self._deposit(SourceMode.DEMO, frame)
            except SOURCE_FAILURES as failure:
                if self._is_stopping():
                    return
                # A demo that won't play is a setup fault, a capture that is missing or malformed,
                # and retrying it would only repeat it. The run stays up so the glasses can still be
                # chosen, and choosing the demo again tries once more.
                log.error("the demo stopped (caught %s), choose it again to retry: %s", type(failure).__name__, failure)
                self._wait_until_demo_is_left()
            except Exception as unexpected_error:
                if self._is_stopping():
                    # A source closed under its own thread at shutdown can fail any way at all.
                    return
                log.error("UNEXPECTED %s on the demo side, may need a handler, choose it again to retry", type(unexpected_error).__name__, exc_info=True)
                self._wait_until_demo_is_left()
            finally:
                if self._take_source(SourceMode.DEMO, source) is not None:
                    self._close_quietly(source)

    def _wait_until_demo_is_left(self) -> None:
        # Without this a failed demo would start over at once, and fail again, as fast as it can.
        with self._lock:
            self._lock.wait_for(lambda: self._stopping or self._mode is not SourceMode.DEMO)

    def _put_source(self, mode: SourceMode, source: RgbSource) -> None:
        with self._lock:
            self._sources[mode] = source

    def _take_source(self, mode: SourceMode, source: RgbSource | None) -> RgbSource | None:
        """
        Take a side's source out, to close it. Returns None when someone else already took it.

        :param source: The source the caller holds, or None for whatever is there.
        """
        with self._lock:
            current = self._sources[mode]
            if current is None or (source is not None and current is not source):
                return None
            self._sources[mode] = None
            return current

    def _next_generation_if(self, mode: SourceMode) -> None:
        with self._lock:
            if self._mode is mode:
                self._generation += 1
                self._newest = None

    def _deposit(self, mode: SourceMode, frame: RgbFrame) -> None:
        """Keep the frame as the newest, if its side is chosen. Otherwise it is read and let go."""
        with self._lock:
            if self._mode is not mode:
                return
            self._newest = dataclasses.replace(frame, source_generation=self._generation)
            self._lock.notify_all()

    def _set_glasses(self, link: GlassesLink, detail: str | None) -> None:
        with self._lock:
            if (link, detail) == (self._glasses, self._glasses_detail):
                return
            self._glasses, self._glasses_detail = link, detail
        self._announce()

    def _announce(self) -> None:
        state = self.state()
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener(state)

    def _is_stopping(self) -> bool:
        with self._lock:
            return self._stopping

    def _close_quietly(self, source: RgbSource) -> None:
        try:
            source.close()
        except SOURCE_FAILURES as failure:
            # Closing a side that already failed can fail again. The side is gone either way.
            log.info("closing a side raised (caught %s, expected): %s", type(failure).__name__, failure)


class CurrentStretchOnly:
    """
    Depth frames from the stretch the switch is on, and none from one it has left.

    The depth model takes about half a second a frame, so a frame from the old side can still be on
    its way when the switch turns. Planned, it would put one arrow from the old side on the page after
    the switch. Dropped here, after the model and before the planner, it never reaches either.
    """

    def __init__(self, source: DepthFrameSource, switch: SwitchableRgbSource) -> None:
        self._source = source
        self._switch = switch

    @property
    def switch(self) -> SwitchableRgbSource:
        """The switch whose stretch is let through, the same one the page turns."""
        return self._switch

    def frames(self) -> Iterator[DepthFrame]:
        for frame in self._source.frames():
            if frame.source_generation == self._switch.generation:
                yield frame

    def close(self) -> None:
        self._source.close()
