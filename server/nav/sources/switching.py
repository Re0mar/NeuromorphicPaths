"""
The live input and a recording behind one depth source, and a switch between them while a run goes on.

A server that serves the page has to stay up whatever the hardware does. The glasses can be off, on
another network or unplugged halfway through a demo, and the Pixel may never connect. So the live
input sits behind this source, which never ends and never raises for want of a device: it keeps
trying, says what it found, and the page shows "No Input" until there is some. Alongside it a run can
hold a recording, a capture replayed through the same route as the glasses, and the page chooses
which side the laptop plans on.

Each side runs on a thread of its own, because a source only gives control back when it has a frame,
and an input that has gone quiet never does. Only the chosen side's frames reach the planner, newest
first, and only its video reaches the page. A side's frames are let go before the depth model while
it isn't chosen, so an unchosen side costs the model nothing, and both sides share one model.

- The live input stays connected while the recording plays, so switching back is immediate. A link
  that won't come up, drops, or goes quiet for a few seconds reads as no input, and is retried.
- The recording starts from the beginning every time it is chosen, and loops when it ends.

Every switch, every reconnect of the live input and every start-over of the recording begins a new
generation. Frames carry it, and the loop clears what the scene and the planner remember from the
last one, since a floor and a previous plan from another room would steer the first frames of this one.
"""

# Standard library imports
import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.sources.config import SourceMode
from nav.sources.rgb import RgbFrame, RgbSource
from nav.sources.scene_video import AccessUnit, SceneVideoFeed, VideoDescription
from nav.types import DepthFrame, DepthFrameSource

log = logging.getLogger(__name__)

# How long the frames generator waits for a frame before looking again. Short, so a stop lands
# quickly, and the stall check below runs at least this often.
FRAME_POLL_SECONDS = 0.25
# How long the live side waits after a failed connect before trying again.
LIVE_RETRY_SECONDS = 5.0
# How long the chosen live input may go without a frame before it reads as no input. The glasses
# plan about 2 frames a second and the Pixel 30, so 3 s is several missed frames on either.
LIVE_STALL_SECONDS = 3.0
# How long a closing side may take to finish.
CLOSE_JOIN_SECONDS = 5.0
# What a side's source can fail with: no glasses on the network, a phone that never connected, a
# calibration the glasses won't give, a capture that won't read. The sources raise these as
# ConnectionError, OSError and ValueError, which covers NeonDeviceError and NeonCalibrationError.
SOURCE_FAILURES: tuple[type[BaseException], ...] = (ConnectionError, OSError, ValueError, EOFError)

# Builds one side's source, given the feed its video should go to and a call that says whether
# that side is chosen right now.
SideFactory = Callable[[SceneVideoFeed | None, Callable[[], bool]], DepthFrameSource]


class InputLink(Enum):
    """How the live input is doing. Values are the strings the page reads."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    NO_INPUT = "no_input"


@dataclass(frozen=True)
class SwitchState:
    """What every page shows: the side in use, how the live input is, and what the run can offer."""

    mode: SourceMode
    live: InputLink
    # Why there is no live input, for the page to say. None otherwise.
    live_detail: str | None
    # Whether the run holds a recording to switch to. Without one the page shows no switch.
    has_recording: bool
    # Whether the live input sends video. The glasses do and the Pixel doesn't.
    live_has_video: bool
    # Every recording the run holds, by name, and the one the recording side plays. The page lists
    # them when there is more than one. Empty and None for a run without a recording.
    recordings: tuple[str, ...] = ()
    recording: str | None = None


class OnlyWhileChosen:
    """An RGB source whose frames are let go before the depth model while its side isn't chosen."""

    def __init__(self, source: RgbSource, chosen: Callable[[], bool]) -> None:
        self._source = source
        self._chosen = chosen

    def frames(self) -> Iterator[RgbFrame]:
        for frame in self._source.frames():
            if self._chosen():
                yield frame

    def close(self) -> None:
        self._source.close()


class SharedEstimator:
    """
    One depth model for both sides: one frame at a time, warmed up once.

    A side starts its source again on every reconnect and every start-over, and each start would
    otherwise warm the model up again, over a second each time. The lock is for the moment of a switch,
    when the side being left can still be finishing a frame as the new one starts.
    """

    def __init__(self, estimator: object) -> None:
        self._estimator = estimator
        self._lock = threading.Lock()
        self._warmed_up = False

    @property
    def device(self) -> str:
        return self._estimator.device

    def warm_up(self) -> None:
        with self._lock:
            if self._warmed_up:
                return
            self._estimator.warm_up()
            self._warmed_up = True

    def estimate(self, image_rgb: np.ndarray) -> object:
        with self._lock:
            return self._estimator.estimate(image_rgb)


class _VideoGate:
    """One side's video, passed on to the page's feed only while that side is chosen."""

    def __init__(self, switch: "SwitchableSource", mode: SourceMode) -> None:
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


class SwitchableSource:
    """Yields the chosen side's depth frames, newest first, until closed. Switched by `request`, from any thread."""

    def __init__(
        self,
        live_factory: SideFactory,
        recordings: Mapping[str, SideFactory],
        initial: SourceMode,
        video_feed: SceneVideoFeed | None = None,
        live_has_video: bool = True,
        live_retry_seconds: float = LIVE_RETRY_SECONDS,
        live_stall_seconds: float = LIVE_STALL_SECONDS,
    ) -> None:
        """
        :param live_factory: Builds the live input's source. Called again on every reconnect.
        :param recordings: Each recording's name and the factory that builds a fresh source for it,
            called on every start-over. The first is the one played until a page picks another.
            Empty for a run with no recording, which then only ever plans on the live input.
        :param initial: The side planned on from the start.
        :param video_feed: The page's video feed, or None when no display shows video.
        :param live_has_video: Whether the live input sends video, for the page to say when it doesn't.
        :param live_retry_seconds: How long to wait between attempts to reach the live input.
        :param live_stall_seconds: How long the chosen live input may go quiet before it reads as no input.
        :raises ValueError: When the run is to start on a recording it doesn't have.
        """
        if initial is SourceMode.RECORDING and not recordings:
            raise ValueError("a run can only start on the recording when it has one")
        self._live_factory = live_factory
        self._recordings: dict[str, SideFactory] = dict(recordings)
        self._recording: str | None = next(iter(self._recordings), None)
        self._video_feed = video_feed
        self._live_has_video = live_has_video
        self._retry_seconds = live_retry_seconds
        self._stall_seconds = live_stall_seconds
        self._lock = threading.Condition()
        self._mode = initial
        self._generation = 0
        self._live = InputLink.CONNECTING
        self._live_detail: str | None = None
        self._last_live_frame = time.monotonic()
        self._newest: DepthFrame | None = None
        self._stopping = False
        self._listeners: list[Callable[[SwitchState], None]] = []
        self._sources: dict[SourceMode, DepthFrameSource | None] = {mode: None for mode in SourceMode}
        # Each side's feed carries its video into a gate, which forwards it only while that side is chosen.
        self._gates = {mode: _VideoGate(self, mode) for mode in SourceMode}
        self._threads: list[threading.Thread] = []

    @property
    def mode(self) -> SourceMode:
        with self._lock:
            return self._mode

    @property
    def video_feed(self) -> SceneVideoFeed | None:
        """The page's video feed, which only the chosen side's video reaches."""
        return self._video_feed

    def state(self) -> SwitchState:
        with self._lock:
            return SwitchState(
                self._mode,
                self._live,
                self._live_detail,
                bool(self._recordings),
                self._live_has_video,
                tuple(self._recordings),
                self._recording,
            )

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
        """Plan on the given side from the next frame on. The side already in use, or a recording the run lacks, changes nothing."""
        with self._lock:
            if mode is self._mode:
                return
            if mode is SourceMode.RECORDING and not self._recordings:
                log.warning("a page asked for the %s, and this run has none", mode.value)
                return
            self._mode = mode
            self._generation += 1
            self._newest = None
            # The live input's quiet spell starts now, not when it was last chosen.
            self._last_live_frame = time.monotonic()
            self._lock.notify_all()
        log.info("switched to the %s", "live input" if mode is SourceMode.LIVE else "recording")
        # The page's decoder starts over on the new stream's description. The recording describes
        # its stream when it starts, which choosing it does.
        description = self._gates[mode].description
        if description is not None and mode is SourceMode.LIVE:
            self.video_feed_describe(description)
        self._announce()

    def request_recording(self, name: str) -> None:
        """
        Play the named recording whenever the recording side is chosen. A name the run lacks changes nothing.

        While the recording side is in use, the new recording starts from its beginning at once, as a
        new stretch of video. The name is checked here, once, against the run's own list.
        """
        with self._lock:
            if name not in self._recordings:
                log.warning("a page asked for recording %r, and this run has no such recording", name)
                return
            if name == self._recording:
                return
            self._recording = name
            if self._mode is SourceMode.RECORDING:
                self._generation += 1
                self._newest = None
            self._lock.notify_all()
        log.info("recording %s chosen", name)
        self._announce()

    def frames(self) -> Iterator[DepthFrame]:
        self._start_threads()
        while True:
            with self._lock:
                while self._newest is None and not self._stopping:
                    self._lock.wait(FRAME_POLL_SECONDS)
                    self._check_live_stall()
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
        sides = [(SourceMode.LIVE, self._run_live)]
        if self._recordings:
            sides.append((SourceMode.RECORDING, self._run_recording))
        for mode, run in sides:
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

    def _chosen(self, mode: SourceMode) -> Callable[[], bool]:
        return lambda: self.mode is mode

    def _run_live(self) -> None:
        """The live side: connect, stream for as long as the link holds, retry when it doesn't."""
        factory = self._live_factory
        while not self._is_stopping():
            source = factory(self._side_feed(SourceMode.LIVE), self._chosen(SourceMode.LIVE))
            self._put_source(SourceMode.LIVE, source)
            self._set_live(InputLink.CONNECTING, None)
            first_frame = True
            try:
                for frame in source.frames():
                    if self._is_stopping():
                        return
                    if first_frame:
                        first_frame = False
                        # A link that came back is a new stretch of video for whoever is planning on it.
                        self._next_generation_if(SourceMode.LIVE)
                    self._saw_live_frame()
                    self._deposit(SourceMode.LIVE, frame)
                reason = "the live input stopped sending"
            except SOURCE_FAILURES as failure:
                if self._is_stopping():
                    return
                # Expected when the glasses are off or on another network, or the phone never
                # connects. The server stays up, and the page says there is no input and why.
                reason = f"{type(failure).__name__}: {failure}"
                log.warning("no live input, trying again in %.0f s (caught %s, expected): %s", self._retry_seconds, type(failure).__name__, failure)
            except Exception as unexpected_error:
                if self._is_stopping():
                    # A source closed under its own thread at shutdown can fail any way at all.
                    return
                # Nobody named this one. Kept as unexpected with its traceback, and the live input is
                # retried rather than the server taken down with it.
                reason = f"{type(unexpected_error).__name__}: {unexpected_error}"
                log.error("UNEXPECTED %s on the live side, may need a handler, trying again", type(unexpected_error).__name__, exc_info=True)
            finally:
                if self._take_source(SourceMode.LIVE, source) is not None:
                    self._close_quietly(source)
            self._set_live(InputLink.NO_INPUT, reason)
            with self._lock:
                self._lock.wait_for(lambda: self._stopping, self._retry_seconds)

    def _run_recording(self) -> None:
        """
        The recording side: idle until chosen, then play the chosen recording from the start, looping,
        until another side or another recording is chosen.
        """
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._stopping or self._mode is SourceMode.RECORDING)
                if self._stopping:
                    return
                name = self._recording
                factory = self._recordings[name]
            source = factory(self._side_feed(SourceMode.RECORDING), self._chosen(SourceMode.RECORDING))
            self._put_source(SourceMode.RECORDING, source)
            # A fresh start is a new stretch of video, even when the recording was already chosen.
            self._next_generation_if(SourceMode.RECORDING)
            log.info("playing recording %s from the start", name)
            try:
                for frame in source.frames():
                    if self._is_stopping() or self.mode is not SourceMode.RECORDING or self._chosen_recording() != name:
                        break
                    self._deposit(SourceMode.RECORDING, frame)
            except SOURCE_FAILURES as failure:
                if self._is_stopping():
                    return
                # A recording that won't play is a setup fault, a capture that is missing or
                # malformed, and retrying it would only repeat it. The run stays up so the live input
                # can still be chosen, and choosing the recording again tries once more.
                log.error("the recording stopped, %s (caught %s), choose it again to retry: %s", name, type(failure).__name__, failure)
                self._wait_until_recording_is_left(name)
            except Exception as unexpected_error:
                if self._is_stopping():
                    # A source closed under its own thread at shutdown can fail any way at all.
                    return
                log.error("UNEXPECTED %s on the recording side, may need a handler, choose it again to retry", type(unexpected_error).__name__, exc_info=True)
                self._wait_until_recording_is_left(name)
            finally:
                if self._take_source(SourceMode.RECORDING, source) is not None:
                    self._close_quietly(source)

    def _wait_until_recording_is_left(self, name: str) -> None:
        # Without this a failed recording would start over at once, and fail again, as fast as it can.
        # Picking another recording leaves it too.
        with self._lock:
            self._lock.wait_for(lambda: self._stopping or self._mode is not SourceMode.RECORDING or self._recording != name)

    def _chosen_recording(self) -> str | None:
        with self._lock:
            return self._recording

    def _check_live_stall(self) -> None:
        """
        Caller holds the lock. A chosen live input that has gone quiet reads as no input.

        A link can stay open with nothing on it: glasses unplugged from the phone keep the stream
        connected and send no frames. Without this the page would show the last frame as if live.
        """
        if self._mode is not SourceMode.LIVE or self._live is not InputLink.CONNECTED:
            return
        quiet_seconds = time.monotonic() - self._last_live_frame
        if quiet_seconds < self._stall_seconds:
            return
        self._live, self._live_detail = InputLink.NO_INPUT, f"no frames for {quiet_seconds:.0f} s"
        log.warning("the live input has sent no frame for %.0f s", quiet_seconds)
        # Announced off this thread, which holds the lock and is the loop's own frame reader.
        threading.Thread(target=self._announce, name="announce-stall", daemon=True).start()

    def _saw_live_frame(self) -> None:
        with self._lock:
            self._last_live_frame = time.monotonic()
            changed = self._live is not InputLink.CONNECTED
            self._live, self._live_detail = InputLink.CONNECTED, None
        if changed:
            self._announce()

    def _put_source(self, mode: SourceMode, source: DepthFrameSource) -> None:
        with self._lock:
            self._sources[mode] = source

    def _take_source(self, mode: SourceMode, source: DepthFrameSource | None) -> DepthFrameSource | None:
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

    def _deposit(self, mode: SourceMode, frame: DepthFrame) -> None:
        """
        Keep the frame as the newest, if its side is still chosen. Otherwise it is let go.

        A frame can finish in the depth model after its side was left. Checked here, at the handover,
        it never reaches the planner.
        """
        with self._lock:
            if self._mode is not mode:
                return
            self._newest = dataclasses.replace(frame, source_generation=self._generation)
            self._lock.notify_all()

    def _set_live(self, link: InputLink, detail: str | None) -> None:
        with self._lock:
            if (link, detail) == (self._live, self._live_detail):
                return
            self._live, self._live_detail = link, detail
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

    def _close_quietly(self, source: DepthFrameSource) -> None:
        try:
            source.close()
        except SOURCE_FAILURES as failure:
            # Closing a side that already failed can fail again. The side is gone either way.
            log.info("closing a side raised (caught %s, expected): %s", type(failure).__name__, failure)
