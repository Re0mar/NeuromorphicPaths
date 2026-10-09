"""
Covers the switch between the glasses and a demo capture, with fake sides that need no hardware.

Each fake side yields frames whose pixels say which side made them, and offers video units that say
the same, so a test can read off which side the planner and the page are getting.
"""

# Standard library imports
import threading
import time
from collections.abc import Iterator

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.config import SourceMode
from nav.sources.rgb import RgbFrame
from nav.sources.scene_video import AccessUnit, SceneVideoFeed, VideoDescription
from nav.sources.switching import CurrentStretchOnly, GlassesLink, SwitchableRgbSource, SwitchState
from nav.types import DepthFrame, Pose

FRAME_INTERVAL_SECONDS = 0.005
WAIT_SECONDS = 3.0
# The pixel value each side's frames carry.
SIDE_PIXEL = {SourceMode.GLASSES: 10, SourceMode.DEMO: 20}


class FakeSide:
    """Yields frames of its own pixel value until closed, or until it has given frame_count of them."""

    def __init__(self, mode: SourceMode, feed: SceneVideoFeed | None, frame_count: int | None = None, fail_with: BaseException | None = None) -> None:
        self.mode = mode
        self.feed = feed
        self.frame_count = frame_count
        self.fail_with = fail_with
        self.closed = threading.Event()

    def frames(self) -> Iterator[RgbFrame]:
        if self.fail_with is not None:
            raise self.fail_with
        if self.feed is not None:
            self.feed.describe(VideoDescription(codec=self.mode.value, parameter_sets=()))
        index = 0
        while not self.closed.is_set() and (self.frame_count is None or index < self.frame_count):
            time.sleep(FRAME_INTERVAL_SECONDS)
            if self.feed is not None:
                self.feed.offer(AccessUnit(timestamp_seconds=float(index), data=self.mode.value.encode(), keyframe=True))
            image = np.full((4, 4, 3), SIDE_PIXEL[self.mode], dtype=np.uint8)
            yield RgbFrame(timestamp_seconds=float(index), image_rgb=image, gaze_pixel=None, pose=None, camera_matrix=None, timing=None)
            index += 1

    def close(self) -> None:
        self.closed.set()


class Factory:
    """Builds fake sides, keeps every one it built, and can be told how the next ones behave."""

    def __init__(self, mode: SourceMode) -> None:
        self.mode = mode
        self.built: list[FakeSide] = []
        self.frame_count: int | None = None
        self.failures: list[BaseException] = []

    def __call__(self, feed: SceneVideoFeed | None) -> FakeSide:
        fail_with = self.failures.pop(0) if self.failures else None
        side = FakeSide(self.mode, feed, self.frame_count, fail_with)
        self.built.append(side)
        return side


class VideoRecorder:
    """The page's end of the video feed: which streams it was told about, and whose units it got."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.lock = threading.Lock()

    def describe(self, description: VideoDescription) -> None:
        with self.lock:
            self.events.append(f"describe {description.codec}")

    def offer(self, unit: AccessUnit) -> None:
        with self.lock:
            self.events.append(unit.data.decode())

    def after(self, marker: str) -> list[str]:
        with self.lock:
            return self.events[len(self.events) - self.events[::-1].index(marker) :]


@pytest.fixture
def factories() -> dict[SourceMode, Factory]:
    return {mode: Factory(mode) for mode in SourceMode}


def _switch(factories, initial: SourceMode = SourceMode.GLASSES, feed: SceneVideoFeed | None = None) -> SwitchableRgbSource:
    return SwitchableRgbSource(factories[SourceMode.GLASSES], factories[SourceMode.DEMO], initial, video_feed=feed, glasses_retry_seconds=0.05)


def _next_from(frames: Iterator[RgbFrame], mode: SourceMode) -> RgbFrame:
    """The next frame from the given side, skipping any the other side had already handed over."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        frame = next(frames)
        if frame.image_rgb[0, 0, 0] == SIDE_PIXEL[mode]:
            return frame
    raise AssertionError(f"no frame from the {mode.value} within {WAIT_SECONDS} s")


def _wait_for(condition, what: str) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def test_frames_come_from_the_side_it_starts_on(factories) -> None:
    switch = _switch(factories, initial=SourceMode.DEMO)
    frames = switch.frames()
    try:
        assert next(frames).image_rgb[0, 0, 0] == SIDE_PIXEL[SourceMode.DEMO]
    finally:
        switch.close()


def test_a_switch_moves_the_frames_and_starts_a_new_stretch_each_time(factories) -> None:
    switch = _switch(factories)
    frames = switch.frames()
    try:
        glasses = _next_from(frames, SourceMode.GLASSES)
        switch.request(SourceMode.DEMO)
        demo = _next_from(frames, SourceMode.DEMO)
        switch.request(SourceMode.GLASSES)
        back = _next_from(frames, SourceMode.GLASSES)

        assert glasses.source_generation < demo.source_generation < back.source_generation
    finally:
        switch.close()


def test_the_glasses_stay_connected_while_the_demo_plays_and_the_demo_starts_over_each_time(factories) -> None:
    switch = _switch(factories)
    frames = switch.frames()
    try:
        _next_from(frames, SourceMode.GLASSES)
        switch.request(SourceMode.DEMO)
        _next_from(frames, SourceMode.DEMO)
        switch.request(SourceMode.GLASSES)
        _next_from(frames, SourceMode.GLASSES)
        _wait_for(lambda: factories[SourceMode.DEMO].built[0].closed.is_set(), "the demo to close when left")
        switch.request(SourceMode.DEMO)
        _next_from(frames, SourceMode.DEMO)

        assert len(factories[SourceMode.GLASSES].built) == 1, "one glasses link for the whole run"
        assert not factories[SourceMode.GLASSES].built[0].closed.is_set()
        assert len(factories[SourceMode.DEMO].built) == 2, "the demo starts from the beginning when chosen again"
    finally:
        switch.close()


def test_the_demo_loops_when_it_ends_as_a_new_stretch(factories) -> None:
    factories[SourceMode.DEMO].frame_count = 3
    switch = _switch(factories, initial=SourceMode.DEMO)
    frames = switch.frames()
    try:
        generations = set()
        deadline = time.monotonic() + WAIT_SECONDS
        while len(factories[SourceMode.DEMO].built) < 3 and time.monotonic() < deadline:
            generations.add(_next_from(frames, SourceMode.DEMO).source_generation)

        assert len(factories[SourceMode.DEMO].built) >= 3, "the demo started over after each end"
        assert len(generations) >= 2
    finally:
        switch.close()


def test_glasses_that_cant_be_reached_are_retried_and_the_state_says_why(factories) -> None:
    factories[SourceMode.GLASSES].failures = [ConnectionError("no Neon found on the network")]
    switch = _switch(factories)
    states: list[SwitchState] = []
    switch.subscribe(states.append)
    frames = switch.frames()
    try:
        frame = _next_from(frames, SourceMode.GLASSES)

        unreachable = [state for state in states if state.glasses is GlassesLink.UNREACHABLE]
        assert unreachable and "no Neon found" in unreachable[0].glasses_detail
        assert states[-1].glasses is GlassesLink.CONNECTED, "the retry got through"
        assert len(factories[SourceMode.GLASSES].built) == 2
        assert frame.image_rgb[0, 0, 0] == SIDE_PIXEL[SourceMode.GLASSES]
    finally:
        switch.close()


def test_a_listener_gets_the_state_at_once_and_on_every_switch(factories) -> None:
    switch = _switch(factories)
    states: list[SwitchState] = []

    switch.subscribe(states.append)
    switch.request(SourceMode.DEMO)
    switch.request(SourceMode.DEMO)

    assert [state.mode for state in states] == [SourceMode.GLASSES, SourceMode.DEMO], "choosing the side in use changes nothing"
    switch.close()


def test_only_the_chosen_sides_video_reaches_the_page_and_a_switch_redescribes_it(factories) -> None:
    feed = SceneVideoFeed()
    page = VideoRecorder()
    feed.subscribe(page)
    switch = _switch(factories, feed=feed)
    frames = switch.frames()
    try:
        _next_from(frames, SourceMode.GLASSES)
        switch.request(SourceMode.DEMO)
        _next_from(frames, SourceMode.DEMO)
        _wait_for(lambda: page.events.count("demo") >= 3, "demo video")

        assert "glasses" not in page.after("describe demo"), "the glasses' units stop reaching the page"
        switch.request(SourceMode.GLASSES)
        # Described again at once, from the stream description the glasses gave at connect.
        assert page.events[-1] == "describe glasses" or "describe glasses" in page.events[-3:]
        _wait_for(lambda: page.after("describe glasses").count("glasses") >= 3, "glasses video again")
        assert "demo" not in page.after("describe glasses")[1:], "the demo's units stop once it is left"
    finally:
        switch.close()


class Collector:
    """Pulls frames on a thread of its own, the way the loop does, so a test can wait without pulling."""

    def __init__(self, switch: SwitchableRgbSource) -> None:
        self.frames: list[RgbFrame] = []
        self._thread = threading.Thread(target=lambda: self.frames.extend(switch.frames()), daemon=True)
        self._thread.start()

    def wait_for_side(self, mode: SourceMode) -> None:
        seen = len(self.frames)
        _wait_for(lambda: any(frame.image_rgb[0, 0, 0] == SIDE_PIXEL[mode] for frame in self.frames[seen:]), f"a frame from the {mode.value}")


def test_a_demo_that_wont_play_waits_to_be_chosen_again_and_the_glasses_still_work(factories, caplog: pytest.LogCaptureFixture) -> None:
    factories[SourceMode.DEMO].failures = [ValueError("capture is malformed: meta.json has no 'sprop'")]
    switch = _switch(factories, initial=SourceMode.DEMO)
    collector = Collector(switch)
    try:
        _wait_for(lambda: "the demo stopped" in caplog.text, "the demo's failure to be logged")
        time.sleep(0.1)
        assert len(factories[SourceMode.DEMO].built) == 1, "a failed demo is not retried on its own"

        switch.request(SourceMode.GLASSES)
        collector.wait_for_side(SourceMode.GLASSES)
        switch.request(SourceMode.DEMO)
        collector.wait_for_side(SourceMode.DEMO)

        assert len(factories[SourceMode.DEMO].built) == 2, "choosing it again tries once more"
    finally:
        switch.close()


def test_close_ends_the_frames_and_closes_both_sides(factories) -> None:
    switch = _switch(factories)
    frames = switch.frames()
    _next_from(frames, SourceMode.GLASSES)
    switch.request(SourceMode.DEMO)
    _next_from(frames, SourceMode.DEMO)

    switch.close()

    assert all(side.closed.is_set() for factory in factories.values() for side in factory.built)
    with pytest.raises(StopIteration):
        next(frames)


def test_a_depth_frame_from_a_stretch_already_left_never_reaches_the_planner() -> None:
    # The frame that was still in the depth model when the switch turned is the one this drops.
    class SwitchAt:
        generation = 2

    def depth_frame(generation: int) -> DepthFrame:
        return DepthFrame(
            timestamp_seconds=float(generation),
            depth_meters=np.ones((2, 2), dtype=np.float32),
            intrinsics=np.eye(3),
            pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False),
            ground_plane=None,
            gaze_pixel=None,
            source_generation=generation,
        )

    class Frames:
        closed = False

        def frames(self):
            yield from (depth_frame(generation) for generation in (1, 1, 2, 2))

        def close(self) -> None:
            self.closed = True

    inner = Frames()
    filtered = CurrentStretchOnly(inner, SwitchAt())

    assert [frame.source_generation for frame in filtered.frames()] == [2, 2]
    filtered.close()
    assert inner.closed
