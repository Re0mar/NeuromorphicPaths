"""
Covers the switch between the live input and a recording, with fake sides that need no hardware.

Each fake side yields depth frames whose timestamps say which side made them, and offers video units
that say the same, so a test can read off which side the planner and the page are getting.
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
from nav.sources.switching import InputLink, OnlyWhileChosen, SharedEstimator, SwitchableSource, SwitchState
from nav.types import DepthFrame, Pose

FRAME_INTERVAL_SECONDS = 0.005
WAIT_SECONDS = 3.0
# The whole-second part of each side's timestamps, so a frame says which side made it.
SIDE_SECONDS = {SourceMode.LIVE: 1000.0, SourceMode.RECORDING: 2000.0}


def _side_of(frame: DepthFrame) -> SourceMode:
    return SourceMode.LIVE if frame.timestamp_seconds < SIDE_SECONDS[SourceMode.RECORDING] else SourceMode.RECORDING


def _depth_frame(seconds: float) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=seconds,
        depth_meters=np.ones((2, 2), dtype=np.float32),
        intrinsics=np.eye(3),
        pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False),
        ground_plane=None,
        gaze_pixel=None,
    )


class FakeSide:
    """Yields depth frames stamped for its side until closed, or until it has given frame_count of them."""

    def __init__(self, mode: SourceMode, feed: SceneVideoFeed | None, frame_count: int | None, fail_with: BaseException | None, flowing: threading.Event) -> None:
        self.mode = mode
        self.feed = feed
        self.frame_count = frame_count
        self.fail_with = fail_with
        self.flowing = flowing
        self.closed = threading.Event()

    def frames(self) -> Iterator[DepthFrame]:
        if self.fail_with is not None:
            raise self.fail_with
        if self.feed is not None:
            self.feed.describe(VideoDescription(codec=self.mode.value, parameter_sets=()))
        index = 0
        while not self.closed.is_set() and (self.frame_count is None or index < self.frame_count):
            time.sleep(FRAME_INTERVAL_SECONDS)
            # A side whose input has gone quiet keeps its link open and sends nothing.
            if not self.flowing.is_set():
                continue
            if self.feed is not None:
                self.feed.offer(AccessUnit(timestamp_seconds=float(index), data=self.mode.value.encode(), keyframe=True))
            yield _depth_frame(SIDE_SECONDS[self.mode] + index * 0.001)
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
        self.flowing = threading.Event()
        self.flowing.set()

    def __call__(self, feed: SceneVideoFeed | None, chosen) -> FakeSide:
        fail_with = self.failures.pop(0) if self.failures else None
        side = FakeSide(self.mode, feed, self.frame_count, fail_with, self.flowing)
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


class Collector:
    """Pulls frames on a thread of its own, the way the loop does, so a test can wait without pulling."""

    def __init__(self, switch: SwitchableSource) -> None:
        self.frames: list[DepthFrame] = []
        self._thread = threading.Thread(target=lambda: self.frames.extend(switch.frames()), daemon=True)
        self._thread.start()

    def wait_for_side(self, mode: SourceMode) -> DepthFrame:
        seen = len(self.frames)
        _wait_for(lambda: any(_side_of(frame) is mode for frame in self.frames[seen:]), f"a frame from the {mode.value}")
        return next(frame for frame in self.frames[seen:] if _side_of(frame) is mode)


@pytest.fixture
def factories() -> dict[SourceMode, Factory]:
    return {mode: Factory(mode) for mode in SourceMode}


def _switch(factories, initial: SourceMode = SourceMode.LIVE, feed: SceneVideoFeed | None = None, with_recording: bool = True, stall_seconds: float = 3.0) -> SwitchableSource:
    return SwitchableSource(
        factories[SourceMode.LIVE],
        {"walk_1": factories[SourceMode.RECORDING]} if with_recording else {},
        initial,
        video_feed=feed,
        live_retry_seconds=0.05,
        live_stall_seconds=stall_seconds,
    )


def _wait_for(condition, what: str) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def test_frames_come_from_the_side_it_starts_on(factories) -> None:
    switch = _switch(factories, initial=SourceMode.RECORDING)
    collector = Collector(switch)
    try:
        assert _side_of(collector.wait_for_side(SourceMode.RECORDING)) is SourceMode.RECORDING
    finally:
        switch.close()


def test_a_switch_moves_the_frames_and_starts_a_new_stretch_each_time(factories) -> None:
    switch = _switch(factories)
    collector = Collector(switch)
    try:
        live = collector.wait_for_side(SourceMode.LIVE)
        switch.request(SourceMode.RECORDING)
        recording = collector.wait_for_side(SourceMode.RECORDING)
        switch.request(SourceMode.LIVE)
        back = collector.wait_for_side(SourceMode.LIVE)

        assert live.source_generation < recording.source_generation < back.source_generation
    finally:
        switch.close()


def test_the_live_input_stays_connected_while_the_recording_plays_and_the_recording_starts_over_each_time(factories) -> None:
    switch = _switch(factories)
    collector = Collector(switch)
    try:
        collector.wait_for_side(SourceMode.LIVE)
        switch.request(SourceMode.RECORDING)
        collector.wait_for_side(SourceMode.RECORDING)
        switch.request(SourceMode.LIVE)
        collector.wait_for_side(SourceMode.LIVE)
        _wait_for(lambda: factories[SourceMode.RECORDING].built[0].closed.is_set(), "the recording to close when left")
        switch.request(SourceMode.RECORDING)
        collector.wait_for_side(SourceMode.RECORDING)

        assert len(factories[SourceMode.LIVE].built) == 1, "one live link for the whole run"
        assert not factories[SourceMode.LIVE].built[0].closed.is_set()
        assert len(factories[SourceMode.RECORDING].built) == 2, "the recording starts from the beginning when chosen again"
    finally:
        switch.close()


def test_the_recording_loops_when_it_ends_as_a_new_stretch(factories) -> None:
    factories[SourceMode.RECORDING].frame_count = 3
    switch = _switch(factories, initial=SourceMode.RECORDING)
    collector = Collector(switch)
    try:
        _wait_for(lambda: len(factories[SourceMode.RECORDING].built) >= 3, "the recording to start over twice")

        generations = {frame.source_generation for frame in collector.frames if _side_of(frame) is SourceMode.RECORDING}
        assert len(generations) >= 2
    finally:
        switch.close()


def test_a_missing_live_input_reads_as_no_input_and_is_retried_until_it_comes(factories) -> None:
    factories[SourceMode.LIVE].failures = [ConnectionError("no Neon found on the network")] * 2
    switch = _switch(factories)
    states: list[SwitchState] = []
    switch.subscribe(states.append)
    collector = Collector(switch)
    try:
        collector.wait_for_side(SourceMode.LIVE)

        no_input = [state for state in states if state.live is InputLink.NO_INPUT]
        assert no_input and "no Neon found" in no_input[0].live_detail
        assert states[-1].live is InputLink.CONNECTED, "the retry got through"
        assert len(factories[SourceMode.LIVE].built) == 3
    finally:
        switch.close()


def test_a_live_input_that_goes_quiet_reads_as_no_input_and_recovers_when_frames_return(factories) -> None:
    switch = _switch(factories, stall_seconds=0.3)
    states: list[SwitchState] = []
    switch.subscribe(states.append)
    collector = Collector(switch)
    try:
        collector.wait_for_side(SourceMode.LIVE)
        factories[SourceMode.LIVE].flowing.clear()

        _wait_for(lambda: states[-1].live is InputLink.NO_INPUT, "the quiet input to read as no input")
        assert "no frames for" in states[-1].live_detail
        factories[SourceMode.LIVE].flowing.set()
        _wait_for(lambda: states[-1].live is InputLink.CONNECTED, "frames to bring the input back")
        assert len(factories[SourceMode.LIVE].built) == 1, "a quiet link is not reconnected, only reported"
    finally:
        switch.close()


def test_a_quiet_live_input_isnt_reported_while_the_recording_plays(factories) -> None:
    switch = _switch(factories, initial=SourceMode.RECORDING, stall_seconds=0.2)
    states: list[SwitchState] = []
    switch.subscribe(states.append)
    collector = Collector(switch)
    try:
        collector.wait_for_side(SourceMode.RECORDING)
        _wait_for(lambda: states[-1].live is InputLink.CONNECTED, "the live input to connect")
        factories[SourceMode.LIVE].flowing.clear()
        time.sleep(0.5)

        assert states[-1].live is InputLink.CONNECTED, "the live side isn't chosen, so its silence isn't news"
    finally:
        switch.close()


def test_a_run_without_a_recording_only_plans_on_the_live_input(factories) -> None:
    switch = _switch(factories, with_recording=False)
    states: list[SwitchState] = []
    switch.subscribe(states.append)
    collector = Collector(switch)
    try:
        collector.wait_for_side(SourceMode.LIVE)
        switch.request(SourceMode.RECORDING)
        collector.wait_for_side(SourceMode.LIVE)

        assert switch.mode is SourceMode.LIVE
        assert not states[-1].has_recording
        assert factories[SourceMode.RECORDING].built == []
    finally:
        switch.close()


def test_a_run_cant_start_on_a_recording_it_doesnt_have(factories) -> None:
    with pytest.raises(ValueError, match="only start on the recording when it has one"):
        _switch(factories, initial=SourceMode.RECORDING, with_recording=False)


def _two_recordings(factories, initial: SourceMode = SourceMode.LIVE) -> tuple[SwitchableSource, Factory]:
    """A switch holding walk_1 from the fixture and a second recording, walk_2, with its own factory."""
    second = Factory(SourceMode.RECORDING)
    switch = SwitchableSource(
        factories[SourceMode.LIVE],
        {"walk_1": factories[SourceMode.RECORDING], "walk_2": second},
        initial,
        live_retry_seconds=0.05,
        live_stall_seconds=3.0,
    )
    return switch, second


def test_the_state_lists_the_recordings_in_order_with_the_first_chosen(factories) -> None:
    switch, _ = _two_recordings(factories)
    without = _switch(factories, with_recording=False)
    try:
        assert (switch.state().recordings, switch.state().recording, switch.state().has_recording) == (("walk_1", "walk_2"), "walk_1", True)
        assert (without.state().recordings, without.state().recording, without.state().has_recording) == ((), None, False)
    finally:
        switch.close()
        without.close()


def test_picking_another_recording_while_it_plays_starts_that_one_from_the_start_as_a_new_stretch(factories) -> None:
    switch, second = _two_recordings(factories, initial=SourceMode.RECORDING)
    collector = Collector(switch)
    states: list[SwitchState] = []
    try:
        before = collector.wait_for_side(SourceMode.RECORDING)
        switch.subscribe(states.append)

        switch.request_recording("walk_2")

        _wait_for(lambda: len(second.built) == 1, "the second recording to start")
        _wait_for(lambda: factories[SourceMode.RECORDING].built[0].closed.is_set(), "the first recording to close")
        _wait_for(lambda: collector.frames and collector.frames[-1].source_generation > before.source_generation, "a frame from the new stretch")
        assert states[-1].recording == "walk_2" and states[-1].mode is SourceMode.RECORDING
        assert len(factories[SourceMode.RECORDING].built) == 1, "the first recording isn't started again"
    finally:
        switch.close()


def test_picking_a_recording_while_on_the_live_input_only_changes_which_plays_next(factories) -> None:
    switch, second = _two_recordings(factories)
    collector = Collector(switch)
    states: list[SwitchState] = []
    try:
        collector.wait_for_side(SourceMode.LIVE)
        switch.subscribe(states.append)

        switch.request_recording("walk_2")

        assert states[-1].recording == "walk_2" and states[-1].mode is SourceMode.LIVE
        assert not second.built and not factories[SourceMode.RECORDING].built, "nothing plays until the recording side is chosen"
        switch.request(SourceMode.RECORDING)
        collector.wait_for_side(SourceMode.RECORDING)
        assert len(second.built) == 1 and not factories[SourceMode.RECORDING].built
    finally:
        switch.close()


def test_a_recording_the_run_lacks_is_ignored_and_said_so(factories, caplog: pytest.LogCaptureFixture) -> None:
    switch, second = _two_recordings(factories)
    states: list[SwitchState] = []
    switch.subscribe(states.append)
    try:
        with caplog.at_level("WARNING"):
            switch.request_recording("walk_9")
            switch.request_recording("walk_1")

        assert [state.recording for state in states] == ["walk_1"], "neither an unknown name nor the one in use announces anything"
        assert "walk_9" in caplog.text and "no such recording" in caplog.text
    finally:
        switch.close()


def test_a_listener_gets_the_state_at_once_and_on_every_switch(factories) -> None:
    switch = _switch(factories)
    states: list[SwitchState] = []

    switch.subscribe(states.append)
    switch.request(SourceMode.RECORDING)
    switch.request(SourceMode.RECORDING)

    assert [state.mode for state in states] == [SourceMode.LIVE, SourceMode.RECORDING], "choosing the side in use changes nothing"
    assert states[0].has_recording and states[0].live is InputLink.CONNECTING
    switch.close()


def test_only_the_chosen_sides_video_reaches_the_page_and_a_switch_redescribes_it(factories) -> None:
    feed = SceneVideoFeed()
    page = VideoRecorder()
    feed.subscribe(page)
    switch = _switch(factories, feed=feed)
    collector = Collector(switch)
    try:
        collector.wait_for_side(SourceMode.LIVE)
        switch.request(SourceMode.RECORDING)
        collector.wait_for_side(SourceMode.RECORDING)
        _wait_for(lambda: page.events.count("recording") >= 3, "recording video")

        assert "live" not in page.after("describe recording"), "the live units stop reaching the page"
        switch.request(SourceMode.LIVE)
        # Described again at once, from the stream description the live input gave at connect.
        _wait_for(lambda: "describe live" in page.events[-5:], "the live stream described again")
        _wait_for(lambda: page.after("describe live").count("live") >= 3, "live video again")
        assert "recording" not in page.after("describe live")[1:], "the recording's units stop once it is left"
    finally:
        switch.close()


def test_a_recording_that_wont_play_waits_to_be_chosen_again_and_the_live_input_still_works(factories, caplog: pytest.LogCaptureFixture) -> None:
    factories[SourceMode.RECORDING].failures = [ValueError("capture is malformed: meta.json has no 'sprop'")]
    switch = _switch(factories, initial=SourceMode.RECORDING)
    collector = Collector(switch)
    try:
        _wait_for(lambda: "the recording stopped" in caplog.text, "the recording's failure to be logged")
        time.sleep(0.1)
        assert len(factories[SourceMode.RECORDING].built) == 1, "a failed recording is not retried on its own"

        switch.request(SourceMode.LIVE)
        collector.wait_for_side(SourceMode.LIVE)
        switch.request(SourceMode.RECORDING)
        collector.wait_for_side(SourceMode.RECORDING)

        assert len(factories[SourceMode.RECORDING].built) == 2, "choosing it again tries once more"
    finally:
        switch.close()


def test_close_ends_the_frames_and_closes_both_sides(factories) -> None:
    switch = _switch(factories)
    frames = switch.frames()
    next(frames)
    switch.request(SourceMode.RECORDING)
    _wait_for(lambda: factories[SourceMode.RECORDING].built, "the recording to start")

    switch.close()

    assert all(side.closed.is_set() for factory in factories.values() for side in factory.built)
    with pytest.raises(StopIteration):
        next(frames)


def test_a_frame_that_finishes_after_its_side_was_left_never_reaches_the_planner() -> None:
    # The depth model takes about half a second a frame, so the side being left can finish one after
    # the switch. Handed over, it would put one arrow from that side on the page.
    release = threading.Event()

    class SlowRecording:
        def frames(self):
            release.wait(WAIT_SECONDS)
            yield _depth_frame(SIDE_SECONDS[SourceMode.RECORDING])

        def close(self) -> None:
            release.set()

    live = Factory(SourceMode.LIVE)
    switch = SwitchableSource(live, {"slow": lambda feed, chosen: SlowRecording()}, SourceMode.RECORDING, live_retry_seconds=0.05)
    collector = Collector(switch)
    try:
        time.sleep(0.2)
        switch.request(SourceMode.LIVE)
        release.set()
        collector.wait_for_side(SourceMode.LIVE)
        time.sleep(0.1)

        assert not any(_side_of(frame) is SourceMode.RECORDING for frame in collector.frames)
    finally:
        switch.close()


def test_frames_of_a_side_that_isnt_chosen_never_reach_the_depth_model() -> None:
    chosen = {"now": True}

    class Pictures:
        def frames(self):
            for index in range(4):
                if index == 2:
                    chosen["now"] = False
                yield RgbFrame(timestamp_seconds=float(index), image_rgb=np.zeros((2, 2, 3), dtype=np.uint8), gaze_pixel=None, pose=None, camera_matrix=None, timing=None)

        def close(self) -> None:
            pass

    gated = OnlyWhileChosen(Pictures(), lambda: chosen["now"])

    assert [frame.timestamp_seconds for frame in gated.frames()] == [0.0, 1.0]


def test_the_shared_model_warms_up_once_and_takes_one_frame_at_a_time() -> None:
    class CountingModel:
        device = "cuda"

        def __init__(self) -> None:
            self.warm_ups = 0
            self.inside = 0
            self.most_inside = 0

        def warm_up(self) -> None:
            self.warm_ups += 1

        def estimate(self, image_rgb):
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
            time.sleep(0.02)
            self.inside -= 1
            return "depth"

    model = CountingModel()
    shared = SharedEstimator(model)
    shared.warm_up()
    shared.warm_up()
    threads = [threading.Thread(target=shared.estimate, args=(None,)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.warm_ups == 1
    assert model.most_inside == 1
    assert shared.device == "cuda"
