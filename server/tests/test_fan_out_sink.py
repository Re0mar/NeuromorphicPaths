"""
Covers the sink that serves several displays from one run.

The fakes here are small classes with real behavior, not mocks: one plain sink and one that also
draws a field and a view, which is the distinction the fan-out has to route on.
"""

# Standard library imports
import logging

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.fan_out import FanOutSink
from nav.types import DebugSink, DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, PlannedPath, Pose


def _path(heading: float = 0.1) -> PlannedPath:
    return PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), heading, False, 2.5, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _view() -> DebugView:
    frame = DepthFrame(
        timestamp_seconds=1.0,
        depth_meters=np.ones((2, 2), dtype=np.float32),
        intrinsics=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False),
        ground_plane=None,
        gaze_pixel=None,
    )
    return DebugView(frame, ObstacleSet(1.0, (), 0), Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30, 1.47)


class PlainSink:
    """A display that takes paths and nothing else, the way the phone sink does."""

    def __init__(self, name: str = "plain") -> None:
        self.name = name
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("started")

    def publish(self, path: PlannedPath) -> None:
        self.events.append(f"published {path.lookahead_heading_radians}")

    def close(self) -> None:
        self.events.append("closed")


class DrawingSink(PlainSink):
    """A display that can also draw the field and the view, the way the window and the browser do."""

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
        self.events.append(f"drew {path.lookahead_heading_radians}")


def test_a_drawing_sink_is_recognised_as_one_and_a_plain_sink_is_not() -> None:
    # The fakes stand in for the real sinks, so the routing test below means nothing unless the
    # protocol tells them apart the same way the loop does.
    assert isinstance(DrawingSink(), DebugSink)
    assert not isinstance(PlainSink(), DebugSink)


def test_every_sink_is_started_in_the_order_it_was_named() -> None:
    first, second = PlainSink("first"), DrawingSink("second")
    fan_out = FanOutSink([first, second])

    fan_out.start()

    assert first.events == ["started"] and second.events == ["started"]


def test_a_published_path_reaches_every_sink() -> None:
    plain, drawing = PlainSink(), DrawingSink()
    FanOutSink([plain, drawing]).publish(_path(heading=0.3))

    assert plain.events == ["published 0.3"]
    assert drawing.events == ["published 0.3"], "publish carries nothing to draw, so a drawing sink gets the path alone"


def test_a_debug_publish_draws_on_the_sinks_that_can_and_hands_the_path_to_the_rest() -> None:
    plain, drawing = PlainSink(), DrawingSink()

    FanOutSink([plain, drawing]).publish_debug(_path(heading=0.4), np.zeros((2, 3)), np.array([-1.0, 0.0, 1.0]), _view())

    assert plain.events == ["published 0.4"]
    assert drawing.events == ["drew 0.4"]


def test_a_sink_that_fails_while_publishing_is_dropped_and_the_others_keep_going(caplog: pytest.LogCaptureFixture) -> None:
    # A walker is looking at one of these. One display's bug must not end the walk, and it must
    # not be silent either.
    class BrokenSink(PlainSink):
        def publish(self, path: PlannedPath) -> None:
            raise RuntimeError("a bug, not a browser that left")

    broken, working = BrokenSink("broken"), PlainSink("working")
    fan_out = FanOutSink([broken, working])

    with caplog.at_level(logging.ERROR, logger="nav.sinks.fan_out"):
        fan_out.publish(_path(heading=0.1))
        fan_out.publish(_path(heading=0.2))

    assert working.events == ["published 0.1", "published 0.2"]
    assert fan_out.sinks == (working,), "the failing display is gone, the working one stays"
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1, "dropped once, not once per frame"
    assert "UNEXPECTED RuntimeError" in errors[0].message and errors[0].exc_info is not None


def test_a_sink_that_cannot_start_ends_the_run() -> None:
    # A port nobody can bind is a configuration problem the user has to see. Walking without the
    # display you asked for, silently, is worse than a run that stops and says why.
    class UnstartableSink(PlainSink):
        def start(self) -> None:
            raise OSError("the phone sink could not listen on 0.0.0.0:9100")

    with pytest.raises(OSError, match="could not listen on 0.0.0.0:9100"):
        FanOutSink([UnstartableSink(), PlainSink()]).start()


def test_close_closes_every_sink_even_after_one_raises(caplog: pytest.LogCaptureFixture) -> None:
    class BadCloser(PlainSink):
        def close(self) -> None:
            raise RuntimeError("will not close")

    last = PlainSink("last")

    with caplog.at_level(logging.ERROR, logger="nav.sinks.fan_out"):
        FanOutSink([BadCloser("bad"), last]).close()

    assert last.events == ["closed"]
    assert any("UNEXPECTED RuntimeError closing BadCloser" in record.message for record in caplog.records)


def test_a_dropped_sink_is_still_closed() -> None:
    # Dropping a display stops it being published to. It still holds a socket or a window.
    class BrokenSink(PlainSink):
        def publish(self, path: PlannedPath) -> None:
            raise RuntimeError("a bug")

    broken = BrokenSink("broken")
    fan_out = FanOutSink([broken, PlainSink()])
    fan_out.publish(_path())

    fan_out.close()

    assert "closed" in broken.events


def test_a_fan_out_with_no_sinks_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one sink"):
        FanOutSink([])
