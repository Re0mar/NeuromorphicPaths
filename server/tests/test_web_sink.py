"""
Covers the web sink against a real aiohttp client over a real socket, not a mock of the app.

The sink runs its server in its own thread. The test drives a client from a short asyncio run
on the test thread, which is exactly the two-loop arrangement the design chose.
"""

# Standard library imports
import asyncio
import json
import time

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.sinks.config import WebConfig
from nav.sinks.web import OUTGOING_QUEUE_LIMIT, WebSink, _quiet_client_resets
from nav.sinks.web_messages import OBSTACLE_KEYS, PLAN_VIEW_KEYS, WebMessageKind
from nav.sources.framecodec import path_message
from nav.types import DebugSink, DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, PlannedPath, Pose

RECEIVE_TIMEOUT_SECONDS = 3.0
VIEW_SIDE = 4


def _path(heading: float = 0.1, alarm: bool = False) -> PlannedPath:
    return PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), heading, alarm, 2.5, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _view() -> DebugView:
    """A 4 by 4 frame of valid depth, no obstacles, a level floor. Enough to render and encode."""
    frame = DepthFrame(
        timestamp_seconds=1.0,
        depth_meters=np.full((VIEW_SIDE, VIEW_SIDE), 2.0, dtype=np.float32),
        intrinsics=np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False),
        ground_plane=None,
        gaze_pixel=None,
    )
    return DebugView(frame, ObstacleSet(1.0, (), 0), Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30)


def _field() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((2, 3)), np.array([-1.0, 0.0, 1.0])


async def _receive_frames(port: int, after_connect, count: int) -> list:
    """The first count websocket frames after connecting, each as a str or bytes."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/ws") as socket:
            after_connect()
            received = []
            for _ in range(count):
                message = await asyncio.wait_for(socket.receive(), RECEIVE_TIMEOUT_SECONDS)
                assert message.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY), message
                received.append(message.data)
            return received


@pytest.fixture
def sink():
    sink = WebSink(WebConfig(port=0))
    sink.start()
    assert sink.port is not None and sink.port > 0
    yield sink
    sink.close()


async def _receive_one(port: int, after_connect) -> str:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/ws") as socket:
            after_connect()
            message = await asyncio.wait_for(socket.receive_str(), RECEIVE_TIMEOUT_SECONDS)
            return message


def test_a_published_path_reaches_a_connected_browser(sink: WebSink) -> None:
    path = _path(heading=0.3, alarm=True)

    # The publish has to happen after the client is connected, or there is nobody to send to.
    received = asyncio.run(_receive_one(sink.port, after_connect=lambda: sink.publish(path)))

    assert json.loads(received) == {"kind": WebMessageKind.PATH.value, **path_message(path)}


def test_the_page_is_served_at_the_root(sink: WebSink) -> None:
    async def fetch() -> str:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{sink.port}/") as response:
                assert response.status == 200
                return await response.text()

    page = asyncio.run(fetch())

    assert "<canvas" in page
    assert "/ws" in page


def test_the_served_page_reads_every_plan_view_key_it_is_sent(sink: WebSink) -> None:
    # A weak check, on purpose: it catches a key renamed on one side, not a wrong drawing. The
    # drawing is checked by eye on a replay, because the repo has no JavaScript runner.
    async def fetch() -> str:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{sink.port}/") as response:
                return await response.text()

    page = asyncio.run(fetch())

    for key in PLAN_VIEW_KEYS + OBSTACLE_KEYS:
        assert f".{key}" in page, f"the page never reads {key}"
    for kind in WebMessageKind:
        assert f'"{kind.value}"' in page, f"the page never dispatches on {kind.value}"


def test_a_client_that_disconnects_does_not_break_the_next_publish(sink: WebSink) -> None:
    async def scenario() -> str:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            leaver = await session.ws_connect(f"ws://127.0.0.1:{sink.port}/ws")
            async with session.ws_connect(f"ws://127.0.0.1:{sink.port}/ws") as stayer:
                await leaver.close()
                # Give the server a moment to notice the leaver has gone.
                await asyncio.sleep(0.2)
                sink.publish(_path(heading=0.2))
                return await asyncio.wait_for(stayer.receive_str(), RECEIVE_TIMEOUT_SECONDS)

    received = asyncio.run(scenario())

    assert json.loads(received)["first_heading_radians"] == pytest.approx(0.2)


def test_close_returns_within_two_seconds() -> None:
    sink = WebSink(WebConfig(port=0))
    sink.start()

    started = time.monotonic()
    sink.close()

    assert time.monotonic() - started < 2.0


def test_publishing_with_no_browser_connected_does_not_raise(sink: WebSink) -> None:
    sink.publish(_path())


def test_a_port_already_in_use_is_reported_rather_than_hung() -> None:
    first = WebSink(WebConfig(port=0))
    first.start()
    try:
        second = WebSink(WebConfig(port=first.port))
        with pytest.raises(ConnectionError, match="could not start"):
            second.start()
    finally:
        first.close()


def test_closing_an_unstarted_sink_does_not_raise() -> None:
    WebSink(WebConfig(port=0)).close()


def test_a_browser_that_stops_reading_cannot_grow_the_queue_without_bound(sink: WebSink, caplog: pytest.LogCaptureFixture) -> None:
    # The pipeline hands over a path, a plan view and a 128 KB picture per planned frame and never waits, while
    # one browser whose window has closed suspends the send loop for every browser. Unbounded, a
    # backgrounded phone browser grew that queue by hundreds of megabytes over a walk.
    async def scenario() -> int:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{sink.port}/ws"):
                await asyncio.sleep(0.3)
                # The client never reads from here on.
                for _ in range(300):
                    sink.publish_debug(_path(), *_field(), _view())
                await asyncio.sleep(1.0)
                assert sink._outgoing is not None
                return sink._outgoing.qsize()

    depth = asyncio.run(scenario())

    assert depth <= OUTGOING_QUEUE_LIMIT, f"the queue grew to {depth}"
    assert sink.dropped > 0, "900 messages through a queue of 32 must have dropped some"

    with caplog.at_level("INFO", logger="nav.sinks.web"):
        sink.close()
    assert any("were dropped because the browsers were not keeping up" in record.message for record in caplog.records)


def test_the_web_sink_is_a_debug_sink() -> None:
    # The loop dispatches on this. Without it the browser would get the path and never the view.
    assert isinstance(WebSink(WebConfig(port=0)), DebugSink)


def test_a_debug_publish_sends_path_then_plan_view_then_png(sink: WebSink) -> None:
    path = _path(heading=0.3)
    field, grid = _field()

    path_text, plan_text, png = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, field, grid, _view()), count=3))

    assert json.loads(path_text) == {"kind": WebMessageKind.PATH.value, **path_message(path)}
    plan_view = json.loads(plan_text)
    assert plan_view["kind"] == WebMessageKind.PLAN_VIEW.value
    assert np.array(plan_view["field"]).shape == field.shape
    assert isinstance(png, bytes)
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    # A 4 by 4 frame scaled up by a whole number to the target width.
    assert decoded.shape == (640, 640, 3)


def test_a_browser_connecting_late_gets_the_latest_of_each_kind_at_once(sink: WebSink) -> None:
    # Published before anyone is connected. A page opened afterwards must not sit blank until the
    # next planned frame, which on a still phone never comes.
    path = _path(heading=0.4)
    field, grid = _field()
    sink.publish_debug(path, field, grid, _view())
    time.sleep(0.3)  # Let the server loop take all three messages off its queue.

    path_text, plan_text, png = asyncio.run(_receive_frames(sink.port, lambda: None, count=3))

    assert json.loads(path_text)["kind"] == WebMessageKind.PATH.value
    assert json.loads(path_text)["first_heading_radians"] == pytest.approx(0.4)
    assert json.loads(plan_text)["kind"] == WebMessageKind.PLAN_VIEW.value
    assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n"


async def _receive_texts_then_check_quiet(port: int, after_connect, texts: int) -> tuple[list, bool]:
    """The first few text frames after connecting, and whether anything at all followed them."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/ws") as socket:
            after_connect()
            received = [await asyncio.wait_for(socket.receive(), RECEIVE_TIMEOUT_SECONDS) for _ in range(texts)]
            try:
                await asyncio.wait_for(socket.receive(), 0.5)
                return [message.data for message in received], True
            except TimeoutError:
                return [message.data for message in received], False


def test_a_png_encode_failure_still_sends_the_path_and_the_plan_view(sink: WebSink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import nav.sinks.web as web_module

    def broken(view, path):
        raise ValueError("no picture today")

    monkeypatch.setattr(web_module, "render_depth_view", broken)
    path = _path(heading=0.5)
    field, grid = _field()

    with caplog.at_level("WARNING", logger="nav.sinks.web"):
        texts, more_arrived = asyncio.run(_receive_texts_then_check_quiet(sink.port, lambda: sink.publish_debug(path, field, grid, _view()), 2))

    assert json.loads(texts[0])["first_heading_radians"] == pytest.approx(0.5)
    assert json.loads(texts[1])["kind"] == WebMessageKind.PLAN_VIEW.value
    assert not more_arrived, "nothing may follow the plan view when the picture failed"
    assert any("depth view not sent" in record.message for record in caplog.records)


def test_a_plan_view_that_cannot_be_built_still_sends_the_path_and_the_png(sink: WebSink, caplog: pytest.LogCaptureFixture) -> None:
    path = _path(heading=0.6)
    field, grid = _field()
    wrong_field = np.zeros((field.shape[0] + 1, field.shape[1]))

    with caplog.at_level("WARNING", logger="nav.sinks.web"):
        received = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, wrong_field, grid, _view()), count=2))

    assert json.loads(received[0])["kind"] == WebMessageKind.PATH.value
    assert isinstance(received[1], bytes), "the picture follows the path straight away"
    assert any("plan view not sent" in record.message for record in caplog.records)
    # The same publish with a field that fits sends all three, so the gap above is the bad field.
    good = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, field, grid, _view()), count=4))
    kinds = [json.loads(message)["kind"] for message in good if isinstance(message, str)]
    assert WebMessageKind.PLAN_VIEW.value in kinds


class _RecordingLoop:
    """Stands in for the loop's default handler, so the test sees what was passed on to it."""

    def __init__(self) -> None:
        self.passed_on: list[dict] = []

    def default_exception_handler(self, context: dict) -> None:
        self.passed_on.append(context)


def test_a_browser_resetting_its_socket_is_logged_as_expected_not_as_an_error(caplog: pytest.LogCaptureFixture) -> None:
    loop = _RecordingLoop()
    reset = {"message": "Exception in callback _call_connection_lost", "exception": ConnectionResetError(10054, "reset")}

    with caplog.at_level("INFO", logger="nav.sinks.web"):
        _quiet_client_resets(loop, reset)

    assert loop.passed_on == []
    assert any("caught ConnectionResetError, expected" in record.message for record in caplog.records)
    # Anything else still reaches asyncio's own handler, so a real fault is not swallowed.
    other = {"message": "something else", "exception": RuntimeError("not a reset")}
    _quiet_client_resets(loop, other)
    assert loop.passed_on == [other]


def test_a_started_sink_installs_the_reset_handler_on_its_loop(sink: WebSink) -> None:
    # The handler only quiets resets if the server's own loop uses it. Tested apart from the handler,
    # because a refactor of _run can drop the one line that installs it with every other test green.
    assert sink._loop is not None
    assert sink._loop.get_exception_handler() is _quiet_client_resets
