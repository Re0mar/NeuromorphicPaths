"""
Covers the web sink against a real aiohttp client over a real socket, not a mock of the app.

The sink runs its server in its own thread. The test drives a client from a short asyncio run
on the test thread, which is exactly the two-loop arrangement the design chose.
"""

# Standard library imports
import asyncio
import dataclasses
import json
import threading
import time

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.config import RunConfig, SinkKind, SourceKind, build_sink, build_source_and_sink
from nav.planner.config import GoalMode
from nav.scene.config import SceneConfig
from nav.sinks.config import WebConfig
from nav.sinks.floor_geometry import floor_hidden_mask
from nav.sinks.web import OUTGOING_QUEUE_LIMIT, WebSink, _quiet_client_resets
from nav.sinks.web_messages import OBSTACLE_KEYS, PLAN_VIEW_KEYS, RING_KEYS, PictureKind, WebMessageKind
from nav.sources.framecodec import path_message
from nav.sources.scene_video import AccessUnit, SceneVideoFeed, VideoDescription, unpack_unit
from nav.types import DebugSink, DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, PlannedPath, Pose
from synthetic_depth import intrinsics, level_floor_depth
from web_samples import sample_field, sample_path, sample_view

RECEIVE_TIMEOUT_SECONDS = 3.0





def _session():
    """A client that accepts the page's self-signed certificate. aiohttp is the web extra, so imported here."""
    import aiohttp

    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))


async def _receive_frames(port: int, after_connect, count: int) -> list:
    """The first count websocket frames after connecting, each as a str or bytes."""
    import aiohttp

    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{port}/ws") as socket:
            after_connect()
            received = []
            for _ in range(count):
                message = await asyncio.wait_for(socket.receive(), RECEIVE_TIMEOUT_SECONDS)
                assert message.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY), message
                received.append(message.data)
            return received


@pytest.fixture
def sink():
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    sink.start()
    assert sink.port is not None and sink.port > 0
    yield sink
    sink.close()


async def _receive_one(port: int, after_connect) -> str:
    import aiohttp

    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{port}/ws") as socket:
            after_connect()
            message = await asyncio.wait_for(socket.receive_str(), RECEIVE_TIMEOUT_SECONDS)
            return message


def test_a_published_path_reaches_a_connected_browser(sink: WebSink) -> None:
    path = sample_path(heading=0.3, alarm=True)

    # The publish has to happen after the client is connected, or there is nobody to send to.
    received = asyncio.run(_receive_one(sink.port, after_connect=lambda: sink.publish(path)))

    assert json.loads(received) == {"kind": WebMessageKind.PATH.value, **path_message(path)}


def test_the_page_is_served_at_the_root(sink: WebSink) -> None:
    async def fetch() -> str:
        import aiohttp

        async with _session() as session:
            async with session.get(f"https://127.0.0.1:{sink.port}/") as response:
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

        async with _session() as session:
            async with session.get(f"https://127.0.0.1:{sink.port}/") as response:
                return await response.text()

    page = asyncio.run(fetch())

    for key in PLAN_VIEW_KEYS + OBSTACLE_KEYS + RING_KEYS:
        assert f".{key}" in page, f"the page never reads {key}"


def test_the_served_page_dispatches_on_every_message_kind(sink: WebSink) -> None:
    async def fetch() -> str:
        async with _session() as session:
            async with session.get(f"https://127.0.0.1:{sink.port}/") as response:
                return await response.text()

    page = asyncio.run(fetch())

    for kind in WebMessageKind:
        assert f'"{kind.value}"' in page, f"the page never dispatches on {kind.value}"


def test_a_client_that_disconnects_does_not_break_the_next_publish(sink: WebSink) -> None:
    async def scenario() -> str:
        import aiohttp

        async with _session() as session:
            leaver = await session.ws_connect(f"wss://127.0.0.1:{sink.port}/ws")
            async with session.ws_connect(f"wss://127.0.0.1:{sink.port}/ws") as stayer:
                await leaver.close()
                # Give the server a moment to notice the leaver has gone.
                await asyncio.sleep(0.2)
                sink.publish(sample_path(heading=0.2))
                return await asyncio.wait_for(stayer.receive_str(), RECEIVE_TIMEOUT_SECONDS)

    received = asyncio.run(scenario())

    assert json.loads(received)["lookahead_heading_radians"] == pytest.approx(0.2)


def test_close_returns_within_two_seconds() -> None:
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    sink.start()

    started = time.monotonic()
    sink.close()

    assert time.monotonic() - started < 2.0


def test_publishing_with_no_browser_connected_does_not_raise(sink: WebSink) -> None:
    sink.publish(sample_path())


def test_a_port_already_in_use_is_reported_rather_than_hung() -> None:
    first = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    first.start()
    try:
        second = WebSink(WebConfig(port=first.port), SceneConfig(), video_feed=None)
        with pytest.raises(ConnectionError, match="could not start"):
            second.start()
    finally:
        first.close()


def test_a_sink_that_failed_to_start_closes_without_raising() -> None:
    """A run whose web port was refused still closes every sink on its way out."""
    first = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    first.start()
    try:
        second = WebSink(WebConfig(port=first.port), SceneConfig(), video_feed=None)
        with pytest.raises(ConnectionError):
            second.start()
        second.close()
    finally:
        first.close()


def test_closing_an_unstarted_sink_does_not_raise() -> None:
    WebSink(WebConfig(port=0), SceneConfig(), video_feed=None).close()


def test_the_page_is_served_over_tls_and_not_over_plain_http(sink: WebSink) -> None:
    # The browser's video decoder only exists on a secure origin, so a plain-http page would be a
    # page with no live video. A plain request must fail at the handshake, never answer 200.
    async def plain_fetch() -> int | None:
        import aiohttp

        async with _session() as session:
            try:
                async with session.get(f"http://127.0.0.1:{sink.port}/") as response:
                    return response.status
            except (aiohttp.ClientConnectorError, aiohttp.ClientResponseError, aiohttp.ServerDisconnectedError):
                return None

    assert asyncio.run(plain_fetch()) != 200


def test_a_missing_certificate_is_refused_by_the_config_with_the_path_in_the_message(tmp_path) -> None:
    missing = tmp_path / "nowhere.crt"

    with pytest.raises(ValueError, match="certificate_path") as refusal:
        WebConfig(port=0, certificate_path=str(missing))

    assert str(missing) in str(refusal.value)


def test_a_damaged_certificate_is_reported_from_start_and_the_sink_still_closes(tmp_path) -> None:
    damaged = tmp_path / "damaged.crt"
    damaged.write_text("not a certificate", encoding="utf-8")
    sink = WebSink(WebConfig(port=0, certificate_path=str(damaged)), SceneConfig(), video_feed=None)

    with pytest.raises(ConnectionError, match="could not start") as refusal:
        sink.start()

    assert str(damaged) in str(refusal.value)
    sink.close()


def test_a_browser_that_stops_reading_cannot_grow_the_queue_without_bound(sink: WebSink, caplog: pytest.LogCaptureFixture) -> None:
    # The pipeline hands over a path per planned frame, plus a plan view and a 128 KB picture up to
    # ten times a second, and never waits, while one browser whose window has closed suspends the
    # send loop for every browser. Unbounded, a backgrounded phone browser grew that queue by
    # hundreds of megabytes over a walk. The publish count is high because a path is about 250
    # bytes, and the socket's own buffers absorb a few hundred before the send loop stalls.
    async def scenario() -> int:
        import aiohttp

        async with _session() as session:
            async with session.ws_connect(f"wss://127.0.0.1:{sink.port}/ws"):
                await asyncio.sleep(0.3)
                # The client never reads from here on.
                for _ in range(20_000):
                    sink.publish_debug(sample_path(), *sample_field(), sample_view())
                await asyncio.sleep(1.0)
                assert sink._outgoing is not None
                return sink._outgoing.qsize()

    depth = asyncio.run(scenario())

    assert depth <= OUTGOING_QUEUE_LIMIT, f"the queue grew to {depth}"
    assert sink.dropped > 0, "about 5 MB to a browser that never reads must have dropped some"

    with caplog.at_level("INFO", logger="nav.sinks.web"):
        sink.close()
    assert any("were dropped because the browsers were not keeping up" in record.message for record in caplog.records)


def test_the_web_sink_is_a_debug_sink() -> None:
    # The loop dispatches on this. Without it the browser would get the path and never the view.
    assert isinstance(WebSink(WebConfig(port=0), SceneConfig(), video_feed=None), DebugSink)


def test_a_debug_publish_sends_path_then_plan_view_then_png(sink: WebSink) -> None:
    path = sample_path(heading=0.3)
    field, grid = sample_field()

    path_text, plan_text, png = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, field, grid, sample_view()), count=3))

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
    path = sample_path(heading=0.4)
    field, grid = sample_field()
    sink.publish_debug(path, field, grid, sample_view())
    time.sleep(0.3)  # Let the server loop take all three messages off its queue.

    path_text, plan_text, png = asyncio.run(_receive_frames(sink.port, lambda: None, count=3))

    assert json.loads(path_text)["kind"] == WebMessageKind.PATH.value
    assert json.loads(path_text)["lookahead_heading_radians"] == pytest.approx(0.4)
    assert json.loads(plan_text)["kind"] == WebMessageKind.PLAN_VIEW.value
    assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n"


async def _receive_texts_then_check_quiet(port: int, after_connect, texts: int) -> tuple[list, bool]:
    """The first few text frames after connecting, and whether anything at all followed them."""
    import aiohttp

    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{port}/ws") as socket:
            after_connect()
            received = [await asyncio.wait_for(socket.receive(), RECEIVE_TIMEOUT_SECONDS) for _ in range(texts)]
            try:
                await asyncio.wait_for(socket.receive(), 0.5)
                return [message.data for message in received], True
            except TimeoutError:
                return [message.data for message in received], False


def test_a_png_encode_failure_still_sends_the_path_and_the_plan_view(sink: WebSink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import nav.sinks.web as web_module

    def broken(view, path, **options):
        raise ValueError("no picture today")

    monkeypatch.setattr(web_module, "render_depth_view", broken)
    path = sample_path(heading=0.5)
    field, grid = sample_field()

    with caplog.at_level("WARNING", logger="nav.sinks.web"):
        texts, more_arrived = asyncio.run(_receive_texts_then_check_quiet(sink.port, lambda: sink.publish_debug(path, field, grid, sample_view()), 2))

    assert json.loads(texts[0])["lookahead_heading_radians"] == pytest.approx(0.5)
    assert json.loads(texts[1])["kind"] == WebMessageKind.PLAN_VIEW.value
    assert not more_arrived, "nothing may follow the plan view when the picture failed"
    assert any("depth view not sent" in record.message for record in caplog.records)


def test_a_plan_view_that_cannot_be_built_still_sends_the_path_and_the_png(sink: WebSink, caplog: pytest.LogCaptureFixture) -> None:
    path = sample_path(heading=0.6)
    field, grid = sample_field()
    wrong_field = np.zeros((field.shape[0] + 1, field.shape[1]))

    with caplog.at_level("WARNING", logger="nav.sinks.web"):
        received = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, wrong_field, grid, sample_view()), count=2))

    assert json.loads(received[0])["kind"] == WebMessageKind.PATH.value
    assert isinstance(received[1], bytes), "the picture follows the path straight away"
    assert any("plan view not sent" in record.message for record in caplog.records)
    # The same publish with a field that fits sends all three, so the gap above is the bad field.
    good = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, field, grid, sample_view()), count=4))
    kinds = [json.loads(message)["kind"] for message in good if isinstance(message, str)]
    assert WebMessageKind.PLAN_VIEW.value in kinds


# A picture no real view draws: a flat 8 by 8 of one color, so a frame can say which renderer made it.
RISK_MARKER_BGR = (11, 22, 33)


def _picture_message(picture: PictureKind) -> str:
    return json.dumps({"kind": "picture", "picture": picture.value})


def _is_risk_marker(png: bytes) -> bool:
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    return decoded.shape == (8, 8, 3) and bool(np.all(decoded == RISK_MARKER_BGR))


@pytest.fixture
def risk_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replaces the risk renderer with one that draws the marker, and records each call."""
    import nav.sinks.web as web_module

    calls = []

    def marker(view, path, field, grid, scene):
        calls.append(path)
        return np.full((8, 8, 3), RISK_MARKER_BGR, dtype=np.uint8)

    monkeypatch.setattr(web_module, "render_risk_view", marker)
    return calls


async def _say_then_receive_until_quiet(port: int, said: list[str], after_saying, quiet_seconds: float) -> list:
    """Every frame after connecting and sending each of said, until none arrives for quiet_seconds."""
    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{port}/ws") as socket:
            for text in said:
                await socket.send_str(text)
            # The server reads the choice on its own loop. A moment's wait makes sure it has before the publish.
            await asyncio.sleep(0.2)
            after_saying()
            received = []
            while True:
                try:
                    message = await asyncio.wait_for(socket.receive(), quiet_seconds)
                except TimeoutError:
                    return received
                received.append(message.data)


def test_a_browser_that_asks_for_risk_gets_the_risk_picture_and_never_the_depth_one(sink: WebSink, risk_calls: list) -> None:
    field, grid = sample_field()

    received = asyncio.run(
        _say_then_receive_until_quiet(sink.port, [_picture_message(PictureKind.RISK)], lambda: sink.publish_debug(sample_path(), field, grid, sample_view()), 0.5)
    )

    pictures = [message for message in received if isinstance(message, bytes)]
    assert len(pictures) == 1 and _is_risk_marker(pictures[0]), "one picture, and it is the risk view"
    assert len(risk_calls) == 1


def test_no_risk_picture_is_drawn_while_no_browser_shows_it(sink: WebSink, risk_calls: list) -> None:
    field, grid = sample_field()

    received = asyncio.run(_say_then_receive_until_quiet(sink.port, [], lambda: sink.publish_debug(sample_path(), field, grid, sample_view()), 0.5))

    pictures = [message for message in received if isinstance(message, bytes)]
    assert len(pictures) == 1 and not _is_risk_marker(pictures[0]), "the depth view, as before"
    assert risk_calls == [], "nobody asked, so it was never drawn"


def test_switching_back_to_depth_stops_the_risk_pictures(sink: WebSink, risk_calls: list) -> None:
    field, grid = sample_field()
    choices = [_picture_message(PictureKind.RISK), _picture_message(PictureKind.DEPTH)]

    received = asyncio.run(_say_then_receive_until_quiet(sink.port, choices, lambda: sink.publish_debug(sample_path(), field, grid, sample_view()), 0.5))

    pictures = [message for message in received if isinstance(message, bytes)]
    assert pictures and not any(_is_risk_marker(picture) for picture in pictures)
    assert risk_calls == []


def test_asking_for_risk_on_a_still_source_redraws_the_last_view(sink: WebSink, risk_calls: list) -> None:
    # One frame, then nothing, as from a phone on a table. The risk view still has to appear.
    field, grid = sample_field()
    sink.publish_debug(sample_path(heading=0.2), field, grid, sample_view())
    time.sleep(0.3)

    received = asyncio.run(_say_then_receive_until_quiet(sink.port, [_picture_message(PictureKind.RISK)], lambda: None, 0.5))

    assert any(isinstance(message, bytes) and _is_risk_marker(message) for message in received)
    assert len(risk_calls) == 1


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("not json", "not JSON"),
        (json.dumps({"kind": "volume", "picture": "risk"}), "does not read"),
        (json.dumps({"kind": "picture", "picture": "x-ray"}), "does not draw"),
    ],
)
def test_a_message_the_laptop_cannot_read_is_logged_and_changes_nothing(sink: WebSink, risk_calls: list, caplog: pytest.LogCaptureFixture, text: str, fragment: str) -> None:
    field, grid = sample_field()

    with caplog.at_level("WARNING", logger="nav.sinks.web"):
        received = asyncio.run(_say_then_receive_until_quiet(sink.port, [text], lambda: sink.publish_debug(sample_path(), field, grid, sample_view()), 0.5))

    pictures = [message for message in received if isinstance(message, bytes)]
    assert len(pictures) == 1 and not _is_risk_marker(pictures[0])
    assert any(fragment in record.message for record in caplog.records), [record.message for record in caplog.records]


def test_the_depth_picture_leaves_the_rings_to_the_page(sink: WebSink, monkeypatch: pytest.MonkeyPatch) -> None:
    import nav.sinks.web as web_module

    asked = []
    real_render = web_module.render_depth_view

    def recording_render(view, path, **options):
        asked.append(options)
        return real_render(view, path, **options)

    monkeypatch.setattr(web_module, "render_depth_view", recording_render)
    field, grid = sample_field()

    asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(sample_path(), field, grid, sample_view()), count=3))

    assert asked == [{"rings": False}]


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


# *******************************************
# The render thread
# *******************************************


def _numbered_path(number: int) -> PlannedPath:
    """A path that names itself through the one number the plan view carries over from it."""
    return PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), 0.1, False, 2.5, scene_information_bits=float(number), avoidance_surprise_bits=0.0)


async def _receive_until_quiet(port: int, after_connect, quiet_seconds: float) -> list:
    """Every frame after connecting, until none arrives for quiet_seconds."""
    import aiohttp

    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{port}/ws") as socket:
            after_connect()
            received = []
            while True:
                try:
                    message = await asyncio.wait_for(socket.receive(), quiet_seconds)
                except TimeoutError:
                    return received
                received.append(message.data)


def test_publish_debug_returns_without_drawing(sink: WebSink, monkeypatch: pytest.MonkeyPatch) -> None:
    """The publisher thread hands over and moves on. A slow picture must not hold up the phone's next path."""
    import nav.sinks.web as web_module

    real_render = web_module.render_depth_view

    def slow_render(view, path, **options):
        time.sleep(0.3)
        return real_render(view, path, **options)

    monkeypatch.setattr(web_module, "render_depth_view", slow_render)
    field, grid = sample_field()

    started = time.perf_counter()
    for _ in range(3):
        sink.publish_debug(sample_path(), field, grid, sample_view())
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, f"three publishes took {elapsed * 1000:.0f} ms"


def test_every_path_goes_out_and_pictures_are_capped_with_the_newest_drawn_last(sink: WebSink) -> None:
    published = 40
    field, grid = sample_field()

    def publish_quickly() -> None:
        # 20 ms apart, faster than the cap of 10 pictures a second.
        for number in range(published):
            sink.publish_debug(_numbered_path(number), field, grid, sample_view())
            time.sleep(0.02)

    started = time.perf_counter()
    received = asyncio.run(_receive_until_quiet(sink.port, publish_quickly, quiet_seconds=0.5))
    elapsed = time.perf_counter() - started

    texts = [json.loads(message) for message in received if isinstance(message, str)]
    paths = [text for text in texts if text["kind"] == WebMessageKind.PATH.value]
    plan_views = [text for text in texts if text["kind"] == WebMessageKind.PLAN_VIEW.value]
    pictures = [message for message in received if isinstance(message, bytes)]

    assert len(paths) == published, "every path goes out, whatever the pictures do"
    # One picture per tenth of a second, plus the first, which has no interval to wait out.
    assert 2 <= len(pictures) <= elapsed * 10 + 1, f"{len(pictures)} pictures in {elapsed:.2f} s"
    assert len(plan_views) == len(pictures)
    assert plan_views[-1]["scene_information_bits"] == published - 1, "the last path published is the last one drawn"


def test_close_stops_the_render_thread_with_a_view_still_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    import nav.sinks.web as web_module

    real_render = web_module.render_depth_view

    def slow_render(view, path, **options):
        time.sleep(0.3)
        return real_render(view, path, **options)

    monkeypatch.setattr(web_module, "render_depth_view", slow_render)
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    sink.start()
    render_thread = sink._render_thread
    field, grid = sample_field()
    for _ in range(3):
        sink.publish_debug(sample_path(), field, grid, sample_view())

    started = time.perf_counter()
    sink.close()

    assert time.perf_counter() - started < 2.0
    assert render_thread is not None and not render_thread.is_alive()


def test_an_unexpected_drawing_error_stops_the_pictures_and_the_paths_keep_going(sink: WebSink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import nav.sinks.web as web_module

    real_render = web_module.render_depth_view
    calls: list[int] = []

    def broken_once(view, path, **options):
        # Only the first drawing fails, so a picture for the second path means the thread carried on.
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("a bug in the drawing")
        return real_render(view, path, **options)

    monkeypatch.setattr(web_module, "render_depth_view", broken_once)
    field, grid = sample_field()

    def publish_twice() -> None:
        sink.publish_debug(_numbered_path(0), field, grid, sample_view())
        time.sleep(0.3)
        sink.publish_debug(_numbered_path(1), field, grid, sample_view())

    with caplog.at_level("ERROR", logger="nav.sinks.web"):
        received = asyncio.run(_receive_until_quiet(sink.port, publish_twice, quiet_seconds=0.6))

    paths = [json.loads(message) for message in received if isinstance(message, str) and json.loads(message)["kind"] == WebMessageKind.PATH.value]
    assert len(paths) == 2
    assert not any(isinstance(message, bytes) for message in received)
    assert any("UNEXPECTED RuntimeError drawing the debug view" in record.message for record in caplog.records)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_a_picture_rate_that_is_not_positive_is_refused(rate: float) -> None:
    with pytest.raises(ValueError, match="max_pictures_per_second"):
        WebConfig(max_pictures_per_second=rate)


def test_a_publish_after_close_starts_no_new_server() -> None:
    """A publisher still inside another display at shutdown can reach this one after it closed."""
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    sink.start()
    sink.close()
    threads_before = {thread.name for thread in threading.enumerate()}

    sink.publish(sample_path())
    sink.publish_debug(sample_path(), *sample_field(), sample_view())

    assert sink._thread is None
    started = {thread.name for thread in threading.enumerate()} - threads_before
    assert not started & {"web-sink", "web-render"}


def test_views_are_not_queued_once_the_render_thread_has_ended(sink: WebSink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import nav.sinks.web as web_module

    def broken(view, path, **options):
        raise RuntimeError("a bug in the drawing")

    monkeypatch.setattr(web_module, "render_depth_view", broken)
    field, grid = sample_field()
    sink.publish_debug(sample_path(), field, grid, sample_view())
    sink._render_thread.join(RECEIVE_TIMEOUT_SECONDS)
    for _ in range(5):
        sink.publish_debug(sample_path(), field, grid, sample_view())

    with caplog.at_level("INFO", logger="nav.sinks.web"):
        sink.close()
    assert not any("replaced by a newer one" in record.message for record in caplog.records)


def test_a_draw_that_outlives_close_is_dropped_quietly(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """
    The draw takes 2.3 s, past close()'s 2 s wait for the render thread, and the server's join is
    held 0.6 s longer, so the draw finishes after the server's loop closed but before the sink lets
    go of it. It used to raise there, logged as an unexpected drawing error.
    """
    import nav.sinks.web as web_module

    real_render = web_module.render_depth_view

    def slow_render(view, path, **options):
        time.sleep(2.3)
        return real_render(view, path, **options)

    monkeypatch.setattr(web_module, "render_depth_view", slow_render)
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=None)
    sink.start()
    server_thread = sink._thread
    real_join = server_thread.join

    def slow_join(timeout=None):
        real_join(timeout)
        time.sleep(0.6)

    server_thread.join = slow_join
    sink.publish_debug(sample_path(), *sample_field(), sample_view())
    time.sleep(0.05)
    with caplog.at_level("DEBUG", logger="nav.sinks.web"):
        sink.close()
        time.sleep(1.0)

    assert not any("UNEXPECTED" in record.message for record in caplog.records)


# *******************************************
# The video socket and the recording
# *******************************************

VIDEO_DESCRIPTION = VideoDescription(codec="avc1.42801f", parameter_sets=(b"\x00\x00\x00\x01\x67sps", b"\x00\x00\x00\x01\x68pps"))
UNIT_INTERVAL_SECONDS = 1.0 / 30.0
# Big enough that a browser which stops reading fills its socket's buffers within a few units.
STALL_UNIT_BYTES = 30_000


def _unit(index: int, keyframe: bool, size: int = 64) -> AccessUnit:
    """Unit number `index` of a stream, its stamp a frame interval apart from its neighbors."""
    return AccessUnit(timestamp_seconds=index * UNIT_INTERVAL_SECONDS, data=bytes([index % 256]) * size, keyframe=keyframe)


def _index_of(unit: AccessUnit) -> int:
    return round(unit.timestamp_seconds / UNIT_INTERVAL_SECONDS)


@pytest.fixture
def feed() -> SceneVideoFeed:
    return SceneVideoFeed()


@pytest.fixture
def video_sink(feed: SceneVideoFeed):
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=feed)
    sink.start()
    yield sink
    sink.close()


async def _receive_video(port: int, after_connect, count: int) -> list:
    """The first count messages on the video socket after connecting, whole, type included."""
    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{port}/video") as socket:
            after_connect()
            return [await asyncio.wait_for(socket.receive(), RECEIVE_TIMEOUT_SECONDS) for _ in range(count)]


async def _read_until_quiet(socket, quiet_seconds: float) -> list:
    received = []
    while True:
        try:
            message = await asyncio.wait_for(socket.receive(), quiet_seconds)
        except TimeoutError:
            return received
        received.append(message)


def _units_in(messages: list) -> list[AccessUnit]:
    import aiohttp

    return [unpack_unit(message.data) for message in messages if message.type == aiohttp.WSMsgType.BINARY]


def test_a_browser_on_the_video_socket_gets_the_description_then_units_from_the_next_keyframe(video_sink: WebSink, feed: SceneVideoFeed) -> None:
    import aiohttp

    feed.describe(VIDEO_DESCRIPTION)

    def offer_two_deltas_then_a_keyframe_and_a_delta() -> None:
        for index, keyframe in ((0, False), (1, False), (2, True), (3, False)):
            feed.offer(_unit(index, keyframe))

    received = asyncio.run(_receive_video(video_sink.port, offer_two_deltas_then_a_keyframe_and_a_delta, count=3))

    assert received[0].type == aiohttp.WSMsgType.TEXT
    description = json.loads(received[0].data)
    assert description["kind"] == WebMessageKind.VIDEO_STREAM.value
    assert description["codec"] == "avc1.42801f"
    units = _units_in(received)
    assert [(_index_of(unit), unit.keyframe) for unit in units] == [(2, True), (3, False)], "the two deltas before the keyframe never go out"


def test_a_description_that_arrives_after_the_browser_connected_is_still_sent_first(video_sink: WebSink, feed: SceneVideoFeed) -> None:
    import aiohttp

    def describe_then_offer() -> None:
        feed.describe(VIDEO_DESCRIPTION)
        feed.offer(_unit(0, keyframe=True))

    received = asyncio.run(_receive_video(video_sink.port, describe_then_offer, count=2))

    assert [message.type for message in received] == [aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY]
    assert json.loads(received[0].data)["kind"] == WebMessageKind.VIDEO_STREAM.value


def test_a_run_without_scene_video_tells_the_browser_so_and_closes(sink: WebSink) -> None:
    import aiohttp

    received = asyncio.run(_receive_video(sink.port, lambda: None, count=2))

    assert received[0].type == aiohttp.WSMsgType.TEXT
    message = json.loads(received[0].data)
    assert message["kind"] == WebMessageKind.VIDEO_UNAVAILABLE.value
    assert "no scene video" in message["reason"]
    assert received[1].type == aiohttp.WSMsgType.CLOSE


def test_two_browsers_each_get_every_unit(video_sink: WebSink, feed: SceneVideoFeed) -> None:
    feed.describe(VIDEO_DESCRIPTION)

    async def scenario() -> tuple[list, list]:
        async with _session() as session:
            async with session.ws_connect(f"wss://127.0.0.1:{video_sink.port}/video") as first:
                async with session.ws_connect(f"wss://127.0.0.1:{video_sink.port}/video") as second:
                    await asyncio.sleep(0.2)
                    for index, keyframe in ((0, True), (1, False), (2, False)):
                        feed.offer(_unit(index, keyframe))
                    return await _read_until_quiet(first, 0.5), await _read_until_quiet(second, 0.5)

    first, second = asyncio.run(scenario())

    assert [_index_of(unit) for unit in _units_in(first)] == [0, 1, 2]
    assert [_index_of(unit) for unit in _units_in(second)] == [0, 1, 2]


STALL_UNITS = 200
AFTER_STALL_UNITS = 60
KEYFRAME_EVERY = 30


async def _one_browser_reads_and_one_stalls(sink: WebSink, feed: SceneVideoFeed) -> tuple[list, list]:
    """
    Two browsers. One reads all along. The other reads nothing while the first batch of units is
    offered, then reads while a second batch, with keyframes in it, follows.
    """
    def offer(index: int) -> None:
        feed.offer(_unit(index, keyframe=index % KEYFRAME_EVERY == 0, size=STALL_UNIT_BYTES))

    async with _session() as session:
        async with session.ws_connect(f"wss://127.0.0.1:{sink.port}/video") as reader:
            async with session.ws_connect(f"wss://127.0.0.1:{sink.port}/video") as staller:
                await asyncio.sleep(0.2)
                feed.describe(VIDEO_DESCRIPTION)
                reading = asyncio.ensure_future(_read_until_quiet(reader, 1.5))
                for index in range(STALL_UNITS):
                    offer(index)
                    await asyncio.sleep(0.002)
                # Only now does the stalled browser read, with the stream still going.
                late_reading = asyncio.ensure_future(_read_until_quiet(staller, 1.5))
                for index in range(STALL_UNITS, STALL_UNITS + AFTER_STALL_UNITS):
                    offer(index)
                    await asyncio.sleep(0.002)
                return await reading, await late_reading


def test_a_video_client_that_stops_reading_is_resynced_at_a_keyframe_and_the_others_are_not(video_sink: WebSink, feed: SceneVideoFeed) -> None:
    read_all_along, read_late = asyncio.run(_one_browser_reads_and_one_stalls(video_sink, feed))

    assert [_index_of(unit) for unit in _units_in(read_all_along)] == list(range(STALL_UNITS + AFTER_STALL_UNITS)), "the reading browser lost nothing"
    assert video_sink._video_dropped > 0, "6 MB to a browser that never reads must have dropped some"
    late_units = _units_in(read_late)
    assert late_units, "the stalled browser gets something once it reads"
    assert len(late_units) < STALL_UNITS + AFTER_STALL_UNITS, "the stalled browser did not get everything, so the queue is bounded"
    gaps = [index for index in range(1, len(late_units)) if _index_of(late_units[index]) != _index_of(late_units[index - 1]) + 1]
    assert gaps, "a stalled browser must have lost a stretch"
    assert all(late_units[index].keyframe for index in gaps), "after every gap the stalled browser resumes on a keyframe"
    assert _index_of(late_units[-1]) == STALL_UNITS + AFTER_STALL_UNITS - 1, "once resynced the stalled browser keeps up to the end"


def test_a_video_client_never_receives_a_delta_after_a_dropped_frame(video_sink: WebSink, feed: SceneVideoFeed) -> None:
    _, read_late = asyncio.run(_one_browser_reads_and_one_stalls(video_sink, feed))

    units = _units_in(read_late)
    for previous, unit in zip(units, units[1:]):
        if _index_of(unit) != _index_of(previous) + 1:
            assert unit.keyframe, f"unit {_index_of(unit)} is a delta sent after unit {_index_of(previous)}, whose successor was dropped"


def test_a_unit_offered_on_the_sources_feed_reaches_a_video_client_through_build_source_and_sink() -> None:
    # End to end from the command line to a browser: the one place that builds both ends hands the
    # feed to each, the source's camera exposes it, the sink subscribes and the socket delivers.
    # The source is built, never connected, so no device process starts.
    import aiohttp
    import dataclasses

    from nav.config import build_run_config, build_source_and_sink
    from stubs import StubDepthEstimator

    # Port 0 is a free port, which the command line refuses on purpose, so it is swapped in after parsing.
    config = dataclasses.replace(
        build_run_config(["--source", "neon_live", "--sink", "web"]),
        web=WebConfig(port=0),
        estimator_factory=lambda estimator_config: StubDepthEstimator(),
    )
    source, sink = build_source_and_sink(config)
    sink.start()
    try:
        feed = source.rgb_source.video_feed

        def describe_and_offer() -> None:
            feed.describe(VIDEO_DESCRIPTION)
            feed.offer(_unit(7, keyframe=True))

        received = asyncio.run(_receive_video(sink.port, describe_and_offer, count=2))
    finally:
        sink.close()

    assert [message.type for message in received] == [aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY]
    assert json.loads(received[0].data)["codec"] == "avc1.42801f"
    assert _index_of(unpack_unit(received[1].data)) == 7


def test_close_returns_within_two_seconds_with_a_video_client_connected(feed: SceneVideoFeed) -> None:
    sink = WebSink(WebConfig(port=0), SceneConfig(), video_feed=feed)
    sink.start()

    async def scenario() -> float:
        async with _session() as session:
            async with session.ws_connect(f"wss://127.0.0.1:{sink.port}/video"):
                await asyncio.sleep(0.2)
                started = time.monotonic()
                await asyncio.get_running_loop().run_in_executor(None, sink.close)
                return time.monotonic() - started

    assert asyncio.run(scenario()) < 2.0


def _recording_file(tmp_path, size: int = 100):
    recording = tmp_path / "demo.mp4"
    recording.write_bytes(bytes(range(size)))
    return recording


def test_the_recording_route_serves_the_file_with_ranges_and_answers_head(tmp_path) -> None:
    recording = _recording_file(tmp_path)
    sink = WebSink(WebConfig(port=0, recording_path=str(recording)), SceneConfig(), video_feed=None)
    sink.start()

    async def fetch() -> tuple:
        async with _session() as session:
            async with session.get(f"https://127.0.0.1:{sink.port}/recording", headers={"Range": "bytes=0-9"}) as partial:
                partial_status, partial_body = partial.status, await partial.read()
            async with session.head(f"https://127.0.0.1:{sink.port}/recording") as head:
                return partial_status, partial_body, head.status, head.headers.get("Content-Length")

    try:
        partial_status, partial_body, head_status, length = asyncio.run(fetch())
    finally:
        sink.close()

    assert partial_status == 206 and partial_body == bytes(range(10))
    assert head_status == 200 and length == "100"


def test_the_recording_status_says_whether_a_recording_exists_without_a_failed_request(tmp_path, sink: WebSink) -> None:
    recording = _recording_file(tmp_path)
    with_recording = WebSink(WebConfig(port=0, recording_path=str(recording)), SceneConfig(), video_feed=None)
    with_recording.start()

    async def status_of(port: int) -> tuple[int, dict]:
        async with _session() as session:
            async with session.get(f"https://127.0.0.1:{port}/recording/status") as response:
                return response.status, await response.json()

    try:
        assert asyncio.run(status_of(with_recording.port)) == (200, {"available": True})
        assert asyncio.run(status_of(sink.port)) == (200, {"available": False})
    finally:
        with_recording.close()


def test_the_recording_route_is_404_without_a_recording(sink: WebSink) -> None:
    async def fetch() -> tuple[int, dict]:
        async with _session() as session:
            async with session.get(f"https://127.0.0.1:{sink.port}/recording") as response:
                return response.status, await response.json()

    status, body = asyncio.run(fetch())

    assert status == 404
    assert "no recording configured" in body["error"]


def test_a_recording_deleted_after_start_is_404_and_logged_once(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    recording = _recording_file(tmp_path)
    sink = WebSink(WebConfig(port=0, recording_path=str(recording)), SceneConfig(), video_feed=None)
    sink.start()
    recording.unlink()

    async def fetch_twice() -> list[int]:
        async with _session() as session:
            statuses = []
            for _ in range(2):
                async with session.get(f"https://127.0.0.1:{sink.port}/recording") as response:
                    statuses.append(response.status)
            return statuses

    try:
        with caplog.at_level("WARNING", logger="nav.sinks.web"):
            statuses = asyncio.run(fetch_twice())
    finally:
        sink.close()

    assert statuses == [404, 404]
    warnings = [record for record in caplog.records if "is gone" in record.getMessage()]
    assert len(warnings) == 1 and str(recording) in warnings[0].getMessage()


def test_a_browser_gets_the_hidden_floor_under_the_runs_scene_config() -> None:
    # End to end through the production path: the run config, build_sink, the render thread, the
    # websocket. The run narrows the usable depth to 20 m, and pixel (64, 86), step 30 straight
    # ahead, reads 25 m. Under the default 30 m that reading is just beyond the floor and hides
    # nothing. Under the run's 20 m it is no reading, so the cell the browser gets must be hidden.
    # A sink that dropped the run's config anywhere along the way would send the default's answer.
    run_scene = dataclasses.replace(SceneConfig(), max_depth_meters=20.0)
    config = RunConfig(
        source_kind=SourceKind.LOGGED,
        sink_kinds=(SinkKind.WEB,),
        goal_mode=GoalMode.AHEAD,
        web=WebConfig(port=0),
        scene=run_scene,
    )
    depth = level_floor_depth()
    depth[86, 64] = 25.0
    frame = DepthFrame(1.0, depth, intrinsics(), Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False), None, None)
    view = DebugView(frame, ObstacleSet(1.0, (), 0), Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30, 1.47)
    times = np.arange(39) * 0.1
    grid = np.linspace(-3.0, 3.0, 61)
    path = PlannedPath(1.0, times, np.zeros(len(times)), 0.0, False, 1.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)

    sink = build_sink(config)
    sink.start()
    try:
        _, plan_text, _ = asyncio.run(_receive_frames(sink.port, lambda: sink.publish_debug(path, np.zeros((len(times), len(grid))), grid, view), count=3))
    finally:
        sink.close()

    hidden = np.array(json.loads(plan_text)["floor_hidden"])
    assert hidden[30, 30]
    assert not floor_hidden_mask(view, times, grid, SceneConfig())[30, 30]
    assert np.array_equal(hidden, floor_hidden_mask(view, times, grid, run_scene))


def test_a_missing_key_file_is_refused_by_the_config_with_the_path_in_the_message(tmp_path) -> None:
    missing = tmp_path / "nowhere.key"

    with pytest.raises(ValueError, match="key_path") as refusal:
        WebConfig(port=0, key_path=str(missing))

    assert str(missing) in str(refusal.value)


@pytest.mark.parametrize("limit", [0, 1, -5])
def test_a_video_queue_limit_under_two_is_refused_naming_the_bound(limit: int) -> None:
    # The resync puts the description back and then one more item, so one slot cannot hold both.
    with pytest.raises(ValueError, match="at least 2"):
        WebConfig(port=0, video_queue_limit=limit)


def test_a_video_queue_limit_of_two_is_accepted() -> None:
    assert WebConfig(port=0, video_queue_limit=2).video_queue_limit == 2


def test_a_second_description_reaching_a_full_two_slot_queue_is_queued_after_the_resync_and_nothing_escapes(feed: SceneVideoFeed, caplog: pytest.LogCaptureFixture) -> None:
    # The glasses reconnecting describe the stream again. With the smallest queue allowed and a
    # browser that reads nothing, the resync's two puts must still fit, or asyncio logs an
    # escaped QueueFull and the browser is left with a half-done resync.
    sink = WebSink(WebConfig(port=0, video_queue_limit=2), SceneConfig(), video_feed=feed)
    sink.start()
    second = VideoDescription(codec="avc1.42801f", parameter_sets=(b"\x00\x00\x00\x01\x67new", b"\x00\x00\x00\x01\x68new"))

    async def stall_then_read() -> list:
        async with _session() as session:
            async with session.ws_connect(f"wss://127.0.0.1:{sink.port}/video") as staller:
                await asyncio.sleep(0.2)
                feed.describe(VIDEO_DESCRIPTION)
                for index in range(STALL_UNITS):
                    feed.offer(_unit(index, keyframe=index % KEYFRAME_EVERY == 0, size=STALL_UNIT_BYTES))
                    await asyncio.sleep(0.002)
                feed.describe(second)
                # The stream goes on with keyframes in it while the browser reads again, as it
                # would on a walk. The resync lands on one of them.
                late_reading = asyncio.ensure_future(_read_until_quiet(staller, 1.5))
                for index in range(STALL_UNITS, STALL_UNITS + AFTER_STALL_UNITS):
                    feed.offer(_unit(index, keyframe=index % KEYFRAME_EVERY == 0, size=STALL_UNIT_BYTES))
                    await asyncio.sleep(0.002)
                return await late_reading

    with caplog.at_level("ERROR"):
        try:
            received = asyncio.run(stall_then_read())
        finally:
            sink.close()

    assert not any("QueueFull" in record.getMessage() or "Exception in callback" in record.getMessage() for record in caplog.records)
    import aiohttp

    text_positions = [position for position, message in enumerate(received) if message.type == aiohttp.WSMsgType.TEXT]
    assert text_positions, "the stalled browser got a description once it read"
    assert json.loads(received[text_positions[-1]].data)["kind"] == WebMessageKind.VIDEO_STREAM.value
    after_last_description = _units_in(received[text_positions[-1] + 1:])
    assert after_last_description, "units follow the last description"
    assert after_last_description[0].keyframe, "after the resync the browser resumes on a keyframe"
