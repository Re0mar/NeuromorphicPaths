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
import numpy as np
import pytest

# Local package imports
from nav.sinks.config import WebConfig
from nav.sinks.web import WebSink
from nav.sources.framecodec import encode_path
from nav.types import PlannedPath

RECEIVE_TIMEOUT_SECONDS = 3.0


def _path(heading: float = 0.1, alarm: bool = False) -> PlannedPath:
    return PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), heading, alarm, 2.5)


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

    assert json.loads(received) == json.loads(encode_path(path))


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
