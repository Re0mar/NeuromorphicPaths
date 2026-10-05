"""
A browser as the display: one static page and one websocket, served from a thread of their own.

The only file in the package allowed to import aiohttp. The server runs its own asyncio loop in
its own thread, and the pipeline's publisher thread hands it messages through
call_soon_threadsafe, so the two never share a loop. No video is sent.

Each planned path goes out as a text frame, the path's JSON with a kind of "path". When the loop
hands this sink a debug view, two more frames follow: a second kind of text message, the plan view,
with the planner's field and what the page needs to draw it top-down, then a binary frame with a
PNG of the depth image the planner saw, groups and path drawn on it. The page dispatches text
frames on their kind. It draws the arrow and turns red on alarm, draws the plan view, and shows the
picture when one arrives.
"""

# Standard library imports
import asyncio
import logging
import threading
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.sinks.config import WebConfig
from nav.sinks.rendering import encode_png, render_depth_view
from nav.sinks.web_messages import WebMessageKind, plan_view_message, web_text_message
from nav.sources.framecodec import path_message
from nav.types import DebugView, PlannedPath

log = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "web_page.html"
STARTUP_TIMEOUT_SECONDS = 5.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0
# How many messages may wait for the browsers. A depth view is about 128 KB and a plan view up to
# about 60 KB, so this is a few megabytes at worst. The queue has to be bounded: the pipeline hands
# over a path, a plan view and a picture per planned frame and never waits, while one browser that
# stops reading suspends the send loop for every browser, so an unbounded queue grows for as long as
# that lasts.
OUTGOING_QUEUE_LIMIT = 32


class _Slot(Enum):
    """Which newest-of-its-kind message a payload replaces. A browser that connects gets one of each, in this order."""

    PATH = 1
    PLAN_VIEW = 2
    DEPTH_PNG = 3


class WebSink:
    """Serves the page and pushes every published path, and the depth view when given one, to every connected browser."""

    def __init__(self, config: WebConfig) -> None:
        self._config = config
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outgoing: asyncio.Queue | None = None
        self._bound_port: int | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Messages discarded because the browsers were not keeping up."""
        return self._dropped

    @property
    def port(self) -> int | None:
        """The port actually bound, once started. Differs from the config's when that was 0."""
        return self._bound_port

    def start(self) -> None:
        """Start the server thread and wait until it is listening, or raise what stopped it."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, name="web-sink", daemon=True)
        self._thread.start()
        self._ready.wait(STARTUP_TIMEOUT_SECONDS)
        if self._startup_error is not None:
            raise ConnectionError(f"web sink could not start on port {self._config.port}: {self._startup_error}") from self._startup_error
        if not self._ready.is_set():
            raise ConnectionError(f"web sink did not start listening on port {self._config.port} in time")

    def publish(self, path: PlannedPath) -> None:
        """Queue the path for every browser. Starts the server on the first call."""
        if self._thread is None:
            self.start()
        self._enqueue(_Slot.PATH, web_text_message(WebMessageKind.PATH, path_message(path)))

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
        """
        Queue the path's JSON, then the plan view, then a PNG of the depth view, for every browser.

        Each one that cannot be built costs itself and nothing else. A plan view that will not
        build still leaves the path and the picture, and a view that will not render still leaves
        the path and the plan view.
        """
        self.publish(path)
        try:
            plan_view = web_text_message(WebMessageKind.PLAN_VIEW, plan_view_message(path, field, grid, view))
        except ValueError as unbuildable:
            log.warning("plan view not sent (caught %s, expected): %s", type(unbuildable).__name__, unbuildable)
        else:
            self._enqueue(_Slot.PLAN_VIEW, plan_view)
        try:
            png = encode_png(render_depth_view(view, path))
        except ValueError as unrenderable:
            log.warning("depth view not sent (caught %s, expected): %s", type(unrenderable).__name__, unrenderable)
            return
        self._enqueue(_Slot.DEPTH_PNG, png)

    def close(self) -> None:
        if self._thread is None:
            return
        self._enqueue(None, None)
        self._thread.join(SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            log.warning("web sink thread did not stop within %.0f s", SHUTDOWN_TIMEOUT_SECONDS)
        self._thread = None
        self._loop = None
        self._outgoing = None
        if self._dropped:
            log.info("%d messages were dropped because the browsers were not keeping up", self._dropped)

    def _enqueue(self, slot: _Slot | None, message: str | bytes | None) -> None:
        # publish runs on the publisher's thread. The queue belongs to the server's loop, so the
        # put is handed to that loop rather than touched from here. Both values go as arguments,
        # never through a closure, so a later frame cannot rebind what this one queued.
        if self._loop is None or self._outgoing is None:
            return
        self._loop.call_soon_threadsafe(self._offer, slot, message)

    def _offer(self, slot: _Slot | None, message: str | bytes | None) -> None:
        """
        Put the message on the queue, making room by discarding the oldest when it is full.

        Runs on the server's loop, which is the only thread allowed to touch the queue. Dropping
        the oldest rather than refusing the newest is what a display wants: a frame nobody has
        managed to send yet is already out of date, and the next one replaces it anyway. A None
        message is the shutdown sentinel.
        """
        if self._outgoing is None:
            return
        while self._outgoing.full():
            try:
                self._outgoing.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._dropped += 1
        self._outgoing.put_nowait(None if message is None else (slot, message))

    def _serve(self) -> None:
        # Optional dependency, present only with the web extra. Imported where it is used so the
        # package imports without it.
        try:
            from aiohttp import web
        except ImportError as missing:
            self._startup_error = missing
            self._ready.set()
            return

        try:
            asyncio.run(self._run(web))
        except OSError as bind_error:
            # The port is taken, or cannot be bound. Reported through start() on the caller's thread.
            self._startup_error = bind_error
            self._ready.set()

    async def _run(self, web) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop.set_exception_handler(_quiet_client_resets)
        self._outgoing = asyncio.Queue(maxsize=OUTGOING_QUEUE_LIMIT)
        sockets: set = set()
        page = PAGE_PATH.read_text(encoding="utf-8")
        # The newest of each kind, written by the send loop below and read when a browser
        # connects, so a page opened mid-run shows something at once rather than waiting for the
        # next planned frame, which on a still phone is never. Loop-thread state only.
        latest: dict[_Slot, str | bytes | None] = {slot: None for slot in _Slot}

        async def serve_page(request):
            return web.Response(text=page, content_type="text/html")

        async def send(socket, message: str | bytes) -> None:
            if isinstance(message, bytes):
                await socket.send_bytes(message)
            else:
                await socket.send_str(message)

        async def serve_socket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            sockets.add(socket)
            log.info("browser connected, %d open", len(sockets))
            try:
                # Enum order: the path, then the plan view, then the picture, as a live frame sends them.
                for slot in _Slot:
                    message = latest[slot]
                    if message is not None:
                        await send(socket, message)
                async for _ in socket:
                    # The browser sends nothing we act on. Reading keeps the socket alive and
                    # notices when it closes.
                    pass
            finally:
                sockets.discard(socket)
                log.info("browser disconnected, %d open", len(sockets))
            return socket

        app = web.Application()
        app.router.add_get("/", serve_page)
        app.router.add_get("/ws", serve_socket)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._config.port)
        await site.start()

        self._bound_port = runner.addresses[0][1] if runner.addresses else self._config.port
        log.info("open http://localhost:%d in a browser", self._bound_port)
        self._ready.set()

        try:
            while True:
                queued = await self._outgoing.get()
                if queued is None:
                    break
                slot, message = queued
                latest[slot] = message
                for socket in list(sockets):
                    try:
                        await send(socket, message)
                    except (ConnectionResetError, RuntimeError) as gone:
                        # The browser went away between the check and the send. Expected on a
                        # phone browser that got backgrounded. Costs that one client only.
                        log.info("dropping a browser (caught %s, expected): %s", type(gone).__name__, gone)
                        sockets.discard(socket)
        finally:
            for socket in list(sockets):
                await socket.close()
            await runner.cleanup()


def _quiet_client_resets(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """
    The server loop's exception handler: a browser that dropped its socket is logged as expected.

    On Windows, asyncio's proactor loop reports a socket the browser reset as an error in its own
    callback, after the socket handler has already seen the browser go. A phone browser that gets
    backgrounded does this routinely. Anything else goes to asyncio's default handler unchanged.

    :param loop: The loop the exception came from.
    :param context: asyncio's description of what failed.
    """
    if isinstance(context.get("exception"), ConnectionResetError):
        log.info("a browser's connection was reset (caught ConnectionResetError, expected): %s", context.get("message"))
        return
    loop.default_exception_handler(context)
