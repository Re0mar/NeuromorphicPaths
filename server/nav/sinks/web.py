"""
A browser as the display: one static page and one websocket, served from a thread of their own.

The only file in the package allowed to import aiohttp. The server runs its own asyncio loop in
its own thread, and the pipeline's publisher thread hands it messages through
call_soon_threadsafe, so the two never share a loop. No video is sent. Each planned path goes out
as a text frame with the path's JSON, and when the loop hands this sink a debug view, a binary
frame follows with a PNG of the depth image the planner saw, groups and path drawn on it. The
page draws the arrow and turns red on alarm, and shows the picture under it when one arrives.
"""

# Standard library imports
import asyncio
import logging
import threading
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.sinks.config import WebConfig
from nav.sinks.rendering import encode_png, render_depth_view
from nav.sources.framecodec import encode_path
from nav.types import DebugView, PlannedPath

log = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "web_page.html"
STARTUP_TIMEOUT_SECONDS = 5.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0


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
        self._enqueue(encode_path(path).decode("utf-8"))

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
        """
        Queue the path's JSON, then a PNG of the depth view, for every browser.

        The field is not sent. The browser gets what a person tuning the planner looks at, which
        is the depth image with the groups and the path on it. A view that will not render costs
        the picture and never the path.
        """
        self.publish(path)
        try:
            png = encode_png(render_depth_view(view, path))
        except ValueError as unrenderable:
            log.warning("depth view not sent (caught %s, expected): %s", type(unrenderable).__name__, unrenderable)
            return
        self._enqueue(png)

    def close(self) -> None:
        if self._thread is None:
            return
        self._enqueue(None)
        self._thread.join(SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            log.warning("web sink thread did not stop within %.0f s", SHUTDOWN_TIMEOUT_SECONDS)
        self._thread = None
        self._loop = None
        self._outgoing = None

    def _enqueue(self, message: str | bytes | None) -> None:
        # publish runs on the publisher's thread. The queue belongs to the server's loop, so the
        # put is handed to that loop rather than touched from here.
        if self._loop is None or self._outgoing is None:
            return
        self._loop.call_soon_threadsafe(self._outgoing.put_nowait, message)

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
        self._outgoing = asyncio.Queue()
        sockets: set = set()
        page = PAGE_PATH.read_text(encoding="utf-8")
        # The newest of each kind, written by the send loop below and read when a browser
        # connects, so a page opened mid-run shows something at once rather than waiting for the
        # next planned frame, which on a still phone is never. Loop-thread state only.
        latest: dict[str, str | bytes | None] = {"text": None, "png": None}

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
                for kind in ("text", "png"):
                    message = latest[kind]
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
                message = await self._outgoing.get()
                if message is None:
                    break
                latest["png" if isinstance(message, bytes) else "text"] = message
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
