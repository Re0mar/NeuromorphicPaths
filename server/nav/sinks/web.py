import asyncio
import logging
import threading
from pathlib import Path

from nav.sinks.config import WebConfig
from nav.sources.framecodec import encode_path
from nav.types import PlannedPath

log = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "web_page.html"
STARTUP_TIMEOUT_SECONDS = 5.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0


class WebSink:
    """Serves the page and pushes every published path to every connected browser."""

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

    def _enqueue(self, message: str | None) -> None:
        # publish runs on the worker thread. The queue belongs to the server's loop, so the put
        # is handed to that loop rather than touched from here.
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

        async def serve_page(request):
            return web.Response(text=page, content_type="text/html")

        async def serve_socket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            sockets.add(socket)
            log.info("browser connected, %d open", len(sockets))
            try:
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
                for socket in list(sockets):
                    try:
                        await socket.send_str(message)
                    except (ConnectionResetError, RuntimeError) as gone:
                        # The browser went away between the check and the send. Expected on a
                        # phone browser that got backgrounded. Costs that one client only.
                        log.info("dropping a browser (caught %s, expected): %s", type(gone).__name__, gone)
                        sockets.discard(socket)
        finally:
            for socket in list(sockets):
                await socket.close()
            await runner.cleanup()