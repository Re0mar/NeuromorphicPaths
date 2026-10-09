"""
A browser as the display: one static page and one websocket, served from a thread of their own.

The only file in the package allowed to import aiohttp. The server runs its own asyncio loop in
its own thread, and the pipeline's publisher thread hands it messages through
call_soon_threadsafe, so the two never share a loop.

Served over TLS with the self-signed certificate committed under tls/. The browser's own video
decoder, WebCodecs, only exists on a secure origin, and a phone opens the page by the laptop's
address, which plain http never makes secure. Each browser accepts the certificate warning once.

Each planned path goes out as a text frame, the path's JSON with a kind of "path". When the loop
hands this sink a debug view, two more frames follow: a second kind of text message, the plan view,
with the planner's field and what the page needs to draw it top-down, then a binary frame with a
PNG of the depth image the planner saw, groups and path drawn on it. The page dispatches text
frames on their kind. It draws the arrow and turns red on alarm, draws the plan view, and shows the
picture when one arrives.

A page can ask for the risk view in place of the depth view, the same picture recolored by the
planner's field. That is the one message a page sends. Each browser gets only the picture it chose,
and the risk view is drawn only while some browser shows it. Neither picture carries the group
rings. The plan view says where they go, and the page draws them, so it can hide them at once.

The plan view and the picture are built on a render thread of their own, from the newest debug view
only, and at most `max_pictures_per_second` times a second. Building them takes about 50 ms. On the
publisher thread that held up the phone's next path, and at every frame it took planning time too.

A second websocket, /video, carries the glasses' compressed video for the page's browser to decode
on its own: the stream's description as a text frame, then one access unit per frame as a binary
frame, from the scene video feed the composition root gave this sink. Each browser has a bounded
queue of its own and starts at a keyframe. A browser that falls behind by more than its queue
loses units up to the next keyframe, so it never gets a delta whose reference it missed, and one
slow browser never holds up another. A run whose source has no video says so on that socket and
closes it. /recording serves the file --demo-recording named, for the page's Recording mode.
"""

# Standard library imports
import asyncio
import logging
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.scene.config import SceneConfig
from nav.sinks.config import WebConfig
from nav.sinks.rendering import encode_png, render_depth_view, render_risk_view
from nav.sinks.web_messages import (
    PictureKind,
    WebMessageKind,
    picture_choice,
    plan_view_message,
    video_stream_message,
    video_unavailable_message,
    web_text_message,
)
from nav.sources.framecodec import path_message
from nav.sources.scene_video import AccessUnit, SceneVideoFeed, VideoDescription, pack_unit
from nav.types import DebugView, PlannedPath

log = logging.getLogger(__name__)

# What the page shows in Live mode on a run whose source cannot offer video.
NO_SCENE_VIDEO_REASON = "this run's source has no scene video"

PAGE_PATH = Path(__file__).parent / "web_page.html"
STARTUP_TIMEOUT_SECONDS = 5.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0
# How many messages may wait for the browsers. A depth view is about 128 KB and a plan view about
# 100 KB on a cluttered room, so this is a few megabytes at worst. The queue has to be bounded: the pipeline hands
# over a path, a plan view and a picture per planned frame and never waits, while one browser that
# stops reading suspends the send loop for every browser, so an unbounded queue grows for as long as
# that lasts.
OUTGOING_QUEUE_LIMIT = 32
# How long the render thread waits for a debug view before checking whether it was asked to stop.
# A new view wakes it at once regardless.
RENDER_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class _DebugPicture:
    """What the render thread needs to build one plan view and one depth picture."""

    path: PlannedPath
    field: np.ndarray
    grid: np.ndarray
    view: DebugView


class _Slot(Enum):
    """Which newest-of-its-kind message a payload replaces. A browser that connects gets one of each, in this order."""

    PATH = 1
    PLAN_VIEW = 2
    DEPTH_PNG = 3
    RISK_PNG = 4


# Which slot carries each picture. A browser gets the one it chose and never the other.
PICTURE_SLOTS = {PictureKind.DEPTH: _Slot.DEPTH_PNG, PictureKind.RISK: _Slot.RISK_PNG}


@dataclass
class _VideoClient:
    """One browser on the video socket: its own queue, and whether it still waits for a keyframe to start on."""

    queue: asyncio.Queue
    waiting_for_keyframe: bool = True


class _LoopVideoListener:
    """
    The sink's end of the scene video feed. Hands everything to the server's loop and does nothing else.

    It runs on the device process's reader thread, which also feeds every other listener, so a
    slow call here would hold the video for them all.
    """

    def __init__(self, sink: "WebSink") -> None:
        self._sink = sink

    def describe(self, description: VideoDescription) -> None:
        self._sink._call_on_loop(self._sink._offer_description, description)

    def offer(self, unit: AccessUnit) -> None:
        self._sink._call_on_loop(self._sink._offer_unit, unit)


class WebSink:
    """Serves the page and pushes every published path, and the depth view when given one, to every connected browser."""

    def __init__(self, config: WebConfig, scene: SceneConfig, *, video_feed: SceneVideoFeed | None) -> None:
        """
        :param config: The port and the certificate.
        :param scene: The run's own scene settings, so the plan view judges hidden floor with the
            inlier distance and depth range the scene used on this run.
        :param video_feed: Where the glasses' compressed video arrives for the page's browser to
            decode. None when the run's source has no video to offer, which the page is told.
            Keyword-only and without a default, so every caller says which it is.
        """
        self._config = config
        self._scene = scene
        self._video_feed = video_feed
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._outgoing: asyncio.Queue | None = None
        self._bound_port: int | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._dropped = 0
        self._render_thread: threading.Thread | None = None
        self._render_wake = threading.Condition()
        self._pending_picture: _DebugPicture | None = None
        # The last view drawn, kept so a browser switching to Risk on a still source gets a picture
        # without waiting for a frame that may never come. Guarded by the render condition.
        self._last_picture: _DebugPicture | None = None
        # Written on the server's loop, read by the render thread. The risk picture is only drawn
        # while some browser shows it, so a run nobody looks at it on pays nothing for it.
        self._risk_wanted = False
        self._render_stopping = False
        self._pictures_replaced = 0
        # Set by close() for good. A publisher still inside a display at shutdown can call in after
        # it, and publish() would otherwise start a second server for nobody.
        self._closed = False
        # The video socket's state. The clients and the description text are loop-thread state,
        # written and read on the server's loop only. The counter is read at close.
        self._video_clients: dict[object, _VideoClient] = {}
        self._video_description_text: str | None = None
        self._video_dropped = 0
        self._unsubscribe_video: Callable[[], None] | None = None
        self._recording_missing_logged = False

    @property
    def dropped(self) -> int:
        """Messages discarded because the browsers were not keeping up."""
        return self._dropped

    @property
    def port(self) -> int | None:
        """The port actually bound, once started. Differs from the config's when that was 0."""
        return self._bound_port

    @property
    def video_feed(self) -> SceneVideoFeed | None:
        """The feed this sink was given for the page's video, or None when the source has none."""
        return self._video_feed

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
        self._render_stopping = False
        self._render_thread = threading.Thread(target=self._render_pictures, name="web-render", daemon=True)
        self._render_thread.start()
        if self._video_feed is not None:
            # Once the loop exists, so the listener always has somewhere to hand units to.
            self._unsubscribe_video = self._video_feed.subscribe(_LoopVideoListener(self))

    def publish(self, path: PlannedPath) -> None:
        """Queue the path for every browser. Starts the server on the first call, and does nothing once closed."""
        if self._closed:
            return
        if self._thread is None:
            self.start()
        self._enqueue(_Slot.PATH, web_text_message(WebMessageKind.PATH, path_message(path)))

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
        """
        Queue the path's JSON now, and leave the view for the render thread to draw.

        Returns without drawing anything. A view still waiting to be drawn is replaced by this one,
        since the page only ever shows the newest.
        """
        self.publish(path)
        if self._render_thread is None or not self._render_thread.is_alive():
            # Closed, never started, or ended by an unexpected drawing error. Nothing would draw it.
            return
        with self._render_wake:
            if self._pending_picture is not None:
                self._pictures_replaced += 1
            self._pending_picture = _DebugPicture(path, field, grid, view)
            self._render_wake.notify_all()

    def close(self) -> None:
        self._closed = True
        if self._unsubscribe_video is not None:
            # First, so no unit is handed to a loop that is about to stop.
            self._unsubscribe_video()
            self._unsubscribe_video = None
        if self._thread is None:
            return
        # The render thread stops first, so nothing it builds is offered to a server that has gone.
        if self._render_thread is not None:
            with self._render_wake:
                self._render_stopping = True
                self._render_wake.notify_all()
            self._render_thread.join(SHUTDOWN_TIMEOUT_SECONDS)
            if self._render_thread.is_alive():
                log.warning("web render thread did not stop within %.0f s", SHUTDOWN_TIMEOUT_SECONDS)
            self._render_thread = None
        self._enqueue(None, None)
        self._thread.join(SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            log.warning("web sink thread did not stop within %.0f s", SHUTDOWN_TIMEOUT_SECONDS)
        self._thread = None
        self._loop = None
        self._outgoing = None
        if self._dropped:
            log.info("%d messages were dropped because the browsers were not keeping up", self._dropped)
        if self._pictures_replaced:
            log.info("%d debug views were replaced by a newer one before they were drawn", self._pictures_replaced)
        if self._video_dropped:
            log.info("%d video units were dropped for browsers that were not keeping up", self._video_dropped)

    def _render_pictures(self) -> None:
        """The render thread. Draws the newest debug view, then waits out the rest of its interval."""
        minimum_interval_seconds = 1.0 / self._config.max_pictures_per_second
        last_render_started: float | None = None
        while True:
            with self._render_wake:
                while self._pending_picture is None and not self._render_stopping:
                    self._render_wake.wait(RENDER_POLL_SECONDS)
                if self._render_stopping:
                    return
            if last_render_started is not None:
                # Waits on the condition rather than sleeping, so a stop is seen at once. A view
                # that arrives during the wait replaces the pending one, and the newest is drawn.
                resume_at = last_render_started + minimum_interval_seconds
                with self._render_wake:
                    while not self._render_stopping and time.perf_counter() < resume_at:
                        self._render_wake.wait(resume_at - time.perf_counter())
                    if self._render_stopping:
                        return
            with self._render_wake:
                picture, self._pending_picture = self._pending_picture, None
                self._last_picture = picture
            last_render_started = time.perf_counter()
            try:
                self._draw(picture)
            except Exception as unexpected_error:
                # Each expected failure is handled inside _draw. Anything else stops the pictures
                # for the rest of the run, and the paths keep going, like a fan-out dropping a sink.
                log.error("UNEXPECTED %s drawing the debug view, no more pictures this run", type(unexpected_error).__name__, exc_info=True)
                return

    def _draw(self, picture: _DebugPicture) -> None:
        """
        Queue the plan view, then a PNG of the depth view, then one of the risk view while a browser shows it.

        Each one that cannot be built costs itself and nothing else. A plan view that will not
        build still leaves the pictures, and a picture that will not render still leaves the rest.
        The path itself has already gone out. Neither picture carries the group rings. The plan
        view says where they go, and the page draws them over whichever picture it shows.
        """
        try:
            plan_view = web_text_message(WebMessageKind.PLAN_VIEW, plan_view_message(picture.path, picture.field, picture.grid, picture.view, self._scene))
        except ValueError as unbuildable:
            log.warning("plan view not sent (caught %s, expected): %s", type(unbuildable).__name__, unbuildable)
        else:
            self._enqueue(_Slot.PLAN_VIEW, plan_view)
        try:
            png = encode_png(render_depth_view(picture.view, picture.path, rings=False))
        except ValueError as unrenderable:
            log.warning("depth view not sent (caught %s, expected): %s", type(unrenderable).__name__, unrenderable)
        else:
            self._enqueue(_Slot.DEPTH_PNG, png)
        if not self._risk_wanted:
            return
        try:
            png = encode_png(render_risk_view(picture.view, picture.path, picture.field, picture.grid, self._scene))
        except ValueError as unrenderable:
            log.warning("risk view not sent (caught %s, expected): %s", type(unrenderable).__name__, unrenderable)
            return
        self._enqueue(_Slot.RISK_PNG, png)

    def _redraw_last(self) -> None:
        """Ask the render thread to draw the last view again, unless a newer one is already waiting."""
        with self._render_wake:
            if self._pending_picture is None and self._last_picture is not None:
                self._pending_picture = self._last_picture
                self._render_wake.notify_all()

    def _call_on_loop(self, callback: Callable[..., None], *arguments: object) -> None:
        """Run a callback on the server's loop from any thread, or drop it when the server is gone."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(callback, *arguments)
        except RuntimeError as closed_loop:
            # The server already stopped. Nothing to hand it to.
            log.debug("video not sent, the server had stopped (caught RuntimeError, expected): %s", closed_loop)

    def _offer_description(self, description: VideoDescription) -> None:
        """Loop thread. Remember the stream's description and send it to every video browser."""
        self._video_description_text = web_text_message(WebMessageKind.VIDEO_STREAM, video_stream_message(description))
        for client in list(self._video_clients.values()):
            self._put_for_client(client, self._video_description_text)

    def _offer_unit(self, unit: AccessUnit) -> None:
        """Loop thread. One unit to every video browser's queue, resyncing a browser whose queue is full."""
        for client in list(self._video_clients.values()):
            self._put_for_client(client, unit)

    def _put_for_client(self, client: _VideoClient, item: str | AccessUnit) -> None:
        try:
            client.queue.put_nowait(item)
        except asyncio.QueueFull:
            # This browser is further behind than one keyframe gap. Everything waiting for it is
            # dropped and it starts again at the next keyframe, so it never sees a delta whose
            # reference frame it missed. The description stays, since a decoder needs it first.
            while not client.queue.empty():
                if not isinstance(client.queue.get_nowait(), str):
                    self._video_dropped += 1
            if not isinstance(item, str):
                self._video_dropped += 1
            client.waiting_for_keyframe = True
            if self._video_description_text is not None:
                client.queue.put_nowait(self._video_description_text)
            if isinstance(item, str):
                client.queue.put_nowait(item)

    def _enqueue(self, slot: _Slot | None, message: str | bytes | None) -> None:
        # publish runs on the publisher's thread. The queue belongs to the server's loop, so the
        # put is handed to that loop rather than touched from here. Both values go as arguments,
        # never through a closure, so a later frame cannot rebind what this one queued.
        # Read once. close() clears the field from another thread, and the loop can close before it does.
        loop = self._loop
        if loop is None or self._outgoing is None:
            return
        try:
            loop.call_soon_threadsafe(self._offer, slot, message)
        except RuntimeError as closed_loop:
            # The server already stopped: a draw that outlived close() by a moment. Nothing to send it to.
            log.debug("message not sent, the server had stopped (caught RuntimeError, expected): %s", closed_loop)

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
            # The loop is closed by now, so close() must not try to hand it a shutdown message.
            self._loop = None
            self._outgoing = None
            self._startup_error = bind_error
            self._ready.set()

    def _tls_context(self) -> ssl.SSLContext:
        """
        The server-side TLS context over the committed certificate.

        :raises OSError: When the certificate or the key cannot be loaded, naming both files. Raised
            as an OSError so it reaches start() through the same route as a port that cannot bind.
        """
        # CLIENT_AUTH is the purpose that builds a context for a server. The other one builds a
        # client context, and the handshake then fails in a way that reads as a bad certificate.
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        try:
            context.load_cert_chain(self._config.certificate_path, self._config.key_path)
        except ssl.SSLError as unloadable:
            raise OSError(
                f"the page's certificate {self._config.certificate_path} or key {self._config.key_path} "
                f"could not be loaded ({unloadable})"
            ) from unloadable
        return context

    async def _run(self, web) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop.set_exception_handler(_quiet_client_resets)
        self._outgoing = asyncio.Queue(maxsize=OUTGOING_QUEUE_LIMIT)
        # Each browser and the picture it shows. Loop-thread state only.
        sockets: dict[object, PictureKind] = {}
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

        def wanted_by(socket, slot: _Slot) -> bool:
            # Every slot goes to every browser except the pictures, which go only where chosen.
            return slot not in PICTURE_SLOTS.values() or PICTURE_SLOTS[sockets[socket]] is slot

        def note_who_wants_risk() -> None:
            self._risk_wanted = PictureKind.RISK in sockets.values()
            if not self._risk_wanted:
                # Stale the moment nobody looks at it. A browser that switches back later gets a
                # fresh one drawn, never a picture from minutes ago.
                latest[_Slot.RISK_PNG] = None

        async def choose_picture(socket, text: str) -> None:
            try:
                choice = picture_choice(text)
            except ValueError as unreadable:
                log.warning("browser message ignored (caught ValueError, expected): %s", unreadable)
                return
            sockets[socket] = choice
            note_who_wants_risk()
            message = latest[PICTURE_SLOTS[choice]]
            if message is not None:
                await send(socket, message)
            else:
                self._redraw_last()

        async def serve_socket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            sockets[socket] = PictureKind.DEPTH
            log.info("browser connected, %d open", len(sockets))
            try:
                # Enum order: the path, then the plan view, then the picture, as a live frame sends them.
                for slot in _Slot:
                    message = latest[slot]
                    if message is not None and wanted_by(socket, slot):
                        await send(socket, message)
                async for received in socket:
                    # Reading also keeps the socket alive and notices when it closes.
                    if received.type == web.WSMsgType.TEXT:
                        await choose_picture(socket, received.data)
            finally:
                sockets.pop(socket, None)
                note_who_wants_risk()
                log.info("browser disconnected, %d open", len(sockets))
            return socket

        async def send_video(socket, client: _VideoClient) -> None:
            """One browser's sender: its queue to its socket, deltas skipped until its first keyframe."""
            try:
                while True:
                    item = await client.queue.get()
                    if isinstance(item, str):
                        await socket.send_str(item)
                        continue
                    if client.waiting_for_keyframe and not item.keyframe:
                        continue
                    client.waiting_for_keyframe = False
                    await socket.send_bytes(pack_unit(item))
            except (ConnectionResetError, RuntimeError) as gone:
                # The browser went away mid-send. Its handler below sees the close and cleans up.
                log.info("dropping a video browser (caught %s, expected): %s", type(gone).__name__, gone)

        async def serve_video(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            if self._video_feed is None:
                await socket.send_str(web_text_message(WebMessageKind.VIDEO_UNAVAILABLE, video_unavailable_message(NO_SCENE_VIDEO_REASON)))
                await socket.close()
                return socket
            client = _VideoClient(queue=asyncio.Queue(maxsize=self._config.video_queue_limit))
            self._video_clients[socket] = client
            if self._video_description_text is not None:
                client.queue.put_nowait(self._video_description_text)
            sender = asyncio.ensure_future(send_video(socket, client))
            log.info("browser connected for video, %d open", len(self._video_clients))
            try:
                async for _ in socket:
                    # The browser sends nothing we act on. Reading notices when it closes.
                    pass
            finally:
                sender.cancel()
                self._video_clients.pop(socket, None)
                log.info("browser disconnected from video, %d open", len(self._video_clients))
            return socket

        def recording_available() -> bool:
            recording = self._config.recording_path
            if recording is not None and Path(recording).is_file():
                return True
            if recording is not None and not self._recording_missing_logged:
                # Was there when the run started and is gone now. Said once, then answered like none.
                self._recording_missing_logged = True
                log.warning("the demo recording %s is gone, the page's Recording mode has nothing to play", recording)
            return False

        async def serve_recording(request):
            if recording_available():
                # FileResponse answers Range requests and HEAD, which a <video> needs to seek and loop.
                return web.FileResponse(self._config.recording_path)
            return web.json_response({"error": "no recording configured"}, status=404)

        async def serve_recording_status(request):
            # Always 200, so the page can ask whether a recording exists without the browser logging
            # a failed request. The page is judged by its console, and an expected 404 would be noise.
            return web.json_response({"available": recording_available()})

        app = web.Application()
        app.router.add_get("/", serve_page)
        app.router.add_get("/ws", serve_socket)
        app.router.add_get("/video", serve_video)
        app.router.add_get("/recording", serve_recording)
        app.router.add_get("/recording/status", serve_recording_status)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._config.port, ssl_context=self._tls_context())
        await site.start()

        self._bound_port = runner.addresses[0][1] if runner.addresses else self._config.port
        log.info("open https://localhost:%d in a browser. Each browser accepts the certificate warning once", self._bound_port)
        self._ready.set()

        try:
            while True:
                queued = await self._outgoing.get()
                if queued is None:
                    break
                slot, message = queued
                if slot is _Slot.RISK_PNG and not self._risk_wanted:
                    # Drawn just before the last browser showing it switched away or left.
                    continue
                latest[slot] = message
                for socket in list(sockets):
                    if socket not in sockets or not wanted_by(socket, slot):
                        continue
                    try:
                        await send(socket, message)
                    except (ConnectionResetError, RuntimeError) as gone:
                        # The browser went away between the check and the send. Expected on a
                        # phone browser that got backgrounded. Costs that one client only.
                        log.info("dropping a browser (caught %s, expected): %s", type(gone).__name__, gone)
                        sockets.pop(socket, None)
        finally:
            for socket in list(sockets):
                await socket.close()
            for socket in list(self._video_clients):
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
