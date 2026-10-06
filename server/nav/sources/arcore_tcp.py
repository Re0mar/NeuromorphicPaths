"""
Depth frames from the Pixel app, over one TCP connection.

The laptop listens, the phone connects, and frames arrive in the wire format that
docs/arcore_wire_format.md describes. One client at a time. When it disconnects this source
ends, and whether to wait for another is the runtime's decision, not this file's. A source that
reconnects forever can never be run to completion in a test.
"""

# Standard library imports
import dataclasses
import logging
import socket
import time
from collections.abc import Iterator

# Local package imports
from nav.clock import laptop_time_seconds
from nav.sources.config import ArCoreConfig
from nav.sources.framecodec import FrameDecodeError, StreamClosedError, decode_frame, read_message
from nav.types import DepthFrame, FrameTiming

log = logging.getLogger(__name__)

# How long one accept waits before returning to Python. Waiting out the whole accept timeout in
# one call makes the run deaf to Ctrl-C for as long as it lasts: a console interrupt is only
# acted on between bytecodes, and a socket call blocked in the operating system is not between
# bytecodes. Launching the app by hand needs a ten minute wait, and ten minutes of a process that
# cannot be stopped is not a wait anybody will sit through.
ACCEPT_POLL_SECONDS = 0.5


class ArCoreTcpSource:
    """Yields DepthFrame objects decoded from whatever connects to the listening port."""

    def __init__(self, config: ArCoreConfig) -> None:
        self._config = config
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None

    @property
    def port(self) -> int:
        """The port actually bound, which differs from the config's when that was 0."""
        if self._listener is None:
            return self._config.port
        return self._listener.getsockname()[1]

    def _listen(self) -> socket.socket:
        if self._listener is not None:
            return self._listener

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows. SO_REUSEADDR there lets a second process bind a port another process is
            # listening on, and the phone's connection then lands on whichever one the kernel
            # picks. A second laptop run did exactly that on the first Pixel session, reporting
            # "listening" while every frame went to the run it was meant to replace. Windows also
            # rebinds a port in TIME_WAIT without any option, so nothing is lost here.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # Everywhere else a restart within a minute of the last run fails with the port in
            # use without this, which reads as the phone being unreachable rather than as a
            # stale socket. A second listener is still refused, which is what the test asserts.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self._config.bind_address, self._config.port))
            listener.listen(1)
        except OSError as taken:
            # Same reasoning as the phone sink's. The bare WinError names no port, and a run has
            # three of them.
            listener.close()
            raise OSError(f"the depth source could not listen on {self._config.bind_address}:{self._config.port}: {taken}") from taken
        listener.settimeout(ACCEPT_POLL_SECONDS)
        self._listener = listener
        log.info("listening for the Pixel app on %s:%d", self._config.bind_address, self.port)
        return listener

    def frames(self) -> Iterator[DepthFrame]:
        listener = self._listen()

        client, peer = self._accept_within_timeout(listener)
        self._client = client
        log.info("sender connected from %s:%d", peer[0], peer[1])

        received = 0
        try:
            while True:
                try:
                    payload = read_message(client)
                except StreamClosedError as closed:
                    if closed.at_message_boundary:
                        log.info("sender disconnected after %d frames", received)
                    else:
                        log.warning("sender disconnected part way through a frame, after %d frames: %s", received, closed)
                    return
                except FrameDecodeError as framing_error:
                    # A zero or over-limit length prefix. The stream is out of step, so the next
                    # bytes are not a frame either. Dropping the connection is the only honest
                    # recovery, and the runtime decides whether to wait for the phone to reconnect.
                    log.warning("stream desynchronised after %d frames, dropping the connection: %s", received, framing_error)
                    return
                except OSError as socket_error:
                    # Reset by peer, or the interface went away. The connection is gone either way.
                    log.info("connection lost after %d frames: %s", received, socket_error)
                    return

                arrival_seconds = laptop_time_seconds()
                try:
                    frame = decode_frame(payload)
                except FrameDecodeError as decode_error:
                    # One bad frame on the wire. Expected on a hotspot, costs one frame, and the
                    # connection stays up. Dropping the connection for it would cost every frame after.
                    log.warning("frame dropped (caught %s, expected): %s", type(decode_error).__name__, decode_error)
                    continue

                # Arrival only. The phone's capture time on this clock needs an offset between the
                # two clocks, which this source does not measure yet.
                frame = dataclasses.replace(
                    frame,
                    timing=FrameTiming(capture_seconds=None, arrival_seconds=arrival_seconds, depth_ready_seconds=None),
                )

                if received == 0:
                    log.info("first frame: depth %s, has_position=%s", frame.depth_meters.shape, frame.pose.has_position)
                received += 1
                yield frame
        finally:
            # The listener stays open, so frames() can be called again for the next connection
            # without holding a dead socket per reconnect until the run ends.
            self._end_connection()

    def _accept_within_timeout(self, listener: socket.socket) -> tuple[socket.socket, tuple]:
        """
        Wait for the sender, in short slices, up to the configured accept timeout.

        Each slice hands control back to Python, which is the only moment a Ctrl-C can be acted
        on, so the wait is interruptible however long it is.

        :param listener: The bound, listening socket.
        :return: The accepted connection and its peer address.
        :rtype: tuple[socket.socket, tuple]
        :raises ConnectionError: When nobody connects within the accept timeout.
        """
        deadline = time.monotonic() + self._config.accept_timeout_seconds
        last_timeout: TimeoutError | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConnectionError(
                    f"no sender connected to port {self.port} within {self._config.accept_timeout_seconds:.0f} s"
                ) from last_timeout
            # The shorter of the two, so a long wait stays interruptible and a short one is still
            # as short as it was asked to be.
            listener.settimeout(min(ACCEPT_POLL_SECONDS, remaining))
            try:
                return listener.accept()
            except TimeoutError as timeout_error:
                last_timeout = timeout_error

    def _end_connection(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def close(self) -> None:
        for name in ("_client", "_listener"):
            sock = getattr(self, name)
            if sock is not None:
                sock.close()
                setattr(self, name, None)
