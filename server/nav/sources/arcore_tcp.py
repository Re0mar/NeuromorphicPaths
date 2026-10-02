"""
Depth frames from the Pixel app, over one TCP connection.

The laptop listens, the phone connects, and frames arrive in the wire format that
docs/arcore_wire_format.md describes. One client at a time. When it disconnects this source
ends, and whether to wait for another is the runtime's decision, not this file's. A source that
reconnects forever can never be run to completion in a test.
"""

# Standard library imports
import logging
import socket
from collections.abc import Iterator

# Local package imports
from nav.sources.config import ArCoreConfig
from nav.sources.framecodec import FrameDecodeError, StreamClosedError, decode_frame, read_message
from nav.types import DepthFrame

log = logging.getLogger(__name__)


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
        listener.bind((self._config.bind_address, self._config.port))
        listener.listen(1)
        listener.settimeout(self._config.accept_timeout_seconds)
        self._listener = listener
        log.info("listening for the Pixel app on %s:%d", self._config.bind_address, self.port)
        return listener

    def frames(self) -> Iterator[DepthFrame]:
        listener = self._listen()

        try:
            client, peer = listener.accept()
        except TimeoutError as timeout_error:
            raise ConnectionError(
                f"no sender connected to port {self.port} within {self._config.accept_timeout_seconds:.0f} s"
            ) from timeout_error
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

                try:
                    frame = decode_frame(payload)
                except FrameDecodeError as decode_error:
                    # One bad frame on the wire. Expected on a hotspot, costs one frame, and the
                    # connection stays up. Dropping the connection for it would cost every frame after.
                    log.warning("frame dropped (caught %s, expected): %s", type(decode_error).__name__, decode_error)
                    continue

                if received == 0:
                    log.info("first frame: depth %s, has_position=%s", frame.depth_meters.shape, frame.pose.has_position)
                received += 1
                yield frame
        finally:
            # The listener stays open, so frames() can be called again for the next connection
            # without holding a dead socket per reconnect until the run ends.
            self._end_connection()

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
