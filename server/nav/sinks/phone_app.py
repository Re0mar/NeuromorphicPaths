"""
The Pixel app as the display, over TCP.

The laptop connects out to the app and sends each path as a length-prefixed JSON message, the
schema in docs/arcore_wire_format.md. When the app goes away the sink says so once and then
drops paths quietly, because a display that has gone is not a reason to stop planning.
"""

# Standard library imports
import logging
import socket

# Local package imports
from nav.sinks.config import PhoneAppConfig
from nav.sources.framecodec import LENGTH_PREFIX, encode_path, write_message
from nav.types import PlannedPath

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 5.0


class PhoneAppSink:
    """Sends every path to the phone. Connects on the first publish, not on construction."""

    def __init__(self, config: PhoneAppConfig) -> None:
        self._config = config
        self._socket: socket.socket | None = None
        self._disconnected = False
        self._dropped = 0

    @property
    def disconnected(self) -> bool:
        return self._disconnected

    def _connect(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        try:
            self._socket = socket.create_connection((self._config.address, self._config.port), timeout=CONNECT_TIMEOUT_SECONDS)
        except OSError as refused:
            raise ConnectionError(f"could not reach the phone app at {self._config.address}:{self._config.port}: {refused}") from refused
        log.info("connected to the phone app at %s:%d", self._config.address, self._config.port)
        return self._socket

    def publish(self, path: PlannedPath) -> None:
        """Send the path, or drop it if the phone has gone, warning once rather than once per frame."""
        if self._disconnected:
            self._dropped += 1
            return

        # Connecting is outside the handler on purpose. A phone that was never reachable is a
        # run started without its display, and that propagates. A phone that goes away mid-run
        # is the case below, and that degrades.
        connection = self._connect()
        message = encode_path(path)
        try:
            write_message(connection, LENGTH_PREFIX.pack(len(message)) + message)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as gone:
            # The phone closed the socket or the hotspot dropped. Expected in the field, and the
            # run keeps going without a display rather than dying for one.
            log.warning("phone app disconnected (caught %s, expected), dropping paths from now on: %s", type(gone).__name__, gone)
            self._disconnected = True
            self._dropped = 1
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def close(self) -> None:
        if self._dropped:
            log.info("%d paths were dropped after the phone app disconnected", self._dropped)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
