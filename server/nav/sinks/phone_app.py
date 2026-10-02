"""
The Pixel app as the display, over TCP.

The laptop listens on a second port and the phone connects to it, the same direction as depth, so
the phone needs one laptop address and the laptop never needs the phone's, which changed three
times on the day of the first walk. Each planned path goes out as a length-prefixed JSON message,
the schema in docs/arcore_wire_format.md. A phone that is not there costs nothing but a count: a
display that has gone, or has not arrived yet, is not a reason to stop planning.
"""

# Standard library imports
import logging
import socket
import threading

# Local package imports
from nav.sinks.config import PhoneAppConfig
from nav.sources.framecodec import LENGTH_PREFIX, encode_path, write_message
from nav.types import PlannedPath

log = logging.getLogger(__name__)

# How long accept() waits before checking whether close() asked the acceptor to stop. Closing a
# listener from another thread does not reliably wake a blocking accept on Windows.
ACCEPT_POLL_SECONDS = 0.5
ACCEPTOR_JOIN_SECONDS = 2.0


class PhoneAppSink:
    """Sends every path to whichever phone is connected. Listens from the first publish, not from construction."""

    def __init__(self, config: PhoneAppConfig) -> None:
        self._config = config
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None
        self._lock = threading.Lock()
        self._acceptor: threading.Thread | None = None
        self._running = False
        self._dropped = 0
        self._phones_accepted = 0

    @property
    def port(self) -> int:
        """The port actually bound, which differs from the config's when that was 0."""
        if self._listener is None:
            return self._config.port
        return self._listener.getsockname()[1]

    @property
    def connected(self) -> bool:
        """Whether a phone's socket is held. A phone that left is only noticed on the next send."""
        with self._lock:
            return self._client is not None

    @property
    def phones_accepted(self) -> int:
        """How many connections the acceptor has taken so far. Rises before the phone is served."""
        with self._lock:
            return self._phones_accepted

    @property
    def dropped(self) -> int:
        """Paths published while no phone was connected, or lost to a phone that had gone."""
        return self._dropped

    def start(self) -> None:
        """
        Bind, listen, and start accepting phones on a thread of its own.

        :raises OSError: When the port cannot be bound, which includes another run holding it.
        """
        if self._listener is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows. SO_REUSEADDR there lets a second run bind the port a first one is listening
            # on, and the phone lands on whichever the kernel picks. Same reasoning as the depth
            # source's listener.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # Everywhere else a restart within a minute of the last run fails with the port in
            # use without this. A second listener is still refused.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._config.bind_address, self._config.port))
        listener.listen(1)
        listener.settimeout(ACCEPT_POLL_SECONDS)
        self._listener = listener
        self._running = True
        self._acceptor = threading.Thread(target=self._accept_phones, args=(listener,), name="phone-acceptor", daemon=True)
        self._acceptor.start()
        log.info("listening for the Pixel app's path connection on %s:%d", self._config.bind_address, self.port)

    def _accept_phones(self, listener: socket.socket) -> None:
        while self._running:
            try:
                client, peer = listener.accept()
            except TimeoutError:
                # Nobody yet. Check the flag and wait again.
                continue
            except OSError:
                # The listener was closed under us, and close() is the only thing that does that.
                return
            # A path is a few hundred bytes once a frame. Nagle would hold it back for nothing.
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                previous = self._client
                self._client = client
                self._phones_accepted += 1
            if previous is not None:
                # One display at a time, and the newest connection is the live one. The old
                # socket is usually a phone that reconnected before its last socket was noticed.
                previous.close()
            log.info("phone connected from %s:%d", peer[0], peer[1])

    def publish(self, path: PlannedPath) -> None:
        """Send the path to the connected phone, or count it as dropped when there is none."""
        if self._listener is None:
            self.start()
        with self._lock:
            client = self._client
        if client is None:
            self._dropped += 1
            return

        message = encode_path(path)
        try:
            write_message(client, LENGTH_PREFIX.pack(len(message)) + message)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as gone:
            # The phone closed the socket or the hotspot dropped. Expected in the field. The run
            # keeps going, and the acceptor hands over the next phone when it arrives.
            log.info("phone disconnected (caught %s, expected), waiting for it: %s", type(gone).__name__, gone)
            self._dropped += 1
            with self._lock:
                if self._client is client:
                    self._client = None
            client.close()

    def close(self) -> None:
        """Stop accepting, drop the phone, and say once how many paths went nowhere."""
        self._running = False
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        if self._acceptor is not None:
            self._acceptor.join(ACCEPTOR_JOIN_SECONDS)
            if self._acceptor.is_alive():
                log.warning("phone acceptor thread did not stop within %.0f s", ACCEPTOR_JOIN_SECONDS)
            self._acceptor = None
        with self._lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()
        if self._dropped:
            log.info("%d paths were dropped while no phone was connected", self._dropped)
