"""Covers the phone sink against a real listening socket standing in for the app."""

# Standard library imports
import socket
import threading

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.config import PhoneAppConfig
from nav.sinks.phone_app import PhoneAppSink
from nav.sources.framecodec import decode_path, read_message
from nav.types import PlannedPath


def _path(heading: float = 0.1) -> PlannedPath:
    return PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), heading, False, 2.5)


class FakePhone:
    """Listens like the app would, and hands the test the accepted connection."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.settimeout(3.0)
        self.port = self._listener.getsockname()[1]
        self.connection: socket.socket | None = None
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        try:
            self.connection, _ = self._listener.accept()
        except OSError:
            self.connection = None

    def wait_for_connection(self) -> socket.socket:
        self._thread.join(3.0)
        assert self.connection is not None, "the sink never connected"
        return self.connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self._listener.close()


def test_a_published_path_arrives_length_prefixed_and_decodes(caplog: pytest.LogCaptureFixture) -> None:
    phone = FakePhone()
    sink = PhoneAppSink(PhoneAppConfig(address="127.0.0.1", port=phone.port))
    try:
        sink.publish(_path(heading=0.25))
        connection = phone.wait_for_connection()
        received = decode_path(read_message(connection))
    finally:
        sink.close()
        phone.close()

    assert received.first_heading_radians == pytest.approx(0.25)
    assert sink.disconnected is False


def test_the_phone_closing_marks_the_sink_disconnected_with_exactly_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    phone = FakePhone()
    sink = PhoneAppSink(PhoneAppConfig(address="127.0.0.1", port=phone.port))
    try:
        sink.publish(_path())
        connection = phone.wait_for_connection()
        read_message(connection)
        connection.close()
        phone.close()

        with caplog.at_level("WARNING"):
            # The first send after the close may still succeed into the socket buffer. Keep
            # publishing until the OS reports the reset, which takes at most a couple of sends.
            for _ in range(20):
                sink.publish(_path())
                if sink.disconnected:
                    break
            for _ in range(5):
                sink.publish(_path())
    finally:
        sink.close()

    assert sink.disconnected is True
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, [record.message for record in warnings]
    assert "dropping paths from now on" in warnings[0].message


def test_nothing_listening_is_reported_by_address() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()
    sink = PhoneAppSink(PhoneAppConfig(address="127.0.0.1", port=free_port))

    with pytest.raises(ConnectionError, match=f"127.0.0.1:{free_port}"):
        sink.publish(_path())


def test_closing_an_unconnected_sink_does_not_raise() -> None:
    PhoneAppSink(PhoneAppConfig(address="127.0.0.1", port=1)).close()
