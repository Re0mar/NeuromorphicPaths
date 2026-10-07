"""Covers the phone sink against real client sockets standing in for the app."""

# Standard library imports
import logging
import socket
import time
from collections.abc import Callable

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.config import PhoneAppConfig
from nav.sinks.phone_app import PhoneAppSink
from nav.sources.framecodec import StreamClosedError, decode_path, read_message
from nav.types import PlannedPath

# Everything here finishes in well under a second. A test that blocks this long is hung.
TEST_TIMEOUT_SECONDS = 3.0


def _path(heading: float = 0.1) -> PlannedPath:
    return PlannedPath(1.0, np.array([0.0, 0.1]), np.array([0.0, 0.05]), heading, False, 2.5, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _listening_sink() -> PhoneAppSink:
    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"))
    sink.start()
    return sink


def _wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


class FakePhone:
    """Connects to the sink the way the app will, and reads framed paths off the socket."""

    def __init__(self, sink: PhoneAppSink) -> None:
        accepted_before = sink.phones_accepted
        self.connection = socket.create_connection(("127.0.0.1", sink.port), timeout=TEST_TIMEOUT_SECONDS)
        # Connected is not the same as accepted. The sink only serves a phone its acceptor has
        # taken, so a publish before that would be counted as dropped and the read would hang.
        _wait_until(lambda: sink.phones_accepted > accepted_before, "the sink to accept the phone")

    def read_path(self) -> PlannedPath:
        return decode_path(read_message(self.connection))

    def close(self) -> None:
        self.connection.close()


def test_a_published_path_arrives_length_prefixed_and_decodes() -> None:
    sink = _listening_sink()
    phone = FakePhone(sink)
    try:
        sink.publish(_path(heading=0.25))
        received = phone.read_path()
    finally:
        phone.close()
        sink.close()

    assert received.lookahead_heading_radians == pytest.approx(0.25)
    assert sink.dropped == 0


def test_three_phones_in_sequence_are_each_served() -> None:
    # One phone proves the first accept. Three prove the acceptor keeps going after a phone
    # leaves, which is the loop a single-instance fixture cannot see stop.
    sink = _listening_sink()
    headings = []
    try:
        for heading in (0.1, 0.2, 0.3):
            phone = FakePhone(sink)
            sink.publish(_path(heading=heading))
            headings.append(phone.read_path().lookahead_heading_radians)
            phone.close()
    finally:
        sink.close()

    assert headings == pytest.approx([0.1, 0.2, 0.3])


def test_a_path_published_with_no_phone_is_dropped_and_counted_once_at_close(caplog: pytest.LogCaptureFixture) -> None:
    sink = _listening_sink()
    with caplog.at_level(logging.INFO, logger="nav.sinks.phone_app"):
        for _ in range(3):
            sink.publish(_path())
        sink.close()

    assert sink.dropped == 3
    dropped_lines = [record.message for record in caplog.records if "dropped" in record.message]
    assert dropped_lines == ["3 paths were dropped while no phone was connected"]


def test_a_phone_that_closes_is_noticed_on_the_next_send_and_the_run_continues(caplog: pytest.LogCaptureFixture) -> None:
    sink = _listening_sink()
    try:
        first = FakePhone(sink)
        sink.publish(_path(heading=0.1))
        first.read_path()
        first.close()

        with caplog.at_level(logging.INFO, logger="nav.sinks.phone_app"):
            # The first send after the close may still succeed into the socket buffer. Keep
            # publishing until the OS reports the reset, which takes at most a couple of sends.
            for _ in range(20):
                sink.publish(_path())
                if not sink.connected:
                    break
            assert not sink.connected, "the sink never noticed the phone had gone"
            sink.publish(_path())

        second = FakePhone(sink)
        sink.publish(_path(heading=0.5))
        received = second.read_path()
        second.close()
    finally:
        sink.close()

    assert received.lookahead_heading_radians == pytest.approx(0.5)
    disconnect_lines = [record for record in caplog.records if "phone disconnected" in record.message]
    assert len(disconnect_lines) == 1, [record.message for record in caplog.records]
    assert disconnect_lines[0].levelno == logging.INFO


def test_a_new_phone_replaces_the_old_one() -> None:
    # A phone that reconnects before its old socket was noticed must get the paths, and the old
    # socket must be closed rather than left to hold the only reference the sink has.
    sink = _listening_sink()
    try:
        old = FakePhone(sink)
        new = FakePhone(sink)
        sink.publish(_path(heading=0.7))
        received = new.read_path()
        with pytest.raises(StreamClosedError):
            old.read_path()
    finally:
        old.close()
        new.close()
        sink.close()

    assert received.lookahead_heading_radians == pytest.approx(0.7)


def test_a_second_sink_on_the_same_port_is_refused() -> None:
    # Two laptop runs on one port must not both report "listening". The depth source learned
    # this on the first Pixel session, and the path port follows the same rule.
    first = _listening_sink()
    second = PhoneAppSink(PhoneAppConfig(port=first.port, bind_address="127.0.0.1"))
    try:
        # The message names the sink and the port. A run listens on three, and the operating
        # system's own text says only that some socket address is in use.
        with pytest.raises(OSError, match=f"phone sink could not listen on 127.0.0.1:{first.port}"):
            second.start()
    finally:
        second.close()
        first.close()


def test_publish_starts_listening_on_its_own() -> None:
    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"))
    try:
        sink.publish(_path())

        assert sink.port != 0
        assert sink.dropped == 1
        phone = FakePhone(sink)
        sink.publish(_path(heading=0.3))
        assert phone.read_path().lookahead_heading_radians == pytest.approx(0.3)
        phone.close()
    finally:
        sink.close()


def test_close_returns_promptly_while_waiting_for_a_phone() -> None:
    sink = _listening_sink()

    started = time.monotonic()
    sink.close()

    # One accept poll at most, so the acceptor thread cannot keep the run from ending.
    assert time.monotonic() - started < 1.5


def test_close_before_start_and_close_twice_do_not_raise() -> None:
    PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1")).close()

    sink = _listening_sink()
    sink.close()
    sink.close()


def test_the_phone_sink_reports_sent_after_the_write() -> None:
    sent: list[float] = []
    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"), on_sent=sent.append)
    sink.start()
    phone = FakePhone(sink)
    try:
        sink.publish(_path())
        assert phone.read_path().timestamp_seconds == 1.0
    finally:
        phone.close()
        sink.close()

    assert sent == [1.0]


def test_a_path_with_no_phone_reports_no_send() -> None:
    """Nothing reached a phone, so a send time here would credit the network with a path it never carried."""
    sent: list[float] = []
    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"), on_sent=sent.append)
    sink.start()
    try:
        sink.publish(_path())
    finally:
        sink.close()

    assert sent == []


def test_a_raising_sent_hook_leaves_the_phone_sink_serving(caplog: pytest.LogCaptureFixture) -> None:
    """A measurement bug must cost the measurement, never the arrow on the walker's phone."""

    def broken_hook(timestamp_seconds: float) -> None:
        raise RuntimeError("a bug in the timing log")

    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"), on_sent=broken_hook)
    sink.start()
    phone = FakePhone(sink)
    try:
        with caplog.at_level(logging.ERROR):
            sink.publish(_path(0.1))
            sink.publish(_path(0.2))
        assert phone.read_path().lookahead_heading_radians == pytest.approx(0.1)
        assert phone.read_path().lookahead_heading_radians == pytest.approx(0.2)
    finally:
        phone.close()
        sink.close()

    errors = [record for record in caplog.records if "sent hook" in record.message]
    assert len(errors) == 1, "logged once, not once per path"


def test_a_failed_write_reports_no_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path that never reached the phone must not be stamped sent, or its time lands in the network share."""
    import nav.sinks.phone_app as phone_app_module

    def reset(client, data) -> None:
        raise ConnectionResetError("the phone went away mid-write")

    sent: list[float] = []
    sink = PhoneAppSink(PhoneAppConfig(port=0, bind_address="127.0.0.1"), on_sent=sent.append)
    sink.start()
    phone = FakePhone(sink)
    try:
        sink.publish(_path())
        monkeypatch.setattr(phone_app_module, "write_message", reset)
        sink.publish(_path())
    finally:
        phone.close()
        sink.close()

    assert sent == [1.0], "only the first path, which went out, is stamped"
