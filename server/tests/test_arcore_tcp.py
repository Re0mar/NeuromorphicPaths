"""
Covers the TCP source against the fake sender, in one process, over a real socket.

Every negative test sends a valid frame through the same sender first and asserts it arrived. A
test that only puts garbage on the wire and asserts nothing came out cannot tell a refused message
from a message that never arrived.

The real Pixel app does not exist yet. Until it does, this source is proven only against the fake
sender, and the Android task owns running the positive case here against the real app.
"""

# Standard library imports
import threading
from collections.abc import Callable

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.arcore_tcp import ArCoreTcpSource
from nav.sources.config import ArCoreConfig
from nav.sources.framecodec import LENGTH_PREFIX, encode_frame
from nav.types import DepthFrame
from fake_arcore_sender import FakeArCoreSender, send_frames, synthetic_frames

FRAME_COUNT = 10
# Everything here finishes in well under a second. A test that blocks this long is hung.
TEST_TIMEOUT_SECONDS = 5.0


def _listening_source(accept_timeout_seconds: float = TEST_TIMEOUT_SECONDS) -> ArCoreTcpSource:
    # Port 0 asks the OS for a free port, and the source reports which one it got.
    source = ArCoreTcpSource(ArCoreConfig(port=0, bind_address="127.0.0.1", accept_timeout_seconds=accept_timeout_seconds))
    source._listen()
    return source


def _in_background(action: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=action, daemon=True)
    thread.start()
    return thread


def _collect(source: ArCoreTcpSource) -> list[DepthFrame]:
    collected: list[DepthFrame] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            collected.extend(source.frames())
        except BaseException as error:  # noqa: BLE001, re-raised on the test thread below
            failure.append(error)

    reader = _in_background(run)
    reader.join(TEST_TIMEOUT_SECONDS)
    assert not reader.is_alive(), "frames() did not finish, the source is hung"
    if failure:
        raise failure[0]
    return collected


def test_every_sent_frame_arrives_equal(caplog: pytest.LogCaptureFixture) -> None:
    source = _listening_source()
    sent = list(synthetic_frames(FRAME_COUNT))
    try:
        _in_background(lambda: send_frames("127.0.0.1", source.port, sent))
        received = _collect(source)
    finally:
        source.close()

    assert len(received) == FRAME_COUNT
    for before, after in zip(sent, received, strict=True):
        assert after.timestamp_seconds == pytest.approx(before.timestamp_seconds)
        assert after.depth_meters == pytest.approx(before.depth_meters)
        assert after.intrinsics == pytest.approx(before.intrinsics)
        assert after.pose.has_position is True
        assert after.pose.position == pytest.approx(before.pose.position)
        assert after.ground_plane is not None
        assert after.ground_plane.offset_meters == pytest.approx(before.ground_plane.offset_meters)


def test_a_corrupt_message_is_skipped_and_the_stream_continues(caplog: pytest.LogCaptureFixture) -> None:
    source = _listening_source()
    frames = list(synthetic_frames(2))

    def send() -> None:
        sender = FakeArCoreSender("127.0.0.1", source.port)
        sender.connect()
        try:
            # Positive control first. Then a well-framed message whose payload is not a frame,
            # then another good one that must still arrive. If the first good frame did not
            # arrive, the test says so instead of passing on an empty result.
            sender.send_frame(frames[0])
            garbage = b"this is not a frame"
            sender.send_raw(LENGTH_PREFIX.pack(len(garbage)) + garbage)
            sender.send_frame(frames[1])
        finally:
            sender.close()

    try:
        _in_background(send)
        with caplog.at_level("WARNING"):
            received = _collect(source)
    finally:
        source.close()

    assert [frame.timestamp_seconds for frame in received] == pytest.approx([frames[0].timestamp_seconds, frames[1].timestamp_seconds])
    assert any("frame dropped" in record.message for record in caplog.records)


def test_a_truncated_message_then_disconnect_ends_the_stream_without_hanging(caplog: pytest.LogCaptureFixture) -> None:
    source = _listening_source()
    frames = list(synthetic_frames(2))

    def send() -> None:
        sender = FakeArCoreSender("127.0.0.1", source.port)
        sender.connect()
        try:
            sender.send_frame(frames[0])
            whole = encode_frame(frames[1])
            sender.send_raw(whole[: len(whole) // 2])
        finally:
            sender.close()

    try:
        _in_background(send)
        with caplog.at_level("WARNING"):
            received = _collect(source)
    finally:
        source.close()

    assert len(received) == 1
    assert any("part way through a frame" in record.message for record in caplog.records)


def test_a_clean_disconnect_between_frames_ends_the_stream_quietly(caplog: pytest.LogCaptureFixture) -> None:
    source = _listening_source()
    try:
        _in_background(lambda: send_frames("127.0.0.1", source.port, synthetic_frames(3)))
        with caplog.at_level("INFO"):
            received = _collect(source)
    finally:
        source.close()

    assert len(received) == 3
    assert not any(record.levelname == "WARNING" for record in caplog.records)
    assert any("disconnected after 3 frames" in record.message for record in caplog.records)


def test_no_sender_within_the_accept_timeout_raises() -> None:
    source = _listening_source(accept_timeout_seconds=0.2)
    try:
        with pytest.raises(ConnectionError, match="no sender connected"):
            list(source.frames())
    finally:
        source.close()


def test_an_absurd_length_prefix_ends_the_connection_rather_than_allocating(caplog: pytest.LogCaptureFixture) -> None:
    source = _listening_source()
    frames = list(synthetic_frames(2))

    def send() -> None:
        sender = FakeArCoreSender("127.0.0.1", source.port)
        sender.connect()
        try:
            sender.send_frame(frames[0])
            sender.send_raw(LENGTH_PREFIX.pack(0xFFFFFFFF))
            # A good frame after the bad prefix. The stream is out of step, so it must not arrive:
            # a reader that kept the connection would read it as the next message and yield it.
            sender.send_frame(frames[1])
        finally:
            sender.close()

    try:
        _in_background(send)
        with caplog.at_level("WARNING"):
            received = _collect(source)
    finally:
        source.close()

    assert [frame.timestamp_seconds for frame in received] == pytest.approx([frames[0].timestamp_seconds])
    assert any("desynchronised" in record.message for record in caplog.records)


def test_frames_can_be_called_again_for_the_next_connection() -> None:
    # The listener outlives a connection, so the runtime's --reconnect reuses the source rather
    # than rebuilding it, and the recording tap around it keeps writing to one log.
    source = _listening_source()
    try:
        _in_background(lambda: send_frames("127.0.0.1", source.port, synthetic_frames(2)))
        first = _collect(source)
        assert source._client is None, "the finished connection's socket must be closed, not kept until close()"

        _in_background(lambda: send_frames("127.0.0.1", source.port, synthetic_frames(3)))
        second = _collect(source)
    finally:
        source.close()

    assert len(first) == 2
    assert len(second) == 3


def test_a_second_listener_on_the_same_port_is_refused() -> None:
    # Two laptop runs on one port must not both report "listening". On Windows SO_REUSEADDR
    # allowed exactly that, and the phone's frames went to the run that was meant to be dead.
    first = _listening_source()
    second = ArCoreTcpSource(ArCoreConfig(port=first.port, bind_address="127.0.0.1"))
    try:
        with pytest.raises(OSError):
            second._listen()
    finally:
        second.close()
        first.close()


def test_closing_twice_and_closing_before_listening_do_not_raise() -> None:
    ArCoreTcpSource(ArCoreConfig(port=0)).close()

    source = _listening_source()
    source.close()
    source.close()


def test_synthetic_frames_carry_a_real_floor_plane() -> None:
    # The plane the sender advertises must be the plane its depth image actually shows, or the
    # scene tests that trust one of them will pass against the other.
    frame = next(synthetic_frames(1))
    height, width = frame.depth_meters.shape
    row = height - 1
    column = width // 2
    depth = frame.depth_meters[row, column]
    focal_y = frame.intrinsics[1, 1]
    principal_y = frame.intrinsics[1, 2]
    # Unproject the bottom-centre pixel and check it satisfies normal . p + offset == 0.
    point = np.array([0.0, (row - principal_y) * depth / focal_y, depth])

    assert frame.ground_plane is not None
    assert frame.ground_plane.normal @ point + frame.ground_plane.offset_meters == pytest.approx(0.0, abs=1e-3)
