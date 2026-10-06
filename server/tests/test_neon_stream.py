"""
Covers the Neon receiver: playing a capture back, handing over frames, matching gaze, the IMU, the
decoder itself, and the live path against a fake client.

Captures are written into tmp_path in the same format examples/capture_neon_stream.py writes,
through the same constants. Most tests use a fake decoder, so they need no PyAV. The ones that use
the real decoder encode their own short H.264 stream, and the live ones patch the client's own
classes, so those skip without the glasses extra.
"""

# Standard library imports
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources import neon_stream
from nav.sources.config import NeonConfig
from nav.sources.neon_stream import (
    GAZE_FILENAME,
    IMU_FILENAME,
    META_FILENAME,
    SCENE_PACKETS_FILENAME,
    NeonStreamDevice,
    _read_packets,
)
from neon_captures import CAMERA_MATRIX, DISTORTION, encode_h264, write_capture


@dataclass
class FakeFrame:
    """Stands in for a decoded PyAV frame. Converts to an image filled with its packet's number."""

    number: int
    conversions: int = 0

    def to_ndarray(self, format: str) -> np.ndarray:  # noqa: A002, the PyAV name
        assert format == "bgr24"
        self.conversions += 1
        return np.full((4, 6, 3), self.number, dtype=np.uint8)


class FakeDecoder:
    """Every packet completes one frame, numbered by the packet's first byte.

    Optionally slow, failing on chosen packet numbers, or holding each frame back until the next
    packet the way the real parser does, so only a flush lets the last one go.
    """

    def __init__(
        self,
        sprop_parameter_sets: list[bytes],
        seconds_per_packet: float = 0.0,
        failures: dict[int, BaseException] | None = None,
        holds_last_frame: bool = False,
    ) -> None:
        self.sprop_parameter_sets = sprop_parameter_sets
        self._seconds_per_packet = seconds_per_packet
        self._failures = failures or {}
        self._holds_last_frame = holds_last_frame
        self._held: tuple[FakeFrame, float] | None = None
        self.frames: list[FakeFrame] = []

    def feed(self, raw: bytes, timestamp_unix_seconds: float) -> tuple[FakeFrame, float] | None:
        time.sleep(self._seconds_per_packet)
        if raw[0] in self._failures:
            raise self._failures[raw[0]]
        frame = FakeFrame(raw[0])
        self.frames.append(frame)
        if not self._holds_last_frame:
            return frame, timestamp_unix_seconds
        previous, self._held = self._held, (frame, timestamp_unix_seconds)
        return previous

    def flush(self) -> tuple[FakeFrame, float] | None:
        held, self._held = self._held, None
        return held


class NoFrameDecoder(FakeDecoder):
    """Takes every packet and never completes a frame, like a stream the decoder cannot make sense of."""

    def feed(self, raw: bytes, timestamp_unix_seconds: float) -> None:
        return None


def _started(capture: Path, decoders: list | None = None, decoder_class: type = FakeDecoder, **decoder_options) -> NeonStreamDevice:
    def build(sprop: list[bytes]) -> FakeDecoder:
        decoder = decoder_class(sprop, **decoder_options)
        if decoders is not None:
            decoders.append(decoder)
        return decoder

    device = NeonStreamDevice(NeonConfig(replay_dir=str(capture)), decoder_factory=build)
    device.start()
    return device


def _frames_until_the_end(device: NeonStreamDevice, timeout_seconds: float = 2.0) -> list[int]:
    """Every frame number handed over before the end of the replay, in order."""
    numbers = []
    for _ in range(100):
        try:
            matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=timeout_seconds)
        except EOFError:
            return numbers
        assert matched is not None, "the replay neither handed over a frame nor ended"
        numbers.append(int(matched.frame.bgr_pixels[0, 0, 0]))
    raise AssertionError("the replay never ended")


def test_packets_written_in_the_capture_format_read_back_unchanged(tmp_path: Path) -> None:
    packets = [(100.0, b"\x01abc"), (100.033, b"\x02"), (100.066, bytes(range(256)))]
    capture = write_capture(tmp_path / "capture", packets)

    assert list(_read_packets(capture / SCENE_PACKETS_FILENAME)) == packets


def test_a_replay_hands_over_the_newest_frame_and_never_the_same_frame_twice(tmp_path: Path) -> None:
    # Frames 1 to 3 within the first 0.1 s, then frame 4 a second later.
    capture = write_capture(tmp_path / "capture", [(50.0, b"\x01"), (50.05, b"\x02"), (50.1, b"\x03"), (51.1, b"\x04")])
    device = _started(capture)
    try:
        time.sleep(0.4)
        newest = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        nothing_new = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.1)
        later = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=2.0)
    finally:
        device.close()

    # The newest of the three, not the first: the pipeline wants the present, not a queue.
    assert np.all(newest.frame.bgr_pixels == 3)
    assert nothing_new is None
    assert np.all(later.frame.bgr_pixels == 4)


def test_only_the_frame_handed_over_is_converted_to_bgr(tmp_path: Path) -> None:
    """Converting all 30 frames a second is what put the client's own decoder seconds behind."""
    decoders: list[FakeDecoder] = []
    capture = write_capture(tmp_path / "capture", [(50.0, b"\x01"), (50.05, b"\x02"), (50.1, b"\x03")])
    device = _started(capture, decoders)
    try:
        time.sleep(0.4)
        device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
    finally:
        device.close()

    assert len(decoders[0].frames) == 3, "all three decoded"
    assert [frame.conversions for frame in decoders[0].frames] == [0, 0, 1]


def test_the_end_of_a_replay_is_raised_only_after_the_last_frame_is_taken(tmp_path: Path) -> None:
    capture = write_capture(tmp_path / "capture", [(10.0, b"\x01"), (10.05, b"\x02")])
    device = _started(capture)
    try:
        time.sleep(0.4)
        last = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        with pytest.raises(EOFError, match="played to the end"):
            device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
    finally:
        device.close()

    assert np.all(last.frame.bgr_pixels == 2)


def test_the_end_of_a_replay_waits_for_every_queued_packet_to_be_decoded(tmp_path: Path) -> None:
    """All ten packets are due at once and take half a second to decode. Ending before that loses the last frames."""
    capture = write_capture(tmp_path / "capture", [(10.0, bytes([index + 1])) for index in range(10)])
    device = _started(capture, seconds_per_packet=0.05)
    try:
        numbers = _frames_until_the_end(device)
    finally:
        device.close()

    assert numbers[-1] == 10


def test_the_frame_the_decoder_holds_back_at_the_end_is_still_handed_over(tmp_path: Path) -> None:
    """The parser lets a frame go only when the next one starts. Without a flush the last frame of every replay is lost."""
    capture = write_capture(tmp_path / "capture", [(10.0, b"\x01"), (10.05, b"\x02"), (10.1, b"\x03")])
    device = _started(capture, holds_last_frame=True)
    try:
        numbers = _frames_until_the_end(device)
    finally:
        device.close()

    assert numbers[-1] == 3


def test_a_replay_that_decodes_no_frames_ends_in_an_error_naming_the_capture(tmp_path: Path) -> None:
    """A run of zero frames that exits cleanly looks exactly like a short walk. A waiting receive raises at once, not at its timeout."""
    capture = write_capture(tmp_path / "unreadable_capture", [(10.0, b"\x01"), (10.3, b"\x02")])
    device = _started(capture, decoder_class=NoFrameDecoder)
    try:
        started = time.monotonic()
        with pytest.raises(ConnectionError, match="unreadable_capture decoded no frames from its 2 scene packets"):
            device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=5.0)
        waited_seconds = time.monotonic() - started
    finally:
        device.close()

    assert waited_seconds < 2.0, "the receive waited out its timeout instead of raising the failure"


def test_a_damaged_packet_is_skipped_and_decoding_goes_on(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """H.264 recovers at the next keyframe, so one bad packet must not stop the stream. A run of them is reported once."""
    failures = {2: ValueError("First bit must be zero (forbidden_zero_bit)"), 3: struct.error("unpack requires a buffer of 1 bytes")}
    capture = write_capture(tmp_path / "capture", [(10.0 + index * 0.05, bytes([index + 1])) for index in range(4)])

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_stream"):
        device = _started(capture, failures=failures)
        try:
            numbers = _frames_until_the_end(device)
        finally:
            device.close()

    assert numbers[-1] == 4, "decoding stopped at the damaged packet"
    warnings = [record.getMessage() for record in caplog.records if "could not read" in record.getMessage()]
    assert len(warnings) == 1
    assert "ValueError" in warnings[0]


def test_an_unknown_decoder_failure_is_raised_by_the_next_receive(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A dead decode thread used to leave the run waiting for good, live and on a replay alike."""
    capture = write_capture(tmp_path / "capture", [(10.0 + index * 0.05, bytes([index + 1])) for index in range(4)])

    with caplog.at_level(logging.ERROR, logger="nav.sources.neon_stream"):
        device = _started(capture, failures={2: KeyError("a failure nobody named")})
        try:
            time.sleep(0.4)
            with pytest.raises(ConnectionError, match="KeyError") as stopped:
                for _ in range(3):
                    device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        finally:
            started = time.monotonic()
            device.close()
            closing_seconds = time.monotonic() - started

    assert "a failure nobody named" in str(stopped.value)
    assert closing_seconds < 2.0
    assert any("UNEXPECTED KeyError" in record.getMessage() and record.exc_info for record in caplog.records)


def test_gaze_is_matched_to_the_nearest_sample_within_a_frame_and_a_half(tmp_path: Path) -> None:
    """A frame is 33 ms, so the bound is 50 ms. Past it, a gaze point belongs to another moment."""
    capture = write_capture(
        tmp_path / "capture",
        [(20.0, b"\x01"), (20.6, b"\x02"), (21.2, b"\x03")],
        gaze=[
            # Frame 1: 10 ms before and 30 ms after. The earlier one is the nearer.
            (19.99, 300.0, 200.0),
            (20.03, 1.0, 1.0),
            # Frame 2: 45 ms before, inside the bound.
            (20.555, 40.0, 50.0),
            # Frame 3: 60 ms before, outside it.
            (21.14, 7.0, 7.0),
        ],
    )
    device = _started(capture)
    try:
        # The pipeline asks for a frame well after it lands. Asking the instant it is decoded
        # would be before the gaze sample 30 ms later has been fed.
        time.sleep(0.2)
        first = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        second = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=2.0)
        third = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=2.0)
    finally:
        device.close()

    assert (first.gaze.x, first.gaze.y) == (300.0, 200.0)
    assert (second.gaze.x, second.gaze.y) == (40.0, 50.0)
    assert third.gaze is None


def test_a_capture_without_gaze_gives_frames_without_gaze(tmp_path: Path) -> None:
    capture = write_capture(tmp_path / "capture", [(30.0, b"\x01")])
    device = _started(capture)
    try:
        matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
    finally:
        device.close()

    assert matched.gaze is None


def test_an_imu_reading_is_handed_over_once(tmp_path: Path) -> None:
    capture = write_capture(tmp_path / "capture", [(40.0, b"\x01"), (40.5, b"\x02")], imu=[(40.0, 0.9, 0.1, 0.2, 0.3)])
    device = _started(capture)
    try:
        device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        first = device.receive_imu_datum(timeout_seconds=0.0)
        again = device.receive_imu_datum(timeout_seconds=0.0)
    finally:
        device.close()

    quaternion = first.quaternion
    assert (quaternion.w, quaternion.x, quaternion.y, quaternion.z) == (0.9, 0.1, 0.2, 0.3)
    assert again is None, "the source carries an orientation forward itself, a repeat would look new"


def test_replayed_stamps_land_on_the_laptop_clock_as_they_are_fed(tmp_path: Path) -> None:
    # Recorded long ago on the Neon's clock, with the laptop 1.3 s ahead. Played back, a frame's
    # stamp plus the offset must be about now, so latency figures measure this laptop.
    capture = write_capture(tmp_path / "capture", [(1_000.0, b"\x01")], offset_ms=1300.0)
    device = _started(capture)
    try:
        matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        received = time.time()
        offset = device.estimate_time_offset()
    finally:
        device.close()

    assert offset.time_offset_ms.median == pytest.approx(1300.0)
    assert matched.frame.timestamp_unix_seconds + offset.time_offset_ms.median / 1000.0 == pytest.approx(received, abs=0.3)


def test_a_capture_recorded_without_a_clock_offset_reports_none_and_not_zero(tmp_path: Path) -> None:
    """An offset of 0 ms would be logged and used as if Time Echo had measured it."""
    capture = write_capture(tmp_path / "capture", [(1_000.0, b"\x01")], offset_ms=None)
    device = _started(capture)
    try:
        offset = device.estimate_time_offset()
    finally:
        device.close()

    assert offset is None


def test_the_calibration_comes_from_the_capture(tmp_path: Path) -> None:
    device = _started(write_capture(tmp_path / "capture", [(1.0, b"\x01")]))
    try:
        calibration = device.get_calibration()
    finally:
        device.close()

    assert calibration.scene_camera_matrix == pytest.approx(np.array(CAMERA_MATRIX))
    assert calibration.scene_distortion_coefficients == pytest.approx(np.array(DISTORTION))


def test_a_capture_without_scene_packets_fails_start_with_a_message(tmp_path: Path) -> None:
    device = NeonStreamDevice(
        NeonConfig(replay_dir=str(write_capture(tmp_path / "capture", []))),
        decoder_factory=FakeDecoder,
    )

    with pytest.raises(ValueError, match="no scene packets"):
        device.start()


def _cut_a_header(capture: Path) -> str:
    packets = capture / SCENE_PACKETS_FILENAME
    size = packets.stat().st_size
    packets.write_bytes(packets.read_bytes() + b"\x00" * 5)
    return f"{SCENE_PACKETS_FILENAME} ends partway through a packet header at byte {size}"


def _cut_a_payload(capture: Path) -> str:
    packets = capture / SCENE_PACKETS_FILENAME
    data = packets.read_bytes()
    packets.write_bytes(data[:-3])
    # Two packets of 12 header bytes and 4 payload bytes each. The second starts at byte 16.
    return f"{SCENE_PACKETS_FILENAME} has a packet at byte 16 of 4 bytes, and only 1 remain"


def _drop_the_parameter_sets(capture: Path) -> str:
    meta_path = capture / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["sprop"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return f"{META_FILENAME} has no 'sprop'"


def _break_a_gaze_line(capture: Path) -> str:
    with (capture / GAZE_FILENAME).open("a", encoding="utf-8") as gaze:
        gaze.write('{"t": 1.0, "x": \n')
    return f"{GAZE_FILENAME} line 2 (JSONDecodeError"


def _drop_an_imu_field(capture: Path) -> str:
    (capture / IMU_FILENAME).write_text(json.dumps({"t": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}) + "\n", encoding="utf-8")
    return f"{IMU_FILENAME} line 1 (KeyError: 'w')"


@pytest.mark.parametrize(
    "damage",
    [_cut_a_header, _cut_a_payload, _drop_the_parameter_sets, _break_a_gaze_line, _drop_an_imu_field],
    ids=["cut header", "cut payload", "meta without sprop", "broken gaze line", "imu line without w"],
)
def test_a_malformed_capture_is_refused_at_start_naming_the_capture_and_the_place(tmp_path: Path, damage) -> None:
    """A damaged capture used to read back short, or fail as UNEXPECTED, or as the stream stopping."""
    capture = write_capture(tmp_path / "damaged_capture", [(1.0, b"\x01abc"), (1.1, b"\x02def")], gaze=[(1.0, 2.0, 3.0)])
    where = damage(capture)
    device = NeonStreamDevice(NeonConfig(replay_dir=str(capture)), decoder_factory=FakeDecoder)

    with pytest.raises(ValueError) as refused:
        device.start()
    device.close()

    message = str(refused.value)
    assert f"capture {capture} is malformed" in message
    assert where in message


def test_a_decoder_that_falls_behind_is_reported_once_per_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Fifty packets at once into a decoder taking 20 ms each. The queue has to show it, and the
    # forty or so packets past the threshold are one warning, not forty.
    monkeypatch.setattr(neon_stream, "DECODE_BACKLOG_WARNING_PACKETS", 5)
    capture = write_capture(tmp_path / "capture", [(5.0, bytes([index + 1])) for index in range(50)])

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_stream"):
        device = _started(capture, seconds_per_packet=0.02)
        try:
            time.sleep(0.3)
        finally:
            device.close()

    warnings = [record for record in caplog.records if "decoder is" in record.message and "behind" in record.message]
    assert len(warnings) == 1


def test_a_real_h264_stream_decodes_through_the_scene_decoder() -> None:
    """Every picture comes out, the last one by the flush, each stamped with its own packet's stamp."""
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    pytest.importorskip("pupil_labs.realtime_api.streaming.nal_unit", reason="the client comes with the glasses extra")
    stream = encode_h264(frame_count=12)
    # The parameter sets as the client hands them over, so this is also the shape a capture stores.
    decoder = neon_stream.SceneDecoder(stream.parameter_sets)

    decoded = []
    for index, picture in enumerate(stream.pictures):
        result = decoder.feed(picture, 100.0 + index / 30)
        if result is not None:
            decoded.append(result)
    flushed = decoder.flush()
    if flushed is not None:
        decoded.append(flushed)

    assert len(decoded) == len(stream.brightness) == 12
    first_frame, first_stamp = decoded[0]
    assert first_frame.to_ndarray(format="bgr24").shape == (48, 64, 3)
    assert first_stamp == pytest.approx(100.0)
    last_frame, last_stamp = decoded[-1]
    assert last_stamp == pytest.approx(100.0 + 11 / 30)
    assert last_frame.to_ndarray(format="bgr24").mean() == pytest.approx(stream.brightness[-1], abs=6)


def test_damaged_packets_in_a_real_replay_are_skipped_and_the_last_frame_still_arrives(tmp_path: Path) -> None:
    """The two ways the client's own NAL helper refuses a packet, in a real stream through the real decoder."""
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    pytest.importorskip("pupil_labs.realtime_api.streaming.nal_unit", reason="the client comes with the glasses extra")
    stream = encode_h264(frame_count=12)
    payloads = list(stream.pictures)
    # A packet with the forbidden bit set, and an empty one, between two pictures.
    payloads[5:5] = [b"\x80damaged", b""]
    capture = write_capture(
        tmp_path / "capture",
        [(100.0 + index / 30, payload) for index, payload in enumerate(payloads)],
        parameter_sets=stream.parameter_sets,
    )
    device = NeonStreamDevice(NeonConfig(replay_dir=str(capture)))
    device.start()
    try:
        brightness = []
        while True:
            try:
                matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=2.0)
            except EOFError:
                break
            assert matched is not None, "the replay neither handed over a frame nor ended"
            brightness.append(float(matched.frame.bgr_pixels.mean()))
    finally:
        device.close()

    assert brightness[-1] == pytest.approx(stream.brightness[-1], abs=6)


def test_closing_mid_replay_with_packets_queued_returns_promptly_and_leaves_no_thread_waiting(tmp_path: Path) -> None:
    # A slow decoder and a full queue, closed before it drains. A join on the queue once waited for
    # good here, on a thread the interpreter then waited for at exit.
    capture = write_capture(tmp_path / "capture", [(5.0, bytes([index + 1])) for index in range(50)])
    before = {thread.ident for thread in threading.enumerate()}
    device = _started(capture, seconds_per_packet=0.02)
    time.sleep(0.2)

    started = time.monotonic()
    device.close()
    closing_seconds = time.monotonic() - started

    time.sleep(0.3)
    left = [thread for thread in threading.enumerate() if thread.ident not in before and thread.is_alive()]
    assert closing_seconds < 2.0
    assert left == [], f"threads still running after close: {[thread.name for thread in left]}"


# The live path, against fake client classes. Each test installs the fake through monkeypatch, so
# the real client is back in place afterwards.


def _live_device(monkeypatch: pytest.MonkeyPatch, script, **config_options) -> NeonStreamDevice:
    pytest.importorskip("pupil_labs.realtime_api", reason="the client comes with the glasses extra")
    from fake_neon_client import install

    install(script, monkeypatch.setattr)
    return NeonStreamDevice(NeonConfig(address="192.0.2.7", **config_options), decoder_factory=FakeDecoder)


def _scene_packets(count: int) -> list[tuple[float, bytes]]:
    return [(1_000.0 + index * 0.02, bytes([index + 1])) for index in range(count)]


def test_a_gaze_stream_that_fails_mid_run_leaves_frames_coming_without_gaze(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Gaze is optional, so losing it must not end a walk that can still plan."""
    from fake_neon_client import FakeNeonScript

    script = FakeNeonScript(
        scene_packets=_scene_packets(40),
        gaze_samples=[(1_000.0, 5.0, 5.0), (1_000.005, 6.0, 6.0)],
        gaze_failure=ConnectionResetError("gaze socket reset"),
    )
    device = _live_device(monkeypatch, script)

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_stream"):
        device.start()
        try:
            time.sleep(0.5)
            matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        finally:
            device.close()

    # The gaze stream failed after about 30 ms. Frame 10 is 200 ms in.
    assert int(matched.frame.bgr_pixels[0, 0, 0]) > 10
    assert matched.gaze is None
    warnings = [record.getMessage() for record in caplog.records if "gaze stream stopped" in record.getMessage()]
    assert len(warnings) == 1
    assert "ConnectionResetError" in warnings[0]


def test_an_imu_stream_that_fails_ends_the_run_naming_the_imu_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Carrying on would hand out the last orientation for good, and the floor would follow a head that has moved."""
    from fake_neon_client import FakeNeonScript

    script = FakeNeonScript(
        scene_packets=_scene_packets(40),
        imu_samples=[(1_000.0, 1.0, 0.0, 0.0, 0.0)],
        imu_failure=ConnectionResetError("imu socket reset"),
    )
    device = _live_device(monkeypatch, script)
    device.start()
    try:
        with pytest.raises(ConnectionError, match="IMU stream stopped") as stopped:
            for _ in range(20):
                device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=0.25)
    finally:
        device.close()

    assert "ConnectionResetError" in str(stopped.value)


def test_a_scene_camera_that_is_not_streaming_fails_start_naming_the_companion_app(monkeypatch: pytest.MonkeyPatch) -> None:
    from fake_neon_client import FakeNeonScript

    device = _live_device(monkeypatch, FakeNeonScript(world_connected=False))

    with pytest.raises(ConnectionError, match="scene camera is not streaming. Is the Companion app open"):
        device.start()
    device.close()


def test_a_time_echo_that_never_answers_is_given_up_after_the_timeout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The client takes no timeout, and the parent's request holds the pipe until this returns."""
    from fake_neon_client import FakeNeonScript

    device = _live_device(monkeypatch, FakeNeonScript(time_echo_never_answers=True), time_echo_timeout_seconds=0.3)
    outcome = {}

    def measure() -> None:
        outcome["value"] = device.estimate_time_offset(number_of_measurements=20)

    # On a thread, so a measurement with no bound fails this test instead of hanging the suite.
    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_stream"):
        started = time.monotonic()
        worker = threading.Thread(target=measure, daemon=True)
        worker.start()
        worker.join(3.0)
        measuring_seconds = time.monotonic() - started

    assert not worker.is_alive(), "the measurement was still running 3 s after a 0.3 s timeout"
    assert outcome["value"] is None
    assert measuring_seconds < 1.3
    assert any("did not answer within 0.3 s" in record.getMessage() for record in caplog.records)


def test_a_time_echo_that_answers_returns_the_phones_medians(monkeypatch: pytest.MonkeyPatch) -> None:
    from fake_neon_client import FakeNeonScript

    device = _live_device(monkeypatch, FakeNeonScript(time_offset_ms=1287.0, round_trip_ms=9.0))

    offset = device.estimate_time_offset(number_of_measurements=20)

    assert offset.time_offset_ms.median == pytest.approx(1287.0)
    assert offset.roundtrip_duration_ms.median == pytest.approx(9.0)
