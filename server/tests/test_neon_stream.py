"""
Covers the Neon receiver: playing a capture back, handing over frames, matching gaze, the IMU, and
the decoder itself.

Captures are written into tmp_path in the same format examples/capture_neon_stream.py writes,
through the same constants. Most tests use a fake decoder, so they need no PyAV. The one that uses
the real decoder encodes its own short H.264 stream and skips without PyAV, which the glasses
extra brings.
"""

# Standard library imports
import json
import logging
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
    PACKET_HEADER,
    SCENE_PACKETS_FILENAME,
    NeonStreamDevice,
    _read_packets,
)

CAMERA_MATRIX = [[890.9, 0.0, 807.3], [0.0, 890.6, 608.5], [0.0, 0.0, 1.0]]
DISTORTION = [-0.13, 0.11, 0.0, 0.0, 0.0, 0.17, 0.05, 0.03]


@dataclass
class FakeFrame:
    """Stands in for a decoded PyAV frame. Converts to an image filled with its packet's number."""

    number: int

    def to_ndarray(self, format: str) -> np.ndarray:  # noqa: A002, the PyAV name
        assert format == "bgr24"
        return np.full((4, 6, 3), self.number, dtype=np.uint8)


class FakeDecoder:
    """Every packet completes one frame, numbered by the packet's first byte. Optionally slow."""

    def __init__(self, sprop_parameter_sets: list[bytes], seconds_per_packet: float = 0.0) -> None:
        self.sprop_parameter_sets = sprop_parameter_sets
        self._seconds_per_packet = seconds_per_packet

    def feed(self, raw: bytes, timestamp_unix_seconds: float):
        time.sleep(self._seconds_per_packet)
        return FakeFrame(raw[0]), timestamp_unix_seconds


def _write_capture(
    capture: Path,
    packets: list[tuple[float, bytes]],
    gaze: list[tuple[float, float, float]] = (),
    imu: list[tuple[float, float, float, float, float]] = (),
    offset_ms: float | None = 1300.0,
) -> Path:
    capture.mkdir()
    with (capture / SCENE_PACKETS_FILENAME).open("wb") as scene:
        for timestamp, raw in packets:
            scene.write(PACKET_HEADER.pack(timestamp, len(raw)))
            scene.write(raw)
    (capture / GAZE_FILENAME).write_text("".join(json.dumps({"t": t, "x": x, "y": y}) + "\n" for t, x, y in gaze), encoding="utf-8")
    (capture / IMU_FILENAME).write_text(
        "".join(json.dumps({"t": t, "w": w, "x": x, "y": y, "z": z}) + "\n" for t, w, x, y, z in imu), encoding="utf-8"
    )
    meta = {
        "scene_camera_matrix": CAMERA_MATRIX,
        "scene_distortion_coefficients": DISTORTION,
        "sprop": [],
        "time_offset": None if offset_ms is None else {"median_ms": offset_ms, "round_trip_median_ms": 7.0},
    }
    (capture / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    return capture


def _started(capture: Path, **decoder_options) -> NeonStreamDevice:
    device = NeonStreamDevice(
        NeonConfig(replay_dir=str(capture)),
        decoder_factory=lambda sprop: FakeDecoder(sprop, **decoder_options),
    )
    device.start()
    return device


def test_packets_written_in_the_capture_format_read_back_unchanged(tmp_path: Path) -> None:
    packets = [(100.0, b"\x01abc"), (100.033, b"\x02"), (100.066, bytes(range(256)))]
    capture = _write_capture(tmp_path / "capture", packets)

    assert list(_read_packets(capture / SCENE_PACKETS_FILENAME)) == packets


def test_a_replay_hands_over_the_newest_frame_and_never_the_same_frame_twice(tmp_path: Path) -> None:
    # Frames 1 to 3 within the first 0.1 s, then frame 4 a second later.
    capture = _write_capture(tmp_path / "capture", [(50.0, b"\x01"), (50.05, b"\x02"), (50.1, b"\x03"), (51.1, b"\x04")])
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


def test_the_end_of_a_replay_is_raised_only_after_the_last_frame_is_taken(tmp_path: Path) -> None:
    capture = _write_capture(tmp_path / "capture", [(10.0, b"\x01"), (10.05, b"\x02")])
    device = _started(capture)
    try:
        time.sleep(0.4)
        last = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        with pytest.raises(EOFError, match="played to the end"):
            device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
    finally:
        device.close()

    assert np.all(last.frame.bgr_pixels == 2)


def test_gaze_is_matched_to_the_nearest_sample_within_a_frame_and_a_half(tmp_path: Path) -> None:
    capture = _write_capture(
        tmp_path / "capture",
        [(20.0, b"\x01"), (20.6, b"\x02")],
        # One sample 10 ms after the first frame. None near the second, the nearest is 0.4 s away.
        gaze=[(20.01, 300.0, 200.0), (20.2, 1.0, 1.0)],
    )
    device = _started(capture)
    try:
        # The pipeline asks for a frame well after it lands. Asking the instant it is decoded
        # would be before the gaze sample 10 ms later has been fed.
        time.sleep(0.2)
        first = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        second = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=2.0)
    finally:
        device.close()

    assert (first.gaze.x, first.gaze.y) == (300.0, 200.0)
    assert second.gaze is None


def test_a_capture_without_gaze_gives_frames_without_gaze(tmp_path: Path) -> None:
    capture = _write_capture(tmp_path / "capture", [(30.0, b"\x01")])
    device = _started(capture)
    try:
        matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
    finally:
        device.close()

    assert matched.gaze is None


def test_an_imu_reading_is_handed_over_once(tmp_path: Path) -> None:
    capture = _write_capture(tmp_path / "capture", [(40.0, b"\x01"), (40.5, b"\x02")], imu=[(40.0, 0.9, 0.1, 0.2, 0.3)])
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
    capture = _write_capture(tmp_path / "capture", [(1_000.0, b"\x01")], offset_ms=1300.0)
    device = _started(capture)
    try:
        matched = device.receive_matched_scene_video_frame_and_gaze(timeout_seconds=1.0)
        received = time.time()
        offset = device.estimate_time_offset()
    finally:
        device.close()

    assert offset.time_offset_ms.median == pytest.approx(1300.0)
    assert matched.frame.timestamp_unix_seconds + offset.time_offset_ms.median / 1000.0 == pytest.approx(received, abs=0.3)


def test_the_calibration_comes_from_the_capture(tmp_path: Path) -> None:
    device = _started(_write_capture(tmp_path / "capture", [(1.0, b"\x01")]))
    try:
        calibration = device.get_calibration()
    finally:
        device.close()

    assert calibration.scene_camera_matrix == pytest.approx(np.array(CAMERA_MATRIX))
    assert calibration.scene_distortion_coefficients == pytest.approx(np.array(DISTORTION))


def test_a_capture_without_scene_packets_fails_start_with_a_message(tmp_path: Path) -> None:
    device = NeonStreamDevice(
        NeonConfig(replay_dir=str(_write_capture(tmp_path / "capture", []))),
        decoder_factory=FakeDecoder,
    )

    with pytest.raises(ValueError, match="no scene packets"):
        device.start()


def test_a_decoder_that_falls_behind_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # Fifty packets at once into a decoder taking 20 ms each. The queue has to show it.
    monkeypatch.setattr(neon_stream, "DECODE_BACKLOG_WARNING_PACKETS", 5)
    capture = _write_capture(tmp_path / "capture", [(5.0, bytes([index + 1])) for index in range(50)])

    with caplog.at_level(logging.WARNING, logger="nav.sources.neon_stream"):
        device = _started(capture, seconds_per_packet=0.02)
        try:
            time.sleep(0.3)
        finally:
            device.close()

    assert any("decoder is" in record.message and "behind" in record.message for record in caplog.records)


def _annex_b_units(stream: bytes) -> list[bytes]:
    """Split an H.264 byte stream on its start codes, the way RTP carries one NAL unit per packet."""
    units, start = [], None
    index = 0
    while index < len(stream) - 3:
        three = stream[index : index + 3] == b"\x00\x00\x01"
        four = stream[index : index + 4] == b"\x00\x00\x00\x01"
        if three or four:
            if start is not None:
                units.append(stream[start:index])
            index += 4 if four else 3
            start = index
            continue
        index += 1
    if start is not None:
        units.append(stream[start:])
    return [unit for unit in units if unit]


def test_a_real_h264_stream_decodes_through_the_scene_decoder() -> None:
    av = pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    pytest.importorskip("pupil_labs.realtime_api.streaming.nal_unit", reason="the client comes with the glasses extra")

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 64, 48, "yuv420p"
    encoder.options = {"tune": "zerolatency", "preset": "ultrafast"}
    stream = b""
    for index in range(12):
        image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        for packet in encoder.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
            stream += bytes(packet)
    for packet in encoder.encode(None):
        stream += bytes(packet)

    units = _annex_b_units(stream)
    # NAL types 7 and 8 are the sequence and picture parameter sets, which the SDP carries live.
    sprop = [unit for unit in units if unit[0] & 0x1F in (7, 8)]
    pictures = [unit for unit in units if unit[0] & 0x1F not in (6, 7, 8)]
    decoder = neon_stream.SceneDecoder(sprop)

    decoded = []
    for index, unit in enumerate(pictures + [pictures[-1]]):
        result = decoder.feed(unit, 100.0 + index / 30)
        if result is not None:
            decoded.append(result)

    assert len(decoded) >= 10
    frame, timestamp = decoded[0]
    image = frame.to_ndarray(format="bgr24")
    assert image.shape == (48, 64, 3)
    assert timestamp == pytest.approx(100.0)


def test_closing_mid_replay_with_packets_queued_returns_promptly_and_leaves_no_thread_waiting(tmp_path: Path) -> None:
    # A slow decoder and a full queue, closed before it drains. A join on the queue once waited for
    # good here, on a thread the interpreter then waited for at exit.
    capture = _write_capture(tmp_path / "capture", [(5.0, bytes([index + 1])) for index in range(50)])
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
