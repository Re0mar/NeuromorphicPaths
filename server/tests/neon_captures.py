"""
Builds Neon capture folders for the tests, in the format examples/capture_neon_stream.py writes.

write_capture puts packets, gaze, IMU and meta into a folder through the same format constants the
replay reads. encode_h264 makes a short real H.264 stream, cut into packets the way the glasses
send it, for the tests that run the real decoder. That one needs PyAV and the client, so callers
importorskip both first.
"""

# Standard library imports
import base64
import json
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.sources.neon_stream import (
    GAZE_FILENAME,
    IMU_FILENAME,
    META_FILENAME,
    PACKET_HEADER,
    SCENE_PACKETS_FILENAME,
)

CAMERA_MATRIX = [[890.9, 0.0, 807.3], [0.0, 890.6, 608.5], [0.0, 0.0, 1.0]]
DISTORTION = [-0.13, 0.11, 0.0, 0.0, 0.0, 0.17, 0.05, 0.03]
# H.264 NAL unit types. 7 and 8 are the parameter sets the stream description carries live, and 6
# is supplemental information no decoder needs.
SEQUENCE_PARAMETER_SET = 7
PICTURE_PARAMETER_SET = 8
SUPPLEMENTAL_INFORMATION = 6
START_CODE = b"\x00\x00\x00\x01"


def write_capture(
    capture: Path,
    packets: list[tuple[float, bytes]],
    gaze: list[tuple[float, float, float]] = (),
    imu: list[tuple[float, float, float, float, float]] = (),
    offset_ms: float | None = 1300.0,
    parameter_sets: list[bytes] = (),
) -> Path:
    """
    A capture folder holding exactly what is passed in.

    :param packets: (stamp, payload) per scene packet.
    :param gaze: (stamp, x, y) per gaze sample.
    :param imu: (stamp, w, x, y, z) per IMU reading.
    :param offset_ms: The recorded Time Echo offset, or None for a capture made without one.
    :param parameter_sets: SPS and PPS as the client's `sprop_parameter_set_payloads` gives them.
    """
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
        "sprop": [base64.b64encode(parameter_set).decode("ascii") for parameter_set in parameter_sets],
        "time_offset": None if offset_ms is None else {"median_ms": offset_ms, "round_trip_median_ms": 7.0},
    }
    (capture / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    return capture


def annex_b_units(stream: bytes) -> list[bytes]:
    """Split an H.264 byte stream on its start codes, the way RTP carries one NAL unit per packet."""
    units, start = [], None
    index = 0
    while index < len(stream) - 3:
        three = stream[index : index + 3] == b"\x00\x00\x01"
        four = stream[index : index + 4] == START_CODE
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


@dataclass(frozen=True)
class EncodedStream:
    """A real H.264 stream, split the way the glasses deliver it."""

    parameter_sets: list[bytes]  # As the client's sprop_parameter_set_payloads, start code included.
    pictures: list[bytes]  # One RTP payload per picture, no start code, as the packets arrive.
    brightness: list[int]  # The gray level each picture was encoded from, in order.


def encode_h264(frame_count: int, height: int = 48, width: int = 64) -> EncodedStream:
    """
    Encode `frame_count` flat gray frames, each brighter than the last, with the settings the
    glasses' live stream is closest to: no B-frames and no lookahead.
    """
    # Optional dependencies. Both come with the glasses extra, and every caller importorskips them.
    import av
    from pupil_labs.realtime_api.streaming.nal_unit import extract_payload_from_nal_unit

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = width, height, "yuv420p"
    encoder.options = {"tune": "zerolatency", "preset": "ultrafast"}
    brightness = [index * 20 for index in range(frame_count)]
    stream = b""
    for level in brightness:
        image = np.full((height, width, 3), level, dtype=np.uint8)
        for packet in encoder.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
            stream += bytes(packet)
    for packet in encoder.encode(None):
        stream += bytes(packet)

    units = annex_b_units(stream)
    parameter_sets = [
        bytes(extract_payload_from_nal_unit(unit))
        for unit in units
        if unit[0] & 0x1F in (SEQUENCE_PARAMETER_SET, PICTURE_PARAMETER_SET)
    ]
    pictures = [
        unit for unit in units if unit[0] & 0x1F not in (SEQUENCE_PARAMETER_SET, PICTURE_PARAMETER_SET, SUPPLEMENTAL_INFORMATION)
    ]
    return EncodedStream(parameter_sets=parameter_sets, pictures=pictures, brightness=brightness)
