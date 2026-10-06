"""
Covers the per-frame timing record and how it travels through the frame codec.

The timing block is optional on the wire in both directions. The Pixel app never sends it, and
every frame log written before it existed lacks it, so the tests that matter most here are the
ones proving a frame without it still decodes as it always did.
"""

# Standard library imports
import dataclasses
import json
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.framecodec import LENGTH_PREFIX, FrameDecodeError, decode_frame, encode_frame, read_message_from_file
from nav.types import DepthFrame, FrameTiming, Pose
from fake_arcore_sender import synthetic_frames

FIXTURES = Path(__file__).parent / "fixtures"


def _payload(message: bytes) -> bytes:
    return message[LENGTH_PREFIX.size :]


def _frame_with(timing: FrameTiming | None) -> DepthFrame:
    return DepthFrame(
        timestamp_seconds=1.0,
        depth_meters=np.ones((4, 4), dtype=np.float32),
        intrinsics=np.eye(3),
        pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False),
        ground_plane=None,
        gaze_pixel=None,
        timing=timing,
    )


def _header_with_timing(timing_block: object) -> bytes:
    """A valid encoded frame whose header's timing key is replaced with whatever the test hands in."""
    payload = _payload(encode_frame(_frame_with(None)))
    header_bytes, _, depth_bytes = payload.partition(b"\n")
    header = json.loads(header_bytes)
    header["timing"] = timing_block
    return json.dumps(header).encode("utf-8") + b"\n" + depth_bytes


def test_frame_timing_accepts_missing_capture_and_depth_ready() -> None:
    timing = FrameTiming(capture_seconds=None, arrival_seconds=10.0, depth_ready_seconds=None)

    assert timing.arrival_seconds == 10.0


def test_timing_round_trips_through_the_codec() -> None:
    timing = FrameTiming(capture_seconds=99.95, arrival_seconds=100.0, depth_ready_seconds=100.1)

    decoded = decode_frame(_payload(encode_frame(_frame_with(timing))))

    assert decoded.timing == timing


def test_a_frame_without_timing_encodes_with_no_timing_key() -> None:
    # So a frame from a source that keeps no timing is byte for byte what it was before the key.
    header_bytes = _payload(encode_frame(_frame_with(None))).partition(b"\n")[0]

    assert "timing" not in json.loads(header_bytes)


def test_a_header_without_timing_decodes_with_timing_none() -> None:
    # The committed frame from the Pixel app's own encoder has no timing block and never will.
    pixel_frame = decode_frame(read_message_from_file(FIXTURES / "pixel_app_frame.bin"))
    synthetic = decode_frame(_payload(encode_frame(next(iter(synthetic_frames(1))))))

    assert pixel_frame.timing is None
    assert synthetic.timing is None


def test_frame_timing_refuses_a_missing_arrival() -> None:
    """
    Every share is measured from arrival, so a record without one has nothing to measure from.

    Left through, it crashed the verbose line on the worker with a TypeError, ending the run.
    """
    with pytest.raises(ValueError, match="arrival_seconds is required"):
        FrameTiming(capture_seconds=None, arrival_seconds=None, depth_ready_seconds=None)


@pytest.mark.parametrize("field_name", ["capture_seconds", "arrival_seconds", "depth_ready_seconds"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_frame_timing_refuses_a_non_finite_value(field_name: str, bad_value: float) -> None:
    """A non-finite stamp turns every share it touches into nan or inf, and the log refuses to write it."""
    values = {"capture_seconds": 1.0, "arrival_seconds": 2.0, "depth_ready_seconds": 3.0}
    values[field_name] = bad_value

    # The finiteness message, not the field name alone. An infinite arrival also trips the order
    # check, whose message names the same field.
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        FrameTiming(**values)


def test_a_literal_timing_block_decodes_to_its_three_values() -> None:
    """
    The key names are what frame logs already on disk were written with.

    A rename on both sides passes the round trip and decodes every old log with its capture lost.
    """
    payload = _header_with_timing({"capture_seconds": 1.0, "arrival_seconds": 2.0, "depth_ready_seconds": 3.0})

    timing = decode_frame(payload).timing

    assert timing == FrameTiming(capture_seconds=1.0, arrival_seconds=2.0, depth_ready_seconds=3.0)


def test_frame_timing_refuses_depth_ready_before_arrival() -> None:
    with pytest.raises(ValueError, match="before arrival_seconds"):
        FrameTiming(capture_seconds=None, arrival_seconds=10.0, depth_ready_seconds=9.0)


def test_a_capture_after_arrival_is_allowed_because_an_offset_can_be_slightly_off() -> None:
    timing = FrameTiming(capture_seconds=10.002, arrival_seconds=10.0, depth_ready_seconds=None)

    assert timing.capture_seconds > timing.arrival_seconds


def test_a_timing_block_without_arrival_is_refused_by_name() -> None:
    with pytest.raises(FrameDecodeError, match="arrival_seconds"):
        decode_frame(_header_with_timing({"capture_seconds": 1.0, "depth_ready_seconds": None}))


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ("soon", "timing must be a JSON object"),
        ({"arrival_seconds": "late"}, "timing.arrival_seconds must be a number"),
        ({"arrival_seconds": 2.0, "depth_ready_seconds": 1.0}, "timing is inconsistent"),
    ],
)
def test_a_malformed_timing_block_is_a_decode_error(block: object, message: str) -> None:
    # A FrameDecodeError costs one frame and the stream continues. Anything else escaping the
    # decoder would be logged as a defect in the laptop.
    with pytest.raises(FrameDecodeError, match=message):
        decode_frame(_header_with_timing(block))


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf")])
def test_a_depth_frame_refuses_a_timestamp_that_is_not_finite(timestamp: float) -> None:
    """
    Every later stage orders frames by their stamp, and the timing log cannot write a NaN.

    Refused where the frame is built, so it names the field rather than failing three stages later.
    """
    with pytest.raises(ValueError, match="timestamp_seconds must be finite"):
        dataclasses.replace(_frame_with(None), timestamp_seconds=timestamp)
