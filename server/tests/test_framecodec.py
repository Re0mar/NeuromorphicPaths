"""
Covers the wire format in both directions.

Round trips prove the encoder and decoder agree with each other. They cannot prove the decoder is
right, because both sides are ours. So every refusal is driven by a payload the test builds by
hand, the way the Android side will build one, rather than by corrupting something encode_frame
produced.
"""

# Standard library imports
import json
import socket
import struct
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.framecodec import (
    HEADER_TERMINATOR,
    LENGTH_PREFIX,
    MAXIMUM_MESSAGE_BYTES,
    WIRE_VERSION,
    DepthWireDtype,
    FrameDecodeError,
    FrameEncodeError,
    decode_frame,
    decode_path,
    encode_frame,
    encode_path,
    read_message,
    read_message_from_file,
    write_message,
)
from nav.types import DepthFrame, Plane, PlannedPath, Pose

ORIENTATION_ONLY_POSE = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False)
POSITIONED_POSE = Pose(
    orientation=np.array([0.0, 1.0, 0.0, 0.0]),
    position=np.array([1.0, 2.0, 3.0]),
    has_position=True,
)
INTRINSICS = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def _frame(**overrides) -> DepthFrame:
    fields = {
        "timestamp_seconds": 12.345,
        "depth_meters": np.arange(12, dtype=np.float32).reshape(3, 4),
        "intrinsics": INTRINSICS,
        "pose": ORIENTATION_ONLY_POSE,
        "ground_plane": None,
        "gaze_pixel": None,
    }
    fields.update(overrides)
    return DepthFrame(**fields)


def _payload_of(message: bytes) -> bytes:
    return message[LENGTH_PREFIX.size :]


def _build_payload(header_overrides: dict | None = None, depth_bytes: bytes | None = None) -> bytes:
    """A valid payload built by hand, with any header field replaced. This is the Android side."""
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    body = depth.tobytes() if depth_bytes is None else depth_bytes
    header = {
        "version": WIRE_VERSION,
        "timestamp_seconds": 1.0,
        "depth": {"dtype": "float32", "shape": [3, 4], "byte_length": len(depth.tobytes())},
        "intrinsics": INTRINSICS.tolist(),
        "pose": {"orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "position_xyz": None, "has_position": False},
        "ground_plane": None,
        "gaze_pixel": None,
    }
    if header_overrides:
        header.update(header_overrides)
    return json.dumps(header).encode("utf-8") + HEADER_TERMINATOR + body


def test_hand_built_payload_decodes() -> None:
    # If this fails, every refusal test below is refusing for the wrong reason.
    frame = decode_frame(_build_payload())

    assert frame.depth_meters.shape == (3, 4)


def test_round_trip_preserves_every_field() -> None:
    original = _frame(
        pose=POSITIONED_POSE,
        ground_plane=Plane(normal=np.array([0.0, 1.0, 0.0]), offset_meters=-1.6),
        gaze_pixel=np.array([12.0, 34.0]),
    )

    decoded = decode_frame(_payload_of(encode_frame(original)))

    assert decoded.timestamp_seconds == pytest.approx(original.timestamp_seconds)
    assert decoded.depth_meters == pytest.approx(original.depth_meters)
    assert decoded.intrinsics == pytest.approx(original.intrinsics)
    assert decoded.pose.orientation == pytest.approx(original.pose.orientation)
    assert decoded.pose.position == pytest.approx(original.pose.position)
    assert decoded.pose.has_position == original.pose.has_position
    assert decoded.ground_plane is not None
    assert decoded.ground_plane.normal == pytest.approx(original.ground_plane.normal)
    assert decoded.ground_plane.offset_meters == pytest.approx(original.ground_plane.offset_meters)
    assert decoded.gaze_pixel == pytest.approx(original.gaze_pixel)


def test_round_trip_keeps_absent_fields_absent() -> None:
    decoded = decode_frame(_payload_of(encode_frame(_frame())))

    assert decoded.ground_plane is None
    assert decoded.gaze_pixel is None
    assert decoded.pose.position is None
    assert decoded.pose.has_position is False


def test_uint16_millimeters_decode_to_float32_meters() -> None:
    # What ARCore actually sends. The conversion lives in the decoder so the Android side is a
    # straight buffer copy.
    millimeters = np.array([[1000, 2500], [0, 65535]], dtype=np.uint16)
    payload = _build_payload(
        header_overrides={"depth": {"dtype": "uint16", "shape": [2, 2], "byte_length": millimeters.nbytes}},
        depth_bytes=millimeters.tobytes(),
    )

    frame = decode_frame(payload)

    assert frame.depth_meters.dtype == np.float32
    assert frame.depth_meters == pytest.approx(np.array([[1.0, 2.5], [0.0, 65.535]]), abs=1e-3)


def test_float16_decodes_to_float32_meters() -> None:
    half = np.array([[1.5, 2.5]], dtype=np.float16)
    payload = _build_payload(
        header_overrides={"depth": {"dtype": "float16", "shape": [1, 2], "byte_length": half.nbytes}},
        depth_bytes=half.tobytes(),
    )

    assert decode_frame(payload).depth_meters == pytest.approx(np.array([[1.5, 2.5]]))


def test_every_wire_dtype_has_a_decode_test() -> None:
    # Keyed by the dtype discriminator, not by position. A fourth member added with no test fails
    # here rather than shipping untested.
    tested = {DepthWireDtype.FLOAT32_METERS, DepthWireDtype.FLOAT16_METERS, DepthWireDtype.UINT16_MILLIMETERS}
    assert tested == set(DepthWireDtype)


@pytest.mark.parametrize(
    ("overrides", "depth_bytes", "expected_in_message"),
    [
        ({"version": 2}, None, "version"),
        ({"version": "one"}, None, "version"),
        ({"depth": {"dtype": "int8", "shape": [3, 4], "byte_length": 12}}, b"\x00" * 12, "depth.dtype"),
        ({"depth": {"dtype": "float32", "shape": [3, 4], "byte_length": 999}}, None, "depth.byte_length"),
        ({"depth": {"dtype": "float32", "shape": [3, 4, 5], "byte_length": 48}}, None, "depth.shape"),
        ({"depth": {"dtype": "float32", "shape": [0, 4], "byte_length": 0}}, b"", r"depth\.shape\[0\]"),
        ({"depth": {"dtype": "float32", "shape": [3, 4], "byte_length": 48}}, b"\x00" * 20, "48"),
        ({"intrinsics": [[1.0, 0.0], [0.0, 1.0]]}, None, "intrinsics"),
        ({"intrinsics": "not a matrix"}, None, "intrinsics"),
        ({"timestamp_seconds": "soon"}, None, "timestamp_seconds"),
        ({"timestamp_seconds": True}, None, "timestamp_seconds"),
        ({"pose": {"orientation_wxyz": [1.0, 0.0, 0.0], "position_xyz": None, "has_position": False}}, None, "orientation_wxyz"),
        ({"pose": {"orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "position_xyz": None, "has_position": True}}, None, "pose"),
        ({"pose": {"orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "position_xyz": None, "has_position": "yes"}}, None, "has_position"),
        ({"ground_plane": {"normal": [0.0, 1.0, 0.0]}}, None, "offset_meters"),
        ({"ground_plane": 5}, None, "ground_plane"),
        ({"gaze_pixel": [1.0, 2.0, 3.0]}, None, "gaze_pixel"),
    ],
)
def test_a_malformed_field_is_refused_by_name(overrides: dict, depth_bytes: bytes | None, expected_in_message: str) -> None:
    with pytest.raises(FrameDecodeError, match=expected_in_message):
        decode_frame(_build_payload(header_overrides=overrides, depth_bytes=depth_bytes))


@pytest.mark.parametrize("missing", ["version", "timestamp_seconds", "depth", "intrinsics", "pose", "ground_plane", "gaze_pixel"])
def test_a_missing_key_is_refused_by_name(missing: str) -> None:
    header = json.loads(_build_payload().split(HEADER_TERMINATOR, 1)[0])
    del header[missing]
    payload = json.dumps(header).encode("utf-8") + HEADER_TERMINATOR + np.arange(12, dtype=np.float32).tobytes()

    # Nullable fields are still required keys. An absent gaze_pixel and a gaze_pixel the producer
    # forgot to send look identical otherwise.
    with pytest.raises(FrameDecodeError, match=missing):
        decode_frame(payload)


def test_a_payload_with_no_header_terminator_is_refused() -> None:
    with pytest.raises(FrameDecodeError, match="no newline"):
        decode_frame(b'{"version": 1}')


def test_a_truncated_json_header_is_refused() -> None:
    with pytest.raises(FrameDecodeError, match="not valid JSON"):
        decode_frame(b'{"version": 1, "depth"' + HEADER_TERMINATOR + b"")


def test_a_header_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(FrameDecodeError, match="must be a JSON object"):
        decode_frame(b"[1, 2, 3]" + HEADER_TERMINATOR + b"")


def test_a_header_that_is_not_utf8_is_refused() -> None:
    with pytest.raises(FrameDecodeError, match="UTF-8"):
        decode_frame(b"\xff\xfe invalid" + HEADER_TERMINATOR + b"")


def _payload_with_raw_timestamp(raw_number: str) -> bytes:
    """A valid payload whose timestamp is replaced by literal JSON text, which json.dumps cannot write."""
    header_bytes, terminator, body = _build_payload({"timestamp_seconds": "RAW"}).partition(HEADER_TERMINATOR)
    return header_bytes.replace(b'"RAW"', raw_number.encode("ascii")) + terminator + body


@pytest.mark.parametrize(
    ("payload", "expected_in_message"),
    [
        # Past a float's range, so float() overflows. numpy raised TypeError on it before.
        (_payload_with_raw_timestamp("1" + "0" * 399), "timestamp_seconds is an integer too large for a float"),
        # Past Python's 4300-digit limit, so json raises a plain ValueError, not a JSONDecodeError.
        (_payload_with_raw_timestamp("1" + "0" * 4999), "header holds a number json will not parse"),
        # Deep enough to exhaust the parser's recursion.
        (b"[" * 100_000 + b"]" * 100_000 + HEADER_TERMINATOR, "header nests too deeply"),
    ],
    ids=["400 digits", "5000 digits", "deep nesting"],
)
def test_a_header_json_cannot_hold_is_a_decode_error(payload: bytes, expected_in_message: str) -> None:
    """
    The port listens on every interface and the source drops a frame only on FrameDecodeError.

    Any other exception from one bad frame ends the run, so each of these has to come out as one.
    """
    with pytest.raises(FrameDecodeError, match=expected_in_message):
        decode_frame(payload)


def test_a_path_json_cannot_hold_is_a_decode_error() -> None:
    """The path decoder reads bytes off a socket too, and goes through the same parse."""
    with pytest.raises(FrameDecodeError, match="path nests too deeply"):
        decode_path(b"[" * 100_000 + b"]" * 100_000)


def test_encoding_a_non_finite_intrinsic_is_our_bug_not_a_decode_error() -> None:
    broken = INTRINSICS.copy()
    broken[0, 0] = np.nan

    # FrameEncodeError, not FrameDecodeError. The receiving loop drops a FrameDecodeError as
    # expected bad input. If encoding raised that too, a systematically broken frame would be
    # dropped silently every frame while the log called it expected.
    with pytest.raises(FrameEncodeError, match="non-finite"):
        encode_frame(_frame(intrinsics=broken))

    assert not isinstance(FrameEncodeError("x"), FrameDecodeError)


def test_encoding_a_non_finite_plane_offset_is_refused() -> None:
    with pytest.raises(FrameEncodeError, match="non-finite"):
        encode_frame(_frame(ground_plane=Plane(normal=np.array([0.0, 1.0, 0.0]), offset_meters=np.inf)))


def test_the_length_prefix_matches_the_payload() -> None:
    message = encode_frame(_frame())
    declared = LENGTH_PREFIX.unpack(message[: LENGTH_PREFIX.size])[0]

    assert declared == len(message) - LENGTH_PREFIX.size


def test_reading_from_a_socket_round_trips() -> None:
    sender, receiver = socket.socketpair()
    try:
        write_message(sender, encode_frame(_frame()))
        frame = decode_frame(read_message(receiver))
    finally:
        sender.close()
        receiver.close()

    assert frame.depth_meters.shape == (3, 4)


def test_a_stream_that_closes_early_raises_instead_of_hanging() -> None:
    sender, receiver = socket.socketpair()
    try:
        message = encode_frame(_frame())
        # Half a message, then the sender goes away. Without the closed-stream check this waits
        # forever, which is the failure that looks like a hung pipeline rather than a bad frame.
        sender.sendall(message[: len(message) // 2])
        sender.close()

        with pytest.raises(FrameDecodeError, match="stream closed"):
            read_message(receiver)
    finally:
        receiver.close()


def test_an_absurd_length_prefix_is_refused_rather_than_allocated() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack(">I", MAXIMUM_MESSAGE_BYTES + 1))
        # The sender closes right after the prefix. Without the limit check the reader would then
        # fail on the closed stream rather than block forever, so this test fails instead of hanging.
        sender.close()
        with pytest.raises(FrameDecodeError, match="limit"):
            read_message(receiver)
    finally:
        sender.close()
        receiver.close()


def test_encoding_a_frame_over_the_message_limit_is_refused() -> None:
    # 4096 by 4096 float32 is exactly 64 MB of depth, and the header puts it over the limit the
    # reader enforces. Refused on this side, so the two sides agree on what can be sent at all.
    huge = _frame(depth_meters=np.zeros((4096, 4096), dtype=np.float32))

    with pytest.raises(FrameEncodeError, match="limit"):
        encode_frame(huge)


def test_a_zero_length_prefix_is_refused() -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack(">I", 0))
        with pytest.raises(FrameDecodeError, match="zero"):
            read_message(receiver)
    finally:
        sender.close()
        receiver.close()


def test_a_frame_file_shorter_than_its_prefix_is_refused(tmp_path) -> None:
    truncated = tmp_path / "frame_000000.bin"
    truncated.write_bytes(b"\x00\x00")

    with pytest.raises(FrameDecodeError, match="too short"):
        read_message_from_file(truncated)


def test_a_frame_file_that_lies_about_its_length_is_refused(tmp_path) -> None:
    lying = tmp_path / "frame_000000.bin"
    lying.write_bytes(struct.pack(">I", 999) + b"short")

    with pytest.raises(FrameDecodeError, match="999"):
        read_message_from_file(lying)


def test_path_round_trip() -> None:
    original = PlannedPath(
        timestamp_seconds=1.0,
        times_seconds=np.array([0.0, 0.1, 0.2]),
        lateral_offsets_meters=np.array([0.0, 0.05, 0.1]),
        first_heading_radians=0.42,
        alarm=True,
        cumulative_cost_bits=12.5,
        scene_information_bits=0.37,
        avoidance_surprise_bits=0.51,
    )

    decoded = decode_path(encode_path(original))

    assert decoded.times_seconds == pytest.approx(original.times_seconds)
    assert decoded.lateral_offsets_meters == pytest.approx(original.lateral_offsets_meters)
    assert decoded.first_heading_radians == pytest.approx(original.first_heading_radians)
    assert decoded.alarm is True
    assert decoded.cumulative_cost_bits == pytest.approx(original.cumulative_cost_bits)
    assert decoded.scene_information_bits == pytest.approx(0.37)
    assert decoded.avoidance_surprise_bits == pytest.approx(0.51)


@pytest.mark.parametrize(
    "missing",
    [
        "timestamp_seconds",
        "times_seconds",
        "lateral_offsets_meters",
        "first_heading_radians",
        "alarm",
        "cumulative_cost_bits",
        "scene_information_bits",
        "avoidance_surprise_bits",
    ],
)
def test_a_path_message_missing_a_field_is_refused(missing: str) -> None:
    message = {
        "timestamp_seconds": 1.0,
        "times_seconds": [0.0, 0.1],
        "lateral_offsets_meters": [0.0, 0.05],
        "first_heading_radians": 0.0,
        "alarm": False,
        "cumulative_cost_bits": 0.0,
        "scene_information_bits": 0.0,
        "avoidance_surprise_bits": 0.0,
    }
    del message[missing]

    with pytest.raises(FrameDecodeError, match=missing):
        decode_path(json.dumps(message).encode("utf-8"))


def test_a_path_with_mismatched_array_lengths_is_refused() -> None:
    # PlannedPath raises a plain ValueError for this. Unwrapped it would escape the handler that
    # exists to treat a bad message as a bad message.
    message = json.dumps(
        {
            "timestamp_seconds": 1.0,
            "times_seconds": [0.0, 0.1],
            "lateral_offsets_meters": [0.0],
            "first_heading_radians": 0.0,
            "alarm": False,
            "cumulative_cost_bits": 0.0,
            "scene_information_bits": 0.0,
            "avoidance_surprise_bits": 0.0,
        }
    ).encode("utf-8")

    with pytest.raises(FrameDecodeError, match="do not make a path"):
        decode_path(message)


def test_the_path_json_matches_the_wire_docs_example() -> None:
    # A round trip passes even when a key is misspelled the same way on both sides. This pins the
    # exact key names and values against the example the document shows the phone's maintainer.
    stated = PlannedPath(
        timestamp_seconds=12.345,
        times_seconds=np.array([0.0, 0.1, 0.2]),
        lateral_offsets_meters=np.array([0.0, 0.05, 0.12]),
        first_heading_radians=0.0423,
        alarm=False,
        cumulative_cost_bits=18.4,
        scene_information_bits=0.37,
        avoidance_surprise_bits=0.51,
    )
    document = (Path(__file__).parent.parent / "docs" / "arcore_wire_format.md").read_text(encoding="utf-8")
    path_section = document.split("## What the laptop sends back", 1)[1].split("\n## ", 1)[0]
    example = json.loads(path_section.split("```json", 1)[1].split("```", 1)[0])

    assert json.loads(encode_path(stated)) == example


@pytest.mark.parametrize(
    ("field", "wrong_value", "expected"),
    [
        ("avoidance_surprise_bits", True, "avoidance_surprise_bits must be a number, got bool"),
        ("scene_information_bits", "0.37", "scene_information_bits must be a number, got str"),
    ],
)
def test_a_path_with_a_wrongly_typed_bits_field_is_refused(field: str, wrong_value: object, expected: str) -> None:
    # Each case breaks exactly one rule, and the match names the field, so the guard is what fired
    # rather than some later error that happens to share the type.
    message = {
        "timestamp_seconds": 1.0,
        "times_seconds": [0.0],
        "lateral_offsets_meters": [0.0],
        "first_heading_radians": 0.0,
        "alarm": False,
        "cumulative_cost_bits": 0.0,
        "scene_information_bits": 0.0,
        "avoidance_surprise_bits": 0.0,
    }
    message[field] = wrong_value

    with pytest.raises(FrameDecodeError, match=expected):
        decode_path(json.dumps(message).encode("utf-8"))


def test_the_wire_is_little_endian_regardless_of_the_host() -> None:
    # The format document states little-endian. tobytes() would have used native order, which
    # happens to agree on every device either side runs on and is still not what the contract says.
    # numpy reports a pinned little-endian dtype as "=" on a little-endian host, so the byte order
    # character proves nothing here. The bytes do.
    for member in DepthWireDtype:
        assert member.numpy_dtype.str.startswith(("<", "|")), member

    frame = _frame(depth_meters=np.array([[1.5]], dtype=np.float32))
    body = _payload_of(encode_frame(frame)).split(HEADER_TERMINATOR, 1)[1]

    assert body == bytes.fromhex("0000c03f")


def test_the_worked_example_in_the_format_document_still_decodes() -> None:
    # The document pastes real encoder output. If the format drifts, that example becomes a lie
    # the Android side implements against, and nothing else would catch it.
    header = (
        b'{"version": 1, "timestamp_seconds": 12.345, '
        b'"depth": {"dtype": "float32", "shape": [2, 2], "byte_length": 16}, '
        b'"intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], '
        b'"pose": {"orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "position_xyz": [0.0, 0.0, 0.0], "has_position": true}, '
        b'"ground_plane": {"normal": [0.0, 1.0, 0.0], "offset_meters": -1.6}, '
        b'"gaze_pixel": null}'
    )
    body = bytes.fromhex("0000c03f000000400000204000004040")

    frame = decode_frame(header + HEADER_TERMINATOR + body)

    assert frame.depth_meters == pytest.approx(np.array([[1.5, 2.0], [2.5, 3.0]]))
    assert frame.pose.has_position is True
    assert frame.ground_plane is not None
    assert frame.ground_plane.offset_meters == pytest.approx(-1.6)


def test_the_format_document_contains_the_generated_example() -> None:
    # The document's example is encoder output, pasted. Generate it again here and insist the
    # document still carries both halves, so the format cannot drift away from what the Android
    # side was told.
    frame = DepthFrame(
        timestamp_seconds=12.345,
        depth_meters=np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float32),
        intrinsics=INTRINSICS,
        # The document's example is a phone's frame, so its orientation is gravity-aligned.
        pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), True, orientation_is_gravity_aligned=True),
        ground_plane=Plane(normal=np.array([0.0, 1.0, 0.0]), offset_meters=-1.6),
        gaze_pixel=None,
    )
    message = encode_frame(frame)
    header, _, body = _payload_of(message).partition(HEADER_TERMINATOR)

    document = (Path(__file__).parent.parent / "docs" / "arcore_wire_format.md").read_text(encoding="utf-8")

    assert message[: LENGTH_PREFIX.size].hex(" ") in document
    assert body.hex(" ") in document
    # The header is shown with indentation it does not have on the wire, so compare its values
    # rather than its bytes.
    for fragment in ('"version": 1', '"byte_length": 16', '"offset_meters": -1.6', '"has_position": true'):
        assert fragment in document, fragment
    assert json.loads(header)["depth"]["byte_length"] == 16


def test_the_format_document_carries_the_path_field_table() -> None:
    # The Kotlin decoder is written from the document's per-field table for the path, the way the
    # frame decoder was written from the frame's. A key the encoder writes and the table does not
    # name is a key the app will not read.
    document = (Path(__file__).parent.parent / "docs" / "arcore_wire_format.md").read_text(encoding="utf-8")
    path_section = document.split("## What the laptop sends back", 1)[1].split("\n## ", 1)[0]
    assert "| Field | Produced by | On the wire | Read by | Value domain | Who enforces it |" in path_section

    written = json.loads(encode_path(PlannedPath(1.0, np.array([0.0]), np.array([0.0]), 0.0, False, 0.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)))
    for key in written:
        assert f"| `{key}` |" in path_section, key


@pytest.mark.parametrize(
    "field",
    ["first_heading_radians", "cumulative_cost_bits", "scene_information_bits", "avoidance_surprise_bits"],
)
def test_a_path_with_a_non_finite_scalar_never_reaches_the_encoder(field: str) -> None:
    # PlannedPath refuses this itself, so encode_path's allow_nan=False is defense in depth that no
    # real PlannedPath can reach. This is the test that proves the first line of defense holds.
    fields = {
        "timestamp_seconds": 1.0,
        "times_seconds": np.array([0.0, 0.1]),
        "lateral_offsets_meters": np.array([0.0, 0.1]),
        "first_heading_radians": 0.0,
        "alarm": False,
        "cumulative_cost_bits": 0.0,
        "scene_information_bits": 0.0,
        "avoidance_surprise_bits": 0.0,
    }
    fields[field] = np.nan

    with pytest.raises(ValueError, match=field):
        PlannedPath(**fields)


@pytest.mark.parametrize("field", ["scene_information_bits", "avoidance_surprise_bits"])
def test_planned_path_refuses_negative_bits(field: str) -> None:
    # Both are divergences or squared ratios, never below zero. A negative one is a bug upstream.
    fields = {
        "timestamp_seconds": 1.0,
        "times_seconds": np.array([0.0]),
        "lateral_offsets_meters": np.array([0.0]),
        "first_heading_radians": 0.0,
        "alarm": False,
        "cumulative_cost_bits": 0.0,
        "scene_information_bits": 0.0,
        "avoidance_surprise_bits": 0.0,
    }
    fields[field] = -0.1

    with pytest.raises(ValueError, match=f"{field} must be finite and zero or more, got -0.1"):
        PlannedPath(**fields)


def _header_and_body(frame: DepthFrame) -> tuple[dict, bytes]:
    """The encoder's header as a dict and the bytes after it, for a test that edits one key."""
    header, _, body = _payload_of(encode_frame(frame)).partition(HEADER_TERMINATOR)
    return json.loads(header), body


def _rebuild(header: dict, body: bytes) -> bytes:
    return json.dumps(header).encode("utf-8") + HEADER_TERMINATOR + body


def test_whether_the_orientation_is_gravity_aligned_survives_a_round_trip() -> None:
    # The scene reads the floor's up from this, so a recording that loses it replays with the
    # defect the flag exists to prevent.
    aligned = _frame(pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), True, orientation_is_gravity_aligned=True))
    unaligned = _frame(pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False, orientation_is_gravity_aligned=False))

    assert decode_frame(_payload_of(encode_frame(aligned))).pose.orientation_is_gravity_aligned is True
    assert decode_frame(_payload_of(encode_frame(unaligned))).pose.orientation_is_gravity_aligned is False


def test_a_header_without_the_gravity_key_is_read_as_the_phones() -> None:
    # Every frame log that existed when the key was added came from the phone, and those are what
    # the planner is tuned against. Reading them as un-aligned would reintroduce the defect.
    header, body = _header_and_body(_frame())
    del header["pose"]["orientation_is_gravity_aligned"]

    assert decode_frame(_rebuild(header, body)).pose.orientation_is_gravity_aligned is True


def test_a_gravity_flag_that_is_not_a_boolean_is_refused_by_name() -> None:
    header, body = _header_and_body(_frame())
    header["pose"]["orientation_is_gravity_aligned"] = 1

    with pytest.raises(FrameDecodeError, match="pose.orientation_is_gravity_aligned must be true or false"):
        decode_frame(_rebuild(header, body))


def test_planned_path_has_no_default_for_either_new_number() -> None:
    # A default would let a construction site that forgot them send a plausible zero on both wires.
    with pytest.raises(TypeError, match="scene_information_bits.*avoidance_surprise_bits"):
        PlannedPath(1.0, np.array([0.0]), np.array([0.0]), 0.0, False, 0.0)
