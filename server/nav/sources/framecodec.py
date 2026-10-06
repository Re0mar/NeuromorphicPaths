"""
The one encoder and decoder for a DepthFrame on the wire or on disk.

The TCP source, the fake sender, the recording tap and the replay source all go through here, so a
recording replays through exactly the code the live Pixel feeds. A second encoder written anywhere
else is a second contract.

Every message is a 4-byte big-endian length, then that many bytes: a UTF-8 JSON header, a newline,
then the raw depth bytes.

Nothing in here casts a field out of the parsed header. Every value is checked and then used,
because a cast raises TypeError, the receiving loop catches FrameDecodeError, and a malformed
frame would otherwise be reported as a defect in us rather than a defect in the input.
"""

# Standard library imports
import json
import socket
import struct
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.types import DepthFrame, FrameTiming, Plane, PlannedPath, Pose

WIRE_VERSION = 1

# Big-endian unsigned 4-byte length prefix, so the Android side can write it with DataOutputStream
# without thinking about byte order.
LENGTH_PREFIX = struct.Struct(">I")
HEADER_TERMINATOR = b"\n"

# A message larger than this is treated as a desynchronised stream rather than a real frame. A
# 4-byte length read at the wrong offset otherwise asks us to allocate gigabytes.
MAXIMUM_MESSAGE_BYTES = 64 * 1024 * 1024

MILLIMETERS_PER_METER = 1000.0

# The frame log's layout. One framed message per file, plus a line per frame in the index.
INDEX_FILENAME = "index.jsonl"
FRAME_FILENAME_TEMPLATE = "frame_{sequence:06d}.bin"
# The same names, for asking whether a directory already holds frames. Beside the template so the
# layout has one home and the two cannot drift.
FRAME_FILENAME_GLOB = "frame_*.bin"


class FrameCodecError(ValueError):
    """Anything this module refuses to encode or decode."""


class FrameEncodeError(FrameCodecError):
    """We tried to send something that is not a valid frame. Our bug, not the sender's."""


class FrameDecodeError(FrameCodecError):
    """What arrived is not a valid frame. The receiving loop drops the frame and carries on."""


class StreamClosedError(FrameDecodeError):
    """The other side went away. Carries how much of the current read was still owed.

    A close with every byte of a message still owed is a clean disconnect between frames. A close
    part way through is a truncated frame. Both end the stream, and only one is worth a warning.
    """

    def __init__(self, bytes_unread: int, bytes_wanted: int) -> None:
        super().__init__(f"stream closed with {bytes_unread} of {bytes_wanted} bytes unread")
        self.bytes_unread = bytes_unread
        self.bytes_wanted = bytes_wanted

    @property
    def at_message_boundary(self) -> bool:
        return self.bytes_unread == self.bytes_wanted


class DepthWireDtype(Enum):
    """How depth is laid out on the wire.

    A string on the wire because that is a serialization boundary, and this enum everywhere after,
    so a typo is a decode error at one place rather than a comparison that silently never matches.
    """

    FLOAT32_METERS = "float32"
    FLOAT16_METERS = "float16"
    UINT16_MILLIMETERS = "uint16"

    @property
    def numpy_dtype(self) -> np.dtype:
        # Explicitly little-endian, not native. The wire format is a contract, and every device on
        # both sides being little-endian today is not the same as the format saying so.
        return np.dtype("<" + np.dtype(self.value).str[1:])

    @property
    def item_bytes(self) -> int:
        return self.numpy_dtype.itemsize

    def to_meters(self, raw: np.ndarray) -> np.ndarray:
        """Convert a decoded array into float32 meters, which is what DepthFrame carries."""
        if self is DepthWireDtype.UINT16_MILLIMETERS:
            # ARCore's depth image is 16-bit millimeters. Sending it as-is halves the bytes and
            # keeps the Android side a straight buffer copy, so the conversion lives here.
            return raw.astype(np.float32) / MILLIMETERS_PER_METER
        return raw.astype(np.float32)


def encode_frame(frame: DepthFrame) -> bytes:
    """
    Serialize a DepthFrame into one length-prefixed message.

    Always writes float32 meters. The other dtypes exist for what the Android side sends us.

    :param frame: The frame to encode.
    :return: The complete message, length prefix included.
    :rtype: bytes
    """
    depth = np.ascontiguousarray(frame.depth_meters, dtype=DepthWireDtype.FLOAT32_METERS.numpy_dtype)
    depth_bytes = depth.tobytes()

    header = {
        "version": WIRE_VERSION,
        "timestamp_seconds": float(frame.timestamp_seconds),
        "depth": {
            "dtype": DepthWireDtype.FLOAT32_METERS.value,
            "shape": list(depth.shape),
            "byte_length": len(depth_bytes),
        },
        "intrinsics": frame.intrinsics.tolist(),
        "pose": {
            "orientation_wxyz": frame.pose.orientation.tolist(),
            "position_xyz": None if frame.pose.position is None else frame.pose.position.tolist(),
            "has_position": frame.pose.has_position,
            "orientation_is_gravity_aligned": frame.pose.orientation_is_gravity_aligned,
        },
        "ground_plane": (
            None
            if frame.ground_plane is None
            else {
                "normal": frame.ground_plane.normal.tolist(),
                "offset_meters": float(frame.ground_plane.offset_meters),
            }
        ),
        "gaze_pixel": None if frame.gaze_pixel is None else frame.gaze_pixel.tolist(),
    }
    # Written only when there is something to write, so a frame from a source that keeps no timing
    # encodes exactly as it did before the key existed.
    if frame.timing is not None:
        header["timing"] = {
            "capture_seconds": frame.timing.capture_seconds,
            "arrival_seconds": frame.timing.arrival_seconds,
            "depth_ready_seconds": frame.timing.depth_ready_seconds,
        }

    try:
        # allow_nan=False turns a non-finite intrinsic or pose into an error here, rather than into
        # a literal NaN in the JSON that no strict parser on the Android side will read back.
        header_bytes = json.dumps(header, allow_nan=False).encode("utf-8")
    except ValueError as non_finite_error:
        raise FrameEncodeError(f"frame has a non-finite value in its header: {non_finite_error}") from non_finite_error

    payload = header_bytes + HEADER_TERMINATOR + depth_bytes
    if len(payload) > MAXIMUM_MESSAGE_BYTES:
        raise FrameEncodeError(f"message is {len(payload)} bytes, over the {MAXIMUM_MESSAGE_BYTES} limit")

    return LENGTH_PREFIX.pack(len(payload)) + payload


def decode_frame(payload: bytes) -> DepthFrame:
    """
    Parse one message payload, without its length prefix, into a DepthFrame.

    :param payload: The bytes between one length prefix and the next.
    :return: The decoded frame.
    :rtype: DepthFrame
    :raises FrameDecodeError: On any malformed field, naming the field.
    """
    header_bytes, terminator, depth_bytes = payload.partition(HEADER_TERMINATOR)
    if not terminator:
        raise FrameDecodeError("payload has no newline after the header, so the header is truncated")

    header = _parse_json(header_bytes, "header")
    if not isinstance(header, dict):
        raise FrameDecodeError(f"header must be a JSON object, got {type(header).__name__}")

    version = _required(header, "version")
    if version != WIRE_VERSION:
        raise FrameDecodeError(f"version is {version!r}, this decoder speaks version {WIRE_VERSION}")

    depth_meters = _decode_depth(_required(header, "depth"), depth_bytes)

    frame_fields = {
        "timestamp_seconds": _number(_required(header, "timestamp_seconds"), "timestamp_seconds"),
        "depth_meters": depth_meters,
        "intrinsics": _matrix(_required(header, "intrinsics"), "intrinsics", rows=3, columns=3),
        "pose": _decode_pose(_required(header, "pose")),
        "ground_plane": _decode_ground_plane(_required(header, "ground_plane")),
        "gaze_pixel": _optional_vector(_required(header, "gaze_pixel"), "gaze_pixel", length=2),
        # Optional. The Pixel app never sends it, and frame logs written before it existed lack it.
        "timing": _decode_timing(header["timing"]) if "timing" in header else None,
    }

    try:
        return DepthFrame(**frame_fields)
    except ValueError as inconsistent_error:
        # DepthFrame's own validation raises plain ValueError. Left alone it would escape the
        # loop's FrameDecodeError handler and be logged as an unexpected defect in us.
        raise FrameDecodeError(f"fields decoded but do not make a frame: {inconsistent_error}") from inconsistent_error


def path_message(path: PlannedPath) -> dict:
    """
    A PlannedPath as the JSON object both wires carry, before serializing.

    The one place the path's keys are spelled. The phone's encoding and the web's envelope are both
    built from it, so neither can drop a key the other sends.

    :param path: The path to describe.
    :return: Plain Python numbers, lists and a bool, ready for json.dumps.
    :rtype: dict
    """
    return {
        "timestamp_seconds": float(path.timestamp_seconds),
        "times_seconds": path.times_seconds.tolist(),
        "lateral_offsets_meters": path.lateral_offsets_meters.tolist(),
        "first_heading_radians": float(path.first_heading_radians),
        "alarm": bool(path.alarm),
        "cumulative_cost_bits": float(path.cumulative_cost_bits),
        "scene_information_bits": float(path.scene_information_bits),
        "avoidance_surprise_bits": float(path.avoidance_surprise_bits),
    }


def encode_path(path: PlannedPath) -> bytes:
    """Serialize a PlannedPath to JSON bytes for the phone and web sinks."""
    message = path_message(path)
    try:
        return json.dumps(message, allow_nan=False).encode("utf-8")
    except ValueError as non_finite_error:
        raise FrameEncodeError(f"path has a non-finite value: {non_finite_error}") from non_finite_error


def decode_path(payload: bytes) -> PlannedPath:
    """Parse a path message. Used by the tests and by anything reading back what a sink sent."""
    message = _parse_json(payload, "path")
    if not isinstance(message, dict):
        raise FrameDecodeError(f"path must be a JSON object, got {type(message).__name__}")

    try:
        return PlannedPath(
            timestamp_seconds=_number(_required(message, "timestamp_seconds"), "timestamp_seconds"),
            times_seconds=_vector(_required(message, "times_seconds"), "times_seconds"),
            lateral_offsets_meters=_vector(_required(message, "lateral_offsets_meters"), "lateral_offsets_meters"),
            first_heading_radians=_number(_required(message, "first_heading_radians"), "first_heading_radians"),
            alarm=_boolean(_required(message, "alarm"), "alarm"),
            cumulative_cost_bits=_number(_required(message, "cumulative_cost_bits"), "cumulative_cost_bits"),
            scene_information_bits=_number(_required(message, "scene_information_bits"), "scene_information_bits"),
            avoidance_surprise_bits=_number(_required(message, "avoidance_surprise_bits"), "avoidance_surprise_bits"),
        )
    except ValueError as inconsistent_error:
        if isinstance(inconsistent_error, FrameDecodeError):
            raise
        raise FrameDecodeError(f"fields decoded but do not make a path: {inconsistent_error}") from inconsistent_error


def read_message(stream: socket.socket) -> bytes:
    """Read one length-prefixed message from a socket, failing rather than hanging on a short read."""
    length = LENGTH_PREFIX.unpack(read_exactly(stream, LENGTH_PREFIX.size))[0]
    if length == 0:
        raise FrameDecodeError("message length prefix is zero")
    if length > MAXIMUM_MESSAGE_BYTES:
        # Almost certainly a length read at the wrong offset. Refusing beats allocating it.
        raise FrameDecodeError(f"message claims {length} bytes, over the {MAXIMUM_MESSAGE_BYTES} limit")
    return read_exactly(stream, length)


def write_message(stream: socket.socket, message: bytes) -> None:
    """Write one already-framed message to a socket."""
    stream.sendall(message)


def read_exactly(stream: socket.socket, byte_count: int) -> bytes:
    """
    Read exactly byte_count bytes, or raise.

    A socket read returns what has arrived, not what was asked for. Looping without the
    closed-stream check is how a receiver waits forever on a sender that went away.
    """
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining > 0:
        chunk = stream.recv(remaining)
        if not chunk:
            raise StreamClosedError(bytes_unread=remaining, bytes_wanted=byte_count)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_message_to_file(path: Path, message: bytes) -> None:
    """Write one framed message as its own file in the frame log. Binary, so no newline translation."""
    path.write_bytes(message)


def read_message_from_file(path: Path) -> bytes:
    """Read one framed message file and return its payload, without the length prefix."""
    raw = path.read_bytes()
    if len(raw) < LENGTH_PREFIX.size:
        raise FrameDecodeError(f"{path.name} is {len(raw)} bytes, too short to hold a length prefix")

    length = LENGTH_PREFIX.unpack(raw[: LENGTH_PREFIX.size])[0]
    payload = raw[LENGTH_PREFIX.size :]
    if len(payload) != length:
        raise FrameDecodeError(f"{path.name} claims {length} bytes and carries {len(payload)}")
    return payload


def _parse_json(raw: bytes, what: str) -> object:
    # Bytes from any host on the network land here, so every way json can refuse them has to come
    # out as FrameDecodeError. That drops one frame, and anything else ends the run.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as encoding_error:
        raise FrameDecodeError(f"{what} is not valid UTF-8: {encoding_error}") from encoding_error
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_error:
        raise FrameDecodeError(f"{what} is not valid JSON: {json_error}") from json_error
    except ValueError as digit_limit_error:
        # Python refuses to parse an integer longer than 4300 digits, with a plain ValueError.
        raise FrameDecodeError(f"{what} holds a number json will not parse: {digit_limit_error}") from digit_limit_error
    except RecursionError as nesting_error:
        raise FrameDecodeError(f"{what} nests too deeply to parse") from nesting_error


def _required(header: dict, key: str) -> object:
    if key not in header:
        raise FrameDecodeError(f"header is missing the key {key!r}")
    return header[key]


def _number(value: object, field: str) -> float:
    # bool is a subclass of int, and a JSON true in a numeric field is a contract error worth
    # reporting rather than silently reading as 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrameDecodeError(f"{field} must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError as overflow_error:
        # A JSON integer has no size limit, and one past a float's range is not a measurement.
        raise FrameDecodeError(f"{field} is an integer too large for a float") from overflow_error
    if not np.isfinite(number):
        raise FrameDecodeError(f"{field} must be finite, got {value}")
    return number


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise FrameDecodeError(f"{field} must be true or false, got {type(value).__name__}")
    return value


def _vector(value: object, field: str, length: int | None = None) -> np.ndarray:
    if not isinstance(value, list):
        raise FrameDecodeError(f"{field} must be a list, got {type(value).__name__}")
    if length is not None and len(value) != length:
        raise FrameDecodeError(f"{field} must have {length} entries, got {len(value)}")
    return np.array([_number(entry, f"{field}[{index}]") for index, entry in enumerate(value)], dtype=np.float64)


def _optional_vector(value: object, field: str, length: int) -> np.ndarray | None:
    if value is None:
        return None
    return _vector(value, field, length=length)


def _matrix(value: object, field: str, rows: int, columns: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != rows:
        raise FrameDecodeError(f"{field} must be a list of {rows} rows, got {_describe(value)}")
    return np.array(
        [_vector(row, f"{field}[{index}]", length=columns) for index, row in enumerate(value)],
        dtype=np.float64,
    )


def _describe(value: object) -> str:
    if isinstance(value, list):
        return f"a list of {len(value)}"
    return type(value).__name__


def _decode_depth(block: object, depth_bytes: bytes) -> np.ndarray:
    if not isinstance(block, dict):
        raise FrameDecodeError(f"depth must be a JSON object, got {type(block).__name__}")

    raw_dtype = _required(block, "dtype")
    try:
        wire_dtype = DepthWireDtype(raw_dtype)
    except ValueError as unknown_dtype_error:
        allowed = ", ".join(member.value for member in DepthWireDtype)
        raise FrameDecodeError(f"depth.dtype is {raw_dtype!r}, allowed values are {allowed}") from unknown_dtype_error

    shape = _required(block, "shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise FrameDecodeError(f"depth.shape must be a list of 2, got {_describe(shape)}")
    for index, extent in enumerate(shape):
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise FrameDecodeError(f"depth.shape[{index}] must be a positive integer, got {extent!r}")

    declared_length = _required(block, "byte_length")
    if isinstance(declared_length, bool) or not isinstance(declared_length, int):
        raise FrameDecodeError(f"depth.byte_length must be an integer, got {type(declared_length).__name__}")

    expected_length = shape[0] * shape[1] * wire_dtype.item_bytes
    if declared_length != expected_length:
        raise FrameDecodeError(
            f"depth.byte_length is {declared_length}, but {shape} of {wire_dtype.value} needs {expected_length}"
        )
    if len(depth_bytes) != declared_length:
        raise FrameDecodeError(f"depth.byte_length is {declared_length}, but {len(depth_bytes)} bytes followed")

    raw = np.frombuffer(depth_bytes, dtype=wire_dtype.numpy_dtype).reshape(shape)
    return wire_dtype.to_meters(raw)


def _decode_pose(block: object) -> Pose:
    if not isinstance(block, dict):
        raise FrameDecodeError(f"pose must be a JSON object, got {type(block).__name__}")

    has_position = _boolean(_required(block, "has_position"), "pose.has_position")
    raw_position = _required(block, "position_xyz")
    position = None if raw_position is None else _vector(raw_position, "pose.position_xyz", length=3)

    # An optional key, and the only one with a default that is not simply the safe answer. Every
    # frame log that existed when this key was added was recorded from the phone, whose world is
    # gravity-aligned, and those recordings are what the planner is tuned against. Reading them as un-aligned would reintroduce the defect the key exists to fix, so
    # an absent key means the sender predates it and is assumed to be the phone. Everything
    # written from now on says so explicitly, including a video file's un-aligned identity pose.
    gravity_aligned = True
    if "orientation_is_gravity_aligned" in block:
        gravity_aligned = _boolean(block["orientation_is_gravity_aligned"], "pose.orientation_is_gravity_aligned")

    try:
        return Pose(
            orientation=_vector(_required(block, "orientation_wxyz"), "pose.orientation_wxyz", length=4),
            position=position,
            has_position=has_position,
            orientation_is_gravity_aligned=gravity_aligned,
        )
    except ValueError as inconsistent_error:
        if isinstance(inconsistent_error, FrameDecodeError):
            raise
        raise FrameDecodeError(f"pose is inconsistent: {inconsistent_error}") from inconsistent_error


def _decode_timing(block: object) -> FrameTiming:
    if not isinstance(block, dict):
        raise FrameDecodeError(f"timing must be a JSON object, got {type(block).__name__}")

    def optional_number(key: str) -> float | None:
        value = block.get(key)
        return None if value is None else _number(value, f"timing.{key}")

    try:
        return FrameTiming(
            capture_seconds=optional_number("capture_seconds"),
            arrival_seconds=_number(_required(block, "arrival_seconds"), "timing.arrival_seconds"),
            depth_ready_seconds=optional_number("depth_ready_seconds"),
        )
    except ValueError as inconsistent_error:
        if isinstance(inconsistent_error, FrameDecodeError):
            raise
        raise FrameDecodeError(f"timing is inconsistent: {inconsistent_error}") from inconsistent_error


def _decode_ground_plane(block: object) -> Plane | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        raise FrameDecodeError(f"ground_plane must be a JSON object or null, got {type(block).__name__}")

    return Plane(
        normal=_vector(_required(block, "normal"), "ground_plane.normal", length=3),
        offset_meters=_number(_required(block, "offset_meters"), "ground_plane.offset_meters"),
    )
