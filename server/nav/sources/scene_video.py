"""
The glasses' compressed video on its way to a display, untouched: units, a feed, and the shapes between.

The device process receives the Neon's H.264 as RTP packets, about twenty to a frame. H.264 calls
one frame's worth an access unit, and the web page's browser can decode one of those on its own.
So the laptop never decodes the video for the page. The assembler here gathers each frame's packets
into one unit on the device process's decode thread, the unit crosses to the pipeline process on a
pipe, and the feed hands it to whichever sink subscribed. Nothing in this module imports the
Pupil Labs client, aiohttp or PyAV, so it imports in the shared layers and in a test environment
with neither extra.

The one piece of the client restated here is its RTP payload extraction, `annex_b_chunk`, because
the assembler needs the same bytes the decoder is fed and may not import the client to get them. A
test pins it to the client's own function.
"""

# Standard library imports
import logging
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Annex B puts this before every NAL unit. The client prepends the four-byte form.
START_CODE = b"\x00\x00\x00\x01"
# The three-byte core of the start code, which the four-byte form also contains. Emulation
# prevention bytes keep it out of every payload, so scanning for it finds unit boundaries only.
START_CODE_CORE = b"\x00\x00\x01"
# One unit on the pipe and on the websocket: flags, then the capture stamp in seconds. Bit 0 of the
# flags marks a keyframe.
UNIT_HEADER = struct.Struct("<Bd")
KEYFRAME_FLAG = 0x01
FORBIDDEN_BIT = 0x80
NAL_TYPE_MASK = 0x1F
NAL_REFERENCE_BITS = 0xE0
FRAGMENT_START_BIT = 0x80


class NalUnitType(IntEnum):
    """The H.264 unit types this module reads. Everything else passes through unread."""

    SLICE = 1
    IDR_SLICE = 5
    SEQUENCE_PARAMETER_SET = 7
    PICTURE_PARAMETER_SET = 8
    # RFC 3984's fragmentation unit A: one large NAL unit split across packets.
    FRAGMENT_A = 28


@dataclass(frozen=True)
class AccessUnit:
    """One frame's worth of H.264 in Annex B form, with the parameter sets prepended on a keyframe."""

    timestamp_seconds: float
    data: bytes
    keyframe: bool


@dataclass(frozen=True)
class VideoDescription:
    """What a decoder needs before the first unit: the codec string and the stream's parameter sets.

    Plain fields only, so it pickles across the device process's pipe.
    """

    codec: str
    parameter_sets: tuple[bytes, ...]


def annex_b_chunk(rtp_payload: bytes) -> bytes:
    """
    An RTP payload as the Annex B bytes a decoder is fed. Restates the client's extraction.

    A whole NAL unit gets a start code. The first fragment of a split unit gets a start code and
    the unit's header rebuilt from the two fragment bytes. A later fragment gets nothing prepended,
    since it continues the unit the first fragment began.

    :raises ValueError: For an empty payload, a set forbidden bit, or a fragment too short to carry
        its two header bytes. The decode thread treats each as one damaged packet.
    """
    if not rtp_payload:
        raise ValueError("empty packet")
    first = rtp_payload[0]
    if first & FORBIDDEN_BIT:
        raise ValueError("First bit must be zero (forbidden_zero_bit)")
    if (first & NAL_TYPE_MASK) != NalUnitType.FRAGMENT_A:
        return START_CODE + rtp_payload
    if len(rtp_payload) < 2:
        raise ValueError("a fragment needs two header bytes")
    fragment_header = rtp_payload[1]
    if fragment_header & FRAGMENT_START_BIT:
        rebuilt_header = (first & NAL_REFERENCE_BITS) | (fragment_header & NAL_TYPE_MASK)
        return START_CODE + bytes((rebuilt_header,)) + rtp_payload[2:]
    return rtp_payload[2:]


def nal_unit_types(chunk: bytes) -> list[int]:
    """The type of every NAL unit that starts in the chunk, found by its start code. A middle fragment has none."""
    types: list[int] = []
    position = chunk.find(START_CODE_CORE)
    while position != -1:
        header_index = position + len(START_CODE_CORE)
        if header_index < len(chunk):
            types.append(chunk[header_index] & NAL_TYPE_MASK)
        position = chunk.find(START_CODE_CORE, header_index)
    return types


def codec_string(sequence_parameter_set: bytes) -> str:
    """
    The `avc1.PPCCLL` string WebCodecs takes, from the SPS: profile, constraint flags and level.

    :raises ValueError: When the bytes are not a sequence parameter set, or too short to carry the three.
    """
    body = sequence_parameter_set.lstrip(b"\x00")
    # The start code's final byte is the one non-zero byte before the NAL header.
    if body[:1] == b"\x01":
        body = body[1:]
    if len(body) < 4 or (body[0] & NAL_TYPE_MASK) != NalUnitType.SEQUENCE_PARAMETER_SET:
        raise ValueError("not a sequence parameter set, so there is no profile and level to read")
    return f"avc1.{body[1]:02x}{body[2]:02x}{body[3]:02x}"


def pack_unit(unit: AccessUnit) -> bytes:
    """The unit as it crosses the pipe and the websocket: the header, then the Annex B bytes."""
    flags = KEYFRAME_FLAG if unit.keyframe else 0
    return UNIT_HEADER.pack(flags, unit.timestamp_seconds) + unit.data


def unpack_unit(payload: bytes) -> AccessUnit:
    """
    The inverse of pack_unit.

    :raises ValueError: For a payload shorter than the header, or flags outside the keyframe bit.
    """
    if len(payload) < UNIT_HEADER.size:
        raise ValueError(f"a unit is at least {UNIT_HEADER.size} bytes, got {len(payload)}")
    flags, timestamp_seconds = UNIT_HEADER.unpack_from(payload)
    if flags & ~KEYFRAME_FLAG:
        raise ValueError(f"unit flags {flags:#04x} set a bit this reader does not know")
    return AccessUnit(timestamp_seconds=timestamp_seconds, data=payload[UNIT_HEADER.size :], keyframe=bool(flags & KEYFRAME_FLAG))


class AccessUnitAssembler:
    """Packets in, one unit out per frame. A new stamp finishes the unit the previous stamp was building."""

    def __init__(self, parameter_sets: list[bytes] | tuple[bytes, ...]) -> None:
        """
        :param parameter_sets: The SPS and PPS as the client's `sprop_parameter_set_payloads` gives
            them, start code included. Prepended to every keyframe, so a decoder can begin on any.
        :raises ValueError: When no sequence parameter set is among them.
        """
        self._parameter_sets = tuple(bytes(parameter_set) for parameter_set in parameter_sets)
        sequence_parameter_sets = [
            parameter_set for parameter_set in self._parameter_sets if NalUnitType.SEQUENCE_PARAMETER_SET in nal_unit_types(parameter_set)
        ]
        if not sequence_parameter_sets:
            raise ValueError("the parameter sets hold no sequence parameter set")
        self._description = VideoDescription(codec=codec_string(sequence_parameter_sets[0]), parameter_sets=self._parameter_sets)
        self._stamp: float | None = None
        self._chunks = bytearray()
        self._keyframe = False

    @property
    def description(self) -> VideoDescription:
        return self._description

    def feed(self, rtp_payload: bytes, timestamp_seconds: float) -> AccessUnit | None:
        """
        One packet in. Returns the previous frame's unit when this packet starts a new frame.

        :raises ValueError: For a packet `annex_b_chunk` refuses. The unit being built is untouched.
        """
        chunk = annex_b_chunk(rtp_payload)
        finished = None
        if self._stamp is not None and timestamp_seconds != self._stamp:
            finished = self._finish()
        self._stamp = timestamp_seconds
        if NalUnitType.IDR_SLICE in nal_unit_types(chunk):
            self._keyframe = True
        self._chunks += chunk
        return finished

    def flush(self) -> AccessUnit | None:
        """The unit still being built, at the end of a stream. None when nothing was fed since the last."""
        if self._stamp is None:
            return None
        return self._finish()

    def _finish(self) -> AccessUnit:
        data = bytes(self._chunks)
        if self._keyframe:
            data = b"".join(self._parameter_sets) + data
        unit = AccessUnit(timestamp_seconds=self._stamp, data=data, keyframe=self._keyframe)
        self._stamp = None
        self._chunks = bytearray()
        self._keyframe = False
        return unit


class SceneVideoListener(Protocol):
    """Anything that receives the video: the feed, a sink's listener, the device process's pipe end."""

    def describe(self, description: VideoDescription) -> None:
        """Called once before any unit, and again if the stream is described again."""

    def offer(self, unit: AccessUnit) -> None:
        """One unit, in stream order, on whatever thread received it."""


@runtime_checkable
class SceneVideoProvider(Protocol):
    """A device that can hand its video to a listener. The device process checks its device for this."""

    def subscribe_video(self, listener: SceneVideoListener) -> None: ...


class SceneVideoFeed:
    """
    The seam between the source that receives video and the sinks that want it.

    Built once by the composition root and given to both ends. A listener subscribed after the
    description is known gets it at once. Listeners are called on the offering thread, in
    subscription order. One that raises is dropped and logged, so one bad sink cannot stop the
    others, and the offering thread, which is the device's pipe reader, never stops on it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: list[SceneVideoListener] = []
        self._description: VideoDescription | None = None

    @property
    def description(self) -> VideoDescription | None:
        with self._lock:
            return self._description

    def describe(self, description: VideoDescription) -> None:
        with self._lock:
            self._description = description
            listeners = list(self._listeners)
        for listener in listeners:
            self._deliver(listener, lambda target: target.describe(description))

    def offer(self, unit: AccessUnit) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            self._deliver(listener, lambda target: target.offer(unit))

    def subscribe(self, listener: SceneVideoListener) -> Callable[[], None]:
        """
        Add a listener and return the call that removes it.

        :raises ValueError: When the same listener is already subscribed, which would feed a sink twice.
        """
        with self._lock:
            if any(existing is listener for existing in self._listeners):
                raise ValueError("this listener is already subscribed to the scene video feed")
            self._listeners.append(listener)
            description = self._description
        if description is not None:
            self._deliver(listener, lambda target: target.describe(description))

        def unsubscribe() -> None:
            with self._lock:
                self._listeners = [existing for existing in self._listeners if existing is not listener]

        return unsubscribe

    def _deliver(self, listener: SceneVideoListener, call: Callable[[SceneVideoListener], None]) -> None:
        try:
            call(listener)
        except Exception as unexpected_error:
            # A listener is a sink's doing, and nobody named what it can raise. Logged as unexpected
            # and dropped, so the thread feeding every other listener goes on.
            log.error(
                "UNEXPECTED %s from a scene video listener, dropped, may need a handler",
                type(unexpected_error).__name__,
                exc_info=True,
            )
            with self._lock:
                self._listeners = [existing for existing in self._listeners if existing is not listener]
