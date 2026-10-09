"""
Covers the scene video module: packet extraction, unit assembly, the wire header, and the feed.

Pure functions and one small pub/sub, so nothing here needs the glasses extra except the test
that pins the extraction to the client's own function, which skips without the client.
"""

# Standard library imports
import logging

# Third party imports
import pytest

# Local package imports
from nav.sources.scene_video import (
    KEYFRAME_FLAG,
    START_CODE,
    UNIT_HEADER,
    AccessUnit,
    AccessUnitAssembler,
    NalUnitType,
    SceneVideoFeed,
    SceneVideoProvider,
    VideoDescription,
    annex_b_chunk,
    codec_string,
    nal_unit_types,
    pack_unit,
    unpack_unit,
)

# The capture's own parameter sets, read off neon_walk_2 on 2026-10-08: Baseline profile 66,
# constraint byte 0x80, level 31.
CAPTURE_SPS = bytes.fromhex("000000016742801fda0190092c") + b"\x00" * 6
CAPTURE_PPS = bytes.fromhex("0000000168ce06f2")
PARAMETER_SETS = [CAPTURE_SPS, CAPTURE_PPS]


def _nal(nal_type: int, body: bytes = b"xyz") -> bytes:
    """A whole NAL unit as an RTP payload: the header byte with the type, then its body."""
    return bytes((0x60 | nal_type,)) + body


def _fragment(nal_type: int, body: bytes, start: bool = False, end: bool = False) -> bytes:
    """One FU-A fragment of a unit of the given type, with the two header bytes RFC 3984 gives it."""
    indicator = 0x60 | NalUnitType.FRAGMENT_A
    header = (0x80 if start else 0) | (0x40 if end else 0) | nal_type
    return bytes((indicator, header)) + body


class _Collector:
    """A listener that keeps what it was given, in order."""

    def __init__(self) -> None:
        self.descriptions: list[VideoDescription] = []
        self.units: list[AccessUnit] = []

    def describe(self, description: VideoDescription) -> None:
        self.descriptions.append(description)

    def offer(self, unit: AccessUnit) -> None:
        self.units.append(unit)


# *******************************************
# Extraction
# *******************************************


def test_annex_b_chunk_matches_the_clients_extraction_on_whole_units_and_fragments() -> None:
    nal_unit = pytest.importorskip("pupil_labs.realtime_api.streaming.nal_unit", reason="the client comes with the glasses extra")

    payloads = [
        _nal(NalUnitType.SLICE, b"slice"),
        _nal(NalUnitType.IDR_SLICE, b"idr"),
        _fragment(NalUnitType.IDR_SLICE, b"first", start=True),
        _fragment(NalUnitType.IDR_SLICE, b"middle"),
        _fragment(NalUnitType.IDR_SLICE, b"last", end=True),
    ]

    for payload in payloads:
        assert annex_b_chunk(payload) == bytes(nal_unit.extract_payload_from_nal_unit(payload)), payload.hex()


def test_a_whole_unit_gets_a_start_code_and_a_fragment_start_gets_its_header_rebuilt() -> None:
    assert annex_b_chunk(_nal(NalUnitType.SLICE, b"ab")) == START_CODE + _nal(NalUnitType.SLICE, b"ab")
    # The rebuilt header keeps the indicator's reference bits (0x60) and takes the fragment's type.
    assert annex_b_chunk(_fragment(NalUnitType.IDR_SLICE, b"ab", start=True)) == START_CODE + bytes((0x65,)) + b"ab"
    assert annex_b_chunk(_fragment(NalUnitType.IDR_SLICE, b"ab")) == b"ab"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"", "empty"),
        (bytes((0x80 | NalUnitType.SLICE,)) + b"x", "forbidden_zero_bit"),
        (bytes((0x60 | NalUnitType.FRAGMENT_A,)), "two header bytes"),
    ],
)
def test_a_payload_the_client_would_refuse_is_refused_with_a_reason(payload: bytes, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        annex_b_chunk(payload)


def test_nal_unit_types_finds_every_unit_a_chunk_starts_and_none_in_a_middle_fragment() -> None:
    chunk = CAPTURE_SPS + CAPTURE_PPS + annex_b_chunk(_nal(NalUnitType.IDR_SLICE))

    assert nal_unit_types(chunk) == [NalUnitType.SEQUENCE_PARAMETER_SET, NalUnitType.PICTURE_PARAMETER_SET, NalUnitType.IDR_SLICE]
    assert nal_unit_types(annex_b_chunk(_fragment(NalUnitType.IDR_SLICE, b"middle"))) == []


# *******************************************
# The codec string
# *******************************************


def test_the_codec_string_of_the_captures_sps_is_avc1_42801f() -> None:
    assert codec_string(CAPTURE_SPS) == "avc1.42801f"


@pytest.mark.parametrize("not_an_sps", [CAPTURE_PPS, START_CODE + bytes((0x67,)), b""])
def test_codec_string_refuses_a_nal_that_is_not_an_sps(not_an_sps: bytes) -> None:
    with pytest.raises(ValueError, match="sequence parameter set"):
        codec_string(not_an_sps)


# *******************************************
# The assembler
# *******************************************


def test_an_assembler_finishes_a_unit_when_the_stamp_changes_and_marks_a_keyframe() -> None:
    assembler = AccessUnitAssembler(PARAMETER_SETS)

    first = assembler.feed(_nal(NalUnitType.SLICE, b"a"), 10.0)
    second = assembler.feed(_nal(NalUnitType.SLICE, b"b"), 10.0)
    third = assembler.feed(_nal(NalUnitType.IDR_SLICE, b"k"), 10.033)
    fourth = assembler.feed(_nal(NalUnitType.SLICE, b"c"), 10.066)
    last = assembler.flush()

    assert first is None and second is None, "a frame is not finished until the next one starts"
    assert third == AccessUnit(10.0, START_CODE + _nal(NalUnitType.SLICE, b"a") + START_CODE + _nal(NalUnitType.SLICE, b"b"), keyframe=False)
    assert fourth is not None and fourth.keyframe and fourth.timestamp_seconds == 10.033
    assert fourth.data == CAPTURE_SPS + CAPTURE_PPS + START_CODE + _nal(NalUnitType.IDR_SLICE, b"k"), "a keyframe carries the parameter sets first"
    assert last == AccessUnit(10.066, START_CODE + _nal(NalUnitType.SLICE, b"c"), keyframe=False)
    assert assembler.flush() is None, "nothing left after the flush"


def test_a_fragmented_keyframe_is_one_unit_with_one_start_code_per_nal() -> None:
    assembler = AccessUnitAssembler(PARAMETER_SETS)
    for payload in (
        _fragment(NalUnitType.IDR_SLICE, b"111", start=True),
        _fragment(NalUnitType.IDR_SLICE, b"222"),
        _fragment(NalUnitType.IDR_SLICE, b"333", end=True),
    ):
        assert assembler.feed(payload, 5.0) is None

    unit = assembler.flush()

    assert unit is not None and unit.keyframe
    assert unit.data == CAPTURE_SPS + CAPTURE_PPS + START_CODE + bytes((0x65,)) + b"111222333"
    assert unit.data.count(START_CODE) == 3, "the two parameter sets and the one rebuilt unit"


def test_the_assembler_describes_the_stream_from_its_parameter_sets() -> None:
    description = AccessUnitAssembler(PARAMETER_SETS).description

    assert description == VideoDescription(codec="avc1.42801f", parameter_sets=(CAPTURE_SPS, CAPTURE_PPS))


def test_an_assembler_without_an_sps_is_refused() -> None:
    with pytest.raises(ValueError, match="sequence parameter set"):
        AccessUnitAssembler([CAPTURE_PPS])


def test_a_refused_packet_leaves_the_unit_being_built_untouched() -> None:
    assembler = AccessUnitAssembler(PARAMETER_SETS)
    assembler.feed(_nal(NalUnitType.SLICE, b"a"), 1.0)

    with pytest.raises(ValueError):
        assembler.feed(b"", 1.0)
    unit = assembler.flush()

    assert unit is not None and unit.data == START_CODE + _nal(NalUnitType.SLICE, b"a")


# *******************************************
# The wire header
# *******************************************


def test_pack_and_unpack_round_trip_a_unit() -> None:
    for unit in (AccessUnit(1_791_471_024.179, b"\x00\x00\x00\x01\x65data", keyframe=True), AccessUnit(2.5, b"", keyframe=False)):
        payload = pack_unit(unit)
        assert len(payload) == UNIT_HEADER.size + len(unit.data)
        assert payload[0] == (KEYFRAME_FLAG if unit.keyframe else 0)
        assert unpack_unit(payload) == unit


def test_a_payload_shorter_than_the_header_is_refused() -> None:
    with pytest.raises(ValueError, match="at least"):
        unpack_unit(b"\x01" * (UNIT_HEADER.size - 1))


def test_flags_outside_the_keyframe_bit_are_refused() -> None:
    with pytest.raises(ValueError, match="flags"):
        unpack_unit(UNIT_HEADER.pack(0x02, 1.0) + b"data")


# *******************************************
# The feed
# *******************************************


def test_a_feed_calls_each_listener_in_order_and_unsubscribe_stops_it() -> None:
    feed = SceneVideoFeed()
    first, second = _Collector(), _Collector()
    description = VideoDescription("avc1.42801f", (CAPTURE_SPS, CAPTURE_PPS))
    unsubscribe_first = feed.subscribe(first)
    feed.subscribe(second)

    feed.describe(description)
    feed.offer(AccessUnit(1.0, b"one", keyframe=True))
    unsubscribe_first()
    feed.offer(AccessUnit(2.0, b"two", keyframe=False))

    assert first.descriptions == [description] and second.descriptions == [description]
    assert [unit.data for unit in first.units] == [b"one"]
    assert [unit.data for unit in second.units] == [b"one", b"two"]
    assert feed.description == description


def test_a_listener_subscribed_after_the_description_gets_it_at_once() -> None:
    feed = SceneVideoFeed()
    description = VideoDescription("avc1.42801f", (CAPTURE_SPS, CAPTURE_PPS))
    feed.describe(description)
    late = _Collector()

    feed.subscribe(late)

    assert late.descriptions == [description]


def test_a_listener_that_raises_is_dropped_and_the_others_still_get_the_unit(caplog: pytest.LogCaptureFixture) -> None:
    feed = SceneVideoFeed()

    class Broken(_Collector):
        def offer(self, unit: AccessUnit) -> None:
            raise RuntimeError("a sink's bug")

    broken, sound = Broken(), _Collector()
    feed.subscribe(broken)
    feed.subscribe(sound)

    with caplog.at_level(logging.ERROR, logger="nav.sources.scene_video"):
        feed.offer(AccessUnit(1.0, b"one", keyframe=True))
        feed.offer(AccessUnit(2.0, b"two", keyframe=False))

    assert [unit.data for unit in sound.units] == [b"one", b"two"]
    errors = [record for record in caplog.records if "UNEXPECTED RuntimeError" in record.getMessage()]
    assert len(errors) == 1, "dropped after the first failure, so logged once"
    assert errors[0].exc_info is not None


def test_subscribing_the_same_listener_twice_is_refused() -> None:
    feed = SceneVideoFeed()
    listener = _Collector()
    feed.subscribe(listener)

    with pytest.raises(ValueError, match="already subscribed"):
        feed.subscribe(listener)


def test_a_feed_is_itself_a_listener_and_a_provider_is_told_apart_at_runtime() -> None:
    # The source hands the feed straight to the device process as the listener, and the device
    # process checks its device with isinstance, so both shapes have to hold at runtime.
    feed = SceneVideoFeed()

    class Provider:
        def subscribe_video(self, listener) -> None:
            listener.describe(VideoDescription("avc1.42801f", (CAPTURE_SPS, CAPTURE_PPS)))

    assert callable(feed.describe) and callable(feed.offer)
    assert isinstance(Provider(), SceneVideoProvider)
    assert not isinstance(_Collector(), SceneVideoProvider)
