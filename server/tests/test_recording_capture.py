"""
Covers turning a Companion recording into a capture: the MP4's parameter sets and samples, the stamps,
and a capture the replay's own readers read back.

The format checks build their bytes by hand, so a wrong offset shows as a wrong number. The end-to-end
tests write a real MP4 with PyAV and decode what the converter wrote, so they skip without the
glasses extra.
"""

# Standard library imports
import json

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.neon_recording import NANOSECONDS_PER_SECOND
from nav.sources.neon_stream import (
    GAZE_FIELDS,
    GAZE_FILENAME,
    IMU_FIELDS,
    IMU_FILENAME,
    META_FILENAME,
    SCENE_PACKETS_FILENAME,
    _read_meta,
    read_capture_packets,
    read_capture_samples,
)
from nav.sources.recording_capture import (
    avcc_parameter_sets,
    capture_scene_packets,
    convert_recording,
    length_prefixed_nal_units,
    scene_video_parts,
    write_capture,
)
from nav.sources.scene_video import START_CODE
from neon_captures import CAMERA_MATRIX, DISTORTION, write_h264_mp4

SPS = bytes([0x67, 0x42, 0x80, 0x1F, 0xDA, 0x01])
PPS = bytes([0x68, 0xCE, 0x06, 0xF2])
# version 1, Baseline (0x42), constraints 0x80, level 3.1 (0x1f), 4-byte lengths, one SPS, one PPS.
AVCC = bytes([1, 0x42, 0x80, 0x1F, 0xFF, 0xE1]) + len(SPS).to_bytes(2, "big") + SPS + bytes([1]) + len(PPS).to_bytes(2, "big") + PPS
FIRST_STAMP_NS = 1_791_472_773_158_428_920
FRAME_NS = 33_333_333


def _stamps(count: int) -> np.ndarray:
    return FIRST_STAMP_NS + FRAME_NS * np.arange(count, dtype=np.int64)


# *******************************************
# The MP4's bytes
# *******************************************


def test_the_avcc_box_gives_its_length_size_and_both_sets_with_start_codes() -> None:
    length_size, parameter_sets = avcc_parameter_sets(AVCC)

    assert length_size == 4
    assert parameter_sets == [START_CODE + SPS, START_CODE + PPS]


def test_a_two_byte_length_size_is_read_from_the_low_bits() -> None:
    two_byte = AVCC[:4] + bytes([0xFD]) + AVCC[5:]

    assert avcc_parameter_sets(two_byte)[0] == 2


@pytest.mark.parametrize(
    ("extradata", "fragment"),
    [
        (START_CODE + SPS, "not an avcC box"),
        (AVCC[:5], "not an avcC box"),
        (AVCC[:7], "inside a parameter set's length"),
        (AVCC[:10], f"inside a {len(SPS)}-byte parameter set"),
        (AVCC[: 8 + len(SPS)], "before its parameter set count"),
    ],
)
def test_extradata_that_is_not_a_whole_avcc_box_is_refused(extradata: bytes, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        avcc_parameter_sets(extradata)


def test_a_sample_splits_into_its_units_without_their_lengths() -> None:
    sample = len(SPS).to_bytes(4, "big") + SPS + len(PPS).to_bytes(4, "big") + PPS

    assert length_prefixed_nal_units(sample, 4) == [SPS, PPS]


@pytest.mark.parametrize(
    ("sample", "fragment"),
    [
        ((100).to_bytes(4, "big") + SPS, "a 100-byte NAL unit at byte 4"),
        ((0).to_bytes(4, "big") + SPS, "a 0-byte NAL unit"),
        (bytes([0, 0]), "inside a NAL length"),
    ],
)
def test_a_sample_whose_lengths_dont_add_up_is_refused(sample: bytes, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        length_prefixed_nal_units(sample, 4)


# *******************************************
# Stamps and parts
# *******************************************


def test_every_unit_of_a_frame_carries_that_frames_stamp_in_seconds() -> None:
    frame = len(SPS).to_bytes(4, "big") + SPS + len(PPS).to_bytes(4, "big") + PPS
    stamps = _stamps(2)

    packets = list(capture_scene_packets([frame, frame], stamps, 4))

    assert [unit for _, unit in packets] == [SPS, PPS, SPS, PPS]
    assert [stamp for stamp, _ in packets] == [stamps[0] / NANOSECONDS_PER_SECOND] * 2 + [stamps[1] / NANOSECONDS_PER_SECOND] * 2


@pytest.mark.parametrize(("frames", "stamps"), [(3, 2), (2, 3)])
def test_frames_and_stamps_that_dont_pair_up_are_refused(frames: int, stamps: int) -> None:
    sample = len(SPS).to_bytes(4, "big") + SPS

    with pytest.raises(ValueError, match="frames"):
        list(capture_scene_packets([sample] * frames, _stamps(stamps), 4))


def test_the_parts_come_in_part_order_not_name_order(tmp_path) -> None:
    for number in (10, 2, 1):
        (tmp_path / f"Neon Scene Camera v1 ps{number}.mp4").write_bytes(b"")
    (tmp_path / "Neon Sensor Module v1 ps1.mp4").write_bytes(b"")

    parts = scene_video_parts(tmp_path)

    assert [part.name for part in parts] == [f"Neon Scene Camera v1 ps{number}.mp4" for number in (1, 2, 10)]


def test_a_folder_without_scene_video_is_refused(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no scene video"):
        scene_video_parts(tmp_path)


# *******************************************
# The capture, read back by the replay's own readers
# *******************************************


def _imu(count: int) -> tuple[np.ndarray, np.ndarray]:
    return _stamps(count) + 1_000_000, np.tile([0.9, 0.1, 0.2, 0.3], (count, 1))


def _gaze(count: int) -> tuple[np.ndarray, np.ndarray]:
    return _stamps(count) + 2_000_000, np.tile([800.0, 600.0], (count, 1))


def test_the_replay_reads_back_exactly_what_was_written(tmp_path) -> None:
    capture = tmp_path / "capture"
    packets = [(1791472773.1584, SPS), (1791472773.1584, PPS), (1791472773.1917, SPS)]

    counts = write_capture(capture, "walk_3", [START_CODE + SPS, START_CODE + PPS], np.array(CAMERA_MATRIX), np.array(DISTORTION), packets, _imu(2), _gaze(3))

    assert counts == {"scene": 3, "gaze": 3, "imu": 2}
    calibration, offset, parameter_sets = _read_meta(capture)
    assert np.allclose(calibration.scene_camera_matrix, CAMERA_MATRIX) and np.allclose(calibration.scene_distortion_coefficients, DISTORTION)
    assert offset is None, "a converted recording has no network clock offset to report"
    assert parameter_sets == [START_CODE + SPS, START_CODE + PPS]
    assert list(read_capture_packets(capture / SCENE_PACKETS_FILENAME)) == packets
    imu = read_capture_samples(capture / IMU_FILENAME, IMU_FIELDS)
    assert imu[0] == pytest.approx((_imu(2)[0][0] / NANOSECONDS_PER_SECOND, 0.9, 0.1, 0.2, 0.3))
    assert read_capture_samples(capture / GAZE_FILENAME, GAZE_FIELDS)[2][1:] == (800.0, 600.0)
    meta = json.loads((capture / META_FILENAME).read_text(encoding="utf-8"))
    assert meta["converted_from"] == "walk_3" and meta["seconds"] == pytest.approx(1791472773.1917 - 1791472773.1584)


def test_a_recording_without_gaze_writes_no_gaze_file(tmp_path) -> None:
    capture = tmp_path / "capture"

    counts = write_capture(capture, "walk", [START_CODE + SPS], np.array(CAMERA_MATRIX), np.array(DISTORTION), [(1.0, SPS)], _imu(1), None)

    assert counts["gaze"] == 0 and not (capture / GAZE_FILENAME).exists()
    assert read_capture_samples(capture / GAZE_FILENAME, GAZE_FIELDS) == [], "the replay reads a missing file as no gaze"


def test_an_existing_capture_is_never_overwritten(tmp_path) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()

    with pytest.raises(FileExistsError):
        write_capture(capture, "walk", [], np.array(CAMERA_MATRIX), np.array(DISTORTION), [], _imu(0), None)


# *******************************************
# End to end, a real MP4
# *******************************************


class _FakeRecording:
    """Everything but the video, which the converter reads from the folder itself."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.closed = False

    def scene_times_ns(self) -> np.ndarray:
        return _stamps(self.frames)

    def scene_camera_matrix(self) -> np.ndarray:
        return np.array(CAMERA_MATRIX)

    def scene_distortion_coefficients(self) -> np.ndarray:
        return np.array(DISTORTION)

    def imu_samples(self) -> tuple[np.ndarray, np.ndarray]:
        return _imu(self.frames)

    def gaze_samples(self) -> tuple[np.ndarray, np.ndarray] | None:
        return _gaze(self.frames)

    def close(self) -> None:
        self.closed = True


def _decoded_frames(capture) -> int:
    # What the replay's decoder does with a single-unit payload: a start code in front, then parse.
    import av

    _, _, parameter_sets = _read_meta(capture)
    decoder = av.CodecContext.create("h264", "r")
    for parameter_set in parameter_sets:
        decoder.parse(parameter_set)
    decoded = 0
    for _, unit in read_capture_packets(capture / SCENE_PACKETS_FILENAME):
        for packet in decoder.parse(START_CODE + unit):
            decoded += len(decoder.decode(packet))
    for packet in decoder.parse(None):
        decoded += len(decoder.decode(packet))
    return decoded + len(decoder.decode(None))


@pytest.mark.parametrize("parts", [1, 2])
def test_a_recordings_video_becomes_packets_the_replays_decoder_plays_in_full(tmp_path, parts: int) -> None:
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    recording = tmp_path / "walk_3"
    recording.mkdir()
    frames_per_part = 12 // parts
    for number in range(1, parts + 1):
        write_h264_mp4(recording / f"Neon Scene Camera v1 ps{number}.mp4", frames_per_part)
    fake = _FakeRecording(12)

    counts = convert_recording(recording, tmp_path / "capture", reader_factory=lambda _: fake)

    assert fake.closed
    assert counts["imu"] == 12 and counts["gaze"] == 12
    stamps = sorted({stamp for stamp, _ in read_capture_packets(tmp_path / "capture" / SCENE_PACKETS_FILENAME)})
    assert stamps == pytest.approx(list(_stamps(12) / NANOSECONDS_PER_SECOND)), "one stamp per frame, in order, across the parts"
    assert _decoded_frames(tmp_path / "capture") == 12


def test_a_video_stored_out_of_display_order_is_refused(tmp_path) -> None:
    pytest.importorskip("av", reason="PyAV comes with the glasses extra")
    recording = tmp_path / "walk"
    recording.mkdir()
    write_h264_mp4(recording / "Neon Scene Camera v1 ps1.mp4", 12, reordered=True)
    fake = _FakeRecording(12)

    with pytest.raises(ValueError, match="out of display order"):
        convert_recording(recording, tmp_path / "capture", reader_factory=lambda _: fake)

    assert fake.closed, "the recording is closed even when the conversion fails"
    assert not (tmp_path / "capture" / META_FILENAME).exists(), "a failed conversion leaves nothing the replay would play"
