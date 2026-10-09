"""
Turns a native Neon recording into a capture, so it replays through the live route.

A capture is what examples/capture_neon_stream.py records off the network: the scene video as the
packets the glasses send, gaze and IMU as lines, and meta.json. `--neon-replay` plays a capture at
the pace it was recorded, through the same decoder, depth model, planner and page video as a live
walk. A recording converted here replays the same way, so a demo from a recording shows the video
the laptop is planning on, in step, as close to live as the laptop can make it.

Depth is never stored. The replay estimates it from the video on every run. So moving the demo to
another recording is converting that recording and pointing `--neon-replay` at the result, and a
change to the depth model reaches every converted recording by itself.

The Companion app stores the scene video as MP4: each frame one sample of length-prefixed NAL units,
with the parameter sets in the file's avcC box. A capture holds RTP payloads instead. Each NAL unit
becomes one single-unit payload here, which the replay reads exactly as it reads the glasses' own
packets, since a single-unit payload is the NAL unit as it is. Fragmenting would only matter on a
network, and a capture's packets never cross one.
"""

# Standard library imports
import base64
import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Protocol

# Third party imports
import numpy as np

# Local package imports
from nav.sources.neon_recording import NANOSECONDS_PER_SECOND, NativeNeonRecordingReader
from nav.sources.neon_stream import GAZE_FILENAME, IMU_FILENAME, META_FILENAME, PACKET_HEADER, SCENE_PACKETS_FILENAME
from nav.sources.scene_video import START_CODE

log = logging.getLogger(__name__)

# The Companion app's scene video, one file per part, "ps1" first. A long recording has several.
SCENE_VIDEO_GLOB = "Neon Scene Camera v1 ps*.mp4"
SCENE_VIDEO_PART_NUMBER = re.compile(r" ps(\d+)\.mp4$")
# avcC, ISO/IEC 14496-15 5.3.3.1: version, profile, compatibility, level, then the length size.
AVCC_VERSION = 1
AVCC_HEADER_BYTES = 5
AVCC_LENGTH_SIZE_MASK = 0x03
AVCC_SEQUENCE_PARAMETER_SET_COUNT_MASK = 0x1F
AVCC_SET_LENGTH_BYTES = 2


class RecordingForCapture(Protocol):
    """What the converter reads from a recording besides its video. NativeNeonRecordingReader is the shipping one."""

    def scene_times_ns(self) -> np.ndarray: ...

    def scene_camera_matrix(self) -> np.ndarray: ...

    def scene_distortion_coefficients(self) -> np.ndarray: ...

    def imu_samples(self) -> tuple[np.ndarray, np.ndarray]: ...

    def gaze_samples(self) -> tuple[np.ndarray, np.ndarray] | None: ...

    def close(self) -> None: ...


def avcc_parameter_sets(extradata: bytes) -> tuple[int, list[bytes]]:
    """
    The NAL length size and the parameter sets from an MP4's avcC box.

    Each set comes back with an Annex B start code in front, the form a capture's `sprop` stores and
    the replay's decoder and the page's decoder are both given.

    :param extradata: The video stream's extradata, which for H.264 in MP4 is the avcC box's body.
    :return: (bytes per NAL length, [SPS..., PPS...]).
    :rtype: tuple[int, list[bytes]]
    :raises ValueError: When the bytes are not an avcC box, or end partway through one.
    """
    if len(extradata) < AVCC_HEADER_BYTES + 1 or extradata[0] != AVCC_VERSION:
        raise ValueError("the video's extradata is not an avcC box, so its parameter sets can't be read")
    length_size = (extradata[4] & AVCC_LENGTH_SIZE_MASK) + 1
    parameter_sets: list[bytes] = []
    position = AVCC_HEADER_BYTES
    # Sequence parameter sets first, their count in the low five bits, then picture parameter sets,
    # their count a whole byte.
    for count_mask in (AVCC_SEQUENCE_PARAMETER_SET_COUNT_MASK, 0xFF):
        if position >= len(extradata):
            raise ValueError(f"the avcC box ends at byte {position}, before its parameter set count")
        count = extradata[position] & count_mask
        position += 1
        for _ in range(count):
            if position + AVCC_SET_LENGTH_BYTES > len(extradata):
                raise ValueError(f"the avcC box ends at byte {position}, inside a parameter set's length")
            length = int.from_bytes(extradata[position : position + AVCC_SET_LENGTH_BYTES], "big")
            position += AVCC_SET_LENGTH_BYTES
            if position + length > len(extradata):
                raise ValueError(f"the avcC box ends at byte {len(extradata)}, inside a {length}-byte parameter set")
            parameter_sets.append(START_CODE + extradata[position : position + length])
            position += length
    return length_size, parameter_sets


def length_prefixed_nal_units(sample: bytes, length_size: int) -> list[bytes]:
    """
    One MP4 sample split into its NAL units, without their length prefixes.

    :param sample: One frame's bytes as the MP4 stores them.
    :param length_size: Bytes per length prefix, from the avcC box.
    :return: The NAL units in order.
    :rtype: list[bytes]
    :raises ValueError: When a length runs past the end of the sample, or a unit is empty.
    """
    units: list[bytes] = []
    position = 0
    while position < len(sample):
        if position + length_size > len(sample):
            raise ValueError(f"a sample of {len(sample)} bytes ends at byte {position}, inside a NAL length")
        length = int.from_bytes(sample[position : position + length_size], "big")
        position += length_size
        if length == 0 or position + length > len(sample):
            raise ValueError(f"a sample of {len(sample)} bytes has a {length}-byte NAL unit at byte {position}")
        units.append(sample[position : position + length])
        position += length
    return units


def scene_video_parts(recording_dir: Path) -> list[Path]:
    """
    The recording's scene video files, in part order.

    :raises FileNotFoundError: When the folder holds none.
    """
    parts = sorted(
        (path for path in recording_dir.glob(SCENE_VIDEO_GLOB) if SCENE_VIDEO_PART_NUMBER.search(path.name)),
        key=lambda path: int(SCENE_VIDEO_PART_NUMBER.search(path.name).group(1)),
    )
    if not parts:
        raise FileNotFoundError(f"{recording_dir} holds no scene video named like '{SCENE_VIDEO_GLOB}'")
    return parts


def mp4_scene_samples(parts: list[Path]) -> tuple[int, list[bytes], Iterator[bytes]]:
    """
    The scene video's length size, its parameter sets, and every frame's sample in order across the parts.

    The parameter sets are the first part's. A Companion recording keeps one camera setting throughout.

    :raises ValueError: When a part's video isn't H.264 in an avcC box, or, while iterating, stores
        its frames out of display order.
    """
    # Optional dependency, present with the glasses extra. Imported here so the module loads without it.
    import av

    with av.open(str(parts[0])) as first:
        context = first.streams.video[0].codec_context
        if context.name != "h264":
            raise ValueError(f"{parts[0].name} is {context.name}, and only H.264 can become a capture")
        length_size, parameter_sets = avcc_parameter_sets(bytes(context.extradata or b""))

    def samples() -> Iterator[bytes]:
        for part in parts:
            with av.open(str(part)) as container:
                stream = container.streams.video[0]
                previous_pts = None
                for packet in container.demux(stream):
                    # The demuxer ends each stream with an empty packet, which holds no frame.
                    if not packet.size:
                        continue
                    # Stamps pair with frames by order, so a video stored out of display order would
                    # stamp frames wrongly. The Companion app never reorders. A video from elsewhere may.
                    if previous_pts is not None and packet.pts is not None and packet.pts < previous_pts:
                        raise ValueError(f"{part.name} stores frames out of display order, so they can't be paired with the recording's stamps")
                    previous_pts = packet.pts
                    yield bytes(packet)

    return length_size, parameter_sets, samples()


def capture_scene_packets(samples: Iterable[bytes], times_ns: np.ndarray, length_size: int) -> Iterator[tuple[float, bytes]]:
    """
    Every NAL unit as a capture packet, stamped with its frame's time in seconds on the Neon clock.

    Frames and stamps pair by order, which holds because the Companion app encodes without reordered
    frames, so storage order is display order.

    :raises ValueError: When the video holds a different number of frames than the recording has stamps.
    """
    stamps = np.asarray(times_ns, dtype=np.int64).reshape(-1)
    frames = 0
    for frames, sample in enumerate(samples, start=1):
        if frames > stamps.size:
            raise ValueError(f"the scene video holds more frames than the recording's {stamps.size} stamps")
        stamp_seconds = int(stamps[frames - 1]) / NANOSECONDS_PER_SECOND
        for unit in length_prefixed_nal_units(sample, length_size):
            yield stamp_seconds, unit
    if frames != stamps.size:
        raise ValueError(f"the scene video holds {frames} frames and the recording has {stamps.size} stamps")


def write_capture(
    out_dir: Path,
    recording_name: str,
    parameter_sets: list[bytes],
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    packets: Iterable[tuple[float, bytes]],
    imu: tuple[np.ndarray, np.ndarray],
    gaze: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, int]:
    """
    Write a capture in the format examples/capture_neon_stream.py writes and `--neon-replay` reads.

    meta.json goes last, so a conversion that fails partway leaves a folder the replay refuses
    rather than one it plays short.

    :param out_dir: A new folder. It must not exist yet.
    :param recording_name: Where the capture came from, kept in meta.json for whoever reads it later.
    :param parameter_sets: SPS and PPS with start codes.
    :param camera_matrix: The scene camera's intrinsics, 3 by 3.
    :param distortion_coefficients: The scene camera's distortion, as the calibration gives it.
    :param packets: (seconds on the Neon clock, NAL unit) in stream order.
    :param imu: (N,) int64 nanosecond times and (N, 4) quaternions, w x y z.
    :param gaze: (N,) int64 nanosecond times and (N, 2) scene pixels, or None for a recording without gaze.
    :return: How many scene packets, gaze samples and IMU readings were written.
    :rtype: dict[str, int]
    :raises FileExistsError: When out_dir exists, so an earlier capture is never overwritten.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    counts = {"scene": 0, "gaze": 0, "imu": 0}
    first_stamp = last_stamp = None
    with (out_dir / SCENE_PACKETS_FILENAME).open("wb") as scene:
        for stamp, unit in packets:
            scene.write(PACKET_HEADER.pack(stamp, len(unit)) + unit)
            counts["scene"] += 1
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp

    imu_times_ns, imu_quaternions = imu
    with (out_dir / IMU_FILENAME).open("w", encoding="utf-8", newline="\n") as lines:
        for time_ns, (w, x, y, z) in zip(np.asarray(imu_times_ns, dtype=np.int64).tolist(), np.asarray(imu_quaternions, dtype=np.float64).tolist()):
            lines.write(json.dumps({"t": time_ns / NANOSECONDS_PER_SECOND, "w": w, "x": x, "y": y, "z": z}) + "\n")
            counts["imu"] += 1
    if gaze is not None:
        gaze_times_ns, gaze_points = gaze
        with (out_dir / GAZE_FILENAME).open("w", encoding="utf-8", newline="\n") as lines:
            for time_ns, (x, y) in zip(np.asarray(gaze_times_ns, dtype=np.int64).tolist(), np.asarray(gaze_points, dtype=np.float64).tolist()):
                lines.write(json.dumps({"t": time_ns / NANOSECONDS_PER_SECOND, "x": x, "y": y}) + "\n")
                counts["gaze"] += 1

    meta = {
        # No glasses on a network, so no address and no clock offset. The replay moves the
        # recording's stamps to now by itself, and a null offset is what it reads as none measured.
        "address": None,
        "converted_from": recording_name,
        "seconds": 0.0 if first_stamp is None else last_stamp - first_stamp,
        "counts": counts,
        "time_offset": None,
        "scene_camera_matrix": np.asarray(camera_matrix, dtype=np.float64).tolist(),
        "scene_distortion_coefficients": np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1).tolist(),
        "encoding": "h264",
        "sprop": [base64.b64encode(parameter_set).decode("ascii") for parameter_set in parameter_sets],
    }
    (out_dir / META_FILENAME).write_bytes(json.dumps(meta, indent=2).encode("utf-8"))
    return counts


def convert_recording(
    recording_dir: Path,
    out_dir: Path,
    reader_factory: Callable[[Path], RecordingForCapture] = NativeNeonRecordingReader,
) -> dict[str, int]:
    """
    Convert one Companion recording folder into a capture folder.

    :param recording_dir: The recording as the Companion app exports it, `info.json` and all.
    :param out_dir: A new folder for the capture.
    :param reader_factory: Opens the recording. The shipping reader unless a test passes a fake.
    :return: The counts written, as write_capture returns them.
    :rtype: dict[str, int]
    :raises FileNotFoundError: When the folder holds no scene video.
    :raises FileExistsError: When out_dir exists.
    :raises ValueError: When the video isn't H.264, or its frames and the recording's stamps disagree.
    """
    parts = scene_video_parts(recording_dir)
    reader = reader_factory(recording_dir)
    try:
        length_size, parameter_sets, samples = mp4_scene_samples(parts)
        counts = write_capture(
            out_dir,
            recording_dir.name,
            parameter_sets,
            reader.scene_camera_matrix(),
            reader.scene_distortion_coefficients(),
            capture_scene_packets(samples, reader.scene_times_ns(), length_size),
            reader.imu_samples(),
            reader.gaze_samples(),
        )
    finally:
        reader.close()
    log.info("converted %s into %s: %s", recording_dir.name, out_dir, counts)
    return counts
