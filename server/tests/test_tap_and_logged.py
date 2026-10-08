"""
Covers recording a run and replaying it.

The pair is tested together because the only thing that makes a replay worth anything is that it
reads back exactly what the tap wrote. Testing either alone would pass while they disagreed.
"""

# Standard library imports
import json
import struct
import time
from collections.abc import Iterator
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.runtime.tap import RecordingTap
from nav.runtime.textio import append_text_lf, write_text_lf
from nav.sources.framecodec import INDEX_FILENAME, FrameDecodeError
from nav.sources.logged import LoggedDepthFrameSource
from nav.types import DepthFrame, Plane, Pose

FRAME_COUNT = 5
INTRINSICS = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


class ListDepthSource:
    """A depth source over frames the test built."""

    def __init__(self, frames: list[DepthFrame]) -> None:
        self._frames = frames
        self.closed = False

    def frames(self) -> Iterator[DepthFrame]:
        yield from self._frames

    def close(self) -> None:
        self.closed = True


def _frames(count: int = FRAME_COUNT) -> list[DepthFrame]:
    return [
        DepthFrame(
            timestamp_seconds=index * 0.1,
            depth_meters=np.full((3, 4), float(index), dtype=np.float32),
            intrinsics=INTRINSICS,
            pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False),
            ground_plane=Plane(normal=np.array([0.0, 1.0, 0.0]), offset_meters=-1.6),
            gaze_pixel=np.array([float(index), 2.0]),
        )
        for index in range(count)
    ]


def _record(log_dir: Path, frames: list[DepthFrame]) -> ListDepthSource:
    inner = ListDepthSource(frames)
    tap = RecordingTap(inner, log_dir)
    passed_through = list(tap.frames())

    assert len(passed_through) == len(frames), "the tap must not swallow or duplicate a frame"
    return inner


def test_recording_writes_a_file_and_an_index_line_per_frame(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())

    assert len(sorted(log_dir.glob("frame_*.bin"))) == FRAME_COUNT
    assert len(log_dir.joinpath(INDEX_FILENAME).read_text(encoding="utf-8").splitlines()) == FRAME_COUNT


def test_a_recording_replays_field_for_field(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    original = _frames()
    _record(log_dir, original)

    replayed = list(LoggedDepthFrameSource(log_dir).frames())

    assert len(replayed) == FRAME_COUNT
    for before, after in zip(original, replayed, strict=True):
        assert after.timestamp_seconds == pytest.approx(before.timestamp_seconds)
        assert after.depth_meters == pytest.approx(before.depth_meters)
        assert after.intrinsics == pytest.approx(before.intrinsics)
        assert after.pose.has_position == before.pose.has_position
        assert after.ground_plane is not None
        assert after.ground_plane.offset_meters == pytest.approx(before.ground_plane.offset_meters)
        assert after.gaze_pixel == pytest.approx(before.gaze_pixel)


def test_the_index_has_no_carriage_returns(tmp_path: Path) -> None:
    # Python's text mode on Windows would write CRLF, and .gitattributes cannot reach a file the
    # program writes at runtime. A log recorded here must be byte-identical to one recorded
    # anywhere else.
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())

    assert b"\r" not in log_dir.joinpath(INDEX_FILENAME).read_bytes()


def test_index_lines_carry_the_sequence_and_the_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())

    lines = log_dir.joinpath(INDEX_FILENAME).read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]

    assert [entry["sequence"] for entry in entries] == list(range(FRAME_COUNT))
    for entry in entries:
        assert (log_dir / entry["file"]).is_file()


def test_closing_the_tap_closes_the_source_underneath(tmp_path: Path) -> None:
    inner = _record(tmp_path / "run", _frames())
    RecordingTap(inner, tmp_path / "unused").close()

    assert inner.closed is True


def test_recording_into_a_directory_with_files_is_refused(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "frame_000000.bin").write_bytes(b"from an earlier run")

    tap = RecordingTap(ListDepthSource(_frames()), log_dir)

    # Two runs interleaved in one log read back as one recording and are not one.
    with pytest.raises(FileExistsError, match="already holds a recording"):
        next(iter(tap.frames()))


def test_a_second_frames_call_continues_the_same_log(tmp_path: Path) -> None:
    # The runtime calls frames() again after the phone reconnects. That is the same run, so the
    # directory the tap already filled is its own and the sequence carries on. Refusing it, as a
    # second tap on the same directory rightly would, ended every reconnecting recording.
    log_dir = tmp_path / "run"
    inner = ListDepthSource(_frames(3))
    tap = RecordingTap(inner, log_dir)
    list(tap.frames())
    list(tap.frames())

    names = sorted(path.name for path in log_dir.glob("frame_*.bin"))
    assert names == [f"frame_{index:06d}.bin" for index in range(6)]
    sequences = [json.loads(line)["sequence"] for line in (log_dir / INDEX_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert sequences == list(range(6))
    assert len(list(LoggedDepthFrameSource(log_dir).frames())) == 6


def test_recording_into_an_empty_existing_directory_is_allowed(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()

    _record(log_dir, _frames(1))

    assert len(sorted(log_dir.glob("frame_*.bin"))) == 1


def test_replaying_a_directory_that_is_not_a_log_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "not_a_log"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match=INDEX_FILENAME):
        next(iter(LoggedDepthFrameSource(empty).frames()))


def test_replaying_an_empty_index_is_refused(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    write_text_lf(log_dir / INDEX_FILENAME, "")

    with pytest.raises(FrameDecodeError, match="nothing to replay"):
        next(iter(LoggedDepthFrameSource(log_dir).frames()))


def test_a_corrupted_frame_file_names_the_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())

    corrupted = log_dir / "frame_000002.bin"
    corrupted.write_bytes(struct.pack(">I", 50) + b"not a frame")

    # Named, and raised rather than skipped. A replay that silently drops frames gives a different
    # answer from the run it is meant to reproduce, and the difference is invisible.
    with pytest.raises(FrameDecodeError, match="frame_000002.bin"):
        list(LoggedDepthFrameSource(log_dir).frames())


def test_an_index_pointing_at_a_missing_file_names_it(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())
    (log_dir / "frame_000003.bin").unlink()

    with pytest.raises(FrameDecodeError, match="frame_000003.bin"):
        list(LoggedDepthFrameSource(log_dir).frames())


def test_a_malformed_index_line_names_the_line(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())
    append_text_lf(log_dir / INDEX_FILENAME, "this is not json\n")

    with pytest.raises(FrameDecodeError, match="line 6"):
        list(LoggedDepthFrameSource(log_dir).frames())


def test_an_index_line_missing_a_key_is_refused(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    write_text_lf(log_dir / INDEX_FILENAME, json.dumps({"sequence": 0, "file": "frame_000000.bin"}) + "\n")

    with pytest.raises(FrameDecodeError, match="timestamp_seconds"):
        list(LoggedDepthFrameSource(log_dir).frames())


def test_frames_replay_in_sequence_order_even_when_the_index_is_shuffled(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())

    index_path = log_dir / INDEX_FILENAME
    lines = index_path.read_text(encoding="utf-8").splitlines()
    write_text_lf(index_path, "\n".join(reversed(lines)) + "\n")

    replayed = list(LoggedDepthFrameSource(log_dir).frames())

    assert [frame.timestamp_seconds for frame in replayed] == pytest.approx([index * 0.1 for index in range(FRAME_COUNT)])


def test_realtime_replay_waits_for_the_recorded_gaps(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    # Two frames a quarter second apart, so the wait is long enough to measure without making the
    # suite slow.
    frames = _frames(1) + [
        DepthFrame(
            timestamp_seconds=0.25,
            depth_meters=np.ones((3, 4), dtype=np.float32),
            intrinsics=INTRINSICS,
            pose=Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False),
            ground_plane=None,
            gaze_pixel=None,
        )
    ]
    _record(log_dir, frames)

    started = time.monotonic()
    list(LoggedDepthFrameSource(log_dir, realtime=True).frames())
    elapsed = time.monotonic() - started

    assert elapsed >= 0.2


def test_replay_without_realtime_does_not_wait(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    _record(log_dir, _frames())

    started = time.monotonic()
    list(LoggedDepthFrameSource(log_dir).frames())

    assert time.monotonic() - started < 0.2


def test_text_writes_use_lf_on_every_platform(tmp_path: Path) -> None:
    target = tmp_path / "written.txt"

    write_text_lf(target, "first\n")
    append_text_lf(target, "second\n")

    assert target.read_bytes() == b"first\nsecond\n"


def test_a_folder_holding_only_timing_and_run_config_is_still_accepted_for_recording(tmp_path: Path) -> None:
    """A run whose phone never connected leaves these two and no frames. That folder holds no recording."""
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "timing.jsonl").write_bytes(b"")
    (log_dir / "run_config.json").write_bytes(b"{}\n")

    _record(log_dir, _frames(1))

    assert len(sorted(log_dir.glob("frame_*.bin"))) == 1
