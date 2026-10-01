"""Covers the recording and IP camera source, which is the only source with no hardware behind it."""

# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from conftest import SYNTHETIC_VIDEO_FPS, SYNTHETIC_VIDEO_FRAME_COUNT, SYNTHETIC_VIDEO_SIZE
from nav.sources.video_file import VideoFileRgbSource


def test_reads_every_frame_with_increasing_timestamps(synthetic_video: Path) -> None:
    source = VideoFileRgbSource(str(synthetic_video))
    try:
        frames = list(source.frames())
    finally:
        source.close()

    assert len(frames) == SYNTHETIC_VIDEO_FRAME_COUNT

    timestamps = [frame.timestamp_seconds for frame in frames]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps), "two frames shared a timestamp"
    # The whole clip is frame count over frame rate. Generous bound, because the container's
    # reported position is not obliged to land exactly on the nominal frame interval.
    assert timestamps[-1] < SYNTHETIC_VIDEO_FRAME_COUNT / SYNTHETIC_VIDEO_FPS + 1.0


def test_frames_are_rgb_the_right_way_round(synthetic_video: Path) -> None:
    source = VideoFileRgbSource(str(synthetic_video))
    try:
        first = next(iter(source.frames()))
    finally:
        source.close()

    width, height = SYNTHETIC_VIDEO_SIZE
    assert first.image_rgb.shape == (height, width, 3)
    assert first.image_rgb.dtype == np.uint8
    # A plain video knows nothing about the wearer.
    assert first.gaze_pixel is None
    assert first.imu_orientation_wxyz is None


def test_missing_file_is_refused_before_opencv_sees_it(tmp_path: Path) -> None:
    source = VideoFileRgbSource(str(tmp_path / "nothing_here.avi"))

    # FileNotFoundError rather than a generic open failure, so the message says which problem it
    # is. OpenCV's own failure for a missing file is indistinguishable from an unplayable codec.
    with pytest.raises(FileNotFoundError, match="nothing_here"):
        next(iter(source.frames()))


def test_a_file_opencv_cannot_decode_is_refused(tmp_path: Path) -> None:
    not_a_video = tmp_path / "broken.avi"
    not_a_video.write_bytes(b"this is not a video")

    source = VideoFileRgbSource(str(not_a_video))
    with pytest.raises(ConnectionError, match="exists but did not open"):
        next(iter(source.frames()))


def test_closing_twice_does_not_raise(synthetic_video: Path) -> None:
    source = VideoFileRgbSource(str(synthetic_video))
    list(source.frames())

    source.close()
    source.close()


def test_closing_without_ever_opening_does_not_raise(tmp_path: Path) -> None:
    # The loop's finally block closes every source, including one that failed to open.
    VideoFileRgbSource(str(tmp_path / "never_opened.avi")).close()
