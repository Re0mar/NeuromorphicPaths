# Standard library imports
from pathlib import Path

# Third party imports
import cv2
import numpy
import pytest
from PIL import Image

# Local package imports
from neuromorphicpaths_analysis.recordings import iter_frames


def write_video(path: Path, frame_count: int, frames_per_second: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), frames_per_second, (64, 48))
    for index in range(frame_count):
        writer.write(numpy.full((48, 64, 3), index * 20, dtype=numpy.uint8))
    writer.release()


def test_folder_frames_come_in_name_order_without_times(tmp_path: Path) -> None:
    for name in ("b.jpg", "a.png", "notes.txt"):
        if name.endswith(".txt"):
            (tmp_path / name).write_text("not a frame", encoding="utf-8")
        else:
            Image.new("RGB", (8, 6)).save(tmp_path / name)

    frames = list(iter_frames(tmp_path))

    assert [frame.name for frame in frames] == ["a.png", "b.jpg"]
    assert all(frame.time_s is None and frame.image.size == (8, 6) for frame in frames)


def test_single_image_is_one_frame(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6)).save(tmp_path / "one.jpg")
    assert [frame.name for frame in iter_frames(tmp_path / "one.jpg")] == ["one.jpg"]


def test_video_keeps_every_nth_frame_and_times_it(tmp_path: Path) -> None:
    video_path = tmp_path / "walk.mp4"
    write_video(video_path, frame_count=7, frames_per_second=10.0)

    frames = list(iter_frames(video_path, every_nth=3))

    assert [frame.name for frame in frames] == ["walk@000000", "walk@000003", "walk@000006"]
    assert [frame.time_s for frame in frames] == pytest.approx([0.0, 0.3, 0.6])
    assert all(frame.image.size == (64, 48) and frame.image.mode == "RGB" for frame in frames)


def test_unsupported_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an image"):
        list(iter_frames(tmp_path / "data.csv"))


def test_every_nth_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(iter_frames(tmp_path, every_nth=0))
