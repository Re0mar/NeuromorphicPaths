"""
Frames from a single image, a folder of images, or a video, all read the same way.

Video frames are timed from their index and the file's frame rate. That is exact for
constant-rate video. Neon recordings carry their own per-frame timestamps, which a Neon reader
should use instead once one exists.
"""

# Standard library imports
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import cv2
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass(frozen=True)
class Frame:
    name: str
    # Seconds from the start of the video. None for still images.
    time_s: float | None
    image: Image.Image


def load_image(path: Path) -> Image.Image:
    # Phones store rotation in EXIF. The app gets upright frames, so this must too.
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def iter_video_frames(path: Path, every_nth: int) -> Iterator[Frame]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{path.name}: OpenCV could not open this video")
    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    if frames_per_second <= 0:
        capture.release()
        raise ValueError(f"{path.name}: video reports no frame rate, so frames cannot be timed")
    try:
        index = 0
        while True:
            # grab() skips decoding, which keeps sampling every nth frame cheap.
            if not capture.grab():
                break
            if index % every_nth == 0:
                decoded, bgr = capture.retrieve()
                if not decoded:
                    raise ValueError(f"{path.name}: frame {index} could not be decoded")
                yield Frame(
                    name=f"{path.stem}@{index:06d}",
                    time_s=index / frames_per_second,
                    image=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                )
            index += 1
    finally:
        capture.release()


def iter_frames(path: Path, every_nth: int = 1) -> Iterator[Frame]:
    """
    Yield frames from an image, a folder of images or a video.

    :param path: An image file, a folder of image files (read in name order) or a video file.
    :param every_nth: For video, keep one frame in every this many. Ignored for images.
    :rtype: Iterator[Frame]
    """
    if every_nth < 1:
        raise ValueError(f"every_nth must be at least 1, got {every_nth}")
    if path.is_dir():
        for image_path in sorted(child for child in path.iterdir() if child.suffix.lower() in IMAGE_SUFFIXES):
            yield Frame(image_path.name, None, load_image(image_path))
        return
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        yield Frame(path.name, None, load_image(path))
    elif suffix in VIDEO_SUFFIXES:
        yield from iter_video_frames(path, every_nth)
    else:
        raise ValueError(
            f"{path.name}: not an image ({', '.join(sorted(IMAGE_SUFFIXES))}), "
            f"a video ({', '.join(sorted(VIDEO_SUFFIXES))}) or a folder"
        )
