"""
Shared constants for the guard tests.

There is no sys.path manipulation here. The package is installed with pip install -e, so the tests
import nav the same way anything else does. A test suite that patches its own import path can pass
against a layout that would not install.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
import nav

NAV_ROOT = Path(nav.__file__).parent

# Floor count of .py files under nav/. The grep tests assert they scanned at least this many,
# because a glob that silently matched nothing reports no violations and reads exactly like a
# clean package. Raise this when files are added.
MINIMUM_FILE_COUNT = 25


# Frames in the synthetic video every source test reads. Small enough to write in milliseconds,
# long enough that an off-by-one in the read loop shows up.
SYNTHETIC_VIDEO_FRAME_COUNT = 10
SYNTHETIC_VIDEO_SIZE = (64, 48)  # width, height
SYNTHETIC_VIDEO_FPS = 10.0


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Write a short video and return its path.

    MJPG into an .avi rather than mp4v, because it is the one writer backend that is present on a
    stock OpenCV wheel on every platform this runs on.
    """
    target = tmp_path / "scene.avi"
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"MJPG"),
        SYNTHETIC_VIDEO_FPS,
        SYNTHETIC_VIDEO_SIZE,
    )
    assert writer.isOpened(), "OpenCV has no MJPG writer, the source tests cannot run"

    width, height = SYNTHETIC_VIDEO_SIZE
    for index in range(SYNTHETIC_VIDEO_FRAME_COUNT):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # A moving block, so a test that mixed frames up could tell.
        frame[:, index * 4 : index * 4 + 4] = 255
        writer.write(frame)
    writer.release()

    assert target.is_file() and target.stat().st_size > 0
    return target
