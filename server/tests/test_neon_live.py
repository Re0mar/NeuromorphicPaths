"""
Covers what the Neon source does with what the client hands it, against a fake device.

The Pupil Labs client itself is absent from the test environment on purpose, so connecting and
discovery are not covered here. What is covered is the part that goes wrong silently: which
quaternion field lands where, what happens between IMU readings, and a frame with no gaze.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.config import NeonConfig
from nav.sources.neon_live import NeonLiveRgbSource


@dataclass
class FakeQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass
class FakeImuDatum:
    quaternion: FakeQuaternion | None


@dataclass
class FakeGaze:
    x: float
    y: float


@dataclass
class FakeScene:
    bgr_pixels: np.ndarray
    timestamp_unix_seconds: float


@dataclass
class FakeMatched:
    scene: FakeScene
    gaze: FakeGaze | None


class FakeDevice:
    """Hands out scripted matched frames and IMU readings, the way the simple client does."""

    def __init__(self, matched: list, imu: list) -> None:
        self._matched = list(matched)
        self._imu = list(imu)
        self.closed = False

    def receive_matched_scene_and_eyes_video_frames_and_gaze(self):
        if not self._matched:
            raise StopIteration
        return self._matched.pop(0)

    def receive_imu_datum(self, timeout_seconds: float):
        return self._imu.pop(0) if self._imu else None

    def close(self) -> None:
        self.closed = True


def _scene(stamp: float) -> FakeScene:
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[..., 0] = 255  # blue in BGR, so the conversion to RGB can be checked
    return FakeScene(bgr_pixels=pixels, timestamp_unix_seconds=stamp)


def _source_with(device: FakeDevice) -> NeonLiveRgbSource:
    source = NeonLiveRgbSource(NeonConfig())
    source._device = device  # the connected device, standing in for _connect()
    return source


def _take(source: NeonLiveRgbSource, count: int) -> list:
    frames = source.frames()
    taken = []
    for _ in range(count):
        try:
            taken.append(next(frames))
        except (StopIteration, RuntimeError):
            break
    return taken


def test_the_imu_quaternion_is_read_by_field_name_into_pose_order() -> None:
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), FakeGaze(100.0, 200.0))],
        imu=[FakeImuDatum(FakeQuaternion(w=0.9, x=0.1, y=0.2, z=0.3))],
    )

    frame = _take(_source_with(device), 1)[0]

    assert frame.imu_orientation_wxyz == pytest.approx([0.9, 0.1, 0.2, 0.3])
    assert frame.gaze_pixel == pytest.approx([100.0, 200.0])
    assert frame.timestamp_seconds == pytest.approx(1.0)
    assert frame.image_rgb[0, 0].tolist() == [0, 0, 255], "BGR in, RGB out"


def test_a_frame_without_a_new_imu_reading_keeps_the_previous_orientation() -> None:
    device = FakeDevice(
        matched=[FakeMatched(_scene(1.0), None), FakeMatched(_scene(2.0), None), FakeMatched(_scene(3.0), None)],
        imu=[FakeImuDatum(FakeQuaternion(1.0, 0.0, 0.0, 0.0)), None, FakeImuDatum(None)],
    )

    frames = _take(_source_with(device), 3)

    assert len(frames) == 3
    assert frames[0].imu_orientation_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])
    # No datum, then a datum with no quaternion. Both carry the last orientation forward rather
    # than dropping to None, which would flip the scene into fitting the floor from scratch.
    assert frames[1].imu_orientation_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert frames[2].imu_orientation_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_before_the_first_imu_reading_the_orientation_is_none_and_no_gaze_is_none() -> None:
    device = FakeDevice(matched=[FakeMatched(_scene(1.0), None)], imu=[])

    frame = _take(_source_with(device), 1)[0]

    assert frame.imu_orientation_wxyz is None
    assert frame.gaze_pixel is None


def test_closing_closes_the_device() -> None:
    device = FakeDevice(matched=[], imu=[])
    source = _source_with(device)

    source.close()

    assert device.closed is True
