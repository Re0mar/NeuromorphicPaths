"""
Decodes a frame the Pixel app's Kotlin encoder produced.

This is the half of the contract the app's own unit tests cannot cover. Their encoder round
tripping through their own code proves nothing about this decoder. The fixture is written by
`FrameEncoderTest.writesTheFixtureTheLaptopDecodes` in `pixel_app`, copied here, and committed.
When the format changes, both sides change and this test says whether they still agree.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.framecodec import LENGTH_PREFIX, decode_frame

FIXTURE = Path(__file__).parent / "fixtures" / "pixel_app_frame.bin"


def test_the_pixel_apps_frame_decodes_with_the_laptops_decoder() -> None:
    assert FIXTURE.is_file(), f"{FIXTURE.name} is missing. Run the pixel_app unit tests and copy build/pixel_app_frame.bin here"
    raw = FIXTURE.read_bytes()
    declared = LENGTH_PREFIX.unpack(raw[: LENGTH_PREFIX.size])[0]
    assert declared == len(raw) - LENGTH_PREFIX.size

    frame = decode_frame(raw[LENGTH_PREFIX.size :])

    # The values the Kotlin test encoded. Millimeters became meters on the way in.
    assert frame.timestamp_seconds == pytest.approx(12.345)
    assert frame.depth_meters == pytest.approx(np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float32))
    assert frame.depth_meters.dtype == np.float32
    assert frame.intrinsics[0, 0] == pytest.approx(500.0)
    assert frame.pose.has_position is True
    assert frame.pose.position == pytest.approx(np.zeros(3))
    # The app says its orientation is measured against gravity, which is what lets the scene read
    # the floor's up from it rather than from the image. Without this the key could stop being
    # sent and the fixture would still decode, with the floor gate quietly back on image-up.
    assert frame.pose.orientation_is_gravity_aligned is True
    assert frame.ground_plane is not None
    assert frame.ground_plane.normal == pytest.approx(np.array([0.0, -1.0, 0.0]))
    assert frame.ground_plane.offset_meters == pytest.approx(1.6)
    assert frame.gaze_pixel is None
