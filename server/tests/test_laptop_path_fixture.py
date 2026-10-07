"""
Writes the path message the Pixel app's Kotlin decoder is tested against, and checks it decodes here.

The other half of the contract from test_pixel_app_fixture.py. The laptop's encoder writes this
fixture, and the app's PathDecoderTest reads the committed copy and asserts the same literals,
so a change on either side shows up as the other side's test going red. The file is written
whenever it is absent or differs, never skipped, so a run on a fresh clone leaves it in place
rather than passing on nothing.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.framecodec import LENGTH_PREFIX, decode_path, encode_path
from nav.types import PlannedPath

FIXTURE = Path(__file__).parent / "fixtures" / "laptop_path.bin"

# The values the Kotlin test asserts as literals. Change them there in the same commit.
STATED = PlannedPath(
    timestamp_seconds=12.345,
    times_seconds=np.array([0.0, 0.1, 0.2]),
    lateral_offsets_meters=np.array([0.0, 0.05, 0.12]),
    lookahead_heading_radians=0.0423,
    alarm=True,
    cumulative_cost_bits=18.4,
    scene_information_bits=0.37,
    avoidance_surprise_bits=0.51,
)


def test_the_laptops_path_fixture_is_written_and_decodes_to_the_stated_values() -> None:
    message = encode_path(STATED)
    framed = LENGTH_PREFIX.pack(len(message)) + message
    if not FIXTURE.is_file() or FIXTURE.read_bytes() != framed:
        # Binary mode, so no layer translates a byte. The .gitattributes entry keeps git out too.
        FIXTURE.write_bytes(framed)

    raw = FIXTURE.read_bytes()
    declared = LENGTH_PREFIX.unpack(raw[: LENGTH_PREFIX.size])[0]
    assert declared == len(raw) - LENGTH_PREFIX.size

    decoded = decode_path(raw[LENGTH_PREFIX.size :])

    assert decoded.timestamp_seconds == pytest.approx(12.345)
    assert decoded.times_seconds == pytest.approx([0.0, 0.1, 0.2])
    assert decoded.lateral_offsets_meters == pytest.approx([0.0, 0.05, 0.12])
    assert decoded.lookahead_heading_radians == pytest.approx(0.0423)
    assert decoded.alarm is True
    assert decoded.cumulative_cost_bits == pytest.approx(18.4)
    assert decoded.scene_information_bits == pytest.approx(0.37)
    assert decoded.avoidance_surprise_bits == pytest.approx(0.51)
    # One line of JSON after the prefix. The Kotlin side reads it with readFully, not by lines,
    # but a newline here would mean the encoder changed shape.
    assert b"\n" not in raw
