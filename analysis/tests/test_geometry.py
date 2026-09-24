"""
Point-of-view estimates checked against a synthetic camera with known height, pitch, heading
and position, looking along a straight path of known width.
"""

# Standard library imports
import dataclasses
import math

# Third party imports
import pytest
from PIL import Image

# Local package imports
from neuromorphicpaths_analysis.detector import PathResult, TracedRow, letterbox
from neuromorphicpaths_analysis.geometry import (
    CAPTURE_PROFILES,
    CaptureProfileName,
    PovStatus,
    estimate_pov,
)

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
BELGIAN = CAPTURE_PROFILES[CaptureProfileName.BELGIAN_DATASET]
FOCAL_LENGTH = BELGIAN.focal_length_px


def project(point: tuple[float, float, float], pitch: float) -> tuple[float, float]:
    """Level-frame point (x right, y down, z forward, camera at origin) to frame pixels."""
    x, y, z = point
    camera_y = y * math.cos(pitch) - z * math.sin(pitch)
    camera_z = y * math.sin(pitch) + z * math.cos(pitch)
    return (
        FRAME_WIDTH / 2 + FOCAL_LENGTH * x / camera_z,
        FRAME_HEIGHT / 2 + FOCAL_LENGTH * camera_y / camera_z,
    )


def synthetic_result(height: float, pitch_deg: float, heading_deg: float, position: float, width: float) -> PathResult:
    """Trace rows of a straight path as a perfect detector would report them."""
    pitch = math.radians(pitch_deg)
    heading = math.radians(heading_deg)
    along = (math.sin(heading), 0.0, math.cos(heading))
    across = (math.cos(heading), 0.0, -math.sin(heading))

    def edge_line(lateral: float) -> tuple[float, float]:
        near, far = (
            project(tuple(lateral * across[i] + distance * along[i] + (height if i == 1 else 0.0) for i in range(3)), pitch)
            for distance in (3.0, 30.0)
        )
        slope = (far[0] - near[0]) / (far[1] - near[1])
        return slope, near[0] - slope * near[1]

    left_slope, left_offset = edge_line(-position * width)
    right_slope, right_offset = edge_line((1 - position) * width)
    geometry = letterbox(Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT)), 320, 320)[1]
    rows = []
    for index, y_px in enumerate(range(FRAME_HEIGHT - 12, FRAME_HEIGHT // 2, -24)):
        rows.append(TracedRow(
            grid_y=index,
            left_cell=0,
            right_cell=0,
            y=y_px / FRAME_HEIGHT,
            left_x=(left_slope * y_px + left_offset) / FRAME_WIDTH,
            right_x=(right_slope * y_px + right_offset) / FRAME_WIDTH,
            left_clipped=False,
            right_clipped=False,
        ))
    return PathResult(geometry, None, rows, 0.9, None, "")


@pytest.mark.parametrize(
    ("height", "pitch_deg", "heading_deg", "position"),
    [
        (0.4, 8.0, 0.0, 0.5),
        (0.3, 4.0, 3.0, 0.3),
        (1.6, 15.0, -5.0, 0.7),
    ],
)
def test_recovers_a_known_camera(height, pitch_deg, heading_deg, position) -> None:
    estimate = estimate_pov(synthetic_result(height, pitch_deg, heading_deg, position, 1.5), BELGIAN)

    assert estimate.status is PovStatus.OK
    assert estimate.pitch_deg == pytest.approx(pitch_deg, abs=1e-6)
    assert estimate.heading_deg == pytest.approx(heading_deg, abs=1e-6)
    assert estimate.position_fraction == pytest.approx(position, abs=1e-6)
    assert estimate.camera_height_m == pytest.approx(height, rel=1e-6)
    assert estimate.path_width_m is None
    assert estimate.reliable


def test_a_bent_edge_is_marked_unreliable() -> None:
    result = synthetic_result(0.4, 8.0, 0.0, 0.5, 1.5)
    # Push the lower left edge points sideways, the way a parked bike cuts into the outline.
    bent = [
        dataclasses.replace(row, left_x=row.left_x + (0.05 if index % 2 else 0.0))
        for index, row in enumerate(result.rows)
    ]

    estimate = estimate_pov(dataclasses.replace(result, rows=bent), BELGIAN)

    assert estimate.status is PovStatus.OK
    assert not estimate.reliable


def test_known_height_gives_path_width_instead() -> None:
    profile = dataclasses.replace(BELGIAN, camera_height_m=1.6, path_width_m=None)

    estimate = estimate_pov(synthetic_result(1.6, 12.0, 2.0, 0.4, 2.0), profile)

    assert estimate.path_width_m == pytest.approx(2.0, rel=1e-6)
    assert estimate.camera_height_m is None


def test_without_focal_length_only_position_is_reported() -> None:
    profile = CAPTURE_PROFILES[CaptureProfileName.NEON]

    estimate = estimate_pov(synthetic_result(1.6, 12.0, 0.0, 0.4, 1.5), profile)

    assert estimate.status is PovStatus.OK
    assert estimate.pitch_deg is None and estimate.camera_height_m is None
    assert estimate.position_fraction == pytest.approx(0.4, abs=1e-6)


def test_clipped_edge_leaves_too_few_rows() -> None:
    result = synthetic_result(0.4, 8.0, 0.0, 0.5, 1.5)
    clipped = [dataclasses.replace(row, left_clipped=True) for row in result.rows]

    estimate = estimate_pov(dataclasses.replace(result, rows=clipped), BELGIAN)

    assert estimate.status is PovStatus.TOO_FEW_ROWS


def test_profile_names_parse_from_their_command_line_spelling() -> None:
    for name in CaptureProfileName:
        assert CaptureProfileName(name.value) is name
        assert CAPTURE_PROFILES[name].name is name
