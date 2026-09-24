# Standard library imports
from pathlib import Path

# Third party imports
import numpy
import pytest
from PIL import Image, ImageOps

# Local package imports
from neuromorphicpaths_analysis.detector import PathDetector, letterbox, trace_path
from neuromorphicpaths_analysis.detector.path_detector import DEFAULT_MODEL_PATH, REPOSITORY_ROOT

GRID_SIZE = 80


@pytest.mark.parametrize(
    ("source_size", "content_size", "padding"),
    [
        ((1920, 1080), (320, 180), (0, 70)),  # dataset stills
        ((1600, 1200), (320, 240), (0, 40)),  # Neon scene camera
        ((504, 896), (180, 320), (70, 0)),  # Meta glasses stream, portrait
    ],
)
def test_letterbox_geometry(source_size, content_size, padding) -> None:
    image = Image.new("RGB", source_size)
    letterboxed, geometry = letterbox(image, 320, 320)
    assert letterboxed.shape == (320, 320, 3)
    assert (geometry.content_width, geometry.content_height) == content_size
    assert (geometry.pad_x, geometry.pad_y) == padding
    # Padding is the Ultralytics gray.
    assert tuple(letterboxed[0, 0]) == (114, 114, 114) or padding == (0, 0)


def portrait_geometry():
    return letterbox(Image.new("RGB", (504, 896)), 320, 320)[1]


def test_trace_follows_the_run_under_the_center_and_ignores_side_blobs() -> None:
    geometry = letterbox(Image.new("RGB", (1920, 1080)), 320, 320)[1]
    grid = numpy.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid[40:70, 30:50] = True  # the path, under the center column
    grid[40:70, 60:65] = True  # a separate patch off to the side

    rows = trace_path(grid, geometry)

    assert [row.grid_y for row in rows] == list(range(69, 39, -1))
    assert all((row.left_cell, row.right_cell) == (30, 49) for row in rows)
    assert not any(row.left_clipped or row.right_clipped for row in rows)


def test_trace_flags_sides_touching_the_image_content_border() -> None:
    geometry = portrait_geometry()
    content_columns = numpy.flatnonzero(geometry.content_cells(GRID_SIZE, GRID_SIZE).any(axis=0))
    first_column, last_column = int(content_columns[0]), int(content_columns[-1])
    grid = numpy.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid[70:80, first_column:45] = True  # runs off the left edge of the image, not of the grid

    rows = trace_path(grid, geometry)

    assert rows and all(row.left_clipped and not row.right_clipped for row in rows)
    assert all(row.left_x == 0.0 for row in rows)
    assert first_column > 0 and last_column < GRID_SIZE - 1


def test_trace_counts_a_run_one_cell_short_of_the_border_as_clipped() -> None:
    geometry = letterbox(Image.new("RGB", (1920, 1080)), 320, 320)[1]
    grid = numpy.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid[50:62, 30:GRID_SIZE - 1] = True  # stops one cell before the last image column

    rows = trace_path(grid, geometry)

    assert rows and all(row.right_clipped and not row.left_clipped for row in rows)


def test_trace_stops_after_too_many_empty_rows() -> None:
    geometry = letterbox(Image.new("RGB", (1920, 1080)), 320, 320)[1]
    grid = numpy.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    grid[70:75, 35:45] = True
    grid[60:66, 35:45] = True  # four empty rows in between, more than the two tolerated

    rows = trace_path(grid, geometry)

    assert min(row.grid_y for row in rows) == 70


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="model file not in this checkout")
def test_real_model_finds_the_sidewalk_in_a_repository_photo() -> None:
    photo_path = REPOSITORY_ROOT / "artifacts" / "images" / "IMG_2316.JPG"
    if not photo_path.exists():
        pytest.skip("test photo not in this checkout")
    image = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")

    result = PathDetector().detect_path(image)

    assert result.top_score >= 0.25, result.debug_info
    assert result.grid_mask.shape == (80, 80)
    assert result.rows, "a detection should trace at least one row"
    assert all(0.0 <= row.left_x <= row.right_x <= 1.0 for row in result.rows)
