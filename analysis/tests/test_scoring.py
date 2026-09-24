# Third party imports
import numpy
import pytest
from PIL import Image

# Local package imports
from neuromorphicpaths_analysis.detector import PathResult, letterbox, trace_path
from neuromorphicpaths_analysis.scoring import edge_errors, grid_iou, label_to_grid, score_frame

GRID_SHAPE = (80, 80)


def landscape_geometry():
    return letterbox(Image.new("RGB", (1920, 1080)), 320, 320)[1]


def label_band(left: int, right: int, top: int = 540) -> numpy.ndarray:
    """A path label covering columns left..right of a 1920x1080 frame, from row top down."""
    label = numpy.zeros((1080, 1920), dtype=bool)
    label[top:, left:right] = True
    return label


def result_for(grid: numpy.ndarray, geometry) -> PathResult:
    return PathResult(geometry, grid, trace_path(grid, geometry), 0.9, None, "")


def test_full_label_fills_exactly_the_content_cells() -> None:
    geometry = landscape_geometry()
    grid = label_to_grid(numpy.ones((1080, 1920), dtype=bool), geometry, GRID_SHAPE)
    assert numpy.array_equal(grid, geometry.content_cells(*GRID_SHAPE))


def test_label_size_must_match_the_frame() -> None:
    with pytest.raises(ValueError):
        label_to_grid(numpy.ones((100, 100), dtype=bool), landscape_geometry(), GRID_SHAPE)


def test_identical_masks_score_perfectly() -> None:
    geometry = landscape_geometry()
    truth = label_to_grid(label_band(480, 1440), geometry, GRID_SHAPE)

    score = score_frame(result_for(truth.copy(), geometry), label_band(480, 1440))

    assert score.iou == 1.0
    assert score.edge_error_mean == 0.0
    assert score.sides_compared > 0 and score.rows_missed == 0


def test_shifted_edge_reports_the_shift_in_path_widths() -> None:
    geometry = landscape_geometry()
    # Cells are 24 px of the original frame. A 960 px wide path is 40 cells. Moving the right
    # edge out by 4 cells is 96 px, a tenth of the path width.
    predicted = label_to_grid(label_band(480, 1440 + 96), geometry, GRID_SHAPE)

    score = score_frame(result_for(predicted, geometry), label_band(480, 1440))

    assert score.edge_error_max == pytest.approx(0.1, abs=1e-6)
    assert score.edge_error_mean == pytest.approx(0.05, abs=1e-6)


def test_clipped_sides_are_not_compared() -> None:
    geometry = landscape_geometry()
    truth_rows = trace_path(label_to_grid(label_band(0, 960), geometry, GRID_SHAPE), geometry)
    predicted_rows = trace_path(label_to_grid(label_band(0, 960), geometry, GRID_SHAPE), geometry)

    errors, _ = edge_errors(predicted_rows, truth_rows)

    # Only the right side counts. The left one is the frame border in both.
    assert len(errors) == len([row for row in truth_rows if row.y >= 0.5])


def test_iou_ignores_padding_cells() -> None:
    content = landscape_geometry().content_cells(*GRID_SHAPE)
    predicted = numpy.ones(GRID_SHAPE, dtype=bool)  # also claims the gray padding
    assert grid_iou(predicted, content.copy(), content) == 1.0
