"""
Covers the debug window's drawing without opening a window.

The two render functions are pure and are what can go wrong. Opening the window is OpenCV's job
and is checked by hand in the replay run.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.debug_window import CANVAS_HEIGHT, CANVAS_WIDTH, FIELD_INSET_SCALE, render_arrow, render_field
from nav.types import PlannedPath

GRID = np.linspace(-3.0, 3.0, 61)
STEPS = 39


def _path(offsets: np.ndarray, alarm: bool = False) -> PlannedPath:
    return PlannedPath(0.0, np.arange(len(offsets)) * 0.1, offsets, 0.0, alarm, 1.0)


def test_the_arrow_canvas_has_the_declared_size_and_turns_red_on_alarm() -> None:
    clear = render_arrow(0.0, _path(np.zeros(STEPS), alarm=False))
    alarm = render_arrow(0.0, _path(np.zeros(STEPS), alarm=True))

    assert clear.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    # BGR. The alarm canvas has far more red than the clear one.
    assert alarm[..., 2].sum() > clear[..., 2].sum()
    assert clear[..., 1].sum() > alarm[..., 1].sum()


def test_a_straight_path_draws_without_error() -> None:
    # The first draft inferred the grid spacing from the path's own offsets, which is undefined
    # for a straight path. The grid is passed in now, and this is the case that caught it.
    field = np.zeros((STEPS, len(GRID)))

    image = render_field(field, GRID, _path(np.zeros(STEPS)))

    assert image.shape == (STEPS * FIELD_INSET_SCALE, len(GRID) * FIELD_INSET_SCALE, 3)


def test_the_path_is_drawn_at_its_lateral_column() -> None:
    field = np.zeros((STEPS, len(GRID)))
    offsets = np.full(STEPS, 1.5)  # column 45 of 61

    image = render_field(field, GRID, _path(offsets))

    # The path line is white. Find the column with the most white, in the unscaled grid.
    white = np.all(image == 255, axis=2)
    column_scores = white.reshape(STEPS * FIELD_INSET_SCALE, len(GRID), FIELD_INSET_SCALE).any(axis=2).sum(axis=0)
    assert int(np.argmax(column_scores)) == 45


def test_a_grid_that_does_not_match_the_field_is_refused() -> None:
    with pytest.raises(ValueError, match="columns"):
        render_field(np.zeros((STEPS, 10)), GRID, _path(np.zeros(STEPS)))


def test_a_capped_point_does_not_black_out_the_rest() -> None:
    # A gentle gradient across the grid, plus one capped cell. Without the percentile clip the
    # cap would own the whole color range and the gradient would render as one color.
    field = np.tile(np.linspace(0.0, 10.0, len(GRID)), (STEPS, 1))
    field[0, 0] = 2.0e4

    image = render_field(field, GRID, _path(np.zeros(STEPS)))

    colors_across = len(np.unique(image[STEPS, :, :].reshape(-1, 3), axis=0))
    assert colors_across > 10, "the gradient must survive one capped cell"
