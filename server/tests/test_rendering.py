"""
Covers the three renderers and the PNG encoder without opening a window or a browser.

The render functions are pure and are what can go wrong. Showing the result is OpenCV's or the
browser's job, checked by eye in a run.
"""

# Third party imports
import cv2
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.pipeline import ScenePipeline
from nav.sinks.rendering import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    COLOR_INVALID_DEPTH,
    DEPTH_VIEW_TARGET_WIDTH,
    FIELD_INSET_SCALE,
    encode_png,
    render_arrow,
    render_depth_view,
    render_field,
)
from nav.types import DebugView, DepthFrame, FloorSource, ObstaclePoint, ObstacleSet, Plane, PlannedPath, Pose
from nav.walker import WalkerConfig
from synthetic_depth import HEIGHT, WIDTH, clean_scene

GRID = np.linspace(-3.0, 3.0, 61)
STEPS = 39
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
CONFIG = SceneConfig()
WALKER = WalkerConfig()
WALKING_SPEED = 1.4
BODY_HALF_WIDTH = 0.30


def _path(offsets: np.ndarray, alarm: bool = False) -> PlannedPath:
    return PlannedPath(0.0, np.arange(len(offsets)) * 0.1, offsets, 0.0, alarm, 1.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _one_step_path() -> PlannedPath:
    """A path whose single step is at the walker's own feet, which projects behind the camera and draws nothing."""
    return PlannedPath(0.0, np.array([0.0]), np.array([0.0]), 0.0, False, 0.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _frame(depth: np.ndarray, intrinsics: np.ndarray) -> DepthFrame:
    return DepthFrame(0.0, depth, intrinsics, Pose(IDENTITY, None, False), None, None)


def _scene_view(obstacles: ObstacleSet | None = None) -> DebugView:
    """The synthetic scene through the real scene layer, so the obstacles carry real camera points."""
    scene = clean_scene(box_lateral_meters=0.5, box_forward_meters=3.0, box_height_meters=1.0, box_half_width_meters=0.1)
    frame = _frame(scene.depth_meters, scene.intrinsics)
    found = obstacles if obstacles is not None else ScenePipeline(CONFIG, WALKER).process(frame)
    return DebugView(frame, found, scene.floor_plane_camera, FloorSource.FITTED, WALKING_SPEED, BODY_HALF_WIDTH)


def _obstacle(camera_point: np.ndarray, clearance: float = 1.0, is_wall: bool = False) -> ObstaclePoint:
    return ObstaclePoint(0.5, 2.9, 1, clearance, 0.01, None, None, is_wall, camera_point)


def _group_at(camera_point: np.ndarray, **kwargs) -> ObstacleSet:
    return ObstacleSet(0.0, (_obstacle(camera_point, **kwargs),), 1)


def _projected(camera_point: np.ndarray, intrinsics: np.ndarray, scale: int) -> tuple[int, int]:
    x, y, z = camera_point
    return int((intrinsics[0, 0] * x / z + intrinsics[0, 2]) * scale), int((intrinsics[1, 1] * y / z + intrinsics[1, 2]) * scale)


# The text line at the top names the group count, so two views that differ only in their groups
# differ there too. Comparisons of the drawing itself start below it.
TEXT_BAND_ROWS = 40


def _below_text(image: np.ndarray) -> np.ndarray:
    return image[TEXT_BAND_ROWS:]


# ------------------------------------------------------------------ the arrow and the field


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


# ------------------------------------------------------------------ the depth view


def test_the_depth_view_has_the_scaled_size_and_a_valid_pixel_is_colored() -> None:
    view = _scene_view()

    image = render_depth_view(view, _path(np.zeros(STEPS)))

    scale = DEPTH_VIEW_TARGET_WIDTH // WIDTH
    assert scale == 5
    assert image.shape == (HEIGHT * scale, WIDTH * scale, 3)
    # Floor, lower left, away from the path down the middle and the text at the top.
    assert tuple(image[90 * scale, 20 * scale]) != COLOR_INVALID_DEPTH


def test_near_depth_is_brighter_than_far_depth_and_gray() -> None:
    # Lower in the image is nearer floor. Column 20 is clear of the path, which is one step here anyway.
    image = render_depth_view(_scene_view(ObstacleSet(0.0, (), 0)), _one_step_path())
    scale = DEPTH_VIEW_TARGET_WIDTH // WIDTH
    near = image[90 * scale, 20 * scale]
    far = image[70 * scale, 20 * scale]

    assert near[0] == near[1] == near[2], "depth is drawn in gray"
    assert int(near[0]) > int(far[0])


def test_invalid_depth_stays_distinguishable_from_the_farthest_gray() -> None:
    # Far depth is near black. A gray marker for no reading would read as far away.
    assert len(set(COLOR_INVALID_DEPTH)) > 1, "the invalid color must not be a gray"


def test_invalid_depth_pixels_are_drawn_in_their_own_color() -> None:
    scene = clean_scene()
    depth = scene.depth_meters.copy()
    depth[40:50, 0:10] = np.nan
    depth[40:50, 118:128] = 0.0
    view = DebugView(_frame(depth, scene.intrinsics), ObstacleSet(0.0, (), 0), scene.floor_plane_camera, FloorSource.FITTED, WALKING_SPEED, BODY_HALF_WIDTH)

    image = render_depth_view(view, _one_step_path())

    scale = DEPTH_VIEW_TARGET_WIDTH // WIDTH
    assert tuple(image[45 * scale, 5 * scale]) == COLOR_INVALID_DEPTH
    assert tuple(image[45 * scale, 123 * scale]) == COLOR_INVALID_DEPTH


def test_a_group_is_drawn_at_its_projected_pixel() -> None:
    camera_point = np.array([0.5, 0.2, 2.9])
    with_group = render_depth_view(_scene_view(_group_at(camera_point)), _one_step_path())
    without = render_depth_view(_scene_view(ObstacleSet(0.0, (), 0)), _one_step_path())
    scale = DEPTH_VIEW_TARGET_WIDTH // WIDTH
    column, row = _projected(camera_point, clean_scene().intrinsics, scale)

    changed = np.any(_below_text(with_group) != _below_text(without), axis=2)
    rows, columns = np.nonzero(changed)
    rows = rows + TEXT_BAND_ROWS

    assert changed.any(), "the group drew nothing"
    distances = np.hypot(rows - row, columns - column)
    assert distances.max() < 50, "the ring is not where the point projects"
    assert distances.min() < 15


def test_a_wall_and_a_post_are_drawn_in_different_colors() -> None:
    camera_point = np.array([0.5, 0.2, 2.9])
    wall = render_depth_view(_scene_view(_group_at(camera_point, is_wall=True)), _one_step_path())
    post = render_depth_view(_scene_view(_group_at(camera_point, is_wall=False)), _one_step_path())

    assert np.any(wall != post)


def test_a_group_behind_the_camera_is_skipped_not_drawn() -> None:
    behind = render_depth_view(_scene_view(_group_at(np.array([0.5, 0.2, -1.0]))), _one_step_path())
    without = render_depth_view(_scene_view(ObstacleSet(0.0, (), 0)), _one_step_path())

    assert np.array_equal(_below_text(behind), _below_text(without))


def _floor_view(focal_pixels: float = 30.0) -> DebugView:
    """A level floor 1.6 m under a camera with no depth readings, so the ribbon is all that changes."""
    intrinsics = np.array([[focal_pixels, 0.0, WIDTH / 2.0], [0.0, focal_pixels, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
    frame = _frame(np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32), intrinsics)
    level_floor = Plane(np.array([0.0, -1.0, 0.0]), 1.6)
    return DebugView(frame, ObstacleSet(0.0, (), 0), level_floor, FloorSource.FITTED, WALKING_SPEED, BODY_HALF_WIDTH)


def _styled_path(offsets: np.ndarray, information: float = 0.0, avoidance: float = 0.0) -> PlannedPath:
    return PlannedPath(0.0, np.arange(len(offsets)) * 0.1, offsets, 0.0, False, 1.0, scene_information_bits=information, avoidance_surprise_bits=avoidance)


def _pixel_on_floor(forward: float, lateral: float, focal_pixels: float = 30.0) -> tuple[int, int]:
    """Row and column, scaled, where a level-floor point lands. Worked out from the pinhole model, not the renderer."""
    scale = DEPTH_VIEW_TARGET_WIDTH // WIDTH
    return int(round((focal_pixels * 1.6 / forward + HEIGHT / 2.0) * scale)), int(round((focal_pixels * lateral / forward + WIDTH / 2.0) * scale))


def test_the_ribbon_is_drawn_the_bodys_width_on_the_floor() -> None:
    # Step 20 is 2.8 m ahead. Its edges at 0.3 m to either side land 16 px either side of the
    # middle at this scale, and each border reaches 2 px past its edge.
    view = _floor_view()
    drawn = render_depth_view(view, _styled_path(np.zeros(STEPS)))
    blank = render_depth_view(view, _one_step_path())
    row, _ = _pixel_on_floor(2.8, 0.0)
    _, left_column = _pixel_on_floor(2.8, -BODY_HALF_WIDTH)
    _, right_column = _pixel_on_floor(2.8, BODY_HALF_WIDTH)

    changed_columns = np.nonzero(np.any(drawn[row] != blank[row], axis=1))[0]

    assert abs(int(changed_columns.min()) - left_column) <= 3
    assert abs(int(changed_columns.max()) - right_column) <= 3


def test_the_fill_fades_to_nothing_at_the_far_end_while_the_borders_stay() -> None:
    view = _floor_view()
    drawn = render_depth_view(view, _styled_path(np.zeros(STEPS))).astype(int)
    blank = render_depth_view(view, _one_step_path()).astype(int)
    near_row, middle = _pixel_on_floor(1.4, 0.0)
    far_row, _ = _pixel_on_floor(5.25, 0.0)
    _, far_border = _pixel_on_floor(5.25, BODY_HALF_WIDTH)

    assert np.abs(drawn[near_row, middle] - blank[near_row, middle]).max() > 50, "the fill is strong near the walker"
    assert np.abs(drawn[far_row, middle] - blank[far_row, middle]).max() <= 3, "the fill has gone by the far end"
    border_window = np.abs(drawn[far_row - 2 : far_row + 3, far_border - 3 : far_border + 4] - blank[far_row - 2 : far_row + 3, far_border - 3 : far_border + 4])
    assert border_window.max() > 30, "the border is still there at the far end"


def test_the_ribbons_color_follows_the_avoidance_surprise() -> None:
    view = _floor_view()
    row, column = _pixel_on_floor(1.4, 0.0)

    calm = render_depth_view(view, _styled_path(np.zeros(STEPS), avoidance=0.0))[row, column].astype(int)
    urgent = render_depth_view(view, _styled_path(np.zeros(STEPS), avoidance=0.72))[row, column].astype(int)

    # BGR. Blue at zero bits, red at 0.72.
    assert calm[0] > urgent[0]
    assert urgent[2] > calm[2]


def test_the_fill_is_more_opaque_with_more_scene_information() -> None:
    view = _floor_view()
    blank = render_depth_view(view, _one_step_path()).astype(int)
    row, column = _pixel_on_floor(1.4, 0.0)

    faint = np.abs(render_depth_view(view, _styled_path(np.zeros(STEPS), information=0.0)).astype(int)[row, column] - blank[row, column]).sum()
    solid = np.abs(render_depth_view(view, _styled_path(np.zeros(STEPS), information=1.0)).astype(int)[row, column] - blank[row, column]).sum()

    assert solid > faint


def test_a_ribbon_crossing_the_image_edge_is_clipped_not_dropped() -> None:
    # A narrower field of view, and a path that drifts 3 m right, so its far end leaves the image
    # on the right. Clipped, it runs right up to the edge.
    view = _floor_view(focal_pixels=120.0)
    drawn = render_depth_view(view, _styled_path(np.linspace(0.0, 3.0, STEPS)))
    blank = render_depth_view(view, _one_step_path())

    changed = np.any(drawn != blank, axis=2)

    assert changed[TEXT_BAND_ROWS:, -1].any(), "the ribbon must reach the right edge"


def test_a_path_step_outside_the_image_is_skipped() -> None:
    view = _scene_view(ObstacleSet(0.0, (), 0))
    far_to_the_side = _path(np.full(STEPS, 50.0))

    image = render_depth_view(view, far_to_the_side)

    assert np.array_equal(image, render_depth_view(view, _one_step_path()))


def test_a_frame_with_no_groups_renders_without_error() -> None:
    image = render_depth_view(_scene_view(ObstacleSet(0.0, (), 0)), _path(np.zeros(STEPS)))

    assert image.shape[2] == 3


def test_the_floor_source_and_the_alarm_are_written_on_the_view() -> None:
    # The text is drawn, not asserted by OCR. What can be asserted is that the two words change
    # the picture, which is what a line that is never drawn would fail.
    view = _scene_view(ObstacleSet(0.0, (), 0))
    supplied = DebugView(view.frame, view.obstacles, view.floor, FloorSource.SUPPLIED, WALKING_SPEED, BODY_HALF_WIDTH)

    assert np.any(render_depth_view(view, _one_step_path()) != render_depth_view(supplied, _one_step_path()))
    assert np.any(render_depth_view(view, _one_step_path()) != render_depth_view(view, PlannedPath(0.0, np.array([0.0]), np.array([0.0]), 0.0, True, 0.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)))


# ------------------------------------------------------------------ the PNG encoder


def test_encode_png_round_trips_through_imdecode() -> None:
    image = render_depth_view(_scene_view(), _path(np.zeros(STEPS)))

    decoded = cv2.imdecode(np.frombuffer(encode_png(image), dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape == image.shape
    assert np.array_equal(decoded, image), "PNG is lossless, the bytes must come back exactly"


def test_encode_png_refuses_an_empty_image() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_png(np.zeros((0, 0, 3), dtype=np.uint8))
