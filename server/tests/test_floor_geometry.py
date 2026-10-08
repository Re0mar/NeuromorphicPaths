"""
Covers the floor points, their projection, and which floor cells the camera saw, on a level floor
with known intrinsics.

Expected pixels are worked out by hand in each test's comment, never by calling the function under
test, so a wrong axis or a swapped sign shows as a wrong number rather than as agreement with itself.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.sinks.floor_geometry import (
    MIN_FLOOR_SCATTER_METERS,
    NEAR_PLANE_METERS,
    clip_segment_to_near_plane,
    clip_to_near_plane,
    floor_hidden_mask,
    floor_point,
    floor_scatter_meters,
    floor_seen_mask,
    project_points,
)
from nav.types import DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, Pose
from synthetic_depth import clean_scene, level_floor_depth, with_box_on_level_floor

# Camera 1.6 m above a level floor. y points down in the camera frame, so the floor is y = 1.6.
LEVEL_FLOOR = Plane(np.array([0.0, -1.0, 0.0]), 1.6)
INTRINSICS = np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])


def test_a_point_on_a_known_floor_projects_to_the_pixel_it_should() -> None:
    # 2 m ahead and 0.5 m right is (0.5, 1.6, 2.0) in the camera frame.
    # Column 100 * 0.5 / 2 + 64 = 89. Row 100 * 1.6 / 2 + 48 = 128.
    point = floor_point(LEVEL_FLOOR, 2.0, 0.5)
    column, row, in_front = project_points(point, INTRINSICS)

    assert point == pytest.approx([0.5, 1.6, 2.0])
    assert bool(in_front)
    assert float(column) == pytest.approx(89.0)
    assert float(row) == pytest.approx(128.0)


def test_floor_points_vectorize_the_same_as_one_at_a_time() -> None:
    forward = np.array([1.0, 2.5, 4.0])
    lateral = np.array([-0.3, 0.0, 0.7])

    together = floor_point(LEVEL_FLOOR, forward, lateral)

    for index in range(3):
        assert together[index] == pytest.approx(floor_point(LEVEL_FLOOR, forward[index], lateral[index]))


def test_a_point_behind_the_camera_is_marked_not_in_front() -> None:
    # The same call carries a point ahead, so "not in front" is the point's and not the function's.
    points = np.array([[0.0, 1.6, -1.0], [0.0, 1.6, 2.0]])

    column, row, in_front = project_points(points, INTRINSICS)

    assert list(in_front) == [False, True]
    assert np.isnan(column[0]) and np.isnan(row[0])
    assert np.isfinite(column[1]) and np.isfinite(row[1])


def test_a_polygon_straddling_the_camera_is_cut_at_the_near_plane() -> None:
    # A floor quadrilateral from 0.5 m behind the camera to 1 m ahead. The part ahead survives.
    quadrilateral = np.array([[-0.3, 1.6, -0.5], [-0.3, 1.6, 1.0], [0.3, 1.6, 1.0], [0.3, 1.6, -0.5]])

    clipped = clip_to_near_plane(quadrilateral)

    assert len(clipped) == 4
    assert clipped[:, 2].min() == pytest.approx(NEAR_PLANE_METERS)
    assert clipped[:, 2].max() == pytest.approx(1.0)


def test_a_polygon_wholly_behind_the_camera_is_dropped() -> None:
    behind = np.array([[-0.3, 1.6, -2.0], [-0.3, 1.6, -1.0], [0.3, 1.6, -1.0]])

    assert len(clip_to_near_plane(behind)) == 0


@pytest.mark.parametrize(("start_z", "end_z", "expected_z"), [(-1.0, 3.0, [NEAR_PLANE_METERS, 3.0]), (3.0, -1.0, [3.0, NEAR_PLANE_METERS]), (1.0, 3.0, [1.0, 3.0])])
def test_a_segment_keeps_its_part_beyond_the_near_plane_in_order(start_z: float, end_z: float, expected_z: list[float]) -> None:
    clipped = clip_segment_to_near_plane(np.array([0.0, 1.6, start_z]), np.array([0.0, 1.6, end_z]))

    assert clipped is not None
    assert clipped[:, 2] == pytest.approx(expected_z)


def test_a_segment_wholly_behind_the_camera_is_dropped() -> None:
    assert clip_segment_to_near_plane(np.array([0.0, 1.6, -2.0]), np.array([0.0, 1.6, -1.0])) is None


TIMES = np.arange(39) * 0.1
GRID = np.linspace(-3.0, 3.0, 61)


def _view(intrinsics: np.ndarray = INTRINSICS, floor: Plane = LEVEL_FLOOR) -> DebugView:
    """A 128 by 96 frame, no obstacles. Only the intrinsics, the image size and the floor matter to the mask."""
    frame = DepthFrame(0.0, np.full((96, 128), 2.0, dtype=np.float32), intrinsics, Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False), None, None)
    return DebugView(frame, ObstacleSet(0.0, (), 0), floor, FloorSource.FITTED, 1.4, 0.3, 1.47)


def test_a_cell_straight_ahead_in_view_is_seen_and_one_far_to_the_side_is_not() -> None:
    # Step 30 is 4.2 m ahead. Straight ahead lands at row 100 * 1.6 / 4.2 + 48 = 86, inside the
    # 96 rows. 3 m to the right lands at column 100 * 3 / 4.2 + 64 = 135, past the 128-pixel edge.
    mask = floor_seen_mask(_view(), TIMES, GRID)

    assert mask[30, 30]
    assert not mask[30, 60]


def test_the_floor_under_the_walker_is_unseen_when_the_camera_looks_ahead() -> None:
    # Step 0 is the walker's own feet, at the camera's plane, so it projects nowhere.
    # Step 5 is 0.7 m ahead, row 100 * 1.6 / 0.7 + 48 = 277, below the 96-row image.
    mask = floor_seen_mask(_view(), TIMES, GRID)

    assert not mask[0].any()
    assert not mask[5, 30]


def test_widening_the_field_of_view_turns_unseen_side_cells_to_seen() -> None:
    # The mask follows the frame's own intrinsics, which is exactly what the Neon check depends on.
    narrow = floor_seen_mask(_view(), TIMES, GRID)
    wide = floor_seen_mask(_view(np.array([[30.0, 0.0, 64.0], [0.0, 30.0, 48.0], [0.0, 0.0, 1.0]])), TIMES, GRID)

    assert wide.sum() > narrow.sum()
    # At a 30-pixel focal length, 3 m right of step 30 lands at column 30 * 3 / 4.2 + 64 = 85.
    assert not narrow[30, 60] and wide[30, 60]


def test_the_mask_has_steps_by_cells() -> None:
    assert floor_seen_mask(_view(), TIMES, GRID).shape == (39, 61)


def test_a_floor_point_behind_the_camera_is_never_seen() -> None:
    # A floor plane 2 m behind the camera, facing it. Every floor point has z = -2.
    behind_plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    behind = floor_point(behind_plane, 1.4 * TIMES[:, None], GRID[None, :])
    assert np.allclose(behind[..., 2], -2.0)
    # Without the in-front check some of these would land inside the image, so the check is what
    # keeps them out, not the bounds.
    raw_column = INTRINSICS[0, 0] * behind[..., 0] / behind[..., 2] + INTRINSICS[0, 2]
    raw_row = INTRINSICS[1, 1] * behind[..., 1] / behind[..., 2] + INTRINSICS[1, 2]
    assert np.any((raw_column >= 0) & (raw_column < 128) & (raw_row >= 0) & (raw_row < 96))

    assert not floor_seen_mask(_view(floor=behind_plane), TIMES, GRID).any()


def test_a_point_exactly_on_the_camera_plane_is_not_in_front() -> None:
    # The walker's own feet sit at z = 0 on a level floor. Counted as in front, they would divide by
    # zero. A point a centimeter ahead is the control.
    _, _, in_front = project_points(np.array([[0.0, 1.6, 0.0], [0.0, 1.6, 0.01]]), INTRINSICS)

    assert list(in_front) == [False, True]


# The occlusion half. A level camera 1.6 m up with a 100-pixel focal length sees floor only
# below its horizon row, 48. Row v reads the floor at depth 100 * 1.6 / (v - 48), so the image's
# bottom row, 95, is floor 3.40 m out, and nothing nearer is in view.
SCENE = SceneConfig()


def _depth_view(depth: np.ndarray, intrinsics: np.ndarray = INTRINSICS, floor: Plane = LEVEL_FLOOR, source: FloorSource = FloorSource.FITTED) -> DebugView:
    frame = DepthFrame(0.0, depth, intrinsics, Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False), None, None)
    return DebugView(frame, ObstacleSet(0.0, (), 0), floor, source, 1.4, 0.3, 1.47)


def test_floor_behind_a_box_is_hidden_and_floor_beside_and_in_front_of_it_is_not() -> None:
    # Step 30 is 4.2 m out, row 48 + 160 / 4.2 = 86.1. Straight ahead, column 64, is inside the box.
    # The box reads 3.6 m there, which on pixel (64, 86)'s ray is y = 38 * 3.6 / 100 = 1.37, so
    # 0.23 m above the floor, far past 4 times the 1 cm floor scatter.
    # 0.6 m right of step 30 is column 64 + 60 / 4.2 = 78.3, beside the box, and reads floor.
    # Step 24, 3.36 m, is row 95.6, below the box's foot at row 92, and reads floor.
    view = _depth_view(with_box_on_level_floor(level_floor_depth()))

    seen = floor_seen_mask(view, TIMES, GRID)
    hidden = floor_hidden_mask(view, TIMES, GRID, SCENE)

    assert seen[30, 30] and seen[30, 36] and seen[24, 30]
    assert hidden[30, 30]
    assert not hidden[30, 36]
    assert not hidden[24, 30]


def test_the_scatter_is_the_rms_height_of_the_floor_inliers() -> None:
    # Pairs of columns lifted 1 cm and lowered 3 cm. The stride takes every other column, which is
    # why the pattern changes every two. RMS: sqrt((0.01^2 + 0.03^2) / 2) = 0.02236 m. Two sizes,
    # because with one size every average agrees: the mean absolute height here is 0.02, not 0.02236.
    heights = np.where((np.arange(128) // 2) % 2 == 0, 0.01, -0.03)[None, :] * np.ones((96, 1))
    view = _depth_view(level_floor_depth(heights))

    assert floor_scatter_meters(view.frame, view.floor, SCENE) == pytest.approx(0.02236, abs=1e-5)


def test_points_beyond_the_inlier_distance_do_not_count_toward_the_scatter() -> None:
    # A box that starts 10 cm up, so none of it is within the fit's 5 cm of the floor. It hides
    # some floor, and the floor left in view still reads 2 cm off.
    heights = np.where((np.arange(128) // 2) % 2 == 0, 0.02, -0.02)[None, :] * np.ones((96, 1))
    view = _depth_view(with_box_on_level_floor(level_floor_depth(heights), bottom_meters=0.1))

    assert floor_scatter_meters(view.frame, view.floor, SCENE) == pytest.approx(0.02, abs=1e-5)


@pytest.mark.parametrize("source", list(FloorSource))
def test_every_floor_source_gets_the_same_scatter_and_mask(source: FloorSource) -> None:
    # On the Pixel the floor is supplied on most frames, and any route can carry the last floor
    # over, so the rule can't depend on which one produced the floor. Every member, so a source added
    # later is checked without anyone remembering to.
    depth = with_box_on_level_floor(level_floor_depth())
    fitted = _depth_view(depth, source=FloorSource.FITTED)
    view = _depth_view(depth, source=source)

    assert floor_scatter_meters(view.frame, view.floor, SCENE) == floor_scatter_meters(fitted.frame, fitted.floor, SCENE)
    assert np.array_equal(floor_hidden_mask(view, TIMES, GRID, SCENE), floor_hidden_mask(fitted, TIMES, GRID, SCENE))


def test_a_quarter_turned_frame_hides_the_same_cells() -> None:
    # The Pixel's depth arrives a quarter turn from how the phone is held. np.rot90 puts old pixel
    # (u, v) at new (v, 127 - u), which is the camera turned so that x' = y and y' = -x. The matrix
    # follows: the new center column is the old center row, 48, and the new center row is
    # 127 - 64 = 63. The floor's normal (0, -1, 0) turns to (-1, 0, 0).
    depth = with_box_on_level_floor(level_floor_depth())
    turned = _depth_view(
        np.rot90(depth).copy(),
        np.array([[100.0, 0.0, 48.0], [0.0, 100.0, 63.0], [0.0, 0.0, 1.0]]),
        Plane(np.array([-1.0, 0.0, 0.0]), 1.6),
    )

    upright = floor_hidden_mask(_depth_view(depth), TIMES, GRID, SCENE)

    assert upright.any()
    assert np.array_equal(floor_hidden_mask(turned, TIMES, GRID, SCENE), upright)


def test_a_pitched_camera_hides_the_floor_behind_a_box_and_not_beside_or_before_it() -> None:
    # Every other fixture here has the floor's normal along a camera axis. A worn camera looks down,
    # so real floors are oblique to every axis. clean_scene: 1.6 m up, pitched 20 degrees down, a box
    # whose front face stands 2.75 m out, from 0.25 to 0.75 m right and from the floor to 1.0 m up.
    # Step 30, 4.2 m out, 0.5 m right: the ray to it crosses the face plane 0.5 * 2.75 / 4.2 = 0.33 m
    # right and 1.6 * (1 - 2.75 / 4.2) = 0.55 m up, inside the face, so hidden.
    # The same distance 0.5 m left crosses at -0.33 m, outside the face, so floor.
    # Step 15, 2.1 m out, is nearer than the face, so floor.
    scene = clean_scene()
    view = _depth_view(scene.depth_meters, scene.intrinsics, scene.floor_plane_camera)
    behind, beside, before = (30, 35), (30, 25), (15, 35)

    seen = floor_seen_mask(view, TIMES, GRID)
    hidden = floor_hidden_mask(view, TIMES, GRID, SCENE)

    assert GRID[35] == pytest.approx(0.5) and GRID[25] == pytest.approx(-0.5)
    assert seen[behind] and seen[beside] and seen[before]
    assert hidden[behind]
    assert not hidden[beside]
    assert not hidden[before]


def test_open_floor_with_noise_is_almost_never_hidden() -> None:
    # Noise of 2 cm in height on every reading, with no box. The rule's cutoff is 4 measured
    # scatters, which a bell curve passes about three times in a hundred thousand readings.
    heights = np.random.default_rng(0).normal(0.0, 0.02, (96, 128))
    view = _depth_view(level_floor_depth(heights))

    seen = floor_seen_mask(view, TIMES, GRID)
    hidden = floor_hidden_mask(view, TIMES, GRID, SCENE)

    assert seen.sum() > 500
    assert hidden.sum() / seen.sum() < 0.01


@pytest.mark.parametrize("reading", [np.nan, 0.0, 40.0])
def test_a_seen_cell_with_no_depth_reading_is_hidden(reading: float) -> None:
    # Not a number, zero, and past the scene's 30 m. None of them is a reading the scene would use,
    # so nobody checked that floor. Step 30 straight ahead is pixel (64, 86). The cell 0.3 m to its
    # right, pixel (71, 86), keeps its floor reading and is the control.
    depth = level_floor_depth()
    depth[86, 64] = reading
    view = _depth_view(depth)

    hidden = floor_hidden_mask(view, TIMES, GRID, SCENE)

    assert hidden[30, 30]
    assert not hidden[30, 33]


def test_a_reading_beyond_the_floor_does_not_hide_its_cell() -> None:
    # A pit: pixel (64, 86) reads 6 m where the floor is 160 / 38 = 4.21 m. On its ray that point is
    # 1.6 * (1 - 6 / 4.21) = 0.68 m below the floor. Holes are not this mask's question.
    depth = level_floor_depth()
    depth[86, 64] = 6.0

    assert not floor_hidden_mask(_depth_view(depth), TIMES, GRID, SCENE)[30, 30]


def test_a_cell_is_never_hidden_where_it_is_unseen() -> None:
    noisy = level_floor_depth(np.random.default_rng(1).normal(0.0, 0.02, (96, 128)))
    holed = level_floor_depth()
    holed[60:96:3, ::3] = np.nan
    for depth in (with_box_on_level_floor(level_floor_depth()), noisy, holed):
        view = _depth_view(depth)
        seen = floor_seen_mask(view, TIMES, GRID)
        hidden = floor_hidden_mask(view, TIMES, GRID, SCENE)

        # Every fixture has unseen cells, so the check below has something to fail on.
        assert (~seen).any()
        assert not (hidden & ~seen).any()


def test_with_too_few_floor_inliers_the_scatter_is_the_inlier_distance() -> None:
    # A wall 2 m out filling the image: every reading is 0.66 m or more above the floor, so nothing
    # is within the fit's 5 cm and the camera saw no floor to measure.
    view = _depth_view(np.full((96, 128), 2.0, dtype=np.float32))

    assert floor_scatter_meters(view.frame, view.floor, SCENE) == SCENE.floor_ransac_distance_meters


def test_an_exact_floor_hides_nothing_on_open_floor() -> None:
    # A perfect floor scatters 0, so the lower bound of 1 cm sets the cutoff at 4 cm. Steps run to
    # 7.9 s, 11.06 m out, where a floor cell lands at row 48 + 160 / 11.06 = 62.5. Rounding a row by
    # half a pixel there moves the floor's depth by about 0.4 m, so a reading put on the cell's own
    # ray instead of its pixel's would stand up to 1.6 * 0.5 / 14.5 = 5.5 cm off the floor.
    far_times = np.arange(80) * 0.1
    view = _depth_view(level_floor_depth())

    assert floor_scatter_meters(view.frame, view.floor, SCENE) == MIN_FLOOR_SCATTER_METERS
    seen = floor_seen_mask(view, far_times, GRID)
    assert seen[70:].any()
    assert not floor_hidden_mask(view, far_times, GRID, SCENE).any()


def test_a_frame_with_no_seen_cells_hides_nothing() -> None:
    # The floor plane 2 m behind the camera, as in the field-of-view test above, so no cell is seen
    # and there is no pixel to read. The mask is all false rather than an error.
    behind_plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    view = _depth_view(level_floor_depth(), floor=behind_plane)

    assert not floor_seen_mask(view, TIMES, GRID).any()
    assert not floor_hidden_mask(view, TIMES, GRID, SCENE).any()


def test_a_cell_at_the_image_edge_reads_the_last_pixel() -> None:
    # 2.6 m right at 4.08 m out lands at column 64 + 260 / 4.08 = 127.7, inside the 128 columns, so
    # seen, and it rounds to 128, one past the last. Row 48 + 160 / 4.08 = 87.2. The last column's
    # pixel (127, 87) reads the floor, 160 / 39 = 4.10 m, so the cell is not hidden.
    times = np.array([4.08 / 1.4])
    grid = np.array([2.6])
    view = _depth_view(level_floor_depth())

    assert floor_seen_mask(view, times, grid)[0, 0]
    assert not floor_hidden_mask(view, times, grid, SCENE)[0, 0]
