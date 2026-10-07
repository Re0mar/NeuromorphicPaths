"""
Covers the floor points and their projection on a level floor with known intrinsics.

Expected pixels are worked out by hand in each test's comment, never by calling the function under
test, so a wrong axis or a swapped sign shows as a wrong number rather than as agreement with itself.
"""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.floor_geometry import (
    NEAR_PLANE_METERS,
    clip_segment_to_near_plane,
    clip_to_near_plane,
    floor_point,
    floor_seen_mask,
    project_points,
)
from nav.types import DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, Pose

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
