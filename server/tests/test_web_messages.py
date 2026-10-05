"""Covers the web sink's messages as plain functions: the plan view's contents and the kind envelope."""

# Standard library imports
import json

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.floor_geometry import floor_seen_mask
from nav.sinks.path_style import BORDER_OPACITY, GROUP_RGB, WALL_RGB, path_color_rgb, path_fill_opacity
from nav.sinks.web_messages import (
    OBSTACLE_KEYS,
    PLAN_VIEW_KEYS,
    WebMessageKind,
    plan_view_message,
    web_text_message,
)
from nav.types import DebugView, DepthFrame, FloorSource, ObstaclePoint, ObstacleSet, Plane, PlannedPath, Pose

STEPS = 4
GRID = np.array([-0.1, 0.0, 0.1])


def _path(information: float = 0.4, avoidance: float = 0.3) -> PlannedPath:
    return PlannedPath(
        timestamp_seconds=2.0,
        times_seconds=np.arange(STEPS) * 0.1,
        lateral_offsets_meters=np.array([0.0, 0.0, 0.1, 0.1]),
        first_heading_radians=0.1,
        alarm=False,
        cumulative_cost_bits=1.5,
        scene_information_bits=information,
        avoidance_surprise_bits=avoidance,
    )


def _view(points: tuple[ObstaclePoint, ...] = ()) -> DebugView:
    frame = DepthFrame(
        timestamp_seconds=2.0,
        depth_meters=np.full((4, 4), 2.0, dtype=np.float32),
        intrinsics=np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False),
        ground_plane=None,
        gaze_pixel=None,
    )
    obstacles = ObstacleSet(2.0, points, len({point.group_id for point in points}))
    return DebugView(frame, obstacles, Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30)


def _obstacle(lateral: float, forward: float, is_wall: bool) -> ObstaclePoint:
    return ObstaclePoint(lateral, forward, 1, 0.5, 0.1, None, None, is_wall, np.zeros(3))


def _field() -> np.ndarray:
    return np.arange(STEPS * len(GRID), dtype=np.float64).reshape(STEPS, len(GRID))


def test_the_plan_view_carries_every_field_the_page_reads() -> None:
    message = plan_view_message(_path(), _field(), GRID, _view((_obstacle(0.2, 1.0, False),)))

    assert set(message) == set(PLAN_VIEW_KEYS)
    assert set(message["obstacles"][0]) == set(OBSTACLE_KEYS)


def test_the_plan_view_field_has_steps_by_cells() -> None:
    message = plan_view_message(_path(), _field(), GRID, _view())

    assert np.array(message["field"]).shape == (STEPS, len(GRID))
    assert message["field"][0] == pytest.approx([0.0, 1.0, 2.0]), "row 0 is now"


def test_obstacles_keep_their_wall_flag_and_order() -> None:
    message = plan_view_message(_path(), _field(), GRID, _view((_obstacle(0.2, 1.0, False), _obstacle(-0.5, 2.0, True))))

    assert [entry["is_wall"] for entry in message["obstacles"]] == [False, True]
    assert message["obstacles"][1]["lateral_meters"] == pytest.approx(-0.5)


def test_the_plan_view_color_and_opacity_come_from_path_style() -> None:
    message = plan_view_message(_path(information=0.4, avoidance=0.3), _field(), GRID, _view())

    assert message["path_color_rgb"] == list(path_color_rgb(0.3))
    assert message["path_fill_opacity"] == pytest.approx(path_fill_opacity(0.4))
    assert message["path_border_opacity"] == pytest.approx(BORDER_OPACITY)
    assert message["group_color_rgb"] == list(GROUP_RGB)
    assert message["wall_color_rgb"] == list(WALL_RGB)


def test_the_path_width_is_the_bodys_width() -> None:
    assert plan_view_message(_path(), _field(), GRID, _view())["path_width_meters"] == pytest.approx(0.6)


def test_the_envelope_adds_the_kind_and_survives_a_json_round_trip() -> None:
    text = web_text_message(WebMessageKind.PLAN_VIEW, plan_view_message(_path(), _field(), GRID, _view()))

    decoded = json.loads(text)
    assert decoded["kind"] == WebMessageKind.PLAN_VIEW.value
    assert set(decoded) == set(PLAN_VIEW_KEYS) | {"kind"}


def test_a_field_that_does_not_match_the_path_and_grid_is_refused() -> None:
    with pytest.raises(ValueError, match=r"field is \(5, 3\), the path and grid need \(4, 3\)"):
        plan_view_message(_path(), np.zeros((STEPS + 1, len(GRID))), GRID, _view())


def test_a_non_finite_field_value_is_refused_at_serialization() -> None:
    field = _field()
    field[1, 1] = np.inf
    message = plan_view_message(_path(), field, GRID, _view())

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        web_text_message(WebMessageKind.PLAN_VIEW, message)


def test_a_body_that_already_has_a_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="the body already carries a kind, 'path'"):
        web_text_message(WebMessageKind.PLAN_VIEW, {"kind": "path"})


def test_the_plan_view_carries_the_floor_mask_for_every_cell() -> None:
    view = _view()
    message = plan_view_message(_path(), _field(), GRID, view)

    mask = np.array(message["floor_seen"])
    assert mask.shape == (STEPS, len(GRID))
    assert mask.dtype == bool
    assert np.array_equal(mask, floor_seen_mask(view, _path().times_seconds, GRID))
