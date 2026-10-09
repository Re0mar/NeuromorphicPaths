"""Covers the web sink's messages as plain functions: the plan view's contents and the kind envelope."""

# Standard library imports
import base64
import dataclasses
import json

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.config import SceneConfig
from nav.sinks.floor_geometry import floor_hidden_mask, floor_seen_mask
from nav.sinks.path_style import BORDER_OPACITY, GROUP_RGB, SURPRISE_HIGH_RGB, WALL_RGB, path_color_rgb, path_fill_opacity
from nav.sinks.web_messages import (
    OBSTACLE_KEYS,
    PLAN_VIEW_KEYS,
    WebMessageKind,
    plan_view_message,
    web_text_message,
)
from nav.types import DebugView, DepthFrame, FloorSource, ObstaclePoint, ObstacleSet, Plane, PlannedPath, Pose
from synthetic_depth import intrinsics, level_floor_depth, with_box_on_level_floor

STEPS = 4
GRID = np.array([-0.1, 0.0, 0.1])
SCENE = SceneConfig()


def _path(information: float = 0.4, avoidance: float = 0.3) -> PlannedPath:
    return PlannedPath(
        timestamp_seconds=2.0,
        times_seconds=np.arange(STEPS) * 0.1,
        lateral_offsets_meters=np.array([0.0, 0.0, 0.1, 0.1]),
        lookahead_heading_radians=0.1,
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
    return DebugView(frame, obstacles, Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30, 1.47)


def _obstacle(lateral: float, forward: float, is_wall: bool) -> ObstaclePoint:
    return ObstaclePoint(lateral, forward, 1, 0.5, 0.1, None, None, is_wall, np.zeros(3))


def _field() -> np.ndarray:
    return np.arange(STEPS * len(GRID), dtype=np.float64).reshape(STEPS, len(GRID))


def test_the_plan_view_carries_every_field_the_page_reads() -> None:
    message = plan_view_message(_path(), _field(), GRID, _view((_obstacle(0.2, 1.0, False),)), SCENE)

    assert set(message) == set(PLAN_VIEW_KEYS)
    assert set(message["obstacles"][0]) == set(OBSTACLE_KEYS)


def test_the_plan_view_field_has_steps_by_cells() -> None:
    message = plan_view_message(_path(), _field(), GRID, _view(), SCENE)

    assert np.array(message["field"]).shape == (STEPS, len(GRID))
    assert message["field"][0] == pytest.approx([0.0, 1.0, 2.0]), "row 0 is now"


def test_obstacles_keep_their_wall_flag_and_order() -> None:
    message = plan_view_message(_path(), _field(), GRID, _view((_obstacle(0.2, 1.0, False), _obstacle(-0.5, 2.0, True))), SCENE)

    assert [entry["is_wall"] for entry in message["obstacles"]] == [False, True]
    assert message["obstacles"][1]["lateral_meters"] == pytest.approx(-0.5)


def test_the_plan_view_color_and_opacity_come_from_path_style() -> None:
    # A red point no constant in the code holds, so the color can only match by reading the view's.
    view = dataclasses.replace(_view(), path_red_from_bits=2.5)
    message = plan_view_message(_path(information=0.4, avoidance=0.3), _field(), GRID, view, SCENE)

    assert message["path_color_rgb"] == list(path_color_rgb(0.3, 2.5))
    assert message["path_fill_opacity"] == pytest.approx(path_fill_opacity(0.4))
    assert message["path_border_opacity"] == pytest.approx(BORDER_OPACITY)
    assert message["group_color_rgb"] == list(GROUP_RGB)
    assert message["wall_color_rgb"] == list(WALL_RGB)


def test_the_path_width_is_the_bodys_width() -> None:
    assert plan_view_message(_path(), _field(), GRID, _view(), SCENE)["path_width_meters"] == pytest.approx(0.6)


def test_the_envelope_adds_the_kind_and_survives_a_json_round_trip() -> None:
    text = web_text_message(WebMessageKind.PLAN_VIEW, plan_view_message(_path(), _field(), GRID, _view(), SCENE))

    decoded = json.loads(text)
    assert decoded["kind"] == WebMessageKind.PLAN_VIEW.value
    assert set(decoded) == set(PLAN_VIEW_KEYS) | {"kind"}


def test_a_field_that_does_not_match_the_path_and_grid_is_refused() -> None:
    with pytest.raises(ValueError, match=r"field is \(5, 3\), the path and grid need \(4, 3\)"):
        plan_view_message(_path(), np.zeros((STEPS + 1, len(GRID))), GRID, _view(), SCENE)


def test_a_non_finite_field_value_is_refused_at_serialization() -> None:
    field = _field()
    field[1, 1] = np.inf
    message = plan_view_message(_path(), field, GRID, _view(), SCENE)

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        web_text_message(WebMessageKind.PLAN_VIEW, message)


def test_a_body_that_already_has_a_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="the body already carries a kind, 'path'"):
        web_text_message(WebMessageKind.PLAN_VIEW, {"kind": "path"})


def test_the_plan_view_carries_the_floor_mask_for_every_cell() -> None:
    view = _view()
    message = plan_view_message(_path(), _field(), GRID, view, SCENE)

    mask = np.array(message["floor_seen"])
    assert mask.shape == (STEPS, len(GRID))
    assert mask.dtype == bool
    assert np.array_equal(mask, floor_seen_mask(view, _path().times_seconds, GRID))


def test_a_debug_view_has_no_default_body_width() -> None:
    # A default would let a construction site that forgot the width draw the path a plausible width.
    view = _view()
    with pytest.raises(TypeError, match="body_half_width_meters"):
        DebugView(view.frame, view.obstacles, view.floor, view.floor_source, view.walking_speed_mps)


def test_a_debug_view_has_no_default_red_point() -> None:
    # A default would be a second copy of the alarm's threshold, free to drift from the planner's.
    view = _view()
    with pytest.raises(TypeError, match="path_red_from_bits"):
        DebugView(view.frame, view.obstacles, view.floor, view.floor_source, view.walking_speed_mps, view.body_half_width_meters)


def test_the_path_is_fully_red_at_the_views_red_point_and_not_before() -> None:
    # 2.5 bits, not the shipped 1.47, so a sink that ignored the view and used the shipped value
    # would already be fully red at 0.9 of it and fail the second assertion.
    view = dataclasses.replace(_view(), path_red_from_bits=2.5)
    at_red = plan_view_message(_path(avoidance=view.path_red_from_bits), _field(), GRID, view, SCENE)
    below_red = plan_view_message(_path(avoidance=view.path_red_from_bits * 0.9), _field(), GRID, view, SCENE)

    assert np.abs(np.array(at_red["path_color_rgb"]) - np.array(SURPRISE_HIGH_RGB)).max() <= 1
    assert np.abs(np.array(below_red["path_color_rgb"]) - np.array(SURPRISE_HIGH_RGB)).max() > 1


def test_the_video_stream_message_carries_the_codec_and_base64_parameter_sets() -> None:
    from nav.sinks.web_messages import video_stream_message, video_unavailable_message
    from nav.sources.scene_video import VideoDescription

    sps, pps = b"\x00\x00\x00\x01\x67\x42\x80\x1f", b"\x00\x00\x00\x01\x68\xce\x06\xf2"
    message = json.loads(web_text_message(WebMessageKind.VIDEO_STREAM, video_stream_message(VideoDescription("avc1.42801f", (sps, pps)))))

    assert message["kind"] == "video_stream"
    assert message["codec"] == "avc1.42801f"
    assert [base64.b64decode(item) for item in message["parameter_sets"]] == [sps, pps]
    assert json.loads(web_text_message(WebMessageKind.VIDEO_UNAVAILABLE, video_unavailable_message("no video"))) == {"kind": "video_unavailable", "reason": "no video"}


# The hidden-floor key needs cells that land in the image, which the 4 by 4 view above has none of.
# These use the level floor from synthetic_depth: a camera 1.6 m up with a 100-pixel focal length
# and a 128 by 96 image, and the planner's own shape of 39 steps by 61 cells.
FLOOR_TIMES = np.arange(39) * 0.1
FLOOR_GRID = np.linspace(-3.0, 3.0, 61)


def _floor_path() -> PlannedPath:
    return PlannedPath(2.0, FLOOR_TIMES, np.zeros(len(FLOOR_TIMES)), 0.0, False, 1.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)


def _floor_view(depth: np.ndarray) -> DebugView:
    frame = DepthFrame(2.0, depth, intrinsics(), Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False), None, None)
    return DebugView(frame, ObstacleSet(2.0, (), 0), Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30, 1.47)


def _floor_message(depth: np.ndarray, scene: SceneConfig = SCENE) -> dict:
    field = np.zeros((len(FLOOR_TIMES), len(FLOOR_GRID)))
    return plan_view_message(_floor_path(), field, FLOOR_GRID, _floor_view(depth), scene)


def test_the_plan_view_carries_the_hidden_floor_mask_for_every_cell() -> None:
    depth = with_box_on_level_floor(level_floor_depth())

    mask = np.array(_floor_message(depth)["floor_hidden"])

    assert mask.shape == (len(FLOOR_TIMES), len(FLOOR_GRID))
    assert mask.dtype == bool
    # The box hides something, so equality below compares a mask that isn't all false.
    assert mask.any()
    assert np.array_equal(mask, floor_hidden_mask(_floor_view(depth), FLOOR_TIMES, FLOOR_GRID, SCENE))


def test_no_cell_in_the_plan_view_is_hidden_where_it_is_unseen() -> None:
    # Read back off the wire, after JSON, the way the page gets it.
    depth = with_box_on_level_floor(level_floor_depth())
    decoded = json.loads(web_text_message(WebMessageKind.PLAN_VIEW, _floor_message(depth)))

    seen = np.array(decoded["floor_seen"])
    hidden = np.array(decoded["floor_hidden"])

    assert (~seen).any() and hidden.any()
    assert not (hidden & ~seen).any()


def test_the_plan_view_uses_the_scene_config_it_is_given() -> None:
    # Step 30 straight ahead is pixel (64, 86), floor 4.21 m out. A reading of 25 m there is beyond
    # the floor, so under the default 30 m range it hides nothing. Under a 20 m range it is no reading
    # at all, so the cell is hidden. Only the config passed in can make that difference.
    depth = level_floor_depth()
    depth[86, 64] = 25.0

    default_range = np.array(_floor_message(depth)["floor_hidden"])
    narrowed_range = np.array(_floor_message(depth, dataclasses.replace(SCENE, max_depth_meters=20.0))["floor_hidden"])

    assert not default_range[30, 30]
    assert narrowed_range[30, 30]
