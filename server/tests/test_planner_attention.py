"""
The planner's half of the gaze switch: whether the eyes are on the danger, and whether the alarm is heard.

Nothing here changes the path. Every test that turns the switch on also checks the alarm itself is
untouched, and one checks the path is identical with the switch on and off.
"""

# Standard library imports
import dataclasses
import time
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.config import GoalMode, RunConfig, SinkKind, SourceKind
from nav.planner.alarm import nearest_corridor_point
from nav.planner.attention import alarm_audible, check_attention_config
from nav.planner.config import PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.runtime.loop import run
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.scene.pipeline import ScenePipeline
from nav.sources.config import LoggedConfig
from nav.types import CameraOutline, DepthFrame, GazeSamples, ObstaclePoint, ObstacleSet, Pose
from nav.walker import WalkerConfig
from synthetic_depth import clean_scene

PLANNER = PlannerConfig()
SWITCH_ON = dataclasses.replace(PLANNER, gaze_silences_alarm=True)
WALKER = WalkerConfig()
FOCAL = 500.0
INTRINSICS = np.array([[FOCAL, 0.0, 320.0], [0.0, FOCAL, 240.0], [0.0, 0.0, 1.0]])
# Clearance over the planner's 1.4 m/s: 0.70 m is 0.5 s to contact, under the 0.7 s alarm and over the
# 0.35 s floor. 0.30 m is 0.21 s, under the floor.
ABOVE_FLOOR = 0.70
UNDER_FLOOR = 0.30
ON_THE_DANGER = GazeSamples(np.array([[320.0, 240.0]]), np.zeros(1), INTRINSICS)
# 30 degrees right of it.
OFF_THE_DANGER = GazeSamples(np.array([[320.0 + FOCAL * np.tan(np.radians(30.0)), 240.0]]), np.zeros(1), INTRINSICS)


def _danger(clearance: float, lateral: float = 0.0) -> ObstaclePoint:
    """A group in the corridor, its outline straight out of the camera, where ON_THE_DANGER looks."""
    return ObstaclePoint(lateral, clearance + 0.35, 1, clearance, 0.05, None, None, False, np.array([0.0, 0.0, 1.0]), CameraOutline(0.0, 0.0, 0.0, 0.0))


def _at(timestamp: float, *points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(timestamp, points, len(points))


# *******************************************
# The pure decision
# *******************************************


def test_alarm_audible_is_the_alarm_unless_every_condition_holds() -> None:
    assert alarm_audible(True, True, 0.5, False, PLANNER)
    assert not alarm_audible(True, True, 0.5, True, PLANNER)
    assert alarm_audible(True, False, 0.5, True, PLANNER)
    assert alarm_audible(True, True, 0.2, True, PLANNER)
    assert not alarm_audible(False, False, None, False, PLANNER)


def test_the_floor_itself_is_above_it() -> None:
    assert not alarm_audible(True, True, PLANNER.attention_floor_seconds, True, PLANNER)
    assert alarm_audible(True, True, PLANNER.attention_floor_seconds - 1e-9, True, PLANNER)


def test_a_held_alarm_with_an_empty_corridor_counts_as_above_the_floor() -> None:
    assert not alarm_audible(True, True, None, True, PLANNER)


# *******************************************
# Through the pipeline
# *******************************************


def test_alarm_audible_follows_alarm_with_the_switch_off() -> None:
    path = PlannerPipeline(PLANNER, WALKER).plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER)

    assert path.alarm and path.danger_noticed
    assert path.alarm_audible


def test_a_noticed_danger_above_the_floor_is_quiet_with_the_switch_on() -> None:
    path = PlannerPipeline(SWITCH_ON, WALKER).plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER)

    assert path.alarm and path.danger_noticed
    assert not path.alarm_audible


def test_a_noticed_danger_under_the_floor_still_sounds() -> None:
    # The same fixture both sides of the floor, so the quiet half proves the gaze arrived and only
    # the floor is holding on the other.
    above = PlannerPipeline(SWITCH_ON, WALKER).plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER)
    under = PlannerPipeline(SWITCH_ON, WALKER).plan(_at(0.0, _danger(UNDER_FLOOR)), gaze=ON_THE_DANGER)

    assert not above.alarm_audible
    assert under.danger_noticed and under.alarm_audible


def test_gaze_clearly_off_the_danger_never_silences_the_alarm() -> None:
    path = PlannerPipeline(SWITCH_ON, WALKER).plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=OFF_THE_DANGER)

    assert not path.danger_noticed
    assert path.alarm_audible


def test_no_gaze_never_silences_the_alarm() -> None:
    path = PlannerPipeline(SWITCH_ON, WALKER).plan(_at(0.0, _danger(ABOVE_FLOOR)))

    assert not path.danger_noticed
    assert path.alarm_audible


def test_the_danger_ear_stays_up_for_a_noticed_danger_with_the_switch_on() -> None:
    # Off to the right, so the alarm pans right and, when heard, drops the right ear to the floor.
    danger = _danger(ABOVE_FLOOR, lateral=0.2)
    heard = PlannerPipeline(PLANNER, WALKER).plan(_at(0.0, danger), gaze=ON_THE_DANGER)
    quiet = PlannerPipeline(SWITCH_ON, WALKER).plan(_at(0.0, danger), gaze=ON_THE_DANGER)

    assert heard.ear_gain_right == pytest.approx(PLANNER.far_ear_floor_gain)
    assert quiet.ear_gain_right > PLANNER.far_ear_floor_gain
    # The side is still sent, only not acted on.
    assert quiet.alarm_pan == heard.alarm_pan and quiet.alarm_pan > 0.0


def test_a_held_alarm_with_an_empty_corridor_is_quiet_when_noticed() -> None:
    planner = PlannerPipeline(SWITCH_ON, WALKER)
    planner.plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER)

    held = planner.plan(_at(0.2))

    assert held.alarm, "the hold keeps the alarm up for its 0.5 s"
    assert held.danger_noticed and not held.alarm_audible


def test_the_switch_can_be_set_after_construction() -> None:
    planner = PlannerPipeline(PLANNER, WALKER)
    assert planner.plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER).alarm_audible

    planner.set_gaze_silences_alarm(True)

    assert planner.gaze_silences_alarm
    assert not planner.plan(_at(0.1, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER).alarm_audible


def test_the_path_never_changes_with_the_switch_on() -> None:
    obstacles = _at(0.0, _danger(ABOVE_FLOOR), _danger(1.5, lateral=0.6))
    off = PlannerPipeline(PLANNER, WALKER).plan(obstacles, gaze=ON_THE_DANGER)
    on = PlannerPipeline(SWITCH_ON, WALKER).plan(obstacles, gaze=ON_THE_DANGER)

    assert np.array_equal(off.lateral_offsets_meters, on.lateral_offsets_meters)
    assert off.lookahead_heading_radians == on.lookahead_heading_radians
    assert off.alarm == on.alarm and off.avoidance_surprise_bits == on.avoidance_surprise_bits


def test_the_same_frame_planned_twice_updates_once() -> None:
    # The second plan of the same stamp carries no gaze. Counted again it would be a new frame with no
    # look and could clear the answer. It keeps the first one instead.
    planner = PlannerPipeline(dataclasses.replace(SWITCH_ON, attention_credit_seconds=0.0), WALKER)
    obstacles = _at(5.0, _danger(ABOVE_FLOOR))
    planner.plan(obstacles, gaze=ON_THE_DANGER)

    again = planner.plan(obstacles)

    assert again.danger_noticed and not again.alarm_audible


def test_a_look_carries_to_the_next_frame_in_credit() -> None:
    planner = PlannerPipeline(SWITCH_ON, WALKER)
    planner.plan(_at(0.0, _danger(ABOVE_FLOOR)), gaze=ON_THE_DANGER)
    # 0.1 s later the danger is 0.14 m nearer at walking pace, and the eyes have moved off it.
    nearer = dataclasses.replace(_danger(ABOVE_FLOOR - 0.14), forward_meters=ABOVE_FLOOR + 0.35 - 0.14)

    path = planner.plan(_at(0.1, nearer), gaze=OFF_THE_DANGER)

    assert path.danger_noticed and not path.alarm_audible


# *******************************************
# Refused at construction
# *******************************************


@pytest.mark.parametrize(
    "field, value, named",
    [
        ("attention_floor_seconds", 0.7, "attention_floor_seconds"),
        ("attention_floor_seconds", 0.0, "attention_floor_seconds"),
        ("attention_window_seconds", 0.25, "attention_window_seconds"),
        ("attention_window_seconds", -0.01, "attention_window_seconds"),
        ("attention_link_meters", 0.0, "attention_link_meters"),
        ("attention_cap_meters", float("nan"), "attention_cap_meters"),
        ("attention_dwell_seconds", -0.1, "attention_dwell_seconds"),
        ("attention_tolerance_degrees", 0.0, "attention tolerance"),
        ("attention_credit_seconds", -1.0, "credit_seconds"),
        ("attention_match_meters", 0.0, "match_meters"),
    ],
)
def test_an_attention_setting_out_of_range_stops_the_planner_at_construction(field: str, value: float, named: str) -> None:
    with pytest.raises(ValueError, match=named):
        PlannerPipeline(dataclasses.replace(PLANNER, **{field: value}), WALKER)


def test_the_shipped_attention_settings_pass_their_check() -> None:
    check_attention_config(PLANNER)


# *******************************************
# Through run(), from a frame log
# *******************************************


def _scene_frames(count: int) -> list[DepthFrame]:
    """A box 1.5 m ahead in the corridor, the gaze on the middle of what the scene calls its danger."""
    scene = clean_scene(box_lateral_meters=0.0, box_forward_meters=1.5, box_height_meters=1.0, box_half_width_meters=0.25)
    plain = DepthFrame(0.0, scene.depth_meters, scene.intrinsics, Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False), scene.floor_plane_camera, None)
    danger = nearest_corridor_point(ScenePipeline(SceneConfig(), WALKER).process(plain), PLANNER)
    assert danger is not None and danger.camera_outline is not None, "the fixture needs a danger to look at"
    outline = danger.camera_outline
    across = (outline.left_tangent + outline.right_tangent) / 2.0
    down = (outline.top_tangent + outline.bottom_tangent) / 2.0
    pixel = np.array([[scene.intrinsics[0, 2] + scene.intrinsics[0, 0] * across, scene.intrinsics[1, 2] + scene.intrinsics[1, 1] * down]])
    return [
        dataclasses.replace(plain, timestamp_seconds=0.1 * index, gaze_pixels=pixel, gaze_sample_offsets_seconds=np.zeros(1))
        for index in range(count)
    ]


def _published_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planner_config: PlannerConfig) -> list:
    import nav.runtime.loop as loop_module

    paths = []

    # The real class, not whatever an earlier call left patched into the loop module.
    class RecordingPlanner(PlannerPipeline):
        def plan(self, *args, **kwargs):
            path = super().plan(*args, **kwargs)
            paths.append(path)
            return path

    monkeypatch.setattr(loop_module, "PlannerPipeline", RecordingPlanner)
    log_dir = tmp_path / "log"
    if not log_dir.exists():
        frames = _scene_frames(5)

        class ListSource:
            def frames(self):
                yield from frames

            def close(self) -> None:
                pass

        list(RecordingTap(ListSource(), log_dir).frames())
    config = RunConfig(
        source_kind=SourceKind.LOGGED,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        logged=LoggedConfig(log_dir=str(log_dir)),
        planner=planner_config,
    )
    assert run(config) == 0
    return paths


def test_a_logged_frame_with_gaze_on_the_danger_is_published_with_alarm_audible_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    switched_off = _published_paths(tmp_path, monkeypatch, PLANNER)
    switched_on = _published_paths(tmp_path, monkeypatch, SWITCH_ON)

    # Positive control first: the logged frames raise the alarm and the gaze reaches the planner.
    assert any(path.alarm and path.danger_noticed for path in switched_off)
    assert all(path.alarm_audible == path.alarm for path in switched_off)
    assert any(path.alarm and not path.alarm_audible for path in switched_on)


# *******************************************
# The run's switch, through run()
# *******************************************


def _run_capturing_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: list[DepthFrame], flip_after: int) -> tuple[list, int]:
    """
    Run the frames through run(), turning the run's gaze switch on after flip_after plans.

    The frames reach the loop as given rather than through a frame log, which keeps no source
    generation, so a change of stretch in them really rebuilds the planner.

    :return: The published paths in order, and how many planners the run built.
    """
    import nav.runtime.loop as loop_module

    switches = []
    paths = []
    planners_built = [0]

    class CapturedSwitch(loop_module.GazeSilencingSwitch):
        def __init__(self, enabled: bool) -> None:
            super().__init__(enabled)
            switches.append(self)

    class RecordingPlanner(PlannerPipeline):
        def __init__(self, *args, **kwargs) -> None:
            planners_built[0] += 1
            super().__init__(*args, **kwargs)

        def plan(self, *args, **kwargs):
            path = super().plan(*args, **kwargs)
            paths.append(path)
            if len(paths) == flip_after:
                # What a page's ask does, on the sink's thread in a live run.
                switches[0].request(True)
            return path

    monkeypatch.setattr(loop_module, "GazeSilencingSwitch", CapturedSwitch)
    monkeypatch.setattr(loop_module, "PlannerPipeline", RecordingPlanner)

    class ListSource:
        def frames(self):
            # One at a time, each once the one before is planned. The loop's worker keeps only the
            # newest frame, so a list handed over at once would be planned as its last frame alone.
            for index, frame in enumerate(frames):
                deadline = time.monotonic() + 5.0
                while len(paths) < index and time.monotonic() < deadline:
                    time.sleep(0.005)
                yield frame

        def close(self) -> None:
            pass

    real_build = loop_module.build_source_and_sink

    def with_these_frames(config, **kwargs):
        # The run's own sink and switch wiring, with the frames handed straight to the loop.
        source, sink = real_build(config, **kwargs)
        source.close()
        return ListSource(), sink

    monkeypatch.setattr(loop_module, "build_source_and_sink", with_these_frames)
    log_dir = tmp_path / "unused_log"
    log_dir.mkdir()
    config = RunConfig(source_kind=SourceKind.LOGGED, sink_kinds=(SinkKind.NONE,), goal_mode=GoalMode.AHEAD, logged=LoggedConfig(log_dir=str(log_dir)), planner=PLANNER)
    assert run(config) == 0
    assert len(switches) == 1, "one switch for the whole run"
    return paths, planners_built[0]


def test_a_switch_request_changes_the_next_published_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _run_capturing_switch(tmp_path, monkeypatch, _scene_frames(4), flip_after=2)

    assert all(path.alarm and path.danger_noticed for path in paths), "the fixture keeps the alarm up and the eyes on it"
    assert [path.alarm_audible for path in paths] == [True, True, False, False]


def test_the_switch_state_survives_a_new_stretch_of_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Flipped on in the first stretch. The third frame starts a second stretch, which rebuilds the planner.
    frames = [dataclasses.replace(frame, source_generation=0 if index < 2 else 1) for index, frame in enumerate(_scene_frames(4))]

    paths, planners_built = _run_capturing_switch(tmp_path, monkeypatch, frames, flip_after=1)

    assert planners_built == 2, "the second stretch must really rebuild the planner for this to test anything"
    assert [path.alarm_audible for path in paths[2:]] == [False, False], "the rebuilt planner reads the run's switch"
