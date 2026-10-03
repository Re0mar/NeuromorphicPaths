"""
The replay command, through its real entry point, on recordings written by the real tap.

No recording from frame_logs/ is ever needed, and nothing here skips when one is absent. The synthetic
recordings supply their floor, so the scene never fits one by RANSAC and every run repeats exactly.
"""

# Standard library imports
import ast
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
import nav.evaluation
from nav.evaluation.__main__ import main
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.overrides import OverrideRefused, apply_overrides
from nav.evaluation.replay import (
    NAV_DIR,
    clone_state,
    evaluate_walk,
    planner_pass,
    scene_pass,
    segment_bounds,
)
from nav.evaluation.track import floor_heading_radians
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.scene.config import SceneConfig
from nav.scene.pipeline import CAMERA_FORWARD
from nav.scene.transform import rotation_matrix_from_quaternion_wxyz
from nav.walker import WalkerConfig
from synthetic_walks import landscape_camera_rotation, quaternion_wxyz, ramp, walking_pose, write_recording

TWO_TURNS = lambda time: ramp(6.0, 2.0, 90.0)(time) + ramp(14.0, 2.0, -90.0)(time)
WALK_SECONDS = 22.0


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    """A 22 s walk with a right 90 and a left 90, written once for the module."""
    return write_recording(tmp_path_factory.mktemp("walk") / "two_turns", TWO_TURNS, WALK_SECONDS)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("replay_cache")


def run(arguments: list[str], capsys) -> tuple[int, str, str]:
    code = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# *******************************************
# Through the real entry point
# *******************************************


def test_replay_through_real_planner_finds_both_turns(recording: Path, capsys) -> None:
    code, out, _ = run([recording, "--passes", "1"], capsys)
    assert code == 0
    assert "Turns: 2 " in out
    assert "right +90, left -90" in out
    # The arrows it scored are the planner's own: plan the same frames with a fresh planner directly.
    scene = scene_pass(recording, SceneConfig(), WalkerConfig(), None)
    frames = planner_pass(scene, PlannerConfig(), WalkerConfig(), GoalMode.AHEAD)
    planner = PlannerPipeline(PlannerConfig(), WalkerConfig())
    direct = [planner.plan(row.obstacles, 0.0, GoalMode.AHEAD, row.gaze_ground_point).first_heading_radians for row in scene.planned]
    assert [frame.arrow_radians for frame in frames] == pytest.approx(direct)
    assert len(frames) == len(scene.planned) > 100


def test_replay_calls_plan_like_the_loop() -> None:
    # The replay re-composes the loop's per-frame call. If the loop's call changes shape, this fails.
    def plan_calls(path: Path) -> list[ast.Call]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "plan"]

    (loop_call,) = plan_calls(NAV_DIR / "runtime" / "loop.py")
    (replay_call,) = plan_calls(NAV_DIR / "evaluation" / "replay.py")
    assert len(loop_call.args) == len(replay_call.args) == 4
    assert not loop_call.keywords and not replay_call.keywords
    assert isinstance(loop_call.args[1], ast.Constant) and loop_call.args[1].value == 0.0
    assert isinstance(replay_call.args[1], ast.Constant) and replay_call.args[1].value == 0.0


def test_cached_pass_matches_a_cold_pass(recording: Path, tmp_path: Path) -> None:
    cold = scene_pass(recording, SceneConfig(), WalkerConfig(), None)
    written = scene_pass(recording, SceneConfig(), WalkerConfig(), tmp_path)
    read = scene_pass(recording, SceneConfig(), WalkerConfig(), tmp_path)
    assert written.cache_state.startswith("miss") and read.cache_state.startswith("hit")
    arrows = [[frame.arrow_radians for frame in planner_pass(each, PlannerConfig(), WalkerConfig(), GoalMode.AHEAD)] for each in (cold, read)]
    assert arrows[0] == arrows[1]
    np.testing.assert_array_equal(cold.pose_positions_world, read.pose_positions_world)


def test_cache_hit_then_miss_on_scene_change(recording: Path, tmp_path: Path) -> None:
    scene_pass(recording, SceneConfig(), WalkerConfig(), tmp_path)
    assert scene_pass(recording, SceneConfig(), WalkerConfig(), tmp_path).cache_state.startswith("hit")
    changed = dataclasses.replace(SceneConfig(), floor_max_tilt_degrees=40.0)
    assert scene_pass(recording, changed, WalkerConfig(), tmp_path).cache_state.startswith("miss")


def test_cache_payload_loads_in_a_fresh_process(recording: Path, tmp_path: Path) -> None:
    scene_pass(recording, SceneConfig(), WalkerConfig(), tmp_path)
    (cache_file,) = tmp_path.glob("*.pkl")
    script = (
        "import pickle, sys\n"
        "times, positions, rows, refused = pickle.load(open(sys.argv[1], 'rb'))\n"
        "print(len(rows), type(rows[0][1]).__name__)\n"
    )
    completed = subprocess.run([sys.executable, "-c", script, str(cache_file)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    count, kind = completed.stdout.split()
    assert int(count) > 100 and kind == "ObstacleSet"


def test_planner_override_moves_only_arrow_scores(recording: Path, cache_dir: Path) -> None:
    scene_config = SceneConfig()
    plain = evaluate_walk(recording, scene_config, "test", PlannerConfig(), EvaluationConfig(), 1, cache_dir)
    tuned_config = dataclasses.replace(PlannerConfig(), lateral_kinetic_weight=2.0)
    tuned = evaluate_walk(recording, scene_config, "test", tuned_config, EvaluationConfig(), 1, cache_dir)
    assert [segment.turns for segment in tuned.segments] == [segment.turns for segment in plain.segments]
    np.testing.assert_array_equal(tuned.spread.change_samples_degrees, plain.spread.change_samples_degrees)
    # The positive control: the override did reach the planner. Without this the test above could
    # pass with the override ignored.
    scene = scene_pass(recording, scene_config, WalkerConfig(), cache_dir)
    plain_arrows = [frame.arrow_radians for frame in planner_pass(scene, PlannerConfig(), WalkerConfig(), GoalMode.AHEAD)]
    tuned_arrows = [frame.arrow_radians for frame in planner_pass(scene, tuned_config, WalkerConfig(), GoalMode.AHEAD)]
    assert plain_arrows != tuned_arrows


def test_spread_only_needs_no_scene_config(tmp_path: Path, capsys) -> None:
    bare = write_recording(tmp_path / "bare", lambda time: 0.0, WALK_SECONDS, write_run_config=False)
    code, out, err = run(["--spread-only", bare, "--eval-set", "min_straight_seconds=5"], capsys)
    # No RecordingRefused for the missing run_config.json. The perfect synthetic walk then fails the
    # dead-band rule, which is the rules working, not the run failing.
    assert "RecordingRefused" not in err
    assert code in (0, 3)
    assert "99th percentile" in out


def test_threshold_rules_are_checked_by_spread_only(recording: Path, capsys) -> None:
    code, out, _ = run(["--spread-only", recording, "--eval-set", "min_straight_seconds=9999"], capsys)
    assert code == 3
    assert "the rule needs 9999 s" in out


def test_markdown_out_is_lf(recording: Path, cache_dir: Path, tmp_path: Path, capsys) -> None:
    out_file = tmp_path / "report.md"
    code, _, _ = run([recording, "--cached", "--cache-dir", cache_dir, "--out", out_file], capsys)
    assert code == 0
    content = out_file.read_bytes()
    assert b"\r" not in content
    assert content.startswith(b"# The planner's arrow")


# *******************************************
# Pieces
# *******************************************


def test_walking_pose_turns_the_way_it_says() -> None:
    straight = walking_pose(np.zeros(3), 0.0, 20.0)
    turned = walking_pose(np.zeros(3), np.radians(10.0), 20.0)
    headings = [
        float(floor_heading_radians(rotation_matrix_from_quaternion_wxyz(pose.orientation) @ CAMERA_FORWARD))
        for pose in (straight, turned)
    ]
    assert np.degrees(headings[1] - headings[0]) == pytest.approx(10.0, abs=1e-6)


def test_quaternion_round_trips() -> None:
    rotation = landscape_camera_rotation(np.radians(37.0), 20.0)
    np.testing.assert_allclose(rotation_matrix_from_quaternion_wxyz(quaternion_wxyz(rotation)), rotation, atol=1e-12)


def test_segment_bounds_at_the_edges() -> None:
    config = EvaluationConfig()
    assert len(segment_bounds(np.array([0.0, 1.0, 6.0, 7.0]), config)) == 1
    assert len(segment_bounds(np.array([0.0, 1.0, 6.1, 7.0]), config)) == 2
    exactly, short = (0.0, 20.0), (0.0, 19.9)
    from nav.evaluation.replay import long_enough
    assert long_enough(exactly, config) and not long_enough(short, config)


def test_overrides_parse_each_type() -> None:
    scene, names = apply_overrides(SceneConfig(), ["depth_stride=3", "floor_max_tilt_degrees=inf"])
    assert scene.depth_stride == 3 and scene.floor_max_tilt_degrees == float("inf")
    assert names == {"depth_stride", "floor_max_tilt_degrees"}
    planner, _ = apply_overrides(PlannerConfig(), ["predict_motion=true"])
    assert planner.predict_motion is True


def test_clone_state_names_the_code_it_ran() -> None:
    probe = NAV_DIR / "evaluation" / "_clone_state_probe.py"
    before = clone_state()
    try:
        probe.write_bytes(b"# probe\n")
        during = clone_state()
    finally:
        probe.unlink()
    assert before != during
    assert "nav code" in during


# *******************************************
# Refusals
# *******************************************


def test_missing_run_config_is_refused_without_flag(recording: Path, cache_dir: Path, tmp_path: Path, capsys) -> None:
    bare = tmp_path / "bare"
    shutil.copytree(recording, bare)
    (bare / "run_config.json").unlink()
    code, _, err = run([bare, "--cached", "--cache-dir", cache_dir], capsys)
    assert code == 1
    assert "RecordingRefused" in err and "--scene-defaults" in err
    code, _, _ = run([bare, "--scene-defaults", "--cached", "--cache-dir", cache_dir], capsys)
    assert code == 0


def test_scene_defaults_with_run_config_is_refused(recording: Path, capsys) -> None:
    code, _, err = run([recording, "--scene-defaults"], capsys)
    assert code == 1
    assert "would contradict" in err


def test_missing_scene_field_is_refused_unless_set(tmp_path: Path, cache_dir: Path, capsys) -> None:
    block = dataclasses.asdict(SceneConfig())
    del block["floor_max_offset_meters"]
    old = write_recording(tmp_path / "old", TWO_TURNS, WALK_SECONDS, run_config_scene=block)
    code, _, err = run([old, "--cached", "--cache-dir", cache_dir], capsys)
    assert code == 1
    assert "floor_max_offset_meters" in err
    code, _, _ = run([old, "--scene-set", "floor_max_offset_meters=inf", "--cached", "--cache-dir", cache_dir], capsys)
    assert code == 0


def test_unknown_scene_key_is_refused(tmp_path: Path, capsys) -> None:
    block = dataclasses.asdict(SceneConfig()) | {"floor_from_the_future": 1.0}
    future = write_recording(tmp_path / "future", TWO_TURNS, 3.0, run_config_scene=block)
    code, _, err = run([future], capsys)
    assert code == 1
    assert "floor_from_the_future" in err


def test_non_aligned_pose_is_refused(tmp_path: Path, capsys) -> None:
    tilted = write_recording(tmp_path / "tilted", TWO_TURNS, 3.0, gravity_aligned_at=lambda time: time < 1.0)
    code, _, err = run([tilted, "--passes", "1"], capsys)
    assert code == 1
    assert "gravity aligned" in err and "1.000 s" in err


def test_missing_index_is_refused(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run([empty, "--scene-defaults", "--passes", "1"], capsys)
    assert code == 1
    assert "FileNotFoundError" in err and "index.jsonl" in err


def test_corrupt_index_is_refused(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "index.jsonl").write_bytes(b"{not json\n")
    code, _, err = run([broken, "--scene-defaults", "--passes", "1"], capsys)
    assert code == 1
    assert "FrameDecodeError" in err


def test_bad_override_is_refused(recording: Path, capsys) -> None:
    for override, named in (("no_such_field=1", "no_such_field"), ("lateral_kinetic_weight=lots", "lateral_kinetic_weight"), ("lateral_kinetic_weight", "FIELD=VALUE")):
        code, _, err = run([recording, "--set", override, "--passes", "1"], capsys)
        assert code == 1, override
        assert "OverrideRefused" in err and named in err


def test_config_refusing_an_override_is_refused_by_name(recording: Path, capsys) -> None:
    code, _, err = run([recording, "--eval-set", "turn_window_seconds=-1", "--passes", "1"], capsys)
    assert code == 1
    assert "OverrideRefused" in err and "turn_window_seconds" in err
    with pytest.raises(OverrideRefused, match="turn_window_seconds"):
        apply_overrides(EvaluationConfig(), ["turn_window_seconds=-1"])


def test_internal_value_error_is_not_reported_as_refused(recording: Path, monkeypatch) -> None:
    # A defect inside the evaluation must keep its traceback, not read as a refused recording.
    def broken(*arguments, **keywords):
        raise ValueError("a defect inside the evaluation")

    monkeypatch.setattr("nav.evaluation.__main__.evaluate_walk", broken)
    with pytest.raises(ValueError, match="a defect inside the evaluation"):
        main([str(recording), "--passes", "1"])


def test_cached_with_several_passes_is_a_usage_error(recording: Path) -> None:
    with pytest.raises(SystemExit) as exited:
        main([str(recording), "--cached", "--passes", "3"])
    assert exited.value.code == 2


def test_no_long_segment_exits_3(tmp_path: Path, capsys) -> None:
    short = write_recording(tmp_path / "short", TWO_TURNS, 5.0)
    code, _, err = run([short, "--passes", "1"], capsys)
    assert code == 3
    assert "no walk had a segment" in err


def test_no_turns_exits_3(tmp_path: Path, capsys) -> None:
    straight = write_recording(tmp_path / "straight", lambda time: 0.0, WALK_SECONDS)
    code, out, err = run([straight, "--passes", "1"], capsys)
    assert code == 3
    assert "no turn was found" in err
    assert "Turns: 0 " in out


def test_refused_scene_frames_are_counted(tmp_path: Path, capsys) -> None:
    late_floor = write_recording(tmp_path / "late_floor", TWO_TURNS, WALK_SECONDS, floorless_until_seconds=1.0)
    scene = scene_pass(late_floor, SceneConfig(), WalkerConfig(), None)
    assert scene.refused_frames == 10
    assert len(scene.planned) == scene.pose_times_seconds.shape[0] - 10
    code, out, _ = run([late_floor, "--passes", "1"], capsys)
    assert code == 0
    assert "refused by the scene 10" in out
