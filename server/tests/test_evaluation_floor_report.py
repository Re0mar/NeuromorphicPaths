"""
`check_planner floor`, on frame logs written by the real tap.

The frames carry no floor of their own, so every floor here comes from the scene's own fit, and the
camera's height is known from the synthetic scene it was cast from. Poses have no position, like the
Neon's, and an orientation that is either gravity aligned or not.
"""

# Standard library imports
import subprocess
import sys
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation.check_planner import EXIT_NOTHING_MEASURED, EXIT_PRINTED, EXIT_REFUSED, main, parse_arguments
from nav.evaluation.floor_report import floor_report, format_floor_report, measured_unit
from nav.evaluation.replay import NAV_DIR, RecordingRefused, UnalignedFrames, scene_pass
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.scene.floor import FloorRefusalCause
from nav.types import DepthFrame, FloorSource, Pose
from nav.walker import WalkerConfig
from synthetic_depth import CAMERA_HEIGHT_METERS, PITCH_DEGREES, clean_scene, degrade_without_floor, pitch_rotation
from synthetic_walks import ListSource, quaternion_wxyz

FRAME_GAP_SECONDS = 0.1
# A level camera's axes (y down, z forward) to a y-up world is a half turn about x. The fixture's
# pitch_rotation maps world to camera, so the pose carries its transpose.
HALF_TURN_ABOUT_X = np.diag([1.0, -1.0, -1.0])
ALIGNED = Pose(
    orientation=quaternion_wxyz(HALF_TURN_ABOUT_X @ pitch_rotation(PITCH_DEGREES).T),
    position=None,
    has_position=False,
    orientation_is_gravity_aligned=True,
)
UNALIGNED = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False, orientation_is_gravity_aligned=False)


def _frame(index: int, camera_height: float = CAMERA_HEIGHT_METERS, floor: bool = True, pose: Pose = ALIGNED) -> DepthFrame:
    scene = clean_scene(camera_height=camera_height)
    depth = scene.depth_meters if floor else degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)
    # No supplied plane, so the floor is always the scene's own fit.
    return DepthFrame(index * FRAME_GAP_SECONDS, depth, scene.intrinsics, pose, None, None)


def _write(log_dir: Path, frames: list[DepthFrame]) -> Path:
    for _ in RecordingTap(ListSource(frames), log_dir).frames():
        pass
    return log_dir


def _report(log_dir: Path, config: SceneConfig = SceneConfig()):
    return floor_report(scene_pass(log_dir, config, WalkerConfig(), None, unaligned_frames=UnalignedFrames.PROCESS))


def _run(arguments: list, capsys) -> tuple[int, str, str]:
    code = main(["floor", *(str(argument) for argument in arguments)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_walk_at_a_known_height_reports_that_height(tmp_path: Path) -> None:
    log_dir = _write(tmp_path / "level", [_frame(index) for index in range(8)])

    report = _report(log_dir)

    assert report.frames == 8
    assert report.by_source[FloorSource.FITTED] == 8
    assert report.heights_meters.size == 8
    assert float(np.median(report.heights_meters)) == pytest.approx(CAMERA_HEIGHT_METERS, abs=0.01)


def test_the_text_names_every_source_and_cause_even_at_zero(tmp_path: Path) -> None:
    log_dir = _write(tmp_path / "level", [_frame(index) for index in range(4)])

    text = format_floor_report("level", _report(log_dir), "defaults")

    for source in FloorSource:
        assert f"  {source.value} " in text
    for cause in FloorRefusalCause:
        assert f"  {cause.value} " in text
    assert f"median {CAMERA_HEIGHT_METERS:.2f} m" in text


def test_previous_floors_are_counted_by_cause_and_left_out_of_the_height(tmp_path: Path) -> None:
    frames = [_frame(index) for index in range(4)] + [_frame(index, floor=False) for index in range(4, 7)]
    log_dir = _write(tmp_path / "floor_then_none", frames)

    report = _report(log_dir)

    assert report.by_source[FloorSource.FITTED] == 4
    assert report.by_source[FloorSource.PREVIOUS] == 3
    assert sum(len(values) for values in report.previous_because.values()) == 3
    # Only the four fitted frames. Counting the three that kept the first floor would repeat its height.
    assert report.heights_meters.size == 4


def test_a_floor_over_the_height_limit_is_refused_as_too_far_by_default(tmp_path: Path) -> None:
    frames = [_frame(index) for index in range(3)] + [_frame(index, camera_height=2.9) for index in range(3, 6)]
    log_dir = _write(tmp_path / "too_high", frames)

    report = _report(log_dir)

    assert report.by_source[FloorSource.PREVIOUS] == 3
    assert len(report.previous_because[FloorRefusalCause.TOO_FAR]) == 3
    assert np.median(report.previous_because[FloorRefusalCause.TOO_FAR]) == pytest.approx(2.9, abs=0.02)


def test_lifting_the_height_limit_fits_the_high_floor(tmp_path: Path, capsys) -> None:
    frames = [_frame(index) for index in range(3)] + [_frame(index, camera_height=2.9) for index in range(3, 6)]
    log_dir = _write(tmp_path / "too_high", frames)

    code, out, _ = _run([log_dir, "--scene-defaults", "--scene-set", "floor_max_offset_meters=10"], capsys)

    assert code == EXIT_PRINTED
    assert "  fitted          6" in out
    assert "  previous        0" in out


def test_frames_with_no_gravity_are_processed_counted_and_left_out(tmp_path: Path) -> None:
    frames = [_frame(index) for index in range(5)] + [_frame(index, pose=UNALIGNED) for index in range(5, 8)]
    log_dir = _write(tmp_path / "imu_gap", frames)

    report = _report(log_dir)

    assert report.frames == 8
    assert report.unaligned_frames == 3
    assert report.heights_meters.size == 5


def test_the_planner_figures_still_refuse_a_walk_with_no_gravity(tmp_path: Path) -> None:
    # Only the floor report keeps such frames. The default for every other caller is unchanged.
    frames = [_frame(index) for index in range(3)] + [_frame(3, pose=UNALIGNED)]
    log_dir = _write(tmp_path / "imu_gap", frames)

    with pytest.raises(RecordingRefused, match="isn't gravity aligned"):
        scene_pass(log_dir, SceneConfig(), WalkerConfig(), None)


def test_a_walk_with_no_floor_at_all_exits_3_after_printing_its_counts(tmp_path: Path, capsys) -> None:
    log_dir = _write(tmp_path / "no_floor", [_frame(index, floor=False) for index in range(3)])

    code, out, err = _run([log_dir, "--scene-defaults"], capsys)

    assert code == EXIT_NOTHING_MEASURED
    assert "frames the scene refused: 3" in out
    assert "no fitted or supplied aligned frame, so no height" in out
    assert "nothing measured" in err


def test_a_missing_log_exits_1_without_a_traceback(tmp_path: Path, capsys) -> None:
    code, _, err = _run([tmp_path / "no_such_walk", "--scene-defaults"], capsys)

    assert code == EXIT_REFUSED
    assert "Traceback" not in err


@pytest.mark.parametrize("extra", [["--segment", "0"], ["--set", "lateral_kinetic_weight=1"]])
def test_floor_takes_no_segment_and_no_planner_override(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as usage:
        parse_arguments(["floor", str(tmp_path), *extra])
    assert usage.value.code == 2


def test_two_processes_print_byte_identical_reports(tmp_path: Path) -> None:
    frames = [_frame(index) for index in range(4)] + [_frame(index, floor=False) for index in range(4, 6)]
    log_dir = _write(tmp_path / "walk", frames)
    command = [sys.executable, "-m", "nav.evaluation.check_planner", "floor", str(log_dir), "--scene-defaults"]

    first = subprocess.run(command, capture_output=True, cwd=NAV_DIR.parent, check=True)
    second = subprocess.run(command, capture_output=True, cwd=NAV_DIR.parent, check=True)

    assert first.stdout == second.stdout
    assert b"median" in first.stdout


def test_every_refusal_cause_has_a_unit_to_print_its_median_in() -> None:
    # A cause added without a unit would raise the first time a walk had one, mid-report.
    assert [measured_unit(cause) for cause in FloorRefusalCause] == ["deg", "m", "m", "points", "", "points"]
