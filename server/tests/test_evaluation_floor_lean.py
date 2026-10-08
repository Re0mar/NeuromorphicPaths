"""
Covers `check_planner floor-lean`: matching a replay's frames back to its capture, and refitting them
with the logged pose and with the pose from the capture's IMU.

Every test here builds its own frame log and capture: level glasses over a flat floor, so the
capture's IMU says level and a logged pose tilted by a known angle should leave the floor leaning by
that angle. The capture's frame intervals mix 33.3333 and 33.3445 ms, as the glasses' do, because
that mix is the only thing that lets a wrong shift be caught at all.
"""

# Standard library imports
import json
from collections.abc import Iterator
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation.check_planner import EXIT_PRINTED, EXIT_REFUSED, main
from nav.evaluation.floor_lean import MAX_STAMP_RESIDUAL_SECONDS, NothingToCompare, StampsDoNotMatch, floor_lean
from nav.pose.imu_orientation import IMU_MATCH_TOLERANCE_SECONDS, multiply_wxyz, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT, rotation_about_x_wxyz
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.types import DepthFrame
from neon_captures import write_capture
from synthetic_depth import clean_scene

# Level glasses look 12 degrees down, which is what the documented mount says (see test_neon_chain).
LEVEL_GLASSES_SCENE = clean_scene(box_lateral_meters=None, pitch_degrees=12.0)
LEVEL_IMU = np.array([1.0, 0.0, 0.0, 0.0])
TILT_DEGREES = 10.0
# The level reading turned 10 degrees about a horizontal axis of the IMU's world, so its up is off by 10.
TILTED_IMU = multiply_wxyz(rotation_about_x_wxyz(TILT_DEGREES), LEVEL_IMU)
SHIFT_SECONDS = 197905.21935606003
FRAME_COUNT = 8
# The two intervals the glasses' capture has, alternating.
INTERVALS_SECONDS = (0.0333333, 0.0333445)


class ListDepthSource:
    def __init__(self, frames: list[DepthFrame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[DepthFrame]:
        yield from self._frames

    def close(self) -> None:
        pass


def _capture_stamps(start: float = 1000.0, count: int = FRAME_COUNT) -> list[float]:
    stamps = [start]
    for index in range(1, count):
        stamps.append(stamps[-1] + INTERVALS_SECONDS[index % 2])
    return stamps


def _level_imu(stamps: list[float], until: float | None = None) -> list[tuple[float, ...]]:
    """Level readings at 110 Hz from the first stamp to `until`, the last stamp by default."""
    end = stamps[-1] if until is None else until
    return [(stamps[0] + index / 110.0, *LEVEL_IMU) for index in range(int((end - stamps[0]) * 110.0) + 2)]


def _capture(directory: Path, stamps: list[float], imu: list[tuple[float, ...]] | None = None) -> Path:
    # The IMU says level throughout, at 110 Hz over the whole capture, unless the test says otherwise.
    return write_capture(directory, [(stamp, b"\x01") for stamp in stamps], imu=_level_imu(stamps) if imu is None else imu)


def _frame_log(directory: Path, stamps: list[float], logged_imu: np.ndarray, shift: float = SHIFT_SECONDS) -> Path:
    frames = [
        DepthFrame(
            timestamp_seconds=stamp + shift,
            depth_meters=LEVEL_GLASSES_SCENE.depth_meters,
            intrinsics=LEVEL_GLASSES_SCENE.intrinsics,
            pose=pose_from_imu(logged_imu, NEON_IMU_MOUNT),
            ground_plane=None,
            gaze_pixel=None,
        )
        for stamp in stamps
    ]
    list(RecordingTap(ListDepthSource(frames), directory).frames())
    return directory


def test_a_tilted_logged_pose_leans_by_its_tilt_and_the_capture_pose_leans_by_nothing(tmp_path: Path) -> None:
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, TILTED_IMU)

    result = floor_lean(log, capture, SHIFT_SECONDS, SceneConfig())

    assert result.largest_stamp_residual_seconds < 1e-6
    assert result.logged.fitted == result.capture.fitted == FRAME_COUNT
    assert np.median(result.logged.leans_degrees) == pytest.approx(TILT_DEGREES, abs=1.0)
    assert np.median(result.capture.leans_degrees) == pytest.approx(0.0, abs=1.0)
    assert result.logged_mismatches is None, "no timing log was written, so nothing was checked"


def test_a_right_logged_pose_gives_the_same_floors_both_ways(tmp_path: Path) -> None:
    # The control for the test above: the same frames with the level pose logged.
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)

    result = floor_lean(log, capture, SHIFT_SECONDS, SceneConfig())

    assert result.logged.leans_degrees == pytest.approx(result.capture.leans_degrees)


def test_a_shift_one_frame_off_is_refused_naming_the_residual(tmp_path: Path) -> None:
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    # Starting two frames in, as a replay's first planned frame does, so a slip of one frame lands on
    # the frame before rather than off the start of the capture.
    log = _frame_log(tmp_path / "log", stamps[2:], LEVEL_IMU)

    with pytest.raises(StampsDoNotMatch, match="us from the nearest frame") as refused:
        floor_lean(log, capture, SHIFT_SECONDS + INTERVALS_SECONDS[0], SceneConfig())
    # About 11 us, the gap between the two intervals: where the slipped frame's interval differs
    # from the one before it. Over the bound, and nowhere near a whole frame.
    assert "is 11." in str(refused.value)
    assert MAX_STAMP_RESIDUAL_SECONDS < 11e-6


def test_a_frame_log_from_a_different_capture_is_refused(tmp_path: Path) -> None:
    stamps = _capture_stamps()
    other = _capture(tmp_path / "other_capture", _capture_stamps(start=2000.0))
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)

    with pytest.raises(StampsDoNotMatch, match="other_capture"):
        floor_lean(log, other, SHIFT_SECONDS, SceneConfig())


def _write_timing(log: Path, stamps: list[float], lines: list[tuple[str, str | None]]) -> None:
    with (log / "timing.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for stamp, (outcome, floor_source) in zip(stamps, lines, strict=True):
            # A planned frame carries its worker times, which the reader insists on. Their values
            # play no part here.
            planned = outcome == "published"
            record = {
                "timestamp_seconds": stamp + SHIFT_SECONDS,
                "capture_seconds": None,
                "arrival_seconds": None,
                "depth_ready_seconds": None,
                "plan_done_seconds": 1.0 if planned else None,
                "floor_source": floor_source,
                "outcome": outcome,
                "started_seconds": 0.5 if planned else None,
                "scene_milliseconds": 1.0 if planned else None,
                "planner_milliseconds": 1.0 if planned else None,
                "usermodel_milliseconds": 1.0 if planned else None,
            }
            handle.write(json.dumps(record) + "\n")


def test_the_logged_side_is_checked_against_the_runs_own_floors_and_frames_it_dropped_are_left_out(tmp_path: Path) -> None:
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)
    # The run dropped the second frame for a newer one, so its scene never saw it, and fitted the rest.
    lines = [("published", "fitted")] * FRAME_COUNT
    lines[1] = ("dropped", None)
    _write_timing(log, stamps, lines)

    result = floor_lean(log, capture, SHIFT_SECONDS, SceneConfig())

    assert result.dropped_by_the_run == 1
    assert result.logged.frames == FRAME_COUNT - 1
    assert result.logged_mismatches == 0


def test_a_log_whose_every_frame_was_dropped_is_refused_naming_the_count(tmp_path: Path) -> None:
    # Nothing on either side. A table of zeros, or a division by them, would pass as a comparison.
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)
    _write_timing(log, stamps, [("dropped", None)] * FRAME_COUNT)

    with pytest.raises(NothingToCompare, match=f"dropped all {FRAME_COUNT} frames"):
        floor_lean(log, capture, SHIFT_SECONDS, SceneConfig())


def test_a_capture_imu_that_stops_early_leaves_the_later_frames_on_both_sides_without_a_capture_pose(tmp_path: Path) -> None:
    # The IMU covers the first half of the capture only. Those frames stay in the comparison, posed
    # level by default on the capture side and counted as having no pose, so both sides keep the
    # same frames. Dropping them instead would compare different frames, which is the thing this
    # refit exists to avoid.
    stamps = _capture_stamps()
    imu = _level_imu(stamps, until=stamps[FRAME_COUNT // 2 - 1])
    capture = _capture(tmp_path / "capture", stamps, imu=imu)
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)
    # The frames more than the tolerance past the last reading. The reading after the halfway
    # stamp still covers the next frame, so this is fewer than half.
    expected_without_pose = sum(1 for stamp in stamps if stamp - imu[-1][0] > IMU_MATCH_TOLERANCE_SECONDS)
    assert 0 < expected_without_pose < FRAME_COUNT

    result = floor_lean(log, capture, SHIFT_SECONDS, SceneConfig())

    assert result.capture.frames == result.logged.frames == FRAME_COUNT
    assert result.capture.frames_without_pose == expected_without_pose
    assert result.logged.frames_without_pose == 0


def test_capture_imu_lines_out_of_order_are_still_matched_by_time(tmp_path: Path) -> None:
    # The file written in reverse. A nearest-stamp search over an unsorted file reads past the
    # right line, and every frame would come back without a capture pose.
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps, imu=list(reversed(_level_imu(stamps))))
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)

    result = floor_lean(log, capture, SHIFT_SECONDS, SceneConfig())

    assert result.capture.frames_without_pose == 0
    assert np.median(result.capture.leans_degrees) == pytest.approx(0.0, abs=1.0)


def test_a_refit_that_disagrees_with_the_run_is_counted(tmp_path: Path) -> None:
    # The negative of the test above: the run says it carried the previous floor where the refit fits one.
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)
    lines = [("published", "fitted")] * FRAME_COUNT
    lines[3] = ("published", "previous")
    _write_timing(log, stamps, lines)

    assert floor_lean(log, capture, SHIFT_SECONDS, SceneConfig()).logged_mismatches == 1


def test_the_command_prints_both_sides_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, TILTED_IMU)

    code = main(["floor-lean", str(log), "--capture", str(capture), "--replay-shift-seconds", repr(SHIFT_SECONDS), "--scene-defaults"])

    printed = capsys.readouterr().out
    assert code == EXIT_PRINTED
    assert "\nlogged " in printed and "\ncapture " in printed


def test_the_command_refuses_a_wrong_shift_with_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    stamps = _capture_stamps()
    capture = _capture(tmp_path / "capture", stamps)
    log = _frame_log(tmp_path / "log", stamps, LEVEL_IMU)

    code = main(["floor-lean", str(log), "--capture", str(capture), "--replay-shift-seconds", repr(SHIFT_SECONDS + 1.0), "--scene-defaults"])

    assert code == EXIT_REFUSED
    assert "StampsDoNotMatch" in capsys.readouterr().err


def test_the_command_refuses_an_empty_frame_log_rather_than_printing_zeros(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    capture = _capture(tmp_path / "capture", _capture_stamps())
    log = tmp_path / "log"
    log.mkdir()
    (log / "index.jsonl").write_text("", encoding="utf-8")

    code = main(["floor-lean", str(log), "--capture", str(capture), "--replay-shift-seconds", repr(SHIFT_SECONDS), "--scene-defaults"])

    assert code == EXIT_REFUSED
    assert "nothing to replay" in capsys.readouterr().err
