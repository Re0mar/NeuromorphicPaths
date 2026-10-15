"""
The attention count, on frames built by hand with the answers worked out here.

The command test writes a synthetic walk through the real tap, as test_evaluation_planner_numbers.py does.
"""

# Standard library imports
import math
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation.attention_count import (
    NEAREST,
    AttentionSetting,
    AttentionWindow,
    GazeCarried,
    attention_count,
    check_windows,
    control_partner,
)
from nav.evaluation.check_planner import format_attention, main
from nav.evaluation.planner_numbers import PlannerInput, ReplayedFrame
from nav.planner.config import PlannerConfig
from nav.types import CameraOutline, ObstaclePoint, ObstacleSet, PlannedPath
from synthetic_walks import write_recording

PLANNER = PlannerConfig()
FOCAL = 500.0
INTRINSICS = np.array([[FOCAL, 0.0, 320.0], [0.0, FOCAL, 240.0], [0.0, 0.0, 1.0]])
# Straight ahead of the camera, so the principal point looks at it.
ON_THE_DANGER = [320.0, 240.0]
# 30 degrees right, well off it at every tolerance swept.
OFF_THE_DANGER = [320.0 + FOCAL * math.tan(math.radians(30.0)), 240.0]
WINDOWS = (NEAREST, AttentionWindow(0.05), AttentionWindow(0.2))


def danger(forward: float = 1.0) -> ObstaclePoint:
    """One group dead ahead in the alarm's corridor, its outline a point straight out of the camera."""
    return ObstaclePoint(0.0, forward, 1, 0.2, 0.05, None, None, False, np.array([0.0, 0.0, 2.0]), CameraOutline(0.0, 0.0, 0.0, 0.0))


def frame(
    time: float,
    alarm: bool = True,
    samples: list | None = None,
    offsets: list | None = None,
    nearest: list | None = None,
    points: tuple[ObstaclePoint, ...] | None = None,
    intrinsics: np.ndarray = INTRINSICS,
) -> ReplayedFrame:
    """A replayed frame with the alarm up or down and the gaze given."""
    points = (danger(),) if points is None else points
    pixels = None if samples is None else np.array(samples, dtype=np.float64).reshape(-1, 2)
    offset_array = None if samples is None else np.array(offsets if offsets is not None else [0.0] * len(samples), dtype=np.float64)
    planner_input = PlannerInput(
        timestamp_seconds=time,
        obstacles=ObstacleSet(time, points, len(points)),
        origin=None,
        lateral_axis=None,
        forward_axis=None,
        gaze_ground_point=None,
        gaze_pixels=pixels,
        gaze_sample_offsets_seconds=offset_array,
        intrinsics=intrinsics,
        gaze_pixel=None if nearest is None else np.array(nearest, dtype=np.float64),
    )
    path = PlannedPath(time, np.zeros(2), np.zeros(2), 0.0, alarm, 0.0, scene_information_bits=0.0, avoidance_surprise_bits=0.0)
    return ReplayedFrame(planner_input, path, raise_decision=alarm)


def count(
    frames: list[ReplayedFrame],
    tolerances: tuple[float, ...] = (5.0,),
    windows: tuple[AttentionWindow, ...] = WINDOWS,
    shift: float = 5.0,
    credits: tuple[float, ...] = (0.0,),
):
    return attention_count(frames, PLANNER, credits, tolerances, windows, 0.5, 0.4, 0.6, shift)


def cell(table: dict, window: AttentionWindow, credit: float = 0.0, tolerance: float = 5.0):
    return table[AttentionSetting(credit, tolerance, window)]


def run(arguments: list, capsys) -> tuple[int, str, str]:
    code = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_attention_table_counts_a_synthetic_log() -> None:
    # One alarm frame looked at, one not, one with no sample at all, and one with the alarm down.
    frames = [
        frame(0.0, samples=[ON_THE_DANGER], nearest=ON_THE_DANGER),
        frame(1.0, samples=[OFF_THE_DANGER], nearest=OFF_THE_DANGER),
        frame(2.0, samples=[]),
        frame(3.0, alarm=False, samples=[ON_THE_DANGER]),
    ]

    counted = count(frames)

    assert (counted.frames, counted.alarm_frames, counted.raised_frames) == (4, 3, 3)
    counts = cell(counted.real, AttentionWindow(0.05))
    assert (counts.noticed, counts.not_noticed, counts.unknown) == (1, 1, 1)
    # Unknown is out of the denominator.
    assert counts.with_a_sample == 2
    assert counted.gaze is GazeCarried.SAMPLES


def test_a_sample_inside_the_wide_window_and_outside_the_narrow_one_counts_only_in_the_wide() -> None:
    counted = count([frame(0.0, samples=[ON_THE_DANGER], offsets=[0.1])])

    narrow, wide = cell(counted.real, AttentionWindow(0.05)), cell(counted.real, AttentionWindow(0.2))
    assert (narrow.noticed, narrow.unknown) == (0, 1)
    assert (wide.noticed, wide.unknown) == (1, 0)


def test_the_nearest_column_reads_the_one_nearest_sample() -> None:
    counted = count([frame(0.0, samples=[OFF_THE_DANGER], nearest=ON_THE_DANGER), frame(1.0, samples=[])])

    nearest = cell(counted.real, NEAREST)
    assert (nearest.noticed, nearest.not_noticed, nearest.unknown) == (1, 0, 1)


def test_an_alarm_held_up_with_an_empty_corridor_is_counted_apart() -> None:
    counted = count([frame(0.0, samples=[ON_THE_DANGER], points=())])

    assert counted.held_without_danger == 1
    assert cell(counted.real, NEAREST).with_a_sample == 0
    assert cell(counted.real, NEAREST).unknown == 0


def test_a_log_without_samples_counts_only_nearest_and_says_so() -> None:
    frames = [frame(0.0, nearest=ON_THE_DANGER)]

    counted = attention_count(frames, PLANNER, (0.0,), (5.0,), (NEAREST,), 0.5, 0.4, 0.6, 5.0)
    text = format_attention("old_walk", counted)

    assert counted.gaze is GazeCarried.NEAREST_ONLY
    assert "only the nearest sample could be counted" in text
    assert "1/1 100%*" in text


def test_a_log_with_no_gaze_prints_no_percentage() -> None:
    counted = count([frame(0.0), frame(1.0)])

    text = format_attention("pixel_walk", counted)

    assert counted.gaze is GazeCarried.NONE
    assert "no gaze at all" in text
    assert "%" not in text


def test_a_cell_with_no_sample_prints_no_percentage() -> None:
    text = format_attention("walk", count([frame(0.0, samples=[ON_THE_DANGER], offsets=[0.1])]))

    assert "no sample [1]" in text


def test_the_shifted_control_pairs_each_frame_with_the_frame_nearest_t_plus_shift() -> None:
    # The gaze is on the danger only at 5 s. Its own frame counts it once. In the control, frame 0
    # borrows frame 5's gaze, and frame 10 runs off the end and looks back to frame 5 as well.
    frames = [frame(float(time), samples=[ON_THE_DANGER if time == 5 else OFF_THE_DANGER]) for time in range(11)]

    counted = count(frames, windows=(AttentionWindow(0.05),))

    real, control = cell(counted.real, AttentionWindow(0.05)), cell(counted.control, AttentionWindow(0.05))
    assert (real.noticed, real.with_a_sample) == (1, 11)
    assert (control.noticed, control.with_a_sample) == (2, 11)
    # 8, 9 and 10 s are within half the shift of the last frame, so those three look back.
    assert counted.control_looked_back == 3


def test_control_partner_looks_forward_and_back_at_the_end() -> None:
    timestamps = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 9.0])

    assert control_partner(timestamps, 0, 5.0) == (3, False)
    # 9 s plus 5 is past the end, so it looks back to the stamp nearest 4 s, which is 5 s.
    assert control_partner(timestamps, 5, 5.0) == (3, True)


def test_control_partner_has_none_when_both_sides_are_nearer_than_half_the_shift() -> None:
    # Forward lands on 2 s and back on 0 s, both 1 s away, so neither is shifted at all.
    assert control_partner(np.array([0.0, 1.0, 2.0]), 1, 5.0) == (None, True)


def test_a_walk_too_short_for_a_partner_counts_the_control_unknown_and_says_how_many() -> None:
    frames = [frame(0.0, samples=[ON_THE_DANGER]), frame(1.0, samples=[ON_THE_DANGER]), frame(2.0, samples=[ON_THE_DANGER])]

    counted = count(frames, windows=(AttentionWindow(0.05),))

    real, control = cell(counted.real, AttentionWindow(0.05)), cell(counted.control, AttentionWindow(0.05))
    assert real.noticed == 3
    # Borrowing gaze from a second earlier would read as the wearer's own look in the control.
    assert (control.noticed, control.unknown) == (0, 3)
    assert (counted.control_unpaired, counted.control_looked_back) == (3, 0)
    assert "3 with no frame half the shift away" in format_attention("short_walk", counted)


def test_the_control_turns_the_partners_gaze_into_rays_with_the_partners_own_matrix() -> None:
    # The partner's camera has its centre 80 px right, so its sample at x 400 is straight ahead.
    # Read through the danger frame's matrix it would sit about 9 degrees off, outside 5.
    shifted = INTRINSICS.copy()
    shifted[0, 2] = 400.0
    frames = [
        frame(0.0, samples=[OFF_THE_DANGER]),
        frame(5.0, alarm=False, samples=[[400.0, 240.0]], intrinsics=shifted),
    ]

    counted = count(frames, windows=(AttentionWindow(0.05),))

    assert cell(counted.control, AttentionWindow(0.05)).noticed == 1


def test_a_walk_whose_every_window_is_empty_has_no_gaze() -> None:
    # Live glasses with the gaze stream stopped carry empty windows, not None.
    counted = count([frame(0.0, samples=[]), frame(1.0, samples=[])])

    assert counted.gaze is GazeCarried.NONE


def test_a_window_wider_than_the_carry_window_is_refused() -> None:
    with pytest.raises(ValueError, match="wider than"):
        check_windows([AttentionWindow(0.25)])
    with pytest.raises(ValueError, match="zero or more"):
        check_windows([AttentionWindow(-0.01)])


def test_a_shift_at_or_under_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="shift"):
        count([frame(0.0)], shift=0.0)


def test_a_tolerance_out_of_range_is_refused_by_the_count() -> None:
    with pytest.raises(ValueError, match="attention tolerance"):
        count([frame(0.0)], tolerances=(0.0,))


def test_a_look_before_the_alarm_counts_on_the_alarm_frame_only_with_credit() -> None:
    # Looked at the danger at 3 m with the alarm down. A second later it is at 1.6 m, the alarm is
    # up and the gaze has moved on. Noticed with a 2 s credit, not on the looking frame alone.
    frames = [
        frame(0.0, alarm=False, samples=[ON_THE_DANGER], points=(danger(3.0),)),
        frame(1.0, samples=[OFF_THE_DANGER], points=(danger(3.0 - PLANNER.walking_speed_mps),)),
    ]

    counted = count(frames, windows=(AttentionWindow(0.05),), credits=(0.0, 2.0))

    without, with_credit = (cell(counted.real, AttentionWindow(0.05), credit) for credit in (0.0, 2.0))
    assert (without.noticed, without.not_noticed) == (0, 1)
    assert (with_credit.noticed, with_credit.not_noticed) == (1, 0)


def test_a_look_in_credit_counts_on_a_frame_with_no_sample_of_its_own() -> None:
    frames = [
        frame(0.0, alarm=False, samples=[ON_THE_DANGER], points=(danger(3.0),)),
        frame(1.0, samples=[], points=(danger(3.0 - PLANNER.walking_speed_mps),)),
    ]

    counted = count(frames, windows=(AttentionWindow(0.05),), credits=(2.0,))

    credited = cell(counted.real, AttentionWindow(0.05), 2.0)
    assert (credited.noticed, credited.unknown) == (1, 0)


def test_planner_input_keeps_its_positional_build() -> None:
    obstacles = ObstacleSet(0.0, (), 0)

    built = PlannerInput(0.0, obstacles, None, None, None, None)

    assert (built.gaze_pixels, built.gaze_sample_offsets_seconds, built.intrinsics, built.gaze_pixel) == (None, None, None, None)


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    """A straight 22 s walk past the synthetic scene's box. It has no gaze."""
    return write_recording(tmp_path_factory.mktemp("walk") / "straight", lambda time: 0.0, 22.0)


def test_attention_command_on_a_walk_with_no_gaze_says_so(recording: Path, capsys) -> None:
    code, out, _ = run(["attention", recording], capsys)

    assert code == 0
    assert "no gaze at all" in out


@pytest.mark.parametrize(
    "flag, value, named",
    [
        ("--windows-seconds", "0.3", "wider than"),
        ("--windows-seconds", "soon", "windows are seconds"),
        ("--tolerances-degrees", "0", "attention tolerance"),
        ("--tolerances-degrees", "wide", "tolerances are numbers"),
        ("--shift-seconds", "0", "shift-seconds"),
        ("--link-meters", "0", "link-meters"),
        ("--cap-meters", "-1", "cap-meters"),
        ("--match-meters", "nan", "match-meters"),
        ("--credits-seconds", "2,-1", "credits-seconds"),
        ("--credits-seconds", "soon", "credits are numbers"),
    ],
)
def test_attention_command_refuses_a_bad_argument_as_a_usage_error(recording: Path, capsys, flag: str, value: str, named: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["attention", str(recording), flag, value])

    assert stopped.value.code == 2
    assert named in capsys.readouterr().err


def test_attention_command_keeps_a_walk_without_gravity_that_numbers_refuses(tmp_path: Path, capsys) -> None:
    # neon_walk_1's IMU sent only zeros, so no frame of it has gravity. The angle needs none.
    walk = write_recording(tmp_path / "no_gravity", lambda time: 0.0, 6.0, gravity_aligned_at=lambda time: False)

    refused_code, _, refused_err = run(["numbers", walk], capsys)
    code, out, _ = run(["attention", walk], capsys)

    assert refused_code == 1 and "gravity" in refused_err
    assert code == 0
    assert "had no gravity and were planned against the picture's own up" in out
