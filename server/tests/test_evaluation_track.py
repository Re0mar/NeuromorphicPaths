"""
The walker's heading from the position track, on synthetic walks from synthetic_walks.py.

Expected values come from the walks' own construction: a walk that starts at heading zero faces -z,
and one that turns right by 90 degrees ends up facing +x, which is `ground_axes`' right.
"""

# Standard library imports
import dataclasses

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.track import floor_heading_radians, heading_direction, walker_track, wrap_radians
from nav.scene.floor import ground_axes
from nav.types import WORLD_UP
from synthetic_walks import FLOOR, POSE_JITTER_METERS, STRIDE_SWAY_METERS, WALKING_SPEED_MPS, ramp, walk

def heading_degrees_near(track, time_seconds: float) -> float:
    return float(np.degrees(track.heading_radians[np.argmin(np.abs(track.times_seconds - time_seconds))]))


# *******************************************
# Heading convention
# *******************************************


def test_floor_heading_matches_ground_axes() -> None:
    for degrees in (0.0, 37.0, 90.0, 170.0, -120.0):
        forward = np.array([np.sin(np.radians(degrees)), 0.0, -np.cos(np.radians(degrees))])
        lateral_axis, _ = ground_axes(FLOOR, forward)
        nudged_right = forward + 0.05 * lateral_axis
        change = float(wrap_radians(floor_heading_radians(nudged_right) - floor_heading_radians(forward)))
        assert change > 0.0, f"turning toward ground_axes' right lowered the heading at {degrees} degrees"


def test_heading_direction_inverts_floor_heading() -> None:
    for degrees in (0.0, 37.0, 90.0, 170.0, -120.0):
        assert float(floor_heading_radians(heading_direction(np.radians(degrees)))) == pytest.approx(np.radians(degrees))


def test_vertical_vector_has_no_heading() -> None:
    assert np.isnan(floor_heading_radians(WORLD_UP))
    assert np.isnan(floor_heading_radians(-WORLD_UP))


def test_wrap_radians_folds_into_half_open_interval() -> None:
    folded = wrap_radians(np.array([np.pi, -np.pi, 3.0 * np.pi, 0.0, -0.5]))
    np.testing.assert_allclose(folded, [np.pi, np.pi, np.pi, 0.0, -0.5])


# *******************************************
# Heading and turn rate
# *******************************************


def test_straight_walk_has_constant_heading_and_zero_turn_rate() -> None:
    times, positions = walk(lambda time: 25.0, 10.0)
    track = walker_track(times, positions, EvaluationConfig())
    interior = slice(10, -10)
    np.testing.assert_allclose(np.degrees(track.heading_radians[interior]), 25.0, atol=0.5)
    assert np.nanmax(np.abs(np.degrees(track.turn_rate_radians_per_second[interior]))) < 1.0


def test_right_turn_is_positive() -> None:
    times, positions = walk(ramp(3.0, 2.0, 90.0), 9.0)
    track = walker_track(times, positions, EvaluationConfig())
    assert heading_degrees_near(track, 8.0) - heading_degrees_near(track, 1.5) == pytest.approx(90.0, abs=2.0)
    assert np.nanmax(track.turn_rate_radians_per_second) > 0.0
    # The literal the convention rests on: a right turn from facing -z ends up walking along +x.
    final_direction = track.positions_floor_meters[-15] - track.positions_floor_meters[-25]
    assert final_direction[0] > 0.9 * np.linalg.norm(final_direction)


def test_left_turn_is_negative() -> None:
    times, positions = walk(ramp(3.0, 2.0, -90.0), 9.0)
    track = walker_track(times, positions, EvaluationConfig())
    assert heading_degrees_near(track, 8.0) - heading_degrees_near(track, 1.5) == pytest.approx(-90.0, abs=2.0)
    assert np.nanmin(track.turn_rate_radians_per_second) < 0.0


def test_loop_unwraps() -> None:
    times, positions = walk(ramp(1.0, 16.0, 720.0), 19.0)
    track = walker_track(times, positions, EvaluationConfig())
    assert heading_degrees_near(track, 18.0) - heading_degrees_near(track, 0.6) == pytest.approx(720.0, abs=5.0)
    steps = np.abs(np.diff(np.degrees(track.heading_radians)))
    assert np.nanmax(steps) < 30.0, "the heading jumped instead of unwrapping"


def test_rotating_the_world_about_up_does_not_change_turn_rate() -> None:
    times, positions = walk(ramp(3.0, 2.0, 60.0), 9.0)
    angle = np.radians(37.0)
    # A rotation about +y.
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    original = walker_track(times, positions, EvaluationConfig())
    rotated = walker_track(times, positions @ rotation.T, EvaluationConfig())
    np.testing.assert_allclose(
        rotated.turn_rate_radians_per_second, original.turn_rate_radians_per_second, atol=1e-6, equal_nan=True
    )


def test_height_changes_do_not_move_heading() -> None:
    times, positions = walk(ramp(3.0, 2.0, 60.0), 9.0)
    bobbing = positions.copy()
    bobbing[:, 1] += 0.15 * np.sin(2.0 * np.pi * 2.0 * times)
    bobbing_track = walker_track(times, bobbing, EvaluationConfig())
    np.testing.assert_allclose(
        bobbing_track.heading_radians,
        walker_track(times, positions, EvaluationConfig()).heading_radians,
        atol=1e-9,
        equal_nan=True,
    )
    # The floor positions carry no height, so a bob can't bend a straight stretch.
    defined = np.isfinite(bobbing_track.positions_floor_meters).all(axis=1)
    np.testing.assert_allclose(bobbing_track.positions_floor_meters[defined] @ WORLD_UP, 0.0, atol=1e-12)


def test_smoothing_keeps_a_real_turn() -> None:
    # The window trades noise for sharpness. This is the floor on the sharpness side.
    times, positions = walk(ramp(5.0, 1.5, 90.0), 12.0, noise_meters=POSE_JITTER_METERS, sway_meters=STRIDE_SWAY_METERS)
    # Jitter and sway at the measured level never break the track. Only the half window at each end,
    # where a full window doesn't fit, is unknown.
    half_window = EvaluationConfig().steps(EvaluationConfig().heading_half_window_seconds) + 1
    assert np.isfinite(walker_track(times, positions, EvaluationConfig()).heading_radians[half_window:-half_window]).all()
    track = walker_track(times, positions, EvaluationConfig())
    around = (track.times_seconds >= 4.75) & (track.times_seconds <= 7.75)
    span = np.degrees(np.nanmax(track.heading_radians[around]) - np.nanmin(track.heading_radians[around]))
    assert span >= 80.0


def test_centered_window_has_no_delay() -> None:
    times, positions = walk(ramp(4.0, 2.0, 80.0), 10.0)
    track = walker_track(times, positions, EvaluationConfig())
    halfway = track.times_seconds[np.nanargmin(np.abs(np.degrees(track.heading_radians) - 40.0))]
    assert halfway == pytest.approx(5.0, abs=EvaluationConfig().resample_step_seconds)


# *******************************************
# Breaks, gaps and unknowns
# *******************************************


def test_tracker_jump_breaks_the_track_and_makes_no_turn() -> None:
    times, positions = walk(lambda time: 0.0, 10.0)
    jumped = positions.copy()
    jumped[150:, 0] += 4.0
    track = walker_track(times, jumped, EvaluationConfig())
    assert track.breaks_for_jumps == 1
    assert set(np.unique(track.piece_index)) == {0, 1}
    defined = np.isfinite(track.heading_radians)
    np.testing.assert_allclose(np.degrees(track.heading_radians[defined]), 0.0, atol=1.0)


def test_long_gap_breaks_the_track() -> None:
    times, positions = walk(lambda time: 0.0, 16.0)
    hole = (times > 6.0) & (times < 10.6)
    track = walker_track(times[~hole], positions[~hole], EvaluationConfig())
    assert track.breaks_for_gaps == 1
    inside = (track.times_seconds > 6.1) & (track.times_seconds < 10.5)
    assert np.isnan(track.heading_radians[inside]).all()
    assert (track.piece_index[inside] == -1).all()


def test_lost_position_breaks_the_track() -> None:
    times, positions = walk(lambda time: 0.0, 10.0)
    lost = positions.copy()
    lost[140:150] = np.nan
    track = walker_track(times, lost, EvaluationConfig())
    assert track.poses_without_position == 10
    assert set(np.unique(track.piece_index[track.piece_index >= 0])) == {0, 1}


def test_short_piece_is_undefined() -> None:
    times, positions = walk(lambda time: 0.0, 8.0)
    jumped = positions.copy()
    # Two jumps leave a 0.5 s piece in the middle, shorter than one 1.1 s window.
    jumped[60:, 0] += 4.0
    jumped[75:, 0] += 4.0
    track = walker_track(times, jumped, EvaluationConfig())
    middle = track.piece_index == 1
    assert middle.any()
    assert np.isnan(track.heading_radians[middle]).all()


def test_duplicate_timestamps_are_dropped_and_counted() -> None:
    times, positions = walk(lambda time: 10.0, 6.0)
    doubled_times = np.repeat(times, 2)
    doubled_positions = np.repeat(positions, 2, axis=0)
    track = walker_track(doubled_times, doubled_positions, EvaluationConfig())
    assert track.duplicates_dropped == times.shape[0]
    assert np.isfinite(track.heading_radians).any()
    assert np.isfinite(track.turn_rate_radians_per_second[np.isfinite(track.turn_rate_radians_per_second)]).all()


def test_standing_still_is_unknown_not_zero() -> None:
    # Shuffling, not frozen. A perfectly still phone has zero velocity and no heading anyway, so it
    # would pass without the speed gate ever running.
    times, positions = walk(
        lambda time: 0.0,
        12.0,
        speed_mps_at=lambda time: 0.15 if 4.0 <= time < 7.0 else WALKING_SPEED_MPS,
        noise_meters=POSE_JITTER_METERS,
    )
    track = walker_track(times, positions, EvaluationConfig())
    standing = (track.times_seconds > 4.8) & (track.times_seconds < 6.2)
    assert np.isnan(track.heading_radians[standing]).all()
    assert np.isnan(track.turn_rate_radians_per_second[standing]).all()
    assert track.samples_too_slow >= int(standing.sum())


def test_backwards_timestamp_is_refused() -> None:
    times, positions = walk(lambda time: 0.0, 3.0)
    times = times.copy()
    times[40] = times[38]
    with pytest.raises(ValueError, match="timestamp"):
        walker_track(times, positions, EvaluationConfig())


def test_too_few_samples_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        walker_track(np.array([0.0]), np.zeros((1, 3)), EvaluationConfig())
    with pytest.raises(ValueError, match="at least two"):
        walker_track(np.array([1.0, 1.0]), np.zeros((2, 3)), EvaluationConfig())


def test_mismatched_shapes_and_infinite_values_are_refused() -> None:
    with pytest.raises(ValueError, match="timestamps but"):
        walker_track(np.arange(4.0), np.zeros((3, 3)), EvaluationConfig())
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        walker_track(np.arange(4.0), np.zeros((4, 2)), EvaluationConfig())
    infinite = np.zeros((4, 3))
    infinite[2, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        walker_track(np.arange(4.0), infinite, EvaluationConfig())
    half_missing = np.zeros((4, 3))
    half_missing[2, 0] = np.nan
    with pytest.raises(ValueError, match="partly NaN"):
        walker_track(np.arange(4.0), half_missing, EvaluationConfig())


def test_config_refuses_non_positive_and_non_whole_windows() -> None:
    for field in dataclasses.fields(EvaluationConfig):
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match=field.name):
                EvaluationConfig(**{field.name: bad})
    with pytest.raises(ValueError, match="heading_half_window_seconds"):
        EvaluationConfig(heading_half_window_seconds=0.25)


def test_config_refuses_a_bool_for_a_number() -> None:
    # bool is an int in Python, so True would otherwise pass as 1 m/s.
    with pytest.raises(ValueError, match="min_walking_speed_mps"):
        EvaluationConfig(min_walking_speed_mps=True)


def test_steps_on_whole_ratios() -> None:
    config = EvaluationConfig()
    assert config.steps(0.3) == 3
    assert config.steps(0.7) == 7


# *******************************************
# Piece ends
# *******************************************


def test_edges_of_a_piece_are_unknown_and_make_no_turn() -> None:
    # A shrunk window at a piece's ends left its edge samples unsmoothed, and on a straight walk with
    # the measured jitter and sway they swung by 6 to 11 degrees, enough to read as a turn on some
    # lengths. A full window or nothing.
    from nav.evaluation.turns import detect_turns
    half_window = EvaluationConfig().steps(EvaluationConfig().heading_half_window_seconds)
    for length_seconds in np.arange(20.0, 20.6, 0.05):
        times, positions = walk(lambda time: 0.0, length_seconds, noise_meters=POSE_JITTER_METERS, sway_meters=STRIDE_SWAY_METERS, seed=3)
        track = walker_track(times, positions, EvaluationConfig())
        assert np.isnan(track.heading_radians[:half_window]).all()
        assert np.isnan(track.heading_radians[-half_window:]).all()
        assert np.isnan(track.turn_rate_radians_per_second[:half_window + 1]).all()
        assert detect_turns(track, EvaluationConfig()) == (), f"a straight {length_seconds:.2f} s walk made a turn"
