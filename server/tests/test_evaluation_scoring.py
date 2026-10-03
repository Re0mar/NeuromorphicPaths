"""
Scoring the planner's arrow against the walker's turns, on synthetic walks and hand-built frames.

The arrow in each frame is set by the test, the way the server would have returned it. Expected values
come from arithmetic on those arrows, never from what the code printed.
"""

# Standard library imports
import ast
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
import nav.evaluation
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.scoring import (
    TurnScore,
    arrow_in_travel_frame,
    find_sidesteps,
    lagged_correlation,
    score_turn,
    summarize,
)
from nav.evaluation.track import walker_track
from nav.evaluation.turns import Turn, TurnSide, TurnTag
from synthetic_walks import phone_frame, ramp, walk

STEP_SECONDS = EvaluationConfig().resample_step_seconds


def frames_for(times, positions, arrow_degrees_at, camera_offset_degrees: float = 0.0, travel_degrees_at=lambda time: 0.0):
    """One frame per recorded sample. The camera points camera_offset_degrees off the walker's travel."""
    return [
        phone_frame(time, position, travel_degrees_at(time) + camera_offset_degrees, arrow_degrees=arrow_degrees_at(time))
        for time, position in zip(times, positions)
    ]


def straight_walk(duration_seconds: float = 12.0):
    times, positions = walk(lambda time: 0.0, duration_seconds)
    return times, positions, walker_track(times, positions, EvaluationConfig())


def right_turn_at(onset_seconds: float) -> Turn:
    return Turn(onset_seconds=onset_seconds, end_seconds=onset_seconds + 2.0, side=TurnSide.RIGHT, change_radians=np.radians(60.0))


def arrow_series(arrow_degrees_at, camera_offset_degrees: float = 0.0, duration_seconds: float = 12.0):
    times, positions, track = straight_walk(duration_seconds)
    frames = frames_for(times, positions, arrow_degrees_at, camera_offset_degrees)
    return track, arrow_in_travel_frame(frames, track, EvaluationConfig())


def value_near(series, time_seconds: float) -> float:
    return float(np.degrees(series.travel_frame_radians[np.argmin(np.abs(series.times_seconds - time_seconds))]))


# *******************************************
# The arrow in the walker's direction of travel
# *******************************************


def test_arrow_in_travel_frame_adds_phone_offset() -> None:
    # Phone 20 degrees left of travel, server arrow 20 degrees right: it sends the walker straight on.
    _, series = arrow_series(lambda time: 20.0, camera_offset_degrees=-20.0)
    assert value_near(series, 6.0) == pytest.approx(0.0, abs=0.5)
    assert float(np.degrees(series.phone_offset_radians[60])) == pytest.approx(-20.0, abs=0.5)


def test_changing_the_arrow_moves_the_series_by_exactly_that() -> None:
    _, plain = arrow_series(lambda time: 3.0, camera_offset_degrees=-8.0)
    _, moved = arrow_series(lambda time: 10.0, camera_offset_degrees=-8.0)
    defined = np.isfinite(plain.travel_frame_radians)
    np.testing.assert_allclose(
        np.degrees(moved.travel_frame_radians[defined] - plain.travel_frame_radians[defined]), 7.0, atol=1e-9
    )


def test_large_phone_offset_is_not_dropped() -> None:
    # The evaluation doesn't drop planner output, however far the phone points off the walk.
    _, series = arrow_series(lambda time: 5.0, camera_offset_degrees=45.0)
    assert value_near(series, 6.0) == pytest.approx(50.0, abs=0.5)


def test_stale_arrow_is_not_used() -> None:
    times, positions, track = straight_walk()
    frames = [frame for frame in frames_for(times, positions, lambda time: 10.0) if not (5.0 < frame.timestamp_seconds < 5.5)]
    series = arrow_in_travel_frame(frames, track, EvaluationConfig())
    assert np.isnan(value_near(series, 5.45))
    assert np.isfinite(value_near(series, 5.2))


# *******************************************
# Per-turn agreement and lead time
# *******************************************


def test_agreeing_arrow_scores_lead_time() -> None:
    _, series = arrow_series(lambda time: 15.0 if 4.5 <= time <= 7.0 else 0.0)
    score = score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.agreed is True
    assert score.lead_seconds == pytest.approx(1.5, abs=STEP_SECONDS)
    assert score.lead_at_limit is False


def test_lead_stops_at_cap_and_at_previous_turn() -> None:
    _, series = arrow_series(lambda time: 15.0, duration_seconds=12.0)
    capped = score_turn(right_turn_at(10.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert capped.lead_seconds == pytest.approx(5.0, abs=STEP_SECONDS)
    assert capped.lead_at_limit is True
    after_turn = score_turn(right_turn_at(10.0), TurnTag.OPEN_AHEAD, series, 8.0, EvaluationConfig())
    assert after_turn.lead_seconds == pytest.approx(2.0, abs=STEP_SECONDS)
    assert after_turn.lead_at_limit is True


def test_lead_back_to_the_start_of_the_recording_is_at_least() -> None:
    # Leaning from the first frame: the lead runs past what was recorded, so it is a floor, not a value.
    _, series = arrow_series(lambda time: 15.0)
    score = score_turn(right_turn_at(3.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.lead_seconds == pytest.approx(3.0, abs=STEP_SECONDS)
    assert score.lead_at_limit is True


def test_no_frames_gives_no_arrow() -> None:
    _, _, track = straight_walk()
    series = arrow_in_travel_frame([], track, EvaluationConfig())
    assert np.isnan(series.travel_frame_radians).all()
    assert score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig()).arrow_read is False


def test_several_turns_scored_in_order() -> None:
    def arrow(time: float) -> float:
        if 2.5 <= time <= 3.0:
            return 20.0  # right before a right turn: agrees
        if 5.5 <= time <= 6.0:
            return -20.0  # left before a right turn: wrong side
        return 0.0  # carry on before the third
    _, series = arrow_series(arrow)
    scores = [score_turn(right_turn_at(onset), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig()) for onset in (3.0, 6.0, 9.0)]
    assert [score.agreed for score in scores] == [True, False, False]
    assert [score.arrow_side for score in scores] == [TurnSide.RIGHT, TurnSide.LEFT, None]


def test_no_arrow_before_turn_is_not_read() -> None:
    times, positions, track = straight_walk()
    frames = [frame for frame in frames_for(times, positions, lambda time: 15.0) if frame.timestamp_seconds > 7.0]
    series = arrow_in_travel_frame(frames, track, EvaluationConfig())
    score = score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.arrow_read is False
    assert score.agreed is None
    assert score.lead_seconds is None


def test_agreeing_mean_with_undefined_onset_has_unknown_lead() -> None:
    times, positions, track = straight_walk()
    # Frames stop 0.5 s before onset, so the arrow is stale at onset but the window's mean agrees.
    frames = [frame for frame in frames_for(times, positions, lambda time: 15.0) if frame.timestamp_seconds < 5.5]
    series = arrow_in_travel_frame(frames, track, EvaluationConfig())
    score = score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.agreed is True
    assert score.lead_seconds is None


def test_agreeing_mean_with_carry_on_at_onset_has_unknown_lead() -> None:
    _, series = arrow_series(lambda time: 20.0 if 5.0 <= time < 5.8 else 0.0)
    score = score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.agreed is True
    assert score.lead_seconds is None


def test_carry_on_arrow_is_not_agreement() -> None:
    _, series = arrow_series(lambda time: 3.0)
    score = score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.arrow_read is True
    assert score.agreed is False
    assert score.arrow_side is None


def test_wrong_side_arrow_has_no_lead() -> None:
    _, series = arrow_series(lambda time: -15.0)
    score = score_turn(right_turn_at(6.0), TurnTag.OPEN_AHEAD, series, None, EvaluationConfig())
    assert score.agreed is False
    assert score.arrow_side == TurnSide.LEFT
    assert score.lead_seconds is None


# *******************************************
# Sidesteps and correlation
# *******************************************


def test_sidestep_while_straight_is_found() -> None:
    _, series = arrow_series(lambda time: 25.0 if 4.0 <= time < 5.0 else 0.0)
    (sidestep,) = find_sidesteps(series, (), EvaluationConfig())
    assert sidestep.side == TurnSide.RIGHT
    assert sidestep.start_seconds == pytest.approx(4.0, abs=STEP_SECONDS)
    assert sidestep.end_seconds == pytest.approx(4.9, abs=2 * STEP_SECONDS)


def test_short_lean_is_not_a_sidestep() -> None:
    _, series = arrow_series(lambda time: 25.0 if 4.0 <= time < 4.3 else 0.0)
    assert find_sidesteps(series, (), EvaluationConfig()) == ()


def test_lean_before_a_turn_is_not_a_sidestep() -> None:
    _, series = arrow_series(lambda time: 25.0 if 4.0 <= time < 5.0 else 0.0)
    assert find_sidesteps(series, (right_turn_at(6.5),), EvaluationConfig()) == ()


def test_correlation_peaks_at_the_built_in_lag() -> None:
    # The walker turns the way the arrow pointed, 1.2 s later.
    lag = 1.2
    arrow_degrees = lambda time: 25.0 * np.sin(2.0 * np.pi * 0.07 * time)
    heading = lambda time: 25.0 / (2.0 * np.pi * 0.07) * 0.6 * (1.0 - np.cos(2.0 * np.pi * 0.07 * (time - lag)))
    times, positions = walk(heading, 60.0)
    track = walker_track(times, positions, EvaluationConfig())
    frames = frames_for(times, positions, arrow_degrees, travel_degrees_at=heading)
    result = lagged_correlation(arrow_in_travel_frame(frames, track, EvaluationConfig()), track, EvaluationConfig())
    assert result.peak_lag_seconds == pytest.approx(lag, abs=STEP_SECONDS + 1e-9)
    assert result.peak_correlation > 0.95


def test_correlation_with_too_few_pairs_is_nan() -> None:
    # A weaving 5 s walk: both series vary, so only the pair count can make the answer unknown.
    heading = lambda time: 20.0 * np.sin(time)
    times, positions = walk(heading, 5.0)
    track = walker_track(times, positions, EvaluationConfig())
    frames = frames_for(times, positions, lambda time: 10.0 * np.sin(time + 1.0), travel_degrees_at=heading)
    result = lagged_correlation(arrow_in_travel_frame(frames, track, EvaluationConfig()), track, EvaluationConfig())
    assert 10 < result.pair_counts[0] < EvaluationConfig().min_correlation_pairs
    assert np.isnan(result.correlation).all()
    assert result.peak_lag_seconds is None


def test_constant_arrow_correlation_is_nan_not_zero() -> None:
    track, series = arrow_series(lambda time: 10.0, duration_seconds=30.0)
    result = lagged_correlation(series, track, EvaluationConfig())
    assert result.pair_counts[0] >= EvaluationConfig().min_correlation_pairs
    assert np.isnan(result.correlation).all()


# *******************************************
# Summary
# *******************************************


def scored(lead: float | None, agreed: bool | None = True, read: bool = True) -> TurnScore:
    side = TurnSide.RIGHT if agreed else None
    return TurnScore(right_turn_at(5.0), TurnTag.OPEN_AHEAD, side, read, agreed, lead, False)


def test_summary_keeps_unknown_out_of_the_median() -> None:
    scores = [scored(1.0), scored(2.0), scored(None), scored(None, agreed=False), scored(None, agreed=None, read=False)]
    summary = summarize(scores, (), 120.0).by_tag[TurnTag.OPEN_AHEAD]
    assert summary.median_lead_seconds == pytest.approx(1.5)
    assert summary.known_leads == 2
    assert summary.turns - summary.known_leads == 3
    assert summary.not_read == 1
    assert summary.carry_on == 1
    all_unknown = summarize([scored(None)], (), 120.0).by_tag[TurnTag.OPEN_AHEAD]
    assert all_unknown.median_lead_seconds is None
    assert all_unknown.known_leads == 0


def test_summary_without_walking_has_no_rate() -> None:
    assert summarize([], (), 0.0).sidesteps_per_minute is None


# *******************************************
# Configuration and imports
# *******************************************


def test_config_refuses_bad_scoring_values() -> None:
    with pytest.raises(ValueError, match="min_correlation_pairs"):
        EvaluationConfig(min_correlation_pairs=1)
    with pytest.raises(ValueError, match="agreement_window_seconds"):
        EvaluationConfig(agreement_window_seconds=1.05)


def test_scoring_imports_nothing_from_the_measured_system() -> None:
    tree = ast.parse((Path(nav.evaluation.__file__).parent / "scoring.py").read_text(encoding="utf-8"))
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert modules, "parsed no imports at all"
    assert not [module for module in modules if module.startswith(("nav.planner", "nav.walker", "nav.runtime"))]
