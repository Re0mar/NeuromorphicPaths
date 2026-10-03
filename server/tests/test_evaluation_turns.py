"""
Turns, the straight-walking spread, and the obstacle-ahead tag, on synthetic walks and phone frames.

The tag tests hold the phone upright and pitched at the floor, as the walks were recorded, and place
obstacles by hand in the phone's own frame, which is how the scene reports them.
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
from nav.evaluation.track import walker_track
from nav.evaluation.turns import (
    Turn,
    TurnSide,
    TurnTag,
    detect_turns,
    straight_stretch_spread,
    tag_turn,
)
from synthetic_walks import (
    FRAME_RATE_HZ,
    POSE_JITTER_METERS,
    WALKING_SPEED_MPS,
    obstacle,
    phone_frame,
    ramp,
    turns_in_sequence,
    walk,
)

# The evaluation's own definitions may not come from the system it measures.
FORBIDDEN_IMPORT_PREFIXES = ("nav.planner", "nav.walker", "nav.runtime")


def straight_track(duration_seconds: float = 10.0):
    times, positions = walk(lambda time: 0.0, duration_seconds)
    return times, positions, walker_track(times, positions, EvaluationConfig())


def frames_before(times, positions, onset_seconds: float, camera_heading_degrees, points_at):
    """Phone frames at the recording's frame rate over the two seconds before onset."""
    frames = []
    for time, position in zip(times, positions):
        if onset_seconds - 2.0 <= time <= onset_seconds:
            frames.append(phone_frame(time, position, camera_heading_degrees, points_at(time)))
    return frames


def turn_at(onset_seconds: float) -> Turn:
    return Turn(onset_seconds=onset_seconds, end_seconds=onset_seconds + 2.0, side=TurnSide.RIGHT, change_radians=1.0)


# *******************************************
# Turns
# *******************************************


def test_three_turns_in_order() -> None:
    heading = turns_in_sequence((3.0, 2.0, 90.0), (9.0, 2.0, -60.0), (15.0, 2.0, 45.0))
    times, positions = walk(heading, 21.0)
    turns = detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    assert [turn.side for turn in turns] == [TurnSide.RIGHT, TurnSide.LEFT, TurnSide.RIGHT]
    np.testing.assert_allclose(np.degrees([turn.change_radians for turn in turns]), [90.0, -60.0, 45.0], atol=3.0)


def test_onset_of_a_constant_rate_turn_is_its_start() -> None:
    times, positions = walk(ramp(5.0, 1.5, 60.0), 10.0)
    (turn,) = detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    assert turn.onset_seconds == pytest.approx(5.0, abs=EvaluationConfig().resample_step_seconds)


def test_onset_of_a_slow_turn_is_not_late() -> None:
    # A departure-threshold onset would land about 0.4 s late here.
    times, positions = walk(ramp(5.0, 2.0, 25.0), 10.0)
    (turn,) = detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    assert turn.onset_seconds == pytest.approx(5.0, abs=0.15)


def test_slow_curve_under_threshold_is_not_a_turn() -> None:
    times, positions = walk(ramp(3.0, 6.0, 15.0), 12.0)
    assert detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig()) == ()


def test_s_bend_is_two_turns() -> None:
    times, positions = walk(turns_in_sequence((3.0, 1.5, 40.0), (4.5, 1.5, -40.0)), 10.0)
    turns = detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    assert [turn.side for turn in turns] == [TurnSide.RIGHT, TurnSide.LEFT]


def test_standing_turn_is_not_detected() -> None:
    # Turning on the spot leaves no direction of travel, so there is no heading to turn. Intended.
    times, positions = walk(ramp(5.0, 2.0, 90.0), 12.0, speed_mps_at=lambda time: 0.1 if 4.0 <= time < 8.0 else WALKING_SPEED_MPS)
    assert detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig()) == ()


def test_turn_made_while_stopped_is_not_scored() -> None:
    # Stop for a second, turn on the spot, walk on. The heading is undefined while standing, so no
    # window spans the stop and no onset could be placed inside it. A stated limit, not an accident.
    stop = lambda time: 0.1 if 5.0 <= time < 6.0 else WALKING_SPEED_MPS
    times, positions = walk(ramp(5.2, 0.6, 60.0), 12.0, speed_mps_at=stop, noise_meters=POSE_JITTER_METERS)
    assert detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig()) == ()


def test_window_across_a_standing_pause_is_not_a_turn() -> None:
    times, positions = walk(
        lambda time: 0.0,
        14.0,
        speed_mps_at=lambda time: 0.15 if 5.0 <= time < 8.0 else WALKING_SPEED_MPS,
        noise_meters=POSE_JITTER_METERS,
    )
    assert detect_turns(walker_track(times, positions, EvaluationConfig()), EvaluationConfig()) == ()


# *******************************************
# Straight-walking spread
# *******************************************


def swaying_heading(amplitude_degrees: float, rate_hz: float):
    return lambda time: amplitude_degrees * np.sin(2.0 * np.pi * rate_hz * time)


def test_straight_spread_on_a_wobbly_straight_walk() -> None:
    amplitude, rate = 3.0, 0.2
    times, positions = walk(swaying_heading(amplitude, rate), 60.0)
    spread = straight_stretch_spread(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    # Over a 2 s window a sinusoid changes by at most 2 A sin(pi f 2), and the 1.1 s centered window
    # scales its amplitude by sin(pi f 1.1) / (pi f 1.1).
    attenuation = np.sin(np.pi * rate * 1.1) / (np.pi * rate * 1.1)
    largest = 2.0 * amplitude * np.sin(np.pi * rate * 2.0) * attenuation
    assert spread.window_count > 0
    assert np.percentile(spread.change_samples_degrees, 99) == pytest.approx(largest, abs=0.3)
    assert np.max(spread.change_samples_degrees) <= largest + 0.3


def test_straight_spread_ignores_a_turn() -> None:
    amplitude, rate = 3.0, 0.2
    times, positions = walk(swaying_heading(amplitude, rate), 60.0)
    plain = straight_stretch_spread(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    turning = lambda time: swaying_heading(amplitude, rate)(time) + ramp(30.0, 2.0, 90.0)(time)
    times, positions = walk(turning, 60.0)
    with_turn = straight_stretch_spread(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    # The largest sample, not a percentile. A turn's tail adds only a few samples, too few to move
    # the 99th percentile, but each one is as large as the cap allows.
    assert np.max(with_turn.change_samples_degrees) <= np.max(plain.change_samples_degrees) + 1.0
    assert with_turn.straight_seconds < plain.straight_seconds


def test_slow_curve_is_not_straight() -> None:
    # 90 degrees over 6 s is 30 degrees per 2 s, past the 25 degree cap. Its gentle start and end
    # would pass the cap on their own, which is why a window either side is kept out too.
    amplitude, rate = 3.0, 0.2
    times, positions = walk(swaying_heading(amplitude, rate), 60.0)
    plain = straight_stretch_spread(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    curving = lambda time: swaying_heading(amplitude, rate)(time) + ramp(30.0, 6.0, 90.0)(time)
    times, positions = walk(curving, 60.0)
    with_curve = straight_stretch_spread(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    assert np.percentile(with_curve.change_samples_degrees, 99) == pytest.approx(
        np.percentile(plain.change_samples_degrees, 99), abs=1.0
    )


def test_change_just_past_the_cap_is_not_straight() -> None:
    # A constant curve changes by rate times 2 s over the turn window. 12.4 deg/s is 24.8, under the
    # 25 degree cap, and 12.6 deg/s is 25.2, over it.
    under = walk(lambda time: 12.4 * time, 30.0)
    over = walk(lambda time: 12.6 * time, 30.0)
    assert straight_stretch_spread(walker_track(*under, EvaluationConfig()), EvaluationConfig()).window_count > 0
    assert straight_stretch_spread(walker_track(*over, EvaluationConfig()), EvaluationConfig()).window_count == 0


def test_no_straight_windows_reports_empty() -> None:
    times, positions = walk(lambda time: 30.0 * time, 20.0)
    spread = straight_stretch_spread(walker_track(times, positions, EvaluationConfig()), EvaluationConfig())
    assert spread.straight_seconds == 0.0
    assert spread.window_count == 0
    assert spread.change_samples_degrees.shape == (0,)


# *******************************************
# The obstacle-ahead tag
# *******************************************


def test_tag_obstacle_ahead() -> None:
    times, positions, track = straight_track()
    frames = frames_before(times, positions, 6.0, 0.0, lambda time: (obstacle(0.0, 1.5),))
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.OBSTACLE_AHEAD


def test_tag_rotates_into_travel_frame() -> None:
    times, positions, track = straight_track()
    offset = np.radians(12.0)
    # The phone points 12 degrees left of travel. Straight ahead of the walker is 12 degrees right of the phone.
    ahead_of_walker = obstacle(1.5 * np.sin(offset), 1.5 * np.cos(offset))
    frames = frames_before(times, positions, 6.0, -12.0, lambda time: (ahead_of_walker,))
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.OBSTACLE_AHEAD
    # Straight ahead of the phone at 2.5 m is 0.52 m off the walker's line, outside the patch.
    ahead_of_phone = obstacle(0.0, 2.5)
    frames = frames_before(times, positions, 6.0, -12.0, lambda time: (ahead_of_phone,))
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.OPEN_AHEAD


def test_tag_unknown_when_patch_out_of_view() -> None:
    times, positions, track = straight_track()
    frames = frames_before(times, positions, 6.0, -60.0, lambda time: ())
    assert frames
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.UNKNOWN


def test_tag_open_when_group_is_beside_or_behind() -> None:
    times, positions, track = straight_track()
    frames = frames_before(times, positions, 6.0, 0.0, lambda time: (obstacle(0.6, 1.5), obstacle(0.0, -1.0, group=2)))
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.OPEN_AHEAD


def test_one_noisy_frame_is_not_an_obstacle() -> None:
    times, positions, track = straight_track()
    noisy_time = times[np.argmin(np.abs(times - 5.0))]
    frames = frames_before(times, positions, 6.0, 0.0, lambda time: (obstacle(0.0, 1.5),) if time == noisy_time else ())
    assert len(frames) > 50
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.OPEN_AHEAD


def test_tag_unknown_without_frames() -> None:
    _, _, track = straight_track()
    assert tag_turn(turn_at(6.0), track, [], EvaluationConfig()) == TurnTag.UNKNOWN


def test_tag_unknown_when_no_world_axis() -> None:
    times, positions, track = straight_track()
    frames = [
        phone_frame(time, position, 0.0, (obstacle(0.0, 1.5),), with_world_pose=False)
        for time, position in zip(times, positions)
        if 4.0 <= time <= 6.0
    ]
    assert tag_turn(turn_at(6.0), track, frames, EvaluationConfig()) == TurnTag.UNKNOWN


# *******************************************
# Configuration and boundaries
# *******************************************


def test_config_refuses_bad_turn_and_tag_values() -> None:
    with pytest.raises(ValueError, match="obstacle_ahead_min_frame_share"):
        EvaluationConfig(obstacle_ahead_min_frame_share=1.5)
    with pytest.raises(ValueError, match="turn_window_seconds"):
        EvaluationConfig(turn_window_seconds=5.0)
    with pytest.raises(ValueError, match="view_check_near_meters"):
        EvaluationConfig(view_check_near_meters=3.0)
    with pytest.raises(ValueError, match="straight_window_seconds"):
        EvaluationConfig(straight_window_seconds=4.05)


def test_evaluation_modules_import_nothing_from_the_measured_system() -> None:
    package = Path(nav.evaluation.__file__).parent
    checked = []
    for name in ("config.py", "track.py", "turns.py", "frames.py"):
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        checked.append(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{name} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{name} imports {alias.name}"
    assert len(checked) == 4


def test_turn_threshold_matches_its_record() -> None:
    # Guards the record, not the measurement. The value and the comment beside it say the same thing,
    # so an edit to one without the other fails here.
    import inspect
    from nav.evaluation import config as config_module
    source = inspect.getsource(config_module)
    assert EvaluationConfig().turn_threshold_degrees == 11.0
    assert "the 99th percentile of heading change over 2 s on straight walking, 10.39" in source
    assert "TEMPORARY" not in source
