"""
The planner's whole-walk numbers on two committed slices of recorded walks, pinned to a tolerance.

Each slice is a contiguous stretch of a walk as the scene handed it to the planner, cut by
`python -m nav.evaluation.fixture`. The test plans it with today's planner and checks every figure.
A planner change moves these numbers, and the commit that makes the change updates them, on purpose,
with the reason in the commit message. A change that moves them by accident shows up here.

The expected values are a snapshot of the planner, not arithmetic. First captured 2026-10-04 at
e983176 plus the commit that added this test. Updated 2026-10-05 for the prior toward the previous
plan (previous_plan_spread_meters 0.25 over the first 1.0 s), which moved every heading figure and no
alarm figure. The full swings, the alarm's hold and raise figures and the near noise were added the
same day, at their values with the prior on. Updated 2026-10-08 for the prior kept across gaps up to
3.0 s with its spread widening 0.05 m² a second, so it holds at the glasses' frame rate: on the
pixel_walk_3 slice the arrow sits at its limit on 420 frames instead of 383 and 5 band frames
instead of 2, because the prior now holds a side through the walk's tracking dropouts. Nothing else
moved on either slice. The arithmetic check beside them is `test_a_hand_built_slice_gives_the_numbers_worked_out_by_hand` in
test_evaluation_fixture.py, which a uniform defect in the planner can't satisfy by accident.

The slices were cut from the recordings in the root clone's server/frame_logs/ with:

    python -m nav.evaluation.fixture cut frame_logs/pixel_walk_3 --scene-set floor_max_offset_meters=2.2
        --scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999
        --start 1537469.2712074 --end 1537502.301757412 --out tests/fixtures/golden_pixel_walk_3.json.gz
    python -m nav.evaluation.fixture cut frame_logs/pixel_display_run --scene-defaults
        --scene-set floor_max_tilt_degrees=50
        --start 1566203.606489937 --end 1566219.202087806 --out tests/fixtures/golden_pixel_display_run.json.gz

The scene settings are the ones that reproduce every recorded figure on those walks. pixel_walk_3's
run_config.json predates three floor settings, so they are set to today's defaults.
"""

# Standard library imports
from dataclasses import dataclass, replace
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.evaluation.config import PlannerNumbersConfig
from nav.evaluation.fixture import read_fixture
from nav.evaluation.planner_numbers import ClearanceBand, PlannerNumbers, whole_walk_numbers
from nav.evaluation.replay import replayed_frames
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.dynamic_programming import plan
from nav.planner.field import lateral_grid, step_count
from nav.planner.goal import goal_position, goal_term, goal_tolerance, terminal_cost_as_rate
from nav.walker import WalkerConfig

FIXTURES = Path(__file__).parent / "fixtures"
SHARE_TOLERANCE_POINTS = 0.5
COUNT_TOLERANCE = 1
DISAGREEMENT_TOLERANCE_METERS = 0.005
# About a tenth of these walks' 34 ms frame step, so one frame more or less of hold shows.
HOLD_TOLERANCE_SECONDS = 0.003
# The noise comes from the scene and no planner change should move it at all.
NOISE_TOLERANCE_METERS = 0.0005


@dataclass(frozen=True)
class Expected:
    """One slice's pinned figures. A share is in percent, and a comment gives its frame base and one frame's step."""

    pinned_all_percent: float
    pinned_band_percent: float
    pinned_clear_percent: float
    # The restated band, as counts: both slices hold under the 100 frames a share needs.
    restated_band_pinned: int
    restated_band_frames: int
    distinct_headings: int
    # Pinned frame pairs whose arrow jumps from one limit to the other.
    full_swings: int
    alarm_on_percent: float
    alarm_changes: int
    raised_past_close: int
    longest_hold_seconds: float
    near_noise_median_meters: float
    near_noise_readings: int
    disagreement_median_meters: float
    disagreement_p90_meters: float


# The planner as it stands. A planner change updates these in the same commit, and says why.
EXPECTED = {
    # pixel_walk_3, last segment, 1537469.27 to 1537502.30 s, 967 frames, outdoors.
    "golden_pixel_walk_3.json.gz": Expected(
        # Before the prior: 564 of 967 pinned, band 64 of 108, p90 1.3046 m.
        # Before the goal could steer (goal as a rate, 1.5 m): 420 of 967 pinned, p90 0.3993 m.
        pinned_all_percent=41.365,  # 400 of 967, one frame 0.10 points
        pinned_band_percent=4.630,  # 5 of 108, one frame 0.93 points
        pinned_clear_percent=0.0,  # 0 of 100, one frame 1.00 point
        restated_band_pinned=0,
        restated_band_frames=52,
        distinct_headings=21,
        full_swings=0,
        alarm_on_percent=47.156,  # 456 of 967
        alarm_changes=26,
        raised_past_close=0,
        longest_hold_seconds=0.4998,
        near_noise_median_meters=0.0986,
        near_noise_readings=8392,
        disagreement_median_meters=0.0668,  # over 964 pairs
        disagreement_p90_meters=0.4591,
    ),
    # pixel_display_run, the classroom walk, last segment, 1566203.61 to 1566219.20 s, 461 frames.
    "golden_pixel_display_run.json.gz": Expected(
        # Before the prior: 227 of 461 pinned, band 48 of 115, 19 distinct headings, p90 1.1095 m.
        # Before the goal could steer: 166 of 461 pinned, band 6 of 115, restated 6 of 54, 16 headings,
        # p90 0.3573 m. The whole classroom walk's band went from 30.3 to 29.2 %, this slice's the other way.
        pinned_all_percent=39.262,  # 181 of 461, one frame 0.22 points
        pinned_band_percent=15.652,  # 18 of 115, one frame 0.87 points
        pinned_clear_percent=0.0,  # 0 of 100, one frame 1.00 point
        restated_band_pinned=0,
        restated_band_frames=54,
        distinct_headings=21,
        full_swings=0,
        alarm_on_percent=27.766,  # 128 of 461
        alarm_changes=12,
        raised_past_close=0,
        longest_hold_seconds=0.2999,
        near_noise_median_meters=0.1244,
        near_noise_readings=1480,
        disagreement_median_meters=0.0610,  # over 460 pairs
        disagreement_p90_meters=0.4160,
    ),
}

# Read at import, so a missing or unreadable slice fails collection with its path, never a skip.
SLICES = {name: read_fixture(FIXTURES / name) for name in EXPECTED}


def numbers_for(name: str) -> PlannerNumbers:
    golden = SLICES[name]
    golden.require_walker(WalkerConfig())
    goal_mode = GoalMode(golden.provenance["cut_from"]["goal_mode"])
    frames = replayed_frames(golden.inputs, PlannerConfig(), WalkerConfig(), goal_mode)
    return whole_walk_numbers(frames, PlannerConfig(), PlannerNumbersConfig())


def percent(count: int, total: int) -> float:
    return 100.0 * count / total


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_planner_gives_the_pinned_numbers_on_the_slice(name: str) -> None:
    numbers = numbers_for(name)
    expected = EXPECTED[name]
    measured = {
        "pinned_all_percent": percent(numbers.pinned_frames, numbers.frames),
        "pinned_band_percent": percent(numbers.pinned_by_band[ClearanceBand.BAND], numbers.frames_by_band[ClearanceBand.BAND]),
        "pinned_clear_percent": percent(numbers.pinned_by_band[ClearanceBand.CLEAR], numbers.frames_by_band[ClearanceBand.CLEAR]),
        "alarm_on_percent": percent(numbers.alarm_on_frames, numbers.frames),
    }
    moved = [
        f"{field} {value:.3f}, pinned {getattr(expected, field):.3f}"
        for field, value in measured.items()
        if abs(value - getattr(expected, field)) > SHARE_TOLERANCE_POINTS
    ]
    counts = (
        ("distinct_headings", numbers.distinct_headings),
        ("full_swings", numbers.full_swings),
        ("alarm_changes", numbers.alarm_changes),
        ("raised_past_close", numbers.raised_past_close),
        ("near_noise_readings", numbers.near_noise_readings),
        ("restated_band_pinned", numbers.restated_band_pinned),
        ("restated_band_frames", numbers.restated_band_frames),
    )
    for field, value in counts:
        if abs(value - getattr(expected, field)) > COUNT_TOLERANCE:
            moved.append(f"{field} {value}, pinned {getattr(expected, field)}")
    for field, value in (("disagreement_median_meters", numbers.disagreement_median_meters), ("disagreement_p90_meters", numbers.disagreement_p90_meters)):
        if abs(value - getattr(expected, field)) > DISAGREEMENT_TOLERANCE_METERS:
            moved.append(f"{field} {value:.4f}, pinned {getattr(expected, field):.4f}")
    if abs(numbers.longest_hold_seconds - expected.longest_hold_seconds) > HOLD_TOLERANCE_SECONDS:
        moved.append(f"longest_hold_seconds {numbers.longest_hold_seconds:.4f}, pinned {expected.longest_hold_seconds:.4f}")
    if numbers.near_noise_median_meters is None or abs(numbers.near_noise_median_meters - expected.near_noise_median_meters) > NOISE_TOLERANCE_METERS:
        moved.append(f"near_noise_median_meters {numbers.near_noise_median_meters}, pinned {expected.near_noise_median_meters:.4f}")
    # Every moved figure in one message, so a planner change shows its whole effect at once.
    assert not moved, f"{name}: " + "; ".join(moved)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_slice_holds_enough_band_and_clear_frames_for_both_shares(name: str) -> None:
    numbers = numbers_for(name)
    needed = PlannerNumbersConfig().min_frames_for_a_share
    assert numbers.frames_by_band[ClearanceBand.BAND] >= needed
    assert numbers.frames_by_band[ClearanceBand.CLEAR] >= needed


def test_the_goal_cost_on_the_chosen_path_matches_a_hand_sum() -> None:
    # Beside the golden values, worked by hand so they are not the code certifying itself. Nothing in
    # view, a gaze goal 1.5 m right, a 0.4 m tolerance and the 6.5 sideways weight, set here rather than
    # read from the defaults. One 0.1 m sidestep a step costs 0.5 x 6.5 x (1 m/s) squared x 0.1 s = 0.325.
    # Ending 1.0 m right takes 10 of them, 3.25, and leaves the goal 0.5 m away: 0.5 x (0.5 / 0.4)
    # squared = 0.78125. 4.03125 in all. One more sidestep would save 0.5 x (0.25 - 0.16) / 0.16 = 0.28125,
    # less than its 0.325, so the plan stops there.
    config = replace(PlannerConfig(), goal_tolerance_meters=0.4, lateral_kinetic_weight=6.5)
    grid = lateral_grid(config)
    field = np.zeros((step_count(config), len(grid)))
    goal = goal_position(GoalMode.GAZE, config, np.array([1.5, 4.0]))
    field[-1] += terminal_cost_as_rate(goal_term(grid, goal, goal_tolerance(config, 0.0)), config)

    offsets, cost = plan(field, 0.0, grid, config)

    assert offsets[-1] == pytest.approx(1.0)
    assert cost == pytest.approx(4.03125, abs=1e-9)
