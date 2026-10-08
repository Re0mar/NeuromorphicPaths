"""The prior toward the previous plan, on its own and inside the pipeline, against behaviors stated in advance."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.field import lateral_grid, step_count
from nav.planner.pipeline import PlannerPipeline
from nav.planner.previous_plan import PreviousPlanPrior
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = replace(PlannerConfig(), previous_plan_spread_meters=0.5, previous_plan_prior_seconds=3.8)
WALKER = WalkerConfig(radius_meters=0.35)
GRID = lateral_grid(CONFIG)
TIMES = np.arange(step_count(CONFIG)) * CONFIG.time_step_seconds
FORWARD = TIMES * CONFIG.walking_speed_mps
FRAME_SECONDS = 1.0 / 30.0
AHEAD = np.array([0.0, CONFIG.goal_distance_meters])


def post(lateral: float, forward: float, timestamp: float, noise: float = 0.05) -> ObstacleSet:
    clearance = float(np.hypot(lateral, forward)) - WALKER.radius_meters
    point = ObstaclePoint(lateral, forward, 1, clearance, noise, None, None, False, np.zeros(3))
    return ObstacleSet(timestamp, (point,), 1)


def straight_plan(offset: float) -> np.ndarray:
    """A sidestep to `offset` at the full sideways speed, then straight on. Every plan starts at the walker."""
    return np.sign(offset) * np.minimum(TIMES * CONFIG.max_lateral_speed_mps, abs(offset))


# *******************************************
# The prior on its own
# *******************************************


def test_the_first_frame_has_no_prior() -> None:
    assert PreviousPlanPrior(CONFIG, TIMES).field(GRID, 0.0, AHEAD) is None


def test_the_prior_is_zero_on_the_previous_plan_and_half_a_unit_one_spread_away() -> None:
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(0.5), 0.0, AHEAD)
    field = prior.field(GRID, 0.0, AHEAD)
    row = field[10]
    assert row[np.argmin(np.abs(GRID - 0.5))] == pytest.approx(0.0)
    # Half of (one spread over the spread) squared.
    assert row[np.argmin(np.abs(GRID - (0.5 + CONFIG.previous_plan_spread_meters)))] == pytest.approx(0.5)
    assert row[np.argmin(np.abs(GRID - (0.5 - 2 * CONFIG.previous_plan_spread_meters)))] == pytest.approx(2.0)


def test_the_previous_plan_is_advanced_by_one_frame_of_walking() -> None:
    # A plan curving right, 0.1 m per meter squared. A third of a second later each row sits 0.47 m
    # further along it (1.4 m/s), and the 0.022 m the walker already went sideways along it comes off.
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(0.1 * FORWARD**2, 0.0, AHEAD)
    elapsed = 1.0 / 3.0
    walked = CONFIG.walking_speed_mps * elapsed
    field = prior.field(GRID, elapsed, AHEAD)
    for row in (5, 20):
        expected = 0.1 * (FORWARD[row] + walked) ** 2 - 0.1 * walked**2
        lowest = GRID[np.argmin(field[row])]
        assert abs(lowest - expected) <= (GRID[1] - GRID[0]) / 2


def test_a_sidestep_already_walked_is_not_asked_for_again() -> None:
    # A sidestep 1 m left at 1 m/s, then straight on. 0.4 s later the walker, at the speeds the
    # planner assumes, is 0.4 m left. A row 0.5 s ahead sat 0.9 s along the old plan, at -0.9 m, so
    # the prior asks for -0.5 m from here. A row 1.5 s ahead asks for -1.0 + 0.4 = -0.6 m.
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(-1.0), 0.0, AHEAD)
    field = prior.field(GRID, 0.4, AHEAD)
    assert GRID[np.argmin(field[5])] == pytest.approx(-0.5, abs=0.051)
    assert GRID[np.argmin(field[15])] == pytest.approx(-0.6, abs=0.051)
    # A full second on, the sidestep is done and walked, and every row asks for straight ahead.
    long_gap = PreviousPlanPrior(replace(CONFIG, previous_plan_max_gap_seconds=2.0), TIMES)
    long_gap.remember(straight_plan(-1.0), 0.0, AHEAD)
    later = long_gap.field(GRID, 1.0, AHEAD)
    for row in (1, 10, 25):
        assert GRID[np.argmin(later[row])] == pytest.approx(0.0, abs=0.051)


def test_a_standing_walker_keeps_the_plan_where_it_was() -> None:
    # Two frames with the same timestamp, as a duplicated frame has. Nothing has been walked.
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(-0.8), 2.0, AHEAD)
    field = prior.field(GRID, 2.0, AHEAD)
    assert GRID[np.argmin(field[15])] == pytest.approx(-0.8)


def test_rows_past_the_previous_plans_reach_get_no_prior() -> None:
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(0.5), 0.0, AHEAD)
    field = prior.field(GRID, 0.3, AHEAD)
    walked = CONFIG.walking_speed_mps * 0.3
    beyond = FORWARD + walked > FORWARD[-1]
    assert beyond.any()
    assert np.all(field[beyond] == 0.0)
    assert np.any(field[~beyond][1:] > 0.0)


def test_rows_past_the_prior_seconds_get_no_prior() -> None:
    prior = PreviousPlanPrior(replace(CONFIG, previous_plan_prior_seconds=1.0), TIMES)
    prior.remember(straight_plan(0.5), 0.0, AHEAD)
    field = prior.field(GRID, 0.0, AHEAD)
    assert np.all(field[TIMES > 1.0 + 1e-9] == 0.0)
    assert np.any(field[(TIMES > 0) & (TIMES <= 1.0)] > 0.0)


def test_row_zero_gets_no_prior() -> None:
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(0.5), 0.0, AHEAD)
    assert np.all(prior.field(GRID, 0.0, AHEAD)[0] == 0.0)


def test_a_clock_that_went_backwards_forgets_the_plan() -> None:
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(0.5), 5.0, AHEAD)
    assert prior.field(GRID, 4.0, AHEAD) is None
    # Forgotten, not just skipped: the next frame forward has nothing either.
    assert prior.field(GRID, 5.1, AHEAD) is None


def test_a_gap_longer_than_the_maximum_forgets_the_plan() -> None:
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(0.5), 0.0, AHEAD)
    assert prior.field(GRID, CONFIG.previous_plan_max_gap_seconds + 0.01, AHEAD) is None
    prior.remember(straight_plan(0.5), 0.0, AHEAD)
    assert prior.field(GRID, CONFIG.previous_plan_max_gap_seconds, AHEAD) is not None


def test_a_goal_that_moved_past_its_tolerance_forgets_the_plan() -> None:
    # A wearer who looks from one gap to the other moves the gaze goal 2.5 m, past the 1.5 m tolerance.
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(-1.25), 0.0, np.array([-1.25, 4.0]))
    assert prior.field(GRID, FRAME_SECONDS, np.array([1.25, 4.0])) is None
    assert prior.field(GRID, 2 * FRAME_SECONDS, np.array([1.25, 4.0])) is None


def test_a_goal_that_moved_only_nearer_or_farther_keeps_the_plan() -> None:
    # The goal term reads only the sideways part, so a gaze sliding from 4 m to 2.2 m out along the same
    # line is the same goal, however far it moved.
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(-1.25), 0.0, np.array([0.3, 4.0]))
    assert prior.field(GRID, FRAME_SECONDS, np.array([0.3, 2.2])) is not None


def test_a_gaze_moving_nearer_and_farther_keeps_a_near_tie_on_its_side() -> None:
    # A post wobbling a millimeter, and a gaze straight ahead that alternates between 2 m and 4 m out,
    # as gaze on the floor does. Forgetting on every forward move would flip the plan every frame.
    pipeline = PlannerPipeline(CONFIG, WALKER)
    sides = []
    for index, scene in enumerate(alternating_posts(12)):
        gaze = np.array([0.0, 2.0 if index % 2 else 4.0])
        sides.append(float(np.sign(pipeline.plan(scene, 0.0, GoalMode.GAZE, gaze).lateral_offsets_meters[-1])))
    assert len(set(sides)) == 1


def test_a_goal_that_moved_within_its_tolerance_keeps_the_plan() -> None:
    # Gaze wanders between fixations. Half a meter is the same place to go.
    prior = PreviousPlanPrior(CONFIG, TIMES)
    prior.remember(straight_plan(-1.25), 0.0, np.array([-1.25, 4.0]))
    assert prior.field(GRID, FRAME_SECONDS, np.array([-0.75, 4.0])) is not None


# *******************************************
# Inside the pipeline
# *******************************************


def alternating_posts(count: int) -> list[ObstacleSet]:
    """A post dead ahead that wobbles a millimeter left and right each frame, the way a measured one does."""
    return [post(0.001 if index % 2 else -0.001, 2.0, index * FRAME_SECONDS) for index in range(count)]


def sides(config: PlannerConfig, scenes: list[ObstacleSet]) -> list[float]:
    pipeline = PlannerPipeline(config, WALKER)
    return [float(np.sign(pipeline.plan(scene).lateral_offsets_meters[-1])) for scene in scenes]


def test_a_near_tie_keeps_its_side() -> None:
    scenes = alternating_posts(12)
    without = sides(replace(CONFIG, previous_plan_prior_enabled=False), scenes)
    # Without memory the millimeter decides, so the plan changes sides with the post.
    assert len(set(without)) == 2
    with_prior = sides(CONFIG, scenes)
    assert len(set(with_prior)) == 1


def test_switching_the_prior_off_leaves_the_plan_what_a_fresh_planner_gives() -> None:
    scenes = alternating_posts(6)
    pipeline = PlannerPipeline(replace(CONFIG, previous_plan_prior_enabled=False), WALKER)
    for scene in scenes:
        fresh = PlannerPipeline(replace(CONFIG, previous_plan_prior_enabled=False), WALKER).plan(scene)
        assert np.array_equal(pipeline.plan(scene).lateral_offsets_meters, fresh.lateral_offsets_meters)


def test_the_first_frame_plans_as_if_there_were_no_prior() -> None:
    scene = post(0.3, 2.0, 0.0)
    with_prior = PlannerPipeline(CONFIG, WALKER).plan(scene)
    without = PlannerPipeline(replace(CONFIG, previous_plan_prior_enabled=False), WALKER).plan(scene)
    assert np.array_equal(with_prior.lateral_offsets_meters, without.lateral_offsets_meters)


def test_the_alarm_does_not_depend_on_the_prior() -> None:
    scenes = [post(0.0, forward, index * FRAME_SECONDS) for index, forward in enumerate(np.linspace(2.0, 0.6, 30))]
    with_prior = PlannerPipeline(CONFIG, WALKER)
    without = PlannerPipeline(replace(CONFIG, previous_plan_prior_enabled=False), WALKER)
    alarms_with = [with_prior.plan(scene).alarm for scene in scenes]
    assert alarms_with == [without.plan(scene).alarm for scene in scenes]
    assert True in alarms_with and False in alarms_with, "the scenes must raise and clear the alarm"


def test_holding_a_side_is_not_counted_as_scene_information() -> None:
    # A post pushes the plan aside, then the view empties. The prior still pulls toward the old
    # sidestep, but that is the walker's memory: with nothing in view, the camera added nothing.
    pipeline = PlannerPipeline(CONFIG, WALKER)
    pipeline.plan(post(0.0, 1.5, 0.0))
    after = pipeline.plan(ObstacleSet(FRAME_SECONDS, (), 0))
    assert after.scene_information_bits == pytest.approx(0.0, abs=1e-9)


def test_the_cost_figure_leaves_out_the_prior() -> None:
    # The same post twice. The second frame remembers the first plan and makes the same plan again, so
    # the scene and the goal charge exactly what a planner with no memory charges for that path. The
    # user model's work figure reads this cost, and holding a side isn't work.
    remembering = PlannerPipeline(CONFIG, WALKER)
    remembering.plan(post(0.0, 1.5, 0.0))
    second = remembering.plan(post(0.0, 1.5, FRAME_SECONDS))
    fresh = PlannerPipeline(replace(CONFIG, previous_plan_prior_enabled=False), WALKER)
    fresh.plan(post(0.0, 1.5, 0.0))
    without = fresh.plan(post(0.0, 1.5, FRAME_SECONDS))
    assert np.array_equal(second.lateral_offsets_meters, without.lateral_offsets_meters)
    assert second.cumulative_cost_bits == pytest.approx(without.cumulative_cost_bits, rel=1e-9)


def test_last_field_includes_the_prior() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)
    empty = ObstacleSet(0.0, (), 0)
    pipeline.plan(empty)
    first = pipeline.last_field.copy()
    pipeline.plan(ObstacleSet(FRAME_SECONDS, (), 0))
    # With nothing in view the field is the goal alone, so any difference is the prior.
    assert np.any(pipeline.last_field != first)


@pytest.mark.parametrize("name, value", [("previous_plan_spread_meters", 0.0), ("previous_plan_spread_meters", -1.0), ("previous_plan_prior_seconds", 0.0), ("previous_plan_max_gap_seconds", -0.1)])
def test_a_bad_prior_constant_stops_the_pipeline_at_construction(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        PlannerPipeline(replace(CONFIG, **{name: value}), WALKER)
    # Even when the prior is off, so switching it on can't start a run with a bad constant.
    with pytest.raises(ValueError, match=name):
        PlannerPipeline(replace(CONFIG, previous_plan_prior_enabled=False, **{name: value}), WALKER)
