"""Covers the work meter on scripted sequences of planner output and observed heading."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.types import PlannedPath
from nav.usermodel.config import UserModelConfig
from nav.usermodel.work import WorkMeter

CONFIG = UserModelConfig(seconds_per_bit=0.25, heading_tolerance_radians=0.05)


def _path(heading: float = 0.0, alarm: bool = False, cost: float = 1.0, timestamp: float = 0.0) -> PlannedPath:
    return PlannedPath(
        timestamp_seconds=timestamp,
        times_seconds=np.array([0.0, 0.1]),
        lateral_offsets_meters=np.array([0.0, 0.0]),
        lookahead_heading_radians=heading,
        alarm=alarm,
        cumulative_cost_bits=cost,
        scene_information_bits=0.0,
        avoidance_surprise_bits=0.0,
    )


def _run(meter: WorkMeter, script: list[tuple[PlannedPath, float]], step: float = 0.1) -> None:
    for index, (path, observed) in enumerate(script):
        meter.observe(path, observed, index * step)


def test_an_alarm_then_a_return_is_one_episode_whose_work_is_the_cost_difference() -> None:
    meter = WorkMeter(CONFIG)
    script = [
        (_path(cost=1.0), 0.0),  # calm
        (_path(alarm=True, heading=0.3, cost=9.0), 0.0),  # planner asks for a turn, cost 9
        (_path(alarm=True, heading=0.3, cost=8.0), 0.2),  # walker turning
        (_path(heading=0.1, cost=5.0), 0.2),  # planner easing
        (_path(heading=0.0, cost=2.0), 0.1),  # planner settled, walker still turned
        (_path(heading=0.0, cost=1.5), 0.0),  # both settled, episode closes at cost 1.5
        (_path(heading=0.0, cost=1.5), 0.0),
    ]

    _run(meter, script)
    episodes = meter.completed_episodes()

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.start_seconds == pytest.approx(0.1)
    assert episode.end_seconds == pytest.approx(0.5)
    assert episode.cost_at_start_bits == pytest.approx(9.0)
    assert episode.cost_at_end_bits == pytest.approx(1.5)
    assert episode.work_bits == pytest.approx(7.5)
    # Walker left tolerance at 0.2 s and was back by 0.5 s.
    assert episode.observed_turn_seconds == pytest.approx(0.3)
    assert episode.predicted_turn_seconds > 0.0
    assert meter.episode_open is False


def test_a_heading_outside_tolerance_opens_an_episode_without_an_alarm() -> None:
    meter = WorkMeter(CONFIG)
    _run(meter, [(_path(heading=0.2, cost=4.0), 0.0), (_path(heading=0.0, cost=1.0), 0.0)])

    assert len(meter.completed_episodes()) == 1
    assert meter.completed_episodes()[0].work_bits == pytest.approx(3.0)


def test_a_sequence_that_never_leaves_tolerance_produces_no_episode() -> None:
    meter = WorkMeter(CONFIG)
    _run(meter, [(_path(heading=0.01, cost=1.0), 0.02)] * 10)

    assert meter.completed_episodes() == []
    assert meter.episode_open is False


def test_an_episode_that_opens_and_never_closes_is_not_completed() -> None:
    meter = WorkMeter(CONFIG)
    _run(meter, [(_path(alarm=True, heading=0.3, cost=9.0), 0.3)] * 10)

    assert meter.completed_episodes() == []
    assert meter.episode_open is True


def test_the_episode_stays_open_while_the_walker_is_still_turned() -> None:
    # Planner settled, walker not yet. The work is not done until the walker is back.
    meter = WorkMeter(CONFIG)
    _run(meter, [(_path(heading=0.3, cost=9.0), 0.0), (_path(heading=0.0, cost=2.0), 0.3), (_path(heading=0.0, cost=2.0), 0.3)])

    assert meter.episode_open is True
    assert meter.completed_episodes() == []


def test_an_episode_with_no_observed_turn_reports_none_for_the_observed_time() -> None:
    # The planner asked, the walker never visibly turned, then the planner stopped asking.
    meter = WorkMeter(CONFIG)
    _run(meter, [(_path(heading=0.3, cost=9.0), 0.0), (_path(heading=0.0, cost=2.0), 0.0)])

    episode = meter.completed_episodes()[0]
    assert episode.observed_turn_seconds is None
    assert episode.work_bits == pytest.approx(7.0)


def test_two_avoidances_make_two_episodes_in_order() -> None:
    meter = WorkMeter(CONFIG)
    one = [(_path(heading=0.3, cost=5.0), 0.0), (_path(cost=1.0), 0.0)]
    two = [(_path(heading=-0.3, cost=7.0), 0.0), (_path(cost=1.0), 0.0)]
    _run(meter, one + two)

    works = [episode.work_bits for episode in meter.completed_episodes()]
    assert works == pytest.approx([4.0, 6.0])


def test_completed_episodes_returns_a_copy() -> None:
    meter = WorkMeter(CONFIG)
    _run(meter, [(_path(heading=0.3, cost=5.0), 0.0), (_path(cost=1.0), 0.0)])

    meter.completed_episodes().clear()

    assert len(meter.completed_episodes()) == 1


def test_the_user_model_imports_nothing_from_the_sinks() -> None:
    import ast
    from pathlib import Path

    import nav.usermodel

    for path in Path(nav.usermodel.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert not any(name.startswith("nav.sinks") for name in imported), f"{path.name} reaches a sink"
