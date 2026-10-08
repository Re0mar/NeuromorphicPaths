"""The danger's side as the pipeline reports it: read when the alarm raises, kept through the hold, cleared after."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.alarm import nearest_corridor_point
from nav.planner.config import PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)


def _point(lateral: float, forward: float, group: int = 1) -> ObstaclePoint:
    clearance = max(0.0, float(np.hypot(lateral, forward)) - WALKER.radius_meters)
    return ObstaclePoint(lateral, forward, group, clearance, 0.1, None, None, False, np.zeros(3))


def _at(timestamp: float, *points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(timestamp_seconds=timestamp, points=tuple(points), groups_in_view=len({p.group_id for p in points}))


def test_the_nearest_corridor_point_is_the_one_with_the_least_clearance() -> None:
    far = _point(0.1, 2.0, group=1)
    near = _point(-0.1, 0.9, group=2)
    beside = _point(1.0, 0.5, group=3)  # Outside the corridor, however near.

    assert nearest_corridor_point(_at(0.0, far, near, beside), CONFIG) is near
    assert nearest_corridor_point(_at(0.0, beside), CONFIG) is None


def test_a_raised_alarm_carries_the_dangers_side_and_ducks_that_ear() -> None:
    # 0.9 m ahead, 0.15 m to the left: raises (0.39 s to contact), halfway to the corridor's left edge.
    path = PlannerPipeline(CONFIG, WALKER).plan(_at(0.0, _point(-0.15, 0.9)))

    assert path.alarm is True
    assert path.alarm_pan == pytest.approx(-0.5)
    assert path.ear_gain_left == pytest.approx(CONFIG.far_ear_floor_gain)
    assert path.ear_gain_right <= 1.0


def test_the_side_is_kept_through_the_hold_and_cleared_when_it_ends() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)
    pipeline.plan(_at(0.0, _point(-0.15, 0.9)))

    held = pipeline.plan(_at(0.1))  # Nothing in the corridor, but the 0.5 s hold keeps the alarm up.
    assert held.alarm is True
    assert held.alarm_pan == pytest.approx(-0.5), "the cue stays on the side the walker last heard it"
    assert held.ear_gain_left == pytest.approx(CONFIG.far_ear_floor_gain)

    cleared = pipeline.plan(_at(1.0))
    assert cleared.alarm is False
    assert cleared.alarm_pan is None
    assert (cleared.ear_gain_left, cleared.ear_gain_right) == (1.0, 1.0)


def test_a_new_danger_on_the_other_side_moves_the_cue_at_once() -> None:
    pipeline = PlannerPipeline(CONFIG, WALKER)
    pipeline.plan(_at(0.0, _point(-0.15, 0.9)))

    moved = pipeline.plan(_at(0.1, _point(0.3, 0.9)))
    assert moved.alarm_pan == pytest.approx(1.0)
    assert moved.ear_gain_right == pytest.approx(CONFIG.far_ear_floor_gain)


def test_an_empty_scene_gives_no_side_and_full_volume_in_both_ears() -> None:
    path = PlannerPipeline(CONFIG, WALKER).plan(_at(0.0))

    assert path.alarm is False
    assert path.alarm_pan is None
    assert (path.ear_gain_left, path.ear_gain_right) == (1.0, 1.0)
