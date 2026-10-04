"""Covers reading the arrow off the path at the lookahead, against angles worked out by hand."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.heading import lookahead_heading, lookahead_step_index
from nav.planner.pipeline import PlannerPipeline
from nav.planner.field import step_count
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)


def _bend(first_moving_step: int, cells_per_step: int = 1, direction: float = 1.0) -> np.ndarray:
    # Straight until first_moving_step, then sideways at a fixed number of cells per step. A path
    # that holds still for its first step reads zero under a first-step rule, so it tells the two apart.
    steps = np.arange(step_count(CONFIG))
    moving = np.maximum(steps - (first_moving_step - 1), 0)
    return direction * moving * cells_per_step * CONFIG.grid_spacing_meters


def test_the_heading_is_the_angle_to_the_path_at_the_lookahead() -> None:
    offsets = _bend(first_moving_step=2)
    assert offsets[1] == 0.0, "the fixture must hold still for its first step"

    # 0.9 m across at step 10, against 10 steps of 0.1 s at 1.4 m/s, so 1.4 m forward.
    assert lookahead_heading(offsets, 10, CONFIG) == pytest.approx(0.571337, abs=1e-6)


def test_the_heading_is_negative_when_the_path_goes_left() -> None:
    assert lookahead_heading(_bend(first_moving_step=2, direction=-1.0), 10, CONFIG) == pytest.approx(-0.571337, abs=1e-6)


def test_a_straight_path_reads_zero() -> None:
    assert lookahead_heading(_bend(first_moving_step=2, cells_per_step=0), 10, CONFIG) == 0.0


def test_the_lookahead_index_rounds_configured_values() -> None:
    # 0.3 / 0.1 and 0.7 / 0.1 come out just under 3 and 7 in floating point, so a floor gets them wrong.
    for seconds, expected in ((1.0, 10), (0.3, 3), (0.7, 7), (3.8, 38)):
        assert lookahead_step_index(replace(CONFIG, heading_lookahead_seconds=seconds)) == expected


def test_a_full_sidestep_reads_exactly_the_sidestep_limit() -> None:
    limit = np.arctan2(CONFIG.max_lateral_speed_mps, CONFIG.walking_speed_mps)
    full_sidestep = _bend(first_moving_step=1)

    assert lookahead_heading(full_sidestep, 10, CONFIG) == pytest.approx(limit)


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_a_lookahead_that_is_not_a_positive_number_is_refused(seconds: float) -> None:
    with pytest.raises(ValueError, match="heading_lookahead_seconds"):
        lookahead_step_index(replace(CONFIG, heading_lookahead_seconds=seconds))


def test_a_lookahead_longer_than_the_horizon_is_refused() -> None:
    with pytest.raises(ValueError, match="heading_lookahead_seconds.*horizon"):
        lookahead_step_index(replace(CONFIG, heading_lookahead_seconds=3.9))


def test_a_lookahead_under_half_a_step_is_refused() -> None:
    with pytest.raises(ValueError, match="heading_lookahead_seconds"):
        lookahead_step_index(replace(CONFIG, heading_lookahead_seconds=0.04))


def test_the_pipeline_refuses_a_bad_lookahead_when_built() -> None:
    with pytest.raises(ValueError, match="heading_lookahead_seconds"):
        PlannerPipeline(replace(CONFIG, heading_lookahead_seconds=0.0), WALKER)
