"""Covers the contact term against values worked out by hand from the normal table, and its refusals."""

# Standard library imports
import warnings
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.contact import ContactSurprise, check_contact_config, contact_time_scale
from nav.planner.field import cost_field, effective_noise, lateral_grid
from nav.planner.surprise import point_surprise
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

CONFIG = PlannerConfig()
WALKER = WalkerConfig(radius_meters=0.35)
TERM = ContactSurprise()
# Twice the 0.30 m body half-width over 1.4 m/s. Hand-worked, so a change to either default shows up here.
TAU_SECONDS = 0.6 / 1.4


def _rate(centre_distance: float, noise: float, config: PlannerConfig = CONFIG) -> float:
    return float(TERM.point_cost(np.array([[centre_distance]]), np.array([noise]), config, WALKER)[0, 0])


def _point(lateral: float, forward: float, group: int, noise: float = 0.0, is_wall: bool = False) -> ObstaclePoint:
    clearance = max(0.0, float(np.hypot(lateral, forward)) - WALKER.radius_meters)
    # A point built by hand has no camera frame. The planner never reads camera_point.
    return ObstaclePoint(lateral, forward, group, clearance, noise, None, None, is_wall, np.zeros(3))


def _set(*points: ObstaclePoint) -> ObstacleSet:
    return ObstacleSet(timestamp_seconds=0.0, points=tuple(points), groups_in_view=len({p.group_id for p in points}))


def test_the_time_scale_is_the_time_to_walk_past_the_body() -> None:
    assert contact_time_scale(CONFIG) == pytest.approx(TAU_SECONDS)


def test_touching_costs_ln_two_per_tau() -> None:
    # Clearance exactly zero: Phi(0) is one half, so the surprise is ln 2.
    assert _rate(CONFIG.body_half_width_meters, 0.0) == pytest.approx(0.693147 / TAU_SECONDS, rel=1e-5)


def test_one_spread_clear_and_one_spread_overlapping_match_the_table() -> None:
    # With N = 0 the spread is the 0.10 m sway. Phi(1) = 0.841345 and Phi(-1) = 0.158655 from the table.
    one_spread_clear = CONFIG.body_half_width_meters + 0.10
    one_spread_overlapping = CONFIG.body_half_width_meters - 0.10
    assert _rate(one_spread_clear, 0.0) == pytest.approx(0.172754 / TAU_SECONDS, rel=1e-5)
    assert _rate(one_spread_overlapping, 0.0) == pytest.approx(1.841022 / TAU_SECONDS, rel=1e-5)


def test_cost_rises_as_the_point_comes_closer() -> None:
    distances = np.linspace(2.0, 0.0, 41)
    rates = [_rate(float(distance), 0.0337) for distance in distances]

    assert all(nearer > farther for farther, nearer in zip(rates, rates[1:]))


def test_a_steady_post_costs_more_to_walk_into_than_his_term_does() -> None:
    # 10 cm of overlap with the classroom's measured near noise. His term floors S at 0.06 m and
    # reads about 0.158 there. Contact reads about 4.11 per second.
    noise = 0.0337
    centre = CONFIG.body_half_width_meters - 0.10
    his = point_surprise(centre - WALKER.radius_meters, noise, CONFIG)

    assert his == pytest.approx(0.1577, abs=1e-3)
    assert _rate(centre, noise) == pytest.approx(4.112, abs=1e-2)
    assert _rate(centre, noise) > 20 * his


def test_a_distant_point_costs_next_to_nothing() -> None:
    assert _rate(CONFIG.body_half_width_meters + 2.0, 0.0337) < 1e-6


def test_a_wall_spreads_its_contact_cost_further() -> None:
    plain = _point(0.0, 0.5, group=1, noise=0.05)
    wall = _point(0.0, 0.5, group=1, noise=0.05, is_wall=True)
    centre = CONFIG.body_half_width_meters + 0.20

    assert _rate(centre, effective_noise(wall, CONFIG)) > 2 * _rate(centre, effective_noise(plain, CONFIG))


@pytest.mark.parametrize("lateral", [0.0, 0.2])
def test_the_total_over_a_fixed_path_does_not_depend_on_the_time_step(lateral: float) -> None:
    # One post 2 m ahead, and the path that holds one lateral offset all the way, priced straight
    # off the field rather than through plan, so a change in how far a step may move cannot leak in.
    # A per-slice form would double the total every time the step halves. As a rate it stays within
    # a Riemann sum's error of a 0.01 s reference: 1.2 % at 0.1 s and 5.7 % at 0.2 s straight
    # through the post, when worked out.
    post = _set(_point(0.0, 2.0, group=1, noise=0.0337))

    def total(time_step: float) -> float:
        config = replace(CONFIG, time_step_seconds=time_step)
        grid = lateral_grid(config)
        field = cost_field(post, grid, config, WALKER, terms=(TERM,))
        return float(field[:, int(np.argmin(np.abs(grid - lateral)))].sum() * time_step)

    reference = total(0.01)
    assert reference > 0.4, "the path must pass close enough to the post to cost something"
    for time_step in (0.05, 0.1, 0.2):
        assert total(time_step) == pytest.approx(reference, rel=0.10)


def test_zero_noise_is_finite() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rate = _rate(0.0, 0.0)

    assert np.isfinite(rate)


def test_the_shipped_sway_never_reaches_the_cap() -> None:
    # Centre distance zero is the deepest overlap there is. -ln Phi(-3) = 6.607726 from the table.
    assert _rate(0.0, 0.0) == pytest.approx(6.607726 / TAU_SECONDS, rel=1e-5)
    assert 6.607726 < CONFIG.contact_surprise_cap


def test_a_tiny_sway_reaches_the_cap_and_stays_finite() -> None:
    # Overlap can never exceed the half-width, so a small sway is the only way to the cap.
    tiny_sway = replace(CONFIG, walker_sway_meters=0.005)

    assert _rate(0.0, 0.0, tiny_sway) == pytest.approx(CONFIG.contact_surprise_cap / TAU_SECONDS)


@pytest.mark.parametrize("sway", [0.0, -0.1, float("nan")])
def test_a_non_positive_sway_is_refused(sway: float) -> None:
    with pytest.raises(ValueError, match="walker_sway_meters"):
        check_contact_config(replace(CONFIG, walker_sway_meters=sway))


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_a_non_positive_cap_is_refused(cap: float) -> None:
    with pytest.raises(ValueError, match="contact_surprise_cap"):
        check_contact_config(replace(CONFIG, contact_surprise_cap=cap))


def test_a_non_positive_body_half_width_is_refused() -> None:
    with pytest.raises(ValueError, match="body_half_width_meters"):
        check_contact_config(replace(CONFIG, body_half_width_meters=0.0))


def test_zero_walking_speed_is_refused_while_the_term_is_on() -> None:
    with pytest.raises(ValueError, match="walking_speed_mps"):
        check_contact_config(replace(CONFIG, walking_speed_mps=0.0))


def test_a_disabled_term_checks_nothing() -> None:
    check_contact_config(replace(CONFIG, contact_term_enabled=False, walker_sway_meters=0.0, walking_speed_mps=0.0))


def test_the_shipped_config_passes_its_own_check() -> None:
    check_contact_config(CONFIG)
