"""The stereo cue's numbers, against the values worked out when the cue was designed, and its refusals."""

# Standard library imports
from dataclasses import replace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.audio import alarm_pan, check_audio_config, ear_gains, far_ear_gain, heading_tolerance_radians
from nav.planner.config import PlannerConfig

CONFIG = PlannerConfig()


def test_the_tolerance_is_the_goal_tolerance_seen_as_an_angle() -> None:
    # atan(1.5 / 4.0), the shipped goal tolerance over the goal distance.
    assert np.degrees(heading_tolerance_radians(CONFIG)) == pytest.approx(20.556, abs=0.01)


@pytest.mark.parametrize(
    "degrees, expected",
    [(0.0, 1.0), (4.0, 0.9812), (10.0, 0.8884), (20.0, 0.6229), (35.5, 0.2251)],
    ids=["straight", "ordinary wobble", "10 deg", "20 deg", "sidestep limit"],
)
def test_the_far_ear_follows_the_surprise_of_the_heading_error(degrees: float, expected: float) -> None:
    assert far_ear_gain(np.radians(degrees), CONFIG) == pytest.approx(expected, abs=0.0005)
    # Either sign of heading gives the same gain. Which ear is far is decided elsewhere.
    assert far_ear_gain(-np.radians(degrees), CONFIG) == pytest.approx(expected, abs=0.0005)


def test_the_far_ear_never_goes_under_the_floor() -> None:
    assert far_ear_gain(np.radians(90.0), CONFIG) == pytest.approx(CONFIG.far_ear_floor_gain)
    # With no floor the Gaussian itself is what is left: e to the minus half of (90 / 20.56) squared, 7e-5.
    assert far_ear_gain(np.radians(90.0), replace(CONFIG, far_ear_floor_gain=0.0)) == pytest.approx(6.9e-5, abs=1e-5)


def test_heading_right_quiets_the_left_ear_and_heading_left_the_right() -> None:
    left, right = ear_gains(np.radians(20.0), False, None, CONFIG)
    assert (left, right) == pytest.approx((0.6229, 1.0), abs=0.0005)
    left, right = ear_gains(-np.radians(20.0), False, None, CONFIG)
    assert (left, right) == pytest.approx((1.0, 0.6229), abs=0.0005)


def test_straight_ahead_leaves_both_ears_at_full() -> None:
    assert ear_gains(0.0, False, None, CONFIG) == (1.0, 1.0)


def test_an_alarm_ducks_the_dangers_ear_to_the_floor() -> None:
    floor = CONFIG.far_ear_floor_gain
    # Danger on the left, path steering right: the left ear is both the far ear and the danger's.
    assert ear_gains(np.radians(20.0), True, -1.0, CONFIG) == pytest.approx((floor, 1.0))
    # Danger on the right while the path still goes right: the right ear drops, the left keeps its steering gain.
    assert ear_gains(np.radians(20.0), True, 1.0, CONFIG) == pytest.approx((0.6229, floor), abs=0.0005)


def test_a_danger_dead_ahead_ducks_both_ears() -> None:
    floor = CONFIG.far_ear_floor_gain
    assert ear_gains(0.0, True, 0.0, CONFIG) == pytest.approx((floor, floor))


def test_an_alarm_with_no_known_side_changes_nothing() -> None:
    # The control for the two tests above: the same heading, alarm up, no pan.
    assert ear_gains(np.radians(20.0), True, None, CONFIG) == ear_gains(np.radians(20.0), False, None, CONFIG)


def test_the_alarm_pan_is_full_at_the_corridors_edge_and_clipped_past_it() -> None:
    edge = CONFIG.body_half_width_meters
    assert alarm_pan(0.0, CONFIG) == 0.0
    assert alarm_pan(edge / 2, CONFIG) == pytest.approx(0.5)
    assert alarm_pan(-edge, CONFIG) == pytest.approx(-1.0)
    assert alarm_pan(3 * edge, CONFIG) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_lateral_or_heading_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        alarm_pan(value, CONFIG)
    with pytest.raises(ValueError, match="finite"):
        ear_gains(value, False, None, CONFIG)


@pytest.mark.parametrize(
    "name, value",
    [
        ("far_ear_floor_gain", -0.1),
        ("far_ear_floor_gain", 1.5),
        ("far_ear_floor_gain", float("nan")),
        ("goal_distance_meters", 0.0),
        ("goal_tolerance_meters", 0.0),
    ],
)
def test_a_bad_cue_constant_is_refused_by_name(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        check_audio_config(replace(CONFIG, **{name: value}))
