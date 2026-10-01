"""Covers the critically damped relaxation against properties the equation guarantees."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.usermodel.config import UserModelConfig
from nav.usermodel.relaxation import INTEGRATION_STEP_SECONDS, predicted_turn_time, relax_step

CONFIG = UserModelConfig(seconds_per_bit=0.25, heading_tolerance_radians=0.05)


def test_a_larger_error_takes_longer() -> None:
    assert predicted_turn_time(0.2, CONFIG) < predicted_turn_time(0.4, CONFIG) < predicted_turn_time(0.8, CONFIG)


def test_an_error_already_inside_tolerance_takes_no_time() -> None:
    assert predicted_turn_time(0.01, CONFIG) == 0.0


def test_the_sign_of_the_error_does_not_matter() -> None:
    assert predicted_turn_time(-0.4, CONFIG) == predicted_turn_time(0.4, CONFIG)


def test_the_solution_never_overshoots_zero() -> None:
    # Critically damped from rest is the fastest settling that never crosses zero. Every Euler
    # step must keep the sign of the starting error.
    error, velocity = 0.5, 0.0
    for _ in range(int(3.0 / INTEGRATION_STEP_SECONDS)):
        error, velocity = relax_step(error, velocity, INTEGRATION_STEP_SECONDS, CONFIG.seconds_per_bit)
        assert error >= -1e-9


def test_time_scales_linearly_with_b_for_a_fixed_error_ratio() -> None:
    # The equation's only time scale is b, so doubling b doubles the time to any fixed fraction.
    slow = UserModelConfig(seconds_per_bit=0.5, heading_tolerance_radians=0.05)

    assert predicted_turn_time(0.4, slow) == pytest.approx(2.0 * predicted_turn_time(0.4, CONFIG), rel=0.02)


def test_the_predicted_time_matches_the_closed_form() -> None:
    # From rest, the critically damped solution is s(t) = s0 (1 + t / b) exp(-t / b). The time
    # to reach the tolerance solves that. Known from outside the integrator.
    error, b, tolerance = 0.4, CONFIG.seconds_per_bit, CONFIG.heading_tolerance_radians
    times = np.arange(0.0, 5.0, 1e-4)
    closed_form = error * (1 + times / b) * np.exp(-times / b)
    expected = float(times[np.argmax(closed_form <= tolerance)])

    assert predicted_turn_time(error, CONFIG) == pytest.approx(expected, abs=0.005)


def test_an_error_that_cannot_settle_reports_infinity() -> None:
    # A b of a hundred seconds relaxes far too slowly to settle inside the maximum.
    glacial = UserModelConfig(seconds_per_bit=100.0, heading_tolerance_radians=0.05)

    assert predicted_turn_time(1.0, glacial) == np.inf


@pytest.mark.parametrize("config", [UserModelConfig(seconds_per_bit=0.0), UserModelConfig(seconds_per_bit=-0.25), UserModelConfig(heading_tolerance_radians=0.0)])
def test_a_degenerate_config_is_refused(config: UserModelConfig) -> None:
    with pytest.raises(ValueError):
        predicted_turn_time(0.4, config)


def test_relax_step_refuses_a_non_positive_b() -> None:
    with pytest.raises(ValueError, match="seconds_per_bit"):
        relax_step(0.4, 0.0, 0.001, 0.0)
