"""The circular exponential filter on the route's relative bearing."""

# Third party imports
import pytest

# Local package imports
from nav.route.bearing_filter import CircularExponentialFilter


def test_the_filter_is_unknown_until_one_sample() -> None:
    bearing_filter = CircularExponentialFilter(2.0)
    assert bearing_filter.value() is None
    bearing_filter.update(42.0, 0.0)
    assert bearing_filter.value() == pytest.approx(42.0)


def test_a_constant_bearing_stays_put_under_noise() -> None:
    bearing_filter = CircularExponentialFilter(2.0)
    for index in range(40):
        bearing_filter.update(90.0 + (8.0 if index % 2 else -8.0), index * 1.0)
    # Alternating plus and minus 8 degrees each second averages near 90, well inside the swing.
    assert bearing_filter.value() == pytest.approx(90.0, abs=4.0)


def test_the_filter_crosses_north_without_a_jump() -> None:
    bearing_filter = CircularExponentialFilter(1.0)
    bearing_filter.update(350.0, 0.0)
    # One time constant later, pulled 1 - 1/e of the way to 10: about 2.6, and never near 180.
    bearing_filter.update(10.0, 1.0)
    value = bearing_filter.value()
    assert value is not None
    assert value == pytest.approx(2.6, abs=0.3)


def test_350_and_10_average_to_0_not_180() -> None:
    bearing_filter = CircularExponentialFilter(1.0)
    bearing_filter.update(350.0, 0.0)
    # After ln 2 time constants the pull is exactly a half, so 350 and 10 weigh the same.
    bearing_filter.update(10.0, 0.6931471805599453)
    value = bearing_filter.value()
    assert value is not None
    assert min(value, 360.0 - value) == pytest.approx(0.0, abs=1e-6)


def test_a_backward_timestamp_resets() -> None:
    bearing_filter = CircularExponentialFilter(10.0)
    bearing_filter.update(0.0, 5.0)
    bearing_filter.update(120.0, 4.0)
    assert bearing_filter.value() == pytest.approx(120.0)


def test_a_zero_time_constant_follows_each_sample() -> None:
    bearing_filter = CircularExponentialFilter(0.0)
    bearing_filter.update(10.0, 0.0)
    bearing_filter.update(200.0, 0.1)
    assert bearing_filter.value() == pytest.approx(200.0)


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_a_negative_time_constant_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="time constant"):
        CircularExponentialFilter(value)
