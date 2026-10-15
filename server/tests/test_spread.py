"""A prior's spread widened by readings of the frame, against values worked out by hand."""

# Third party imports
import pytest

# Local package imports
from nav.planner.spread import spread_meters, spread_variance


def test_with_no_growth_the_spread_is_its_base() -> None:
    assert spread_meters(0.37, []) == pytest.approx(0.37)
    assert spread_variance(0.37, [(5.0, 0.0)]) == pytest.approx(0.1369)


def test_growths_add_their_variance() -> None:
    # 0.3 squared is 0.09, plus 2 x 0.5 is 1.0, plus 0.25 x 4 is 1.0: 2.09, a spread of 1.4457.
    assert spread_variance(0.3, [(2.0, 0.5), (0.25, 4.0)]) == pytest.approx(2.09)
    assert spread_meters(0.3, [(2.0, 0.5), (0.25, 4.0)]) == pytest.approx(1.4457, abs=1e-4)


@pytest.mark.parametrize("base", [0.0, -0.2, float("nan")])
def test_a_spread_not_above_zero_is_refused(base: float) -> None:
    with pytest.raises(ValueError, match="spread"):
        spread_variance(base, [])


@pytest.mark.parametrize("growth", [(-1.0, 1.0), (float("inf"), 1.0)])
def test_a_negative_or_infinite_growth_rate_is_refused(growth: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="growth"):
        spread_variance(0.4, [growth])


@pytest.mark.parametrize("growth", [(1.0, -0.5), (1.0, float("nan"))])
def test_a_negative_or_unknown_reading_is_refused(growth: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="reading"):
        spread_variance(0.4, [growth])
