"""Covers the one conversion from the planner's natural-log values to bits."""

# Standard library imports
import math

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.units import NATS_PER_BIT, bits_from_nats


def test_one_bit_is_ln_2_nats() -> None:
    assert NATS_PER_BIT == pytest.approx(math.log(2.0))
    assert bits_from_nats(math.log(2.0)) == pytest.approx(1.0)


def test_arrays_convert_elementwise() -> None:
    nats = np.array([0.0, math.log(2.0), math.log(8.0)])

    np.testing.assert_allclose(bits_from_nats(nats), [0.0, 1.0, 3.0])


def test_a_negative_value_stays_negative_rather_than_being_masked() -> None:
    # The helper refuses nothing. A sign error upstream has to reach the caller's own check.
    assert bits_from_nats(-math.log(2.0)) == pytest.approx(-1.0)
