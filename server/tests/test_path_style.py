"""Covers the path's color ramp and fill opacity against the module's own constants and a passed red point."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.path_style import (
    FILL_OPACITY_FLOOR,
    FILL_OPACITY_FULL_AT_BITS,
    SURPRISE_HIGH_RGB,
    SURPRISE_LOW_RGB,
    path_color_rgb,
    path_fill_opacity,
)

# Neither 0.72 nor the shipped 1.47, so a test can only pass by using the red point it is handed.
RED = 1.1


def test_zero_surprise_is_the_low_color_and_red_threshold_is_the_high_color() -> None:
    # Through OKLab and back, so a channel may come back one step off its starting value.
    assert np.abs(np.array(path_color_rgb(0.0, RED)) - np.array(SURPRISE_LOW_RGB)).max() <= 1
    assert np.abs(np.array(path_color_rgb(RED, RED)) - np.array(SURPRISE_HIGH_RGB)).max() <= 1


def test_surprise_past_the_red_threshold_stays_red() -> None:
    assert path_color_rgb(RED * 10.0, RED) == path_color_rgb(RED, RED)


def test_the_color_changes_monotonically_toward_red() -> None:
    steps = [path_color_rgb(bits, RED) for bits in np.linspace(0.0, RED, 9)]
    reds = [color[0] for color in steps]
    blues = [color[2] for color in steps]

    assert reds == sorted(reds)
    assert blues == sorted(blues, reverse=True)


def test_the_middle_of_the_ramp_is_not_darker_than_either_end() -> None:
    # The reason to blend in OKLab. A plain sRGB blend of this blue and red sags in relative
    # luminance (the light that actually reaches the eye) to 0.136 in the middle against 0.192 and
    # 0.200 at the ends. Gamma-encoded luma hides that sag, so it is not the measure used here.
    def luminance(color: tuple[int, int, int]) -> float:
        encoded = np.array(color, dtype=np.float64) / 255.0
        linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
        return float(linear @ np.array([0.2126, 0.7152, 0.0722]))

    middle = luminance(path_color_rgb(RED / 2.0, RED))

    assert middle >= min(luminance(SURPRISE_LOW_RGB), luminance(SURPRISE_HIGH_RGB)) - 0.01


def test_fill_opacity_is_the_floor_at_zero_bits_and_solid_from_full() -> None:
    assert path_fill_opacity(0.0) == pytest.approx(FILL_OPACITY_FLOOR)
    assert path_fill_opacity(FILL_OPACITY_FULL_AT_BITS) == pytest.approx(1.0)
    assert path_fill_opacity(FILL_OPACITY_FULL_AT_BITS * 4.0) == pytest.approx(1.0)
    assert FILL_OPACITY_FLOOR < path_fill_opacity(FILL_OPACITY_FULL_AT_BITS / 2.0) < 1.0


def test_a_negative_surprise_is_refused() -> None:
    with pytest.raises(ValueError, match="avoidance_surprise_bits must be finite and zero or more, got -0.5"):
        path_color_rgb(-0.5, RED)


def test_a_non_finite_information_is_refused() -> None:
    with pytest.raises(ValueError, match="scene_information_bits must be finite and zero or more, got nan"):
        path_fill_opacity(float("nan"))


def test_the_red_point_is_the_one_passed_in() -> None:
    # Below the red point the path isn't fully red yet. At it, it is. So the threshold moves with the argument.
    assert path_color_rgb(1.0, 2.0) != path_color_rgb(2.0, 2.0)
    assert path_color_rgb(1.0, 1.0) == path_color_rgb(2.0, 2.0)


@pytest.mark.parametrize("red_from_bits", [0.0, -1.0, float("nan"), float("inf")])
def test_a_red_point_that_is_not_a_positive_number_is_refused(red_from_bits: float) -> None:
    with pytest.raises(ValueError, match="red_from_bits must be finite and above zero"):
        path_color_rgb(0.5, red_from_bits)
