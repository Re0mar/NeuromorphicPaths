"""Covers the path's color ramp and fill opacity against the module's own constants."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sinks.path_style import (
    FILL_OPACITY_FLOOR,
    FILL_OPACITY_FULL_AT_BITS,
    SURPRISE_HIGH_RGB,
    SURPRISE_LOW_RGB,
    SURPRISE_RED_BITS,
    path_color_rgb,
    path_fill_opacity,
)


def test_zero_surprise_is_the_low_color_and_red_threshold_is_the_high_color() -> None:
    # Through OKLab and back, so a channel may come back one step off its starting value.
    assert np.abs(np.array(path_color_rgb(0.0)) - np.array(SURPRISE_LOW_RGB)).max() <= 1
    assert np.abs(np.array(path_color_rgb(SURPRISE_RED_BITS)) - np.array(SURPRISE_HIGH_RGB)).max() <= 1


def test_surprise_past_the_red_threshold_stays_red() -> None:
    assert path_color_rgb(SURPRISE_RED_BITS * 10.0) == path_color_rgb(SURPRISE_RED_BITS)


def test_the_color_changes_monotonically_toward_red() -> None:
    steps = [path_color_rgb(bits) for bits in np.linspace(0.0, SURPRISE_RED_BITS, 9)]
    reds = [color[0] for color in steps]
    blues = [color[2] for color in steps]

    assert reds == sorted(reds)
    assert blues == sorted(blues, reverse=True)


def test_the_middle_of_the_ramp_is_not_darker_than_either_end() -> None:
    # The reason to blend in OKLab. A plain sRGB blend of this blue and red sags in the middle.
    def luma(color: tuple[int, int, int]) -> float:
        return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

    middle = luma(path_color_rgb(SURPRISE_RED_BITS / 2.0))

    assert middle >= min(luma(SURPRISE_LOW_RGB), luma(SURPRISE_HIGH_RGB)) - 1.0


def test_fill_opacity_is_the_floor_at_zero_bits_and_solid_from_full() -> None:
    assert path_fill_opacity(0.0) == pytest.approx(FILL_OPACITY_FLOOR)
    assert path_fill_opacity(FILL_OPACITY_FULL_AT_BITS) == pytest.approx(1.0)
    assert path_fill_opacity(FILL_OPACITY_FULL_AT_BITS * 4.0) == pytest.approx(1.0)
    assert FILL_OPACITY_FLOOR < path_fill_opacity(FILL_OPACITY_FULL_AT_BITS / 2.0) < 1.0


def test_a_negative_surprise_is_refused() -> None:
    with pytest.raises(ValueError, match="avoidance_surprise_bits must be finite and zero or more, got -0.5"):
        path_color_rgb(-0.5)


def test_a_non_finite_information_is_refused() -> None:
    with pytest.raises(ValueError, match="scene_information_bits must be finite and zero or more, got nan"):
        path_fill_opacity(float("nan"))
