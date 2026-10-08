"""
How the planned path and the obstacles look, defined once for every display that draws them.

Color runs blue to red with the avoidance surprise, and the fill gets more solid the more the scene
shaped the plan. The borders stay at one opacity whatever the fill does, so the path's direction is
visible after the fill has faded. The depth view and the web page both read these, the page as
values the laptop sends, so neither can drift from the other.

The red point is passed in rather than set here. It's the alarm's threshold in bits, which only the
planner's config knows, and a constant here would drift the first time that threshold changed.

Plain numpy. No OpenCV, so the web sink's message builder can use it, and no aiohttp.
"""

# Third party imports
import numpy as np

FILL_OPACITY_FLOOR = 0.7  # Never fainter than this, so a plan nothing shaped is still clearly drawn.
# A display choice, unrelated to the alarm: one bit of scene information is when the fill goes solid.
FILL_OPACITY_FULL_AT_BITS = 1.0
BORDER_OPACITY = 0.6
# sRGB, 0 to 255.
SURPRISE_LOW_RGB = (47, 111, 255)
SURPRISE_HIGH_RGB = (235, 48, 48)
# Obstacle groups amber and walls magenta, in the depth view and on the page alike.
GROUP_RGB = (255, 200, 0)
WALL_RGB = (255, 0, 255)

# Linear sRGB to and from OKLab, Björn Ottosson's published matrices (bottosson.github.io/posts/oklab).
# Blending there rather than in sRGB keeps the middle of the ramp from going dark and muddy.
_LINEAR_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_CUBE_ROOT_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_OKLAB_TO_LMS_CUBE_ROOT = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_LMS_TO_LINEAR = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


def path_color_rgb(avoidance_surprise_bits: float, red_from_bits: float) -> tuple[int, int, int]:
    """
    The path's color for this much avoidance surprise: blue at 0 bits, red from `red_from_bits`.

    :param avoidance_surprise_bits: From the planned path.
    :param red_from_bits: Where the path is fully red, the alarm's threshold from `path_red_from_bits`.
    :return: sRGB, each 0 to 255.
    :rtype: tuple[int, int, int]
    :raises ValueError: When the bits are negative or not finite, or the red point is not above zero.
    """
    _check_bits(avoidance_surprise_bits, "avoidance_surprise_bits")
    if not np.isfinite(red_from_bits) or red_from_bits <= 0.0:
        raise ValueError(f"red_from_bits must be finite and above zero, got {red_from_bits}")
    fraction = min(1.0, avoidance_surprise_bits / red_from_bits)
    low = _srgb_to_oklab(np.array(SURPRISE_LOW_RGB, dtype=np.float64))
    high = _srgb_to_oklab(np.array(SURPRISE_HIGH_RGB, dtype=np.float64))
    blended = _oklab_to_srgb(low + fraction * (high - low))
    return int(blended[0]), int(blended[1]), int(blended[2])


def path_fill_opacity(scene_information_bits: float) -> float:
    """
    How solid the path's fill is: FILL_OPACITY_FLOOR with nothing shaping the plan, solid from one bit.

    :param scene_information_bits: From the planned path.
    :return: Opacity, FILL_OPACITY_FLOOR to 1.
    :rtype: float
    :raises ValueError: When the bits are negative or not finite.
    """
    _check_bits(scene_information_bits, "scene_information_bits")
    return FILL_OPACITY_FLOOR + (1.0 - FILL_OPACITY_FLOOR) * min(1.0, scene_information_bits / FILL_OPACITY_FULL_AT_BITS)


def _check_bits(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and zero or more, got {value}")


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    encoded = rgb / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    return _LMS_CUBE_ROOT_TO_OKLAB @ np.cbrt(_LINEAR_TO_LMS @ linear)


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    linear = np.clip(_LMS_TO_LINEAR @ (_OKLAB_TO_LMS_CUBE_ROOT @ lab) ** 3, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1.0 / 2.4) - 0.055)
    return np.round(encoded * 255.0)
