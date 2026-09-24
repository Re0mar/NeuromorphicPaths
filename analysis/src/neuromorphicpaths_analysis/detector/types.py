"""
Result types of the path detector.

Normalized coordinates run 0 to 1 across the ORIGINAL frame, with the letterbox padding
removed, the same convention as the app's PathResult.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy


@dataclass(frozen=True)
class LetterboxGeometry:
    """Where the original frame sits inside the square model input."""

    source_width: int
    source_height: int
    input_width: int
    input_height: int
    content_width: int
    content_height: int
    pad_x: int
    pad_y: int

    def to_image_x(self, input_fraction: float) -> float:
        """Map an x fraction of the model input to an x fraction of the original frame."""
        return min(max((input_fraction * self.input_width - self.pad_x) / self.content_width, 0.0), 1.0)

    def to_image_y(self, input_fraction: float) -> float:
        """Map a y fraction of the model input to a y fraction of the original frame."""
        return min(max((input_fraction * self.input_height - self.pad_y) / self.content_height, 0.0), 1.0)

    def content_cells(self, grid_height: int, grid_width: int) -> numpy.ndarray:
        """
        Mask-grid cells whose center lies on the real image rather than on the padding.

        Mirrors isEmptyImageCell in the app, including its inclusive bounds.

        :return: Boolean array of shape (grid_height, grid_width).
        :rtype: numpy.ndarray
        """
        center_x = (numpy.arange(grid_width) + 0.5) / grid_width * self.input_width
        center_y = (numpy.arange(grid_height) + 0.5) / grid_height * self.input_height
        inside_x = (center_x >= self.pad_x) & (center_x <= self.pad_x + self.content_width)
        inside_y = (center_y >= self.pad_y) & (center_y <= self.pad_y + self.content_height)
        return numpy.outer(inside_y, inside_x)


@dataclass(frozen=True)
class TracedRow:
    """
    One row of the path outline, traced upward from the bottom center.

    A clipped side means the run reached the edge of the image content, so that x is the
    frame border and not the path edge. Anything measuring position or fitting edge lines
    has to skip clipped sides.
    """

    grid_y: int
    left_cell: int
    right_cell: int
    y: float
    left_x: float
    right_x: float
    left_clipped: bool
    right_clipped: bool


@dataclass(frozen=True)
class PathResult:
    """What one frame produced. grid_mask is empty when nothing was detected."""

    geometry: LetterboxGeometry
    grid_mask: numpy.ndarray
    rows: list[TracedRow]
    top_score: float
    box: tuple[float, float, float, float] | None
    debug_info: str
    # Left, top, right, bottom of the region the mask was allowed into: the detection box plus
    # any margin the decoder adds. A mask edge running along one of its sides is the box, not
    # the path. None for label outlines, which have no box.
    mask_bounds: tuple[float, float, float, float] | None = None
