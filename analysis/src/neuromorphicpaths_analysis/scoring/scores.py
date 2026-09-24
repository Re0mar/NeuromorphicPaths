"""
Scores comparing the detector's mask with a hand-checked label.

Both are compared on the model's own prototype grid, so the model is not charged for detail
it cannot produce at its resolution. Two numbers per frame:

- IoU (overlap divided by union) over grid cells on the real image. The standard score, but
  dominated by the easy middle of the path.
- Edge error: how far the traced left and right edges sit from the label's, in path widths,
  over the lower part of the frame. This is the number the rider-position math reads.

Given a capture profile, it also runs the point-of-view estimate on both outlines and reports
how far the model moves pitch, position and height away from what the label gives. That turns
edge accuracy into the units the rest of the project uses.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import cv2
import numpy

# Local package imports
from neuromorphicpaths_analysis.detector import LetterboxGeometry, PathResult, TracedRow, trace_path
from neuromorphicpaths_analysis.geometry import CaptureProfile, PovEstimate, PovStatus, estimate_pov

# Only the part of the frame near the rider feeds the position math.
DEFAULT_MIN_IMAGE_Y = 0.5

# A grid cell counts as path when at least half its footprint is path in the label.
LABEL_CELL_THRESHOLD = 0.5


@dataclass(frozen=True)
class PovDifference:
    """Model estimate minus label estimate. None where either side could not be estimated."""

    pitch_deg: float | None
    position_fraction: float | None
    camera_height_m: float | None
    # Only a difference between two reliable estimates says anything about the model.
    both_reliable: bool = False


@dataclass(frozen=True)
class FrameScore:
    iou: float
    edge_error_mean: float | None
    edge_error_max: float | None
    sides_compared: int
    rows_missed: int
    pov_difference: PovDifference | None = None


def difference(model_value: float | None, label_value: float | None) -> float | None:
    if model_value is None or label_value is None:
        return None
    return model_value - label_value


def pov_difference(model: PovEstimate, label: PovEstimate) -> PovDifference:
    if model.status is not PovStatus.OK or label.status is not PovStatus.OK:
        return PovDifference(None, None, None)
    return PovDifference(
        pitch_deg=difference(model.pitch_deg, label.pitch_deg),
        position_fraction=difference(model.position_fraction, label.position_fraction),
        camera_height_m=difference(model.camera_height_m, label.camera_height_m),
        both_reliable=model.reliable and label.reliable,
    )


def label_to_grid(label_mask: numpy.ndarray, geometry: LetterboxGeometry, grid_shape: tuple[int, int]) -> numpy.ndarray:
    """
    Shrink a full-resolution label onto the model's mask grid.

    The label goes through the same letterbox as the frame, then each cell takes the share of
    its area that is path.

    :param label_mask: Boolean label at the original frame's resolution.
    :param grid_shape: (height, width) of the model's prototype grid.
    :return: Boolean grid, False on padding cells.
    :rtype: numpy.ndarray
    """
    if label_mask.shape != (geometry.source_height, geometry.source_width):
        raise ValueError(
            f"Label is {label_mask.shape[1]}x{label_mask.shape[0]} but the frame was "
            f"{geometry.source_width}x{geometry.source_height}"
        )
    content = cv2.resize(
        label_mask.astype(numpy.float32),
        (geometry.content_width, geometry.content_height),
        interpolation=cv2.INTER_AREA,
    )
    letterboxed = numpy.zeros((geometry.input_height, geometry.input_width), dtype=numpy.float32)
    letterboxed[
        geometry.pad_y:geometry.pad_y + geometry.content_height,
        geometry.pad_x:geometry.pad_x + geometry.content_width,
    ] = content
    grid_height, grid_width = grid_shape
    share = cv2.resize(letterboxed, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
    return (share >= LABEL_CELL_THRESHOLD) & geometry.content_cells(grid_height, grid_width)


def grid_iou(predicted: numpy.ndarray, truth: numpy.ndarray, content: numpy.ndarray) -> float:
    """
    Overlap divided by union over cells on the real image.

    :return: 1.0 when both are empty, since there is nothing to disagree about.
    :rtype: float
    """
    predicted = predicted & content
    truth = truth & content
    union = int((predicted | truth).sum())
    if union == 0:
        return 1.0
    return int((predicted & truth).sum()) / union


def edge_errors(
    predicted_rows: list[TracedRow],
    truth_rows: list[TracedRow],
    min_image_y: float = DEFAULT_MIN_IMAGE_Y,
) -> tuple[list[float], int]:
    """
    Signed edge offsets in path widths, row by row, for rows both traces reached.

    A side is skipped when either trace was clipped there, because a frame border is not an
    edge to measure against. Width comes from the label, so a detector that widens the path
    cannot shrink its own error.

    :return: The per-side errors, and how many label rows the detector's trace never reached.
    :rtype: tuple[list[float], int]
    """
    predicted_by_row = {row.grid_y: row for row in predicted_rows}
    errors: list[float] = []
    rows_missed = 0
    for truth in truth_rows:
        if truth.y < min_image_y:
            continue
        predicted = predicted_by_row.get(truth.grid_y)
        if predicted is None:
            rows_missed += 1
            continue
        path_width = truth.right_x - truth.left_x
        if path_width <= 0:
            continue
        if not (truth.left_clipped or predicted.left_clipped):
            errors.append((predicted.left_x - truth.left_x) / path_width)
        if not (truth.right_clipped or predicted.right_clipped):
            errors.append((predicted.right_x - truth.right_x) / path_width)
    return errors, rows_missed


def score_frame(
    result: PathResult,
    label_mask: numpy.ndarray,
    min_image_y: float = DEFAULT_MIN_IMAGE_Y,
    profile: CaptureProfile | None = None,
) -> FrameScore:
    """
    Score one detector result against its label.

    :param result: Detector output for the frame.
    :param label_mask: Boolean label at the frame's original resolution.
    :param profile: When given, the point-of-view estimates are compared too.
    :rtype: FrameScore
    """
    grid_shape = result.grid_mask.shape
    content = result.geometry.content_cells(*grid_shape)
    truth_grid = label_to_grid(label_mask, result.geometry, grid_shape)
    truth_rows = trace_path(truth_grid, result.geometry)

    errors, rows_missed = edge_errors(result.rows, truth_rows, min_image_y)
    absolute = [abs(error) for error in errors]

    pov = None
    if profile is not None:
        # The label is traced exactly like the model's mask, so both estimates see the same kind of input.
        label_result = PathResult(result.geometry, truth_grid, truth_rows, 1.0, None, "label")
        pov = pov_difference(estimate_pov(result, profile, min_image_y), estimate_pov(label_result, profile, min_image_y))

    return FrameScore(
        iou=grid_iou(result.grid_mask, truth_grid, content),
        edge_error_mean=sum(absolute) / len(absolute) if absolute else None,
        edge_error_max=max(absolute) if absolute else None,
        sides_compared=len(errors),
        rows_missed=rows_missed,
        pov_difference=pov,
    )
