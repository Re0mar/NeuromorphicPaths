"""
Python copy of the first app's PathDetector, now kept at
OldAppEnvrionmentStuff/app/src/main/java/com/example/neuromorphicpaths/vision/PathDetector.kt.

It runs the same .tflite file with the same preprocessing and decoding. Constants keep the
Kotlin names in snake case. When one changes there, change it here too.

SidewalkVision took over these settings on 2026-09-25, so the default preset now matches both
apps. The one addition over the first app, which SidewalkVision also has, is clipping: each
traced row says whether a side touched the image border. The other preset keeps SidewalkVision's
decoding as it first shipped, to reproduce the comparison between the two.
"""

# Standard library imports
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Third party imports
import cv2
import numpy
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

# Local package imports
from neuromorphicpaths_analysis.detector.types import LetterboxGeometry, PathResult, TracedRow

# detector/ -> neuromorphicpaths_analysis/ -> src/ -> analysis/ -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
# The model SidewalkVision ships. Byte-identical to the one the first app used.
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "SidewalkVision" / "app" / "src" / "main" / "assets" / "best_int8.tflite"

# Mirrors of the PathDetector.kt companion object.
CONF_THRESHOLD = 0.25
BOXES_NORMALIZED = True
PAD_GRAY = 114
MIN_CENTER_Y = 0.0
MIN_BOTTOM_Y = 0.0
PRIOR_X = 0.30
PRIOR_Y = 0.30
MAX_GAP_ROWS = 2

# Not in either app. The model's box is clamped a little inside the frame, so a path running off
# the side often stops one cell short of the last image column. Within this many cells of the
# border, a side counts as clipped.
CLIP_MARGIN_CELLS = 1

# SidewalkVision's decoding as it first shipped, in PathDetector.kt at commit e8b81c2, each citing
# the Kotlin it copies. Kept only to reproduce the comparison with the first app's settings.
SIDEWALK_VISION_ORIGINAL_CONF_THRESHOLD = 0.001  # val CONF_THRESHOLD = 0.001f
SIDEWALK_VISION_ORIGINAL_MASK_PROBABILITY = 0.35  # sigmoid > 0.35f
SIDEWALK_VISION_ORIGINAL_BOX_MARGIN = 0.1  # ys >= (t - 0.1f) and so on, in fractions of the model input


class DecoderPreset(Enum):
    FIRST_APP = "first-app"
    SIDEWALK_VISION_ORIGINAL = "sidewalk-vision-original"


class EdgeRule(Enum):
    # Start from the run under the bottom center and follow it upward, row by row.
    TRACED_RUN = "traced-run"
    # The first and last mask cell in each row, wherever they are.
    OUTERMOST = "outermost"


@dataclass(frozen=True)
class DecoderSettings:
    """How one app turns the model's raw output into a mask and two edges."""

    preset: DecoderPreset
    conf_threshold: float
    # Weight detections toward the bottom center of the frame, where the ground ahead is.
    prefer_ground_ahead: bool
    mask_probability: float
    box_margin: float
    # A single bilinear shrink from a large frame aliases. Antialiasing avoids that.
    antialiased_resize: bool
    edge_rule: EdgeRule


DECODER_SETTINGS: dict[DecoderPreset, DecoderSettings] = {
    DecoderPreset.FIRST_APP: DecoderSettings(
        preset=DecoderPreset.FIRST_APP,
        conf_threshold=CONF_THRESHOLD,
        prefer_ground_ahead=True,
        mask_probability=0.5,
        box_margin=0.0,
        antialiased_resize=True,
        edge_rule=EdgeRule.TRACED_RUN,
    ),
    DecoderPreset.SIDEWALK_VISION_ORIGINAL: DecoderSettings(
        preset=DecoderPreset.SIDEWALK_VISION_ORIGINAL,
        conf_threshold=SIDEWALK_VISION_ORIGINAL_CONF_THRESHOLD,
        prefer_ground_ahead=False,
        mask_probability=SIDEWALK_VISION_ORIGINAL_MASK_PROBABILITY,
        box_margin=SIDEWALK_VISION_ORIGINAL_BOX_MARGIN,
        # Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true), filtered, in one step.
        antialiased_resize=False,
        edge_rule=EdgeRule.OUTERMOST,
    ),
}


# *******************************************
# Helper Functions
# *******************************************


def kotlin_round(value: float) -> int:
    """
    Round half up, like Kotlin's roundToInt.

    Python's round() sends halves to the even neighbor, which would move the letterbox
    padding by a pixel on some frame sizes.
    """
    return math.floor(value + 0.5)


def letterbox(
    image: Image.Image,
    input_width: int,
    input_height: int,
    antialiased: bool = True,
) -> tuple[numpy.ndarray, LetterboxGeometry]:
    """
    Fit a frame into the model input with gray padding, keeping its aspect ratio.

    PIL's bilinear resize is antialiased when shrinking. The first app gets the same effect by
    halving repeatedly before its final bilinear draw, so pixel values differ slightly but not the
    shape. SidewalkVision as first shipped shrank in one bilinear step, which OpenCV's plain
    bilinear resize matches.

    :param image: RGB frame at original resolution.
    :param antialiased: False to copy the original SidewalkVision's single-step shrink.
    :return: The padded RGB input as a uint8 array, and where the frame sits inside it.
    :rtype: tuple[numpy.ndarray, LetterboxGeometry]
    """
    scale = min(input_width / image.width, input_height / image.height)
    content_width = min(max(kotlin_round(image.width * scale), 1), input_width)
    content_height = min(max(kotlin_round(image.height * scale), 1), input_height)
    pad_x = kotlin_round((input_width - content_width) / 2)
    pad_y = kotlin_round((input_height - content_height) / 2)

    if antialiased:
        content = image.resize((content_width, content_height), Image.Resampling.BILINEAR)
    else:
        content = Image.fromarray(
            cv2.resize(numpy.asarray(image), (content_width, content_height), interpolation=cv2.INTER_LINEAR)
        )
    canvas = Image.new("RGB", (input_width, input_height), (PAD_GRAY, PAD_GRAY, PAD_GRAY))
    canvas.paste(content, (pad_x, pad_y))

    geometry = LetterboxGeometry(
        source_width=image.width,
        source_height=image.height,
        input_width=input_width,
        input_height=input_height,
        content_width=content_width,
        content_height=content_height,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return numpy.asarray(canvas), geometry


def trace_path(grid_mask: numpy.ndarray, geometry: LetterboxGeometry) -> list[TracedRow]:
    """
    Trace the path outline upward from the bottom center of a mask grid.

    Same rules as the app: on the first row with mask cells take the run nearest the center
    column, then on each row above keep only the run overlapping the one below, tolerating
    MAX_GAP_ROWS empty rows. Scoring runs this on label grids too, so both sides are traced
    the same way.

    :param grid_mask: Boolean mask on the model's prototype grid.
    :return: One entry per traced row, bottom row first.
    :rtype: list[TracedRow]
    """
    grid_height, grid_width = grid_mask.shape
    content_columns = numpy.flatnonzero(geometry.content_cells(grid_height, grid_width).any(axis=0))
    first_content_column = int(content_columns[0])
    last_content_column = int(content_columns[-1])

    center_column = grid_width // 2
    previous_left = -1
    previous_right = -1
    gap_rows = 0
    rows: list[TracedRow] = []

    for grid_y in range(grid_height - 1, -1, -1):
        row = grid_mask[grid_y]
        chosen_left = -1
        chosen_right = -1
        chosen_rank = -math.inf

        x = 0
        while x < grid_width:
            if not row[x]:
                x += 1
                continue
            start = x
            while x < grid_width and row[x]:
                x += 1
            end = x - 1

            if previous_left == -1:
                # Starting row: the run containing the center column wins, then the nearest.
                rank = 0 if start <= center_column <= end else -min(abs(start - center_column), abs(end - center_column))
            else:
                # Later rows: must overlap the run below, and the larger overlap wins.
                rank = min(end, previous_right) - max(start, previous_left) + 1
                if rank <= 0:
                    continue
            if rank > chosen_rank:
                chosen_rank = rank
                chosen_left = start
                chosen_right = end

        if chosen_left == -1:
            if previous_left != -1:
                gap_rows += 1
                if gap_rows > MAX_GAP_ROWS:
                    break
            continue
        gap_rows = 0
        previous_left = chosen_left
        previous_right = chosen_right

        rows.append(
            TracedRow(
                grid_y=grid_y,
                left_cell=chosen_left,
                right_cell=chosen_right,
                y=geometry.to_image_y((grid_y + 0.5) / grid_height),
                left_x=geometry.to_image_x(chosen_left / grid_width),
                right_x=geometry.to_image_x((chosen_right + 1) / grid_width),
                left_clipped=chosen_left <= first_content_column + CLIP_MARGIN_CELLS,
                right_clipped=chosen_right >= last_content_column - CLIP_MARGIN_CELLS,
            )
        )
    return rows


def outermost_edges(grid_mask: numpy.ndarray, geometry: LetterboxGeometry) -> list[TracedRow]:
    """
    The first and last mask cell of every row, as SidewalkVision first read its edges.

    Unlike trace_path, this follows no run. Two sidewalks in one row give one outline spanning
    both and the road between them. Clipping is flagged the same way as in trace_path.

    :return: One entry per row holding mask cells on the image, bottom row first.
    :rtype: list[TracedRow]
    """
    grid_height, grid_width = grid_mask.shape
    content = geometry.content_cells(grid_height, grid_width)
    content_columns = numpy.flatnonzero(content.any(axis=0))
    first_content_column = int(content_columns[0])
    last_content_column = int(content_columns[-1])

    rows: list[TracedRow] = []
    for grid_y in range(grid_height - 1, -1, -1):
        columns = numpy.flatnonzero(grid_mask[grid_y] & content[grid_y])
        if columns.size == 0:
            continue
        left_cell, right_cell = int(columns[0]), int(columns[-1])
        rows.append(
            TracedRow(
                grid_y=grid_y,
                left_cell=left_cell,
                right_cell=right_cell,
                y=geometry.to_image_y((grid_y + 0.5) / grid_height),
                left_x=geometry.to_image_x(left_cell / grid_width),
                right_x=geometry.to_image_x((right_cell + 1) / grid_width),
                left_clipped=left_cell <= first_content_column + CLIP_MARGIN_CELLS,
                right_clipped=right_cell >= last_content_column - CLIP_MARGIN_CELLS,
            )
        )
    return rows


# *******************************************
# Detector
# *******************************************


class PathDetector:
    """Loads the YOLO segmentation model once and runs it on frames, decoding like one of the apps."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, preset: DecoderPreset = DecoderPreset.FIRST_APP) -> None:
        self.settings = DECODER_SETTINGS[preset]
        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        self._input = self._interpreter.get_input_details()[0]
        shape = self._input["shape"]
        # Channels-first when the second axis holds the colors, as the app decides it.
        self._input_is_nchw = int(shape[1]) in (1, 3)
        if self._input_is_nchw:
            self.input_height, self.input_width = int(shape[2]), int(shape[3])
        else:
            self.input_height, self.input_width = int(shape[1]), int(shape[2])

        # Rank 3 is the detections, rank 4 the prototype masks. Resolved from shapes, not order.
        outputs = self._interpreter.get_output_details()
        self._detection_output = next(detail for detail in outputs if len(detail["shape"]) == 3)
        self._proto_output = next(detail for detail in outputs if len(detail["shape"]) == 4)

    def detect_path(self, image: Image.Image) -> PathResult:
        """
        Run the model on one frame and trace the path it finds.

        :param image: RGB frame at original resolution, already upright.
        :return: Mask grid, traced outline and score, in the app's conventions.
        :rtype: PathResult
        """
        letterboxed, geometry = letterbox(
            image, self.input_width, self.input_height, antialiased=self.settings.antialiased_resize
        )
        self._interpreter.set_tensor(self._input["index"], self._to_input_tensor(letterboxed))
        self._interpreter.invoke()

        detections = self._read_output(self._detection_output)[0]
        prototypes = self._read_output(self._proto_output)[0]
        # Both layouts appear in YOLO exports. Normalize to (channels, anchors) and (masks, y, x).
        if detections.shape[0] > detections.shape[1]:
            detections = detections.T
        prototypes_channels_first = prototypes.shape[0] == 32 or prototypes.shape[0] < prototypes.shape[1]
        if not prototypes_channels_first:
            prototypes = numpy.moveaxis(prototypes, -1, 0)
        return self._decode(detections, prototypes, geometry)

    def _to_input_tensor(self, letterboxed: numpy.ndarray) -> numpy.ndarray:
        scaled = letterboxed.astype(numpy.float32) / 255.0
        if self._input_is_nchw:
            scaled = numpy.transpose(scaled, (2, 0, 1))
        scaled = scaled[numpy.newaxis]

        dtype = self._input["dtype"]
        if dtype == numpy.float32:
            return scaled
        # Quantized input, the same formula as the app's lookup table.
        scale, zero_point = self._input["quantization"]
        limits = numpy.iinfo(dtype)
        quantized = numpy.floor(scaled / scale + zero_point + 0.5)
        return numpy.clip(quantized, limits.min, limits.max).astype(dtype)

    def _read_output(self, detail: dict) -> numpy.ndarray:
        raw = self._interpreter.get_tensor(detail["index"])
        if raw.dtype == numpy.float32:
            return raw
        scale, zero_point = detail["quantization"]
        return (raw.astype(numpy.float32) - zero_point) * scale

    def _decode(self, detections: numpy.ndarray, prototypes: numpy.ndarray, geometry: LetterboxGeometry) -> PathResult:
        settings = self.settings
        input_width = self.input_width
        input_height = self.input_height
        proto_count, grid_height, grid_width = prototypes.shape
        empty_grid = numpy.zeros((grid_height, grid_width), dtype=bool)

        # Already a probability in this export, so no sigmoid.
        scores = detections[4]
        center_x, center_y, box_width, box_height = detections[0], detections[1], detections[2], detections[3]
        if not BOXES_NORMALIZED:
            center_x, box_width = center_x / input_width, box_width / input_width
            center_y, box_height = center_y / input_height, box_height / input_height

        image_center_x = (center_x * input_width - geometry.pad_x) / geometry.content_width
        image_center_y = (center_y * input_height - geometry.pad_y) / geometry.content_height
        image_bottom = ((center_y + box_height / 2) * input_height - geometry.pad_y) / geometry.content_height

        if settings.prefer_ground_ahead:
            accepted = (
                (scores >= settings.conf_threshold)
                & (image_center_x >= 0) & (image_center_x <= 1)
                & (image_center_y >= MIN_CENTER_Y) & (image_center_y <= 1)
                & (image_bottom >= MIN_BOTTOM_Y)
            )
            # Prefer boxes low and central in the frame.
            prior = (1 - PRIOR_X * numpy.abs(image_center_x - 0.5) * 2) * (1 - PRIOR_Y * (1 - image_center_y))
            adjusted = numpy.where(accepted, scores * prior, 0.0)
        else:
            # The single highest score anywhere, with no regard for where the box is.
            accepted = scores >= settings.conf_threshold
            adjusted = numpy.where(accepted, scores, 0.0)

        top_score = float(scores.max())
        if not accepted.any() or adjusted.max() <= 0:
            over_threshold = int((scores >= settings.conf_threshold).sum())
            debug_info = f"No walkway (top score {top_score * 100:.1f}%, {over_threshold} over threshold)"
            return PathResult(geometry, empty_grid, [], top_score, None, debug_info)

        best = int(numpy.argmax(adjusted))
        left = min(max(center_x[best] - box_width[best] / 2, 0.0), 1.0)
        top = min(max(center_y[best] - box_height[best] / 2, 0.0), 1.0)
        right = min(max(center_x[best] + box_width[best] / 2, 0.0), 1.0)
        bottom = min(max(center_y[best] + box_height[best] / 2, 0.0), 1.0)

        # Coefficients times prototypes, kept only inside the box. The sum is the mask's logit, so
        # comparing it with the logit of the probability threshold avoids a sigmoid per cell.
        # Cell centers are tested so the grid lines up with the input image.
        coefficient_count = min(proto_count, detections.shape[0] - 5)
        coefficients = detections[5:5 + coefficient_count, best]
        mask_sum = numpy.tensordot(coefficients, prototypes[:coefficient_count], axes=1)
        logit_threshold = math.log(settings.mask_probability / (1 - settings.mask_probability))
        margin = settings.box_margin
        cell_x = (numpy.arange(grid_width) + 0.5) / grid_width
        cell_y = (numpy.arange(grid_height) + 0.5) / grid_height
        inside_box = numpy.outer(
            (cell_y >= top - margin) & (cell_y <= bottom + margin),
            (cell_x >= left - margin) & (cell_x <= right + margin),
        )
        grid_mask = inside_box & (mask_sum > logit_threshold)

        box = (
            geometry.to_image_x(left),
            geometry.to_image_y(top),
            geometry.to_image_x(right),
            geometry.to_image_y(bottom),
        )
        debug_info = f"Walkway {scores[best] * 100:.1f}% @anchor {best} | mask {int(grid_mask.sum())} cells"
        mask_bounds = (
            geometry.to_image_x(max(left - margin, 0.0)),
            geometry.to_image_y(max(top - margin, 0.0)),
            geometry.to_image_x(min(right + margin, 1.0)),
            geometry.to_image_y(min(bottom + margin, 1.0)),
        )
        read_edges = trace_path if settings.edge_rule is EdgeRule.TRACED_RUN else outermost_edges
        return PathResult(
            geometry,
            grid_mask,
            read_edges(grid_mask, geometry),
            float(scores[best]),
            box,
            debug_info,
            mask_bounds,
        )
