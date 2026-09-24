"""
Camera point of view from the path's two edges.

Each edge is fitted as a straight line x = slope * y + offset in frame pixels. Where the two
lines meet is the vanishing point, which sits on the horizon. From there:

- pitch, how far the camera tilts down, is the angle from the frame's center row up to the
  horizon: atan((cy - vp_y) / f)
- heading, the angle between where the camera points and where the path goes, is the horizontal
  angle of the vanishing point: atan((vp_x - cx) * cos(pitch) / f). Positive means the path
  heads off to the right of where the camera points.
- position across the path, 0 at the left edge and 1 at the right, is where the camera's own
  ground point falls between the lines: (-left_slope - tan(heading) * sin(pitch)) /
  (right_slope - left_slope). Without a focal length the correction term is dropped, which is
  exact when either pitch or heading is zero and close when both are small.
- camera height and path width trade against each other:
  height = width * cos(pitch) / ((right_slope - left_slope) * cos(heading))

Assumes flat ground, a straight path over the fitted rows, and no roll. The Neon IMU can
supply roll later. Position needs no focal length. Pitch and heading do.
"""

# Standard library imports
import math
from dataclasses import dataclass, replace
from enum import Enum

# Third party imports
import numpy

# Local package imports
from neuromorphicpaths_analysis.detector import PathResult, TracedRow
from neuromorphicpaths_analysis.geometry.capture_profiles import CaptureProfile

# The lower part of the frame is nearest the camera, where the path is most likely straight.
DEFAULT_MIN_IMAGE_Y = 0.5
MIN_ROWS_PER_EDGE = 4
# Below this difference in slope the edges are parallel in the image and meet nowhere useful.
MIN_SLOPE_DIFFERENCE = 1e-3
# Edge points come from the model's mask grid, so distances are judged in grid cells. A straight
# edge traced on the grid steps by up to a cell and scatters by about a third of one.
# Points further than this from the fitted line are dropped and the line refitted. They are stray
# cells where the mask frays at the frame border or catches something beside the path.
OUTLIER_DISTANCE_CELLS = 1.5
OUTLIER_REFITS = 3
# After dropping outliers, a straight edge fits within this. Past it, the edge is curved.
MAX_RELIABLE_RMS_CELLS = 0.75
# A line through a handful of rows swings a lot once extended to the horizon. Two estimates of
# the same frame from nine rows each disagreed by 17 degrees of pitch.
MIN_RELIABLE_ROWS_PER_EDGE = 12
# If more than this share of an edge's points had to be dropped, the edge is not one straight line.
MAX_OUTLIER_SHARE = 0.4
# An edge that stays within this of the mask region's side over its whole length is the side of
# the detection box cutting the mask off, not the path. It fits a line perfectly, so the checks
# above would pass it. With SidewalkVision's decoding it gave pitches of 72 and 4 degrees on
# photos taken at about 20.
BOX_SIDE_TOLERANCE_CELLS = 1.0


class PovStatus(Enum):
    OK = "ok"
    TOO_FEW_ROWS = "too few unclipped rows on one edge"
    PARALLEL_EDGES = "edges are parallel in the image"
    VANISHING_POINT_BELOW_PATH = "edges meet below the path, not ahead of it"


@dataclass(frozen=True)
class EdgeFit:
    slope: float
    offset: float
    # Over the rows kept after dropping outliers.
    rms_px: float
    rows_used: int
    rows_offered: int
    cell_px: float
    # Vertical extent of the kept rows, in frame pixels.
    top_px: float
    bottom_px: float
    on_box_side: bool = False

    def x_at(self, y_px: float) -> float:
        return self.slope * y_px + self.offset

    def runs_along(self, x_px: float) -> bool:
        """Whether the fitted line stays within BOX_SIDE_TOLERANCE_CELLS of a vertical line at x_px."""
        # The fit is a straight line, so its largest distance from a vertical line is at an end.
        distance = max(abs(self.x_at(self.top_px) - x_px), abs(self.x_at(self.bottom_px) - x_px))
        return distance <= BOX_SIDE_TOLERANCE_CELLS * self.cell_px

    @property
    def rms_cells(self) -> float:
        """Scatter in mask-grid cells, the unit the straightness limit uses on every camera."""
        return self.rms_px / self.cell_px

    @property
    def straight(self) -> bool:
        return (
            self.rms_px <= MAX_RELIABLE_RMS_CELLS * self.cell_px
            and self.rows_used >= MIN_RELIABLE_ROWS_PER_EDGE
            and self.rows_used >= (1 - MAX_OUTLIER_SHARE) * self.rows_offered
            and not self.on_box_side
        )


@dataclass(frozen=True)
class PovEstimate:
    status: PovStatus
    left: EdgeFit | None = None
    right: EdgeFit | None = None
    vanishing_point_px: tuple[float, float] | None = None
    pitch_deg: float | None = None
    heading_deg: float | None = None
    position_fraction: float | None = None
    camera_height_m: float | None = None
    path_width_m: float | None = None
    # False unless both edges are straight by EdgeFit.straight. The numbers are still reported,
    # but marked.
    reliable: bool = False


def grid_cell_px(result: PathResult) -> float:
    """Width of one mask-grid cell, in pixels of the original frame."""
    geometry = result.geometry
    grid_width = result.grid_mask.shape[1]
    return geometry.input_width / grid_width * geometry.source_width / geometry.content_width


def fit_edge(
    rows: list[TracedRow],
    side_is_left: bool,
    frame_width: int,
    frame_height: int,
    cell_px: float,
) -> EdgeFit | None:
    """
    Line through one edge's unclipped points, in frame pixels, refitted without outliers.

    :param cell_px: Size of one mask-grid cell in frame pixels, the unit for outlier distance.
    :return: None when fewer than MIN_ROWS_PER_EDGE rows are usable.
    :rtype: EdgeFit | None
    """
    points = [
        (row.y * frame_height, (row.left_x if side_is_left else row.right_x) * frame_width)
        for row in rows
        if not (row.left_clipped if side_is_left else row.right_clipped)
    ]
    if len(points) < MIN_ROWS_PER_EDGE:
        return None
    y_values, x_values = numpy.array(points).T
    kept = numpy.ones(len(points), dtype=bool)
    for _ in range(OUTLIER_REFITS + 1):
        slope, offset = numpy.polyfit(y_values[kept], x_values[kept], 1)
        distance = numpy.abs(x_values - (slope * y_values + offset))
        next_kept = distance <= OUTLIER_DISTANCE_CELLS * cell_px
        # Too few left to fit a line means the edge is not straight. Keep the last fit and let
        # the reliability check say so.
        if next_kept.sum() < MIN_ROWS_PER_EDGE or numpy.array_equal(next_kept, kept):
            break
        kept = next_kept
    residuals = x_values[kept] - (slope * y_values[kept] + offset)
    return EdgeFit(
        slope=float(slope),
        offset=float(offset),
        rms_px=float(numpy.sqrt(numpy.mean(residuals ** 2))),
        rows_used=int(kept.sum()),
        rows_offered=len(points),
        cell_px=cell_px,
        top_px=float(y_values[kept].min()),
        bottom_px=float(y_values[kept].max()),
    )


def estimate_pov(
    result: PathResult,
    profile: CaptureProfile,
    min_image_y: float = DEFAULT_MIN_IMAGE_Y,
) -> PovEstimate:
    """
    Estimate pitch, heading, position across the path and height or width for one frame.

    :param result: Detector output, or a label traced the same way.
    :param profile: The capture setup. Unknown fields leave the matching outputs as None.
    :param min_image_y: Only rows below this fraction of the frame height are fitted.
    :rtype: PovEstimate
    """
    frame_width = result.geometry.source_width
    frame_height = result.geometry.source_height
    rows = [row for row in result.rows if row.y >= min_image_y]

    cell_px = grid_cell_px(result)
    left = fit_edge(rows, True, frame_width, frame_height, cell_px)
    right = fit_edge(rows, False, frame_width, frame_height, cell_px)
    if left is None or right is None:
        return PovEstimate(PovStatus.TOO_FEW_ROWS, left, right)
    if result.mask_bounds is not None:
        bounds_left, _, bounds_right, _ = result.mask_bounds
        left = replace(left, on_box_side=left.runs_along(bounds_left * frame_width))
        right = replace(right, on_box_side=right.runs_along(bounds_right * frame_width))

    slope_difference = right.slope - left.slope
    if abs(slope_difference) < MIN_SLOPE_DIFFERENCE:
        return PovEstimate(PovStatus.PARALLEL_EDGES, left, right)

    vanishing_y = (left.offset - right.offset) / slope_difference
    vanishing_x = left.x_at(vanishing_y)
    # The path widens toward the camera, so the lines must meet above the rows they came from.
    if vanishing_y >= min(row.y for row in rows) * frame_height or slope_difference < 0:
        return PovEstimate(PovStatus.VANISHING_POINT_BELOW_PATH, left, right, (vanishing_x, vanishing_y))

    reliable = left.straight and right.straight
    position_fraction = -left.slope / slope_difference
    estimate = PovEstimate(
        status=PovStatus.OK,
        left=left,
        right=right,
        vanishing_point_px=(vanishing_x, vanishing_y),
        position_fraction=position_fraction,
        reliable=reliable,
    )

    focal_length = profile.focal_length_for(frame_width)
    if focal_length is None:
        return estimate

    pitch = math.atan((frame_height / 2 - vanishing_y) / focal_length)
    heading = math.atan((vanishing_x - frame_width / 2) * math.cos(pitch) / focal_length)
    # Height per meter of path width. Same for every row, since both lines share the vanishing point.
    height_per_width = math.cos(pitch) / (slope_difference * math.cos(heading))

    return PovEstimate(
        status=PovStatus.OK,
        left=left,
        right=right,
        vanishing_point_px=(vanishing_x, vanishing_y),
        pitch_deg=math.degrees(pitch),
        heading_deg=math.degrees(heading),
        position_fraction=(-left.slope - math.tan(heading) * math.sin(pitch)) / slope_difference,
        camera_height_m=None if profile.path_width_m is None else profile.path_width_m * height_per_width,
        path_width_m=None if profile.camera_height_m is None else profile.camera_height_m / height_per_width,
        reliable=reliable,
    )
