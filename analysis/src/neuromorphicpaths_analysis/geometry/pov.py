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
from dataclasses import dataclass
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
# A straight edge traced on the model grid scatters by about a third of a cell, roughly 0.4% of
# the frame width. Well past that, the edge is curved or blocked (a parked bike, a hedge) and the
# estimate should not be trusted.
MAX_RELIABLE_FIT_RMS_FRACTION = 0.01
# A line through a handful of rows swings a lot once extended to the horizon. Two estimates of
# the same frame from nine rows each were seen to disagree by 17 degrees of pitch.
MIN_RELIABLE_ROWS_PER_EDGE = 12


class PovStatus(Enum):
    OK = "ok"
    TOO_FEW_ROWS = "too few unclipped rows on one edge"
    PARALLEL_EDGES = "edges are parallel in the image"
    VANISHING_POINT_BELOW_PATH = "edges meet below the path, not ahead of it"


@dataclass(frozen=True)
class EdgeFit:
    slope: float
    offset: float
    rms_px: float
    rows_used: int

    def x_at(self, y_px: float) -> float:
        return self.slope * y_px + self.offset


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
    # False when either edge fits its line poorly or rests on too few rows. The numbers are
    # still reported, but marked.
    reliable: bool = False


def fit_edge(rows: list[TracedRow], side_is_left: bool, frame_width: int, frame_height: int) -> EdgeFit | None:
    """
    Least-squares line through one edge's unclipped points, in frame pixels.

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
    slope, offset = numpy.polyfit(y_values, x_values, 1)
    residuals = x_values - (slope * y_values + offset)
    return EdgeFit(float(slope), float(offset), float(numpy.sqrt(numpy.mean(residuals ** 2))), len(points))


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

    left = fit_edge(rows, True, frame_width, frame_height)
    right = fit_edge(rows, False, frame_width, frame_height)
    if left is None or right is None:
        return PovEstimate(PovStatus.TOO_FEW_ROWS, left, right)

    slope_difference = right.slope - left.slope
    if abs(slope_difference) < MIN_SLOPE_DIFFERENCE:
        return PovEstimate(PovStatus.PARALLEL_EDGES, left, right)

    vanishing_y = (left.offset - right.offset) / slope_difference
    vanishing_x = left.x_at(vanishing_y)
    # The path widens toward the camera, so the lines must meet above the rows they came from.
    if vanishing_y >= min(row.y for row in rows) * frame_height or slope_difference < 0:
        return PovEstimate(PovStatus.VANISHING_POINT_BELOW_PATH, left, right, (vanishing_x, vanishing_y))

    reliable = (
        max(left.rms_px, right.rms_px) <= MAX_RELIABLE_FIT_RMS_FRACTION * frame_width
        and min(left.rows_used, right.rows_used) >= MIN_RELIABLE_ROWS_PER_EDGE
    )
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
