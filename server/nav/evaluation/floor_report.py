"""
The floor a recorded walk was given: where each frame's floor came from, and how high the camera sat above it.

Read from one scene pass, so the heights are the scene's own planes and the reasons are the floor
fit's own refusals. Nothing here fits a plane or judges one. A camera height near eye level is the
check that a depth source is in meters, which is what this report is for.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.replay import ScenePass
from nav.scene.floor import FloorRefusalCause
from nav.types import FloorSource

# The heights are reported at these percentiles beside the median.
LOW_PERCENTILE = 10.0
HIGH_PERCENTILE = 90.0


@dataclass(frozen=True)
class FloorReport:
    """Every floor figure of one walk, each with the frame count behind it."""

    frames: int
    by_source: dict[FloorSource, int]
    unaligned_frames: int
    refused_frames: int
    refusal_reasons: dict[str, int]
    # Over gravity-aligned frames whose floor was fitted or supplied, in meters.
    heights_meters: np.ndarray
    # Each previous-floor frame's measured value, by why its fit gave nothing. None where the cause has no number.
    previous_because: dict[FloorRefusalCause, list[float | None]]


def floor_report(scene: ScenePass) -> FloorReport:
    """
    The floor figures of one scene pass.

    :param scene: A pass over the whole walk.
    :return: Counts, heights and refusal causes.
    :rtype: FloorReport
    """
    by_source = {source: 0 for source in FloorSource}
    previous_because: dict[FloorRefusalCause, list[float | None]] = {cause: [] for cause in FloorRefusalCause}
    heights: list[float] = []
    unaligned = 0
    for row in scene.planned:
        by_source[row.floor_source] += 1
        if not row.gravity_aligned:
            # The floor was fitted against the picture's up, which is wrong on a tilted head, so its
            # height isn't the camera's. Counted, so a walk with many of these says so.
            unaligned += 1
        elif row.floor_source in (FloorSource.FITTED, FloorSource.SUPPLIED):
            # A previous floor repeats an earlier frame's height. Counting it would weight the median
            # toward whatever floor stood before each run of refusals.
            heights.append(row.camera_height_meters)
        # Only frames with gravity: a lean measured against the picture's up isn't the same
        # quantity as one measured against gravity, so the two never share a median.
        if row.gravity_aligned and row.floor_source is FloorSource.PREVIOUS and row.floor_refusal is not None:
            previous_because[row.floor_refusal.cause].append(row.floor_refusal.measured)
    return FloorReport(
        frames=len(scene.planned) + scene.refused_frames,
        by_source=by_source,
        unaligned_frames=unaligned,
        refused_frames=scene.refused_frames,
        refusal_reasons=dict(scene.refusal_reasons),
        heights_meters=np.asarray(heights, dtype=np.float64),
        previous_because=previous_because,
    )


def measured_unit(cause: FloorRefusalCause) -> str:
    """The unit a refusal cause's measured value is in, for printing beside its median."""
    match cause:
        case FloorRefusalCause.LEANS:
            return "deg"
        case FloorRefusalCause.TOO_CLOSE | FloorRefusalCause.TOO_FAR:
            return "m"
        case FloorRefusalCause.TOO_FEW_CANDIDATES:
            return "points"
        case FloorRefusalCause.NO_PLANE | FloorRefusalCause.NOT_FINITE:
            return ""
        case _:
            # Unreachable while every member is handled. Loud, so a new cause is never printed unitless.
            raise ValueError(f"no unit for {cause}")


def has_heights(report: FloorReport) -> bool:
    """Whether the report measured a camera height at all."""
    return report.heights_meters.size > 0


def format_floor_report(name: str, report: FloorReport, scene_source: str) -> str:
    """
    One plain-text block for one walk.

    Every source and cause is printed, zero or not, in enum order. Refusal reasons go by count and
    then by text. So the same pass prints the same text, byte for byte. Which refusal reasons get a
    line of their own is still the scene pass's first few, so that part does follow frame order.

    :param name: The walk's name.
    :param report: Its figures.
    :param scene_source: Where the scene settings came from.
    :return: The block, ending in a newline.
    :rtype: str
    """
    lines = [f"=== {name}", f"{report.frames} frames, scene settings from {scene_source}"]
    lines.append("floor by source:")
    for source in FloorSource:
        lines.append(f"  {source.value:<10} {report.by_source[source]:>6}")
    lines.append(f"frames not gravity aligned: {report.unaligned_frames}, the picture's up used, left out of the height")

    if has_heights(report):
        median = float(np.median(report.heights_meters))
        low, high = np.percentile(report.heights_meters, [LOW_PERCENTILE, HIGH_PERCENTILE])
        lines.append(
            f"camera above the floor: median {median:.2f} m, {LOW_PERCENTILE:.0f}th {low:.2f} m, "
            f"{HIGH_PERCENTILE:.0f}th {high:.2f} m, over {report.heights_meters.size} fitted or supplied aligned frames"
        )
    else:
        lines.append("camera above the floor: no fitted or supplied aligned frame, so no height")

    lines.append("fits refused on frames with gravity, previous floor kept:")
    for cause in FloorRefusalCause:
        measured = [value for value in report.previous_because[cause] if value is not None]
        median_text = f", median {float(np.median(measured)):.2f} {measured_unit(cause)}" if measured else ""
        lines.append(f"  {cause.value:<19} {len(report.previous_because[cause]):>6}{median_text}")

    lines.append(f"frames the scene refused: {report.refused_frames}")
    for reason, count in sorted(report.refusal_reasons.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count:>6}  {reason}")
    return "\n".join(lines) + "\n"
