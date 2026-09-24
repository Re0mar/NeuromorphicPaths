"""
Estimate the camera's point of view for an image, a folder of images, or a video.

Usage:
    python -m neuromorphicpaths_analysis.geometry <image | folder | video> [--every N] [--csv out.csv]
        [--profile belgian-dataset] [--focal-length PX] [--camera-height M] [--path-width M]

Pitch is positive when the camera tilts down. Heading is positive when the path heads off to
the right of where the camera points. Position runs from 0 at the left edge to 1 at the right.
--csv writes every field, including the vanishing point and the edge fits, one row per frame.
"""

# Standard library imports
import argparse
import csv
from pathlib import Path

# Local package imports
from neuromorphicpaths_analysis.detector import PathDetector
from neuromorphicpaths_analysis.detector.path_detector import DEFAULT_MODEL_PATH
from neuromorphicpaths_analysis.geometry.capture_profiles import CaptureProfile, CaptureProfileName
from neuromorphicpaths_analysis.geometry.pov import PovEstimate, PovStatus, estimate_pov
from neuromorphicpaths_analysis.geometry.profile_arguments import add_profile_arguments, profile_from_arguments
from neuromorphicpaths_analysis.recordings import Frame, iter_frames

CSV_COLUMNS = [
    "frame", "time_s", "status", "reliable", "detection_score",
    "pitch_deg", "heading_deg", "position", "camera_height_m", "path_width_m",
    "vanishing_x_px", "vanishing_y_px",
    "left_rms_px", "right_rms_px", "left_rows", "right_rows",
]


def format_optional(value: float | None, width: int, precision: int) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:{width}.{precision}f}"


def csv_row(frame: Frame, detection_score: float, estimate: PovEstimate) -> dict:
    vanishing = estimate.vanishing_point_px
    return {
        "frame": frame.name,
        "time_s": frame.time_s,
        "status": estimate.status.name,
        "reliable": estimate.reliable,
        "detection_score": round(detection_score, 4),
        "pitch_deg": estimate.pitch_deg,
        "heading_deg": estimate.heading_deg,
        "position": estimate.position_fraction,
        "camera_height_m": estimate.camera_height_m,
        "path_width_m": estimate.path_width_m,
        "vanishing_x_px": None if vanishing is None else vanishing[0],
        "vanishing_y_px": None if vanishing is None else vanishing[1],
        "left_rms_px": None if estimate.left is None else estimate.left.rms_px,
        "right_rms_px": None if estimate.right is None else estimate.right.rms_px,
        "left_rows": None if estimate.left is None else estimate.left.rows_used,
        "right_rows": None if estimate.right is None else estimate.right.rows_used,
    }


def print_row(frame: Frame, estimate: PovEstimate) -> None:
    time_text = format_optional(frame.time_s, 8, 2)
    if estimate.status is not PovStatus.OK:
        print(f"{frame.name:<22} {time_text} {estimate.status.value}")
        return
    print(
        f"{frame.name:<22} {time_text} {format_optional(estimate.pitch_deg, 9, 1)} "
        f"{format_optional(estimate.heading_deg, 11, 1)} {format_optional(estimate.position_fraction, 9, 2)} "
        f"{format_optional(estimate.camera_height_m, 9, 2)} {format_optional(estimate.path_width_m, 8, 2)}  "
        f"{estimate.left.rms_px:.1f} / {estimate.right.rms_px:.1f}"
        f"{'' if estimate.reliable else '  UNRELIABLE, edge curved, blocked or too short'}"
    )


def run(source: Path, profile: CaptureProfile, detector: PathDetector, every_nth: int, csv_path: Path | None) -> int:
    """
    Estimate every frame of a source, print a table, and optionally write a CSV.

    :return: How many frames were processed.
    :rtype: int
    """
    print(f"profile {profile.name.value}: {profile.note}")
    print(f"{'frame':<22} {'time s':>8} {'pitch deg':>9} {'heading deg':>11} {'position':>9} "
          f"{'height m':>9} {'width m':>8}  fit rms px")
    rows = []
    for frame in iter_frames(source, every_nth):
        result = detector.detect_path(frame.image)
        estimate = estimate_pov(result, profile)
        print_row(frame, estimate)
        rows.append(csv_row(frame, result.top_score, estimate))

    if csv_path is not None:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate camera pitch, heading, position and height per frame")
    parser.add_argument("source", type=Path, help="an image, a folder of images, or a video")
    parser.add_argument("--every", type=int, default=1, help="for video, keep one frame in every N")
    parser.add_argument("--csv", type=Path, help="also write all fields to this CSV file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    add_profile_arguments(parser, CaptureProfileName.BELGIAN_DATASET)
    arguments = parser.parse_args()

    run(arguments.source, profile_from_arguments(arguments), PathDetector(arguments.model), arguments.every, arguments.csv)


if __name__ == "__main__":
    main()
