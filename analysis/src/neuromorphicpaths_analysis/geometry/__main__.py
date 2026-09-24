"""
Estimate the camera's point of view for each frame in a folder.

Usage:
    python -m neuromorphicpaths_analysis.geometry --frames <dir> [--profile belgian-dataset]
        [--focal-length PX] [--camera-height M] [--path-width M]

Pitch is positive when the camera tilts down. Heading is positive when the path heads off to
the right of where the camera points. Position runs from 0 at the left edge to 1 at the right.
"""

# Standard library imports
import argparse
from pathlib import Path

# Third party imports
from PIL import Image, ImageOps

# Local package imports
from neuromorphicpaths_analysis.detector import PathDetector
from neuromorphicpaths_analysis.detector.path_detector import DEFAULT_MODEL_PATH
from neuromorphicpaths_analysis.geometry.capture_profiles import CaptureProfileName
from neuromorphicpaths_analysis.geometry.pov import PovStatus, estimate_pov
from neuromorphicpaths_analysis.geometry.profile_arguments import add_profile_arguments, profile_from_arguments

FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def format_optional(value: float | None, width: int, precision: int) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:{width}.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate camera pitch, heading, position and height per frame")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    add_profile_arguments(parser, CaptureProfileName.BELGIAN_DATASET)
    arguments = parser.parse_args()

    profile = profile_from_arguments(arguments)
    detector = PathDetector(arguments.model)
    print(f"profile {profile.name.value}: {profile.note}")
    print(f"{'frame':<16} {'pitch deg':>9} {'heading deg':>11} {'position':>9} {'height m':>9} {'width m':>8}  fit rms px")

    for frame_path in sorted(path for path in arguments.frames.iterdir() if path.suffix.lower() in FRAME_SUFFIXES):
        image = ImageOps.exif_transpose(Image.open(frame_path)).convert("RGB")
        estimate = estimate_pov(detector.detect_path(image), profile)
        if estimate.status is not PovStatus.OK:
            print(f"{frame_path.name:<16} {estimate.status.value}")
            continue
        print(
            f"{frame_path.name:<16} {format_optional(estimate.pitch_deg, 9, 1)} "
            f"{format_optional(estimate.heading_deg, 11, 1)} {format_optional(estimate.position_fraction, 9, 2)} "
            f"{format_optional(estimate.camera_height_m, 9, 2)} {format_optional(estimate.path_width_m, 8, 2)}  "
            f"{estimate.left.rms_px:.1f} / {estimate.right.rms_px:.1f}"
            f"{'' if estimate.reliable else '  UNRELIABLE, edge curved, blocked or too short'}"
        )


if __name__ == "__main__":
    main()
