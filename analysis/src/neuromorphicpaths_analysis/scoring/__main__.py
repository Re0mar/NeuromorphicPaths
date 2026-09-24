"""
Score the app's model on a folder of frames with labels.

Usage:
    python -m neuromorphicpaths_analysis.scoring --frames <dir> --labels <dir> [--model <tflite>]
        [--profile belgian-dataset] [--focal-length PX] [--camera-height M] [--path-width M]

A frame <name>.jpg is scored when <labels>/<name>_mask.png exists, the file the labeling tool
writes. Frames without a label are listed and skipped. The last three columns are the model's
point-of-view estimate minus the label's: pitch in degrees, position in path widths, height in
meters.
"""

# Standard library imports
import argparse
from pathlib import Path

# Third party imports
import numpy
from PIL import Image, ImageOps

# Local package imports
from neuromorphicpaths_analysis.detector import PathDetector
from neuromorphicpaths_analysis.detector.decoder_arguments import add_decoder_argument, decoder_from_arguments
from neuromorphicpaths_analysis.detector.path_detector import DEFAULT_MODEL_PATH
from neuromorphicpaths_analysis.geometry import CaptureProfileName
from neuromorphicpaths_analysis.geometry.profile_arguments import add_profile_arguments, profile_from_arguments
from neuromorphicpaths_analysis.scoring.scores import DEFAULT_MIN_IMAGE_Y, score_frame

FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def format_optional(value: float | None, width: int = 6, precision: int = 3) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:{width}.{precision}f}"


def mean_absolute(values: list[float | None]) -> float | None:
    present = [abs(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the path model against labeled frames")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-y", type=float, default=DEFAULT_MIN_IMAGE_Y,
                        help="only rows below this fraction of frame height feed the edge error")
    add_profile_arguments(parser, CaptureProfileName.BELGIAN_DATASET)
    add_decoder_argument(parser)
    arguments = parser.parse_args()

    profile = profile_from_arguments(arguments)
    detector = PathDetector(arguments.model, decoder_from_arguments(arguments))
    frame_paths = sorted(path for path in arguments.frames.iterdir() if path.suffix.lower() in FRAME_SUFFIXES)

    print(f"{'frame':<16} {'score':>6} {'IoU':>6} {'edge mean':>10} {'edge max':>9} {'sides':>6} {'missed':>7}"
          f" {'pitch diff':>9} {'pos diff':>7} {'height diff':>11}")
    ious: list[float] = []
    edge_means: list[float | None] = []
    pitch_differences: list[float | None] = []
    position_differences: list[float | None] = []
    height_differences: list[float | None] = []
    for frame_path in frame_paths:
        label_path = arguments.labels / f"{frame_path.stem}_mask.png"
        if not label_path.exists():
            print(f"{frame_path.name:<16} no label, skipped")
            continue
        # Phones store rotation in EXIF. The app gets upright frames, so this must too.
        image = ImageOps.exif_transpose(Image.open(frame_path)).convert("RGB")
        label = numpy.asarray(ImageOps.exif_transpose(Image.open(label_path)).convert("L")) > 127

        result = detector.detect_path(image)
        score = score_frame(result, label, arguments.min_y, profile)
        pov = score.pov_difference
        ious.append(score.iou)
        edge_means.append(score.edge_error_mean)
        if pov.both_reliable:
            pitch_differences.append(pov.pitch_deg)
            position_differences.append(pov.position_fraction)
            height_differences.append(pov.camera_height_m)
        print(
            f"{frame_path.name:<16} {result.top_score:6.3f} {score.iou:6.3f} "
            f"{format_optional(score.edge_error_mean, 10)} {format_optional(score.edge_error_max, 9)} "
            f"{score.sides_compared:6d} {score.rows_missed:7d} {format_optional(pov.pitch_deg, 9, 2)} "
            f"{format_optional(pov.position_fraction, 7)} {format_optional(pov.camera_height_m, 11)}"
            f"{'' if pov.both_reliable or pov.pitch_deg is None else '  (unreliable, not in mean)'}"
        )

    if ious:
        print(
            f"\n{len(ious)} frames. Mean IoU {sum(ious) / len(ious):.3f}. "
            f"Mean edge error {format_optional(mean_absolute(edge_means)).strip()} path widths.\n"
            f"Mean absolute model-minus-label over {len(pitch_differences)} reliable frames: "
            f"pitch {format_optional(mean_absolute(pitch_differences), 0, 2)} deg, "
            f"position {format_optional(mean_absolute(position_differences), 0)} path widths, "
            f"height {format_optional(mean_absolute(height_differences), 0)} m."
        )


if __name__ == "__main__":
    main()
