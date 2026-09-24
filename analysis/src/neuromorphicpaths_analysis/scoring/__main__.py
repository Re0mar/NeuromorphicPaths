"""
Score the app's model on a folder of frames with labels.

Usage:
    python -m neuromorphicpaths_analysis.scoring --frames <dir> --labels <dir> [--model <tflite>]

A frame <name>.jpg is scored when <labels>/<name>_mask.png exists, the file the labeling tool
writes. Frames without a label are listed and skipped.
"""

# Standard library imports
import argparse
from pathlib import Path

# Third party imports
import numpy
from PIL import Image, ImageOps

# Local package imports
from neuromorphicpaths_analysis.detector import PathDetector
from neuromorphicpaths_analysis.detector.path_detector import DEFAULT_MODEL_PATH
from neuromorphicpaths_analysis.scoring.scores import DEFAULT_MIN_IMAGE_Y, score_frame

FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def format_optional(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the path model against labeled frames")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-y", type=float, default=DEFAULT_MIN_IMAGE_Y,
                        help="only rows below this fraction of frame height feed the edge error")
    arguments = parser.parse_args()

    detector = PathDetector(arguments.model)
    frame_paths = sorted(path for path in arguments.frames.iterdir() if path.suffix.lower() in FRAME_SUFFIXES)

    print(f"{'frame':<16} {'score':>6} {'IoU':>6} {'edge mean':>10} {'edge max':>9} {'sides':>6} {'missed':>7}")
    ious: list[float] = []
    edge_means: list[float] = []
    for frame_path in frame_paths:
        label_path = arguments.labels / f"{frame_path.stem}_mask.png"
        if not label_path.exists():
            print(f"{frame_path.name:<16} no label, skipped")
            continue
        # Phones store rotation in EXIF. The app gets upright frames, so this must too.
        image = ImageOps.exif_transpose(Image.open(frame_path)).convert("RGB")
        label = numpy.asarray(ImageOps.exif_transpose(Image.open(label_path)).convert("L")) > 127

        result = detector.detect_path(image)
        score = score_frame(result, label, arguments.min_y)
        ious.append(score.iou)
        if score.edge_error_mean is not None:
            edge_means.append(score.edge_error_mean)
        print(
            f"{frame_path.name:<16} {result.top_score:6.3f} {score.iou:6.3f} "
            f"{format_optional(score.edge_error_mean):>10} {format_optional(score.edge_error_max):>9} "
            f"{score.sides_compared:6d} {score.rows_missed:7d}"
        )

    if ious:
        mean_edge = sum(edge_means) / len(edge_means) if edge_means else None
        print(f"\n{len(ious)} frames. Mean IoU {sum(ious) / len(ious):.3f}. "
              f"Mean edge error {format_optional(mean_edge).strip()} path widths.")


if __name__ == "__main__":
    main()
