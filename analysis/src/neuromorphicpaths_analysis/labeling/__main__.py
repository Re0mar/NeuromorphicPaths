"""
Label path masks with SAM 2, one click at a time.

Usage:
    python -m neuromorphicpaths_analysis.labeling propose <frame> [<frame> ...] --out <dir>
    python -m neuromorphicpaths_analysis.labeling refine <frame> --out <dir> [--pos X,Y ...] [--neg X,Y ...] [--reset]

propose starts each frame from one click at the bottom center. refine adds clicks to what the
frame already has. Coordinates are original-frame pixels, read off the overlay's grid.
"""

# Standard library imports
import argparse
from pathlib import Path

# Local package imports
from neuromorphicpaths_analysis.labeling.sam_labeler import (
    SamLabeler,
    default_prompts,
    empty_prompts,
    load_frame,
    load_prompts,
)


def parse_point(text: str) -> list[int]:
    """
    Parse an "X,Y" command line argument into integer pixel coordinates.

    :param text: Coordinates as written on the command line.
    :return: [x, y] in original frame pixels.
    :rtype: list[int]
    """
    x_text, y_text = text.split(",")
    return [int(float(x_text)), int(float(y_text))]


def command_propose(arguments: argparse.Namespace) -> None:
    labeler = SamLabeler()
    for frame_path in arguments.frames:
        prompts = default_prompts(load_frame(frame_path))
        score = labeler.label_frame(frame_path, arguments.out, prompts)
        print(f"{frame_path.name}: proposed, SAM score {score:.3f}")


def command_refine(arguments: argparse.Namespace) -> None:
    frame_path = arguments.frames[0]
    prompts = empty_prompts() if arguments.reset else load_prompts(arguments.out / f"{frame_path.stem}.json")
    for point_text in arguments.pos:
        prompts["points"].append(parse_point(point_text))
        prompts["labels"].append(1)
    for point_text in arguments.neg:
        prompts["points"].append(parse_point(point_text))
        prompts["labels"].append(0)
    if not prompts["points"]:
        raise SystemExit(f"{frame_path.name}: no clicks to run. Give --pos or --neg.")

    score = SamLabeler().label_frame(frame_path, arguments.out, prompts)
    print(f"{frame_path.name}: refined with {len(prompts['points'])} clicks, SAM score {score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM 2 assisted path labeling")
    parser.add_argument("command", choices=["propose", "refine"])
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pos", action="append", default=[], help="positive click as X,Y")
    parser.add_argument("--neg", action="append", default=[], help="negative click as X,Y")
    parser.add_argument("--reset", action="store_true", help="drop earlier clicks for this frame")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    {"propose": command_propose, "refine": command_refine}[arguments.command](arguments)


if __name__ == "__main__":
    main()
