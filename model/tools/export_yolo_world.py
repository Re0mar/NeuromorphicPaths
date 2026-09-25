"""
Export YOLO-World with this project's vocabulary to ONNX, for the app's detector.

Reads the prompts from the vocabulary file the app ships, so the model's class order and the
app's class order are the same file. Writes the model next to it.

Run from a Python environment with ultralytics, onnx and onnxslim installed:

    python model/tools/export_yolo_world.py
    python model/tools/export_yolo_world.py --weights yolov8m-worldv2.pt --input-size 416

The first run downloads the base weights and the CLIP text encoder ultralytics uses to embed
the prompts. Nothing is trained.
"""

# Standard library imports
import argparse
import shutil
from pathlib import Path

# Third party imports
from ultralytics import YOLOWorld

MODEL_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = MODEL_DIR / "src" / "main" / "assets" / "yolo_world"
VOCABULARY_PATH = ASSET_DIR / "vocabulary.tsv"
OUTPUT_PATH = ASSET_DIR / "yolo_world.onnx"

DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"
DEFAULT_INPUT_SIZE = 320


def read_prompts(vocabulary_path: Path) -> list[str]:
    """
    Return the prompts in file order, skipping blank lines and comments.

    :param vocabulary_path: The tab-separated vocabulary file the app also reads.
    :return: Prompts in the order the model's output classes will use.
    :rtype: list[str]
    """
    prompts: list[str] = []
    for line_number, raw_line in enumerate(vocabulary_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{vocabulary_path}:{line_number}: expected 'prompt<TAB>CLASS', got {raw_line!r}")
        prompts.append(fields[0].strip())
    if not prompts:
        raise ValueError(f"{vocabulary_path} has no prompts")
    return prompts


def export(weights: str, input_size: int) -> Path:
    """
    Embed the vocabulary into the model and export it to ONNX at the app's asset path.

    :param weights: Ultralytics YOLO-World checkpoint name or path.
    :param input_size: Square input size in pixels. Must match the detector's input size.
    :return: Path of the written ONNX file.
    :rtype: Path
    """
    prompts = read_prompts(VOCABULARY_PATH)
    model = YOLOWorld(weights)
    model.set_classes(prompts)
    # Static shapes, fp32 and opset 12 keep the graph inside what ONNX Runtime Mobile runs
    # on every Android CPU without extra providers.
    exported = Path(model.export(format="onnx", imgsz=input_size, half=False, dynamic=False, simplify=True, opset=12))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(OUTPUT_PATH))
    return OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO-World checkpoint, default yolov8s-worldv2.pt")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="square input size, default 320")
    arguments = parser.parse_args()
    output = export(arguments.weights, arguments.input_size)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"wrote {output} ({size_mb:.1f} MB), {len(read_prompts(VOCABULARY_PATH))} classes, input {arguments.input_size}")


if __name__ == "__main__":
    main()
