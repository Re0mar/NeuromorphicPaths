"""
Export SegFormer-B0 trained on ADE20K to ONNX, for the app's surface segmenter.

The checkpoint comes from the Hugging Face hub on the first run and lands in its cache. Nothing
is trained. The graph takes one RGB image in 0..1, planar, at a fixed square size, folds the
ImageNet normalization in, and returns the winning class index per cell of a grid a quarter of
the input size, so the phone does no arithmetic on 150 channels of logits.

The class table beside the model, ade20k_classes.tsv, is checked against the checkpoint's own
label list before anything is written, so the app's index-to-class mapping cannot drift from
the model's output order.

Run from a Python environment with torch, transformers, onnx and onnxruntime installed:

    python model/tools/export_segformer.py
    python model/tools/export_segformer.py --input-size 384 --output /tmp/segformer_384.onnx
"""

# Standard library imports
import argparse
from pathlib import Path

# Third party imports
import numpy
import onnx
import onnxruntime
import torch
from transformers import SegformerForSemanticSegmentation

MODEL_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = MODEL_DIR / "src" / "main" / "assets" / "segformer"
CLASSES_PATH = ASSET_DIR / "ade20k_classes.tsv"
OUTPUT_PATH = ASSET_DIR / "segformer_ade20k.onnx"

DEFAULT_WEIGHTS = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_INPUT_SIZE = 256

# The normalization SegFormer's image processor applies, folded into the graph.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# LayerNormalization arrived in opset 17. Earlier opsets decompose it into a dozen small ops,
# which ONNX Runtime 1.30 runs fine but slower.
OPSET = 17


def read_class_table(classes_path: Path) -> list[tuple[int, str]]:
    """
    Return (index, label) pairs from the class table, in file order.

    :param classes_path: The tab-separated table the app also reads.
    :return: Index and label per line, comments and blanks skipped.
    :rtype: list[tuple[int, str]]
    """
    rows: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(classes_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{classes_path}:{line_number}: expected 'index<TAB>label<TAB>SCENE_CLASS', got {raw_line!r}")
        rows.append((int(fields[0]), fields[1].strip()))
    if not rows:
        raise ValueError(f"{classes_path} has no classes")
    return rows


def check_class_table(model: SegformerForSemanticSegmentation, classes_path: Path) -> None:
    """
    Fail loudly when the class table and the checkpoint disagree on any index or label.

    :param model: The loaded checkpoint, whose config carries id2label.
    :param classes_path: The table to check.
    """
    expected = {index: label.strip() for index, label in model.config.id2label.items()}
    rows = read_class_table(classes_path)
    if [index for index, _ in rows] != list(range(len(expected))):
        raise ValueError(f"{classes_path} must list indices 0..{len(expected) - 1} in order, once each")
    for index, label in rows:
        if expected[index] != label:
            raise ValueError(f"{classes_path}: index {index} is {label!r} but the checkpoint says {expected[index]!r}")


class NormalizeAndArgmax(torch.nn.Module):
    """Wraps the checkpoint so the graph takes 0..1 RGB and returns class indices."""

    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=(rgb - self.mean) / self.std).logits
        return logits.argmax(dim=1)


def export(weights: str, input_size: int, output_path: Path = OUTPUT_PATH) -> Path:
    """
    Write the ONNX model and check it runs, at the app's asset path by default.

    :param weights: Hugging Face checkpoint name or local path.
    :param input_size: Square input size in pixels. The app reads it back from the model.
    :param output_path: Where to write the model. Anywhere but the asset path is for comparing
        exports, since the app only loads the one in its assets.
    :return: Path of the written ONNX file.
    :rtype: Path
    """
    model = SegformerForSemanticSegmentation.from_pretrained(weights).eval()
    check_class_table(model, CLASSES_PATH)
    wrapped = NormalizeAndArgmax(model).eval()
    example = torch.zeros(1, 3, input_size, input_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The legacy exporter rather than the dynamo one: static shapes, no external data files, and
    # a graph ONNX Runtime Mobile runs on the CPU provider without extra passes.
    torch.onnx.export(
        wrapped,
        (example,),
        str(output_path),
        input_names=["rgb"],
        output_names=["classes"],
        opset_version=OPSET,
        dynamic_axes=None,
        dynamo=False,
    )
    onnx.checker.check_model(str(output_path))
    # Prove the file loads and answers with the grid size the app expects, a quarter of the input.
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    classes = session.run(None, {"rgb": numpy.zeros((1, 3, input_size, input_size), dtype=numpy.float32)})[0]
    expected_shape = (1, input_size // 4, input_size // 4)
    if classes.shape != expected_shape:
        raise RuntimeError(f"exported model answers {classes.shape}, expected {expected_shape}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"checkpoint, default {DEFAULT_WEIGHTS}")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help=f"square input size, default {DEFAULT_INPUT_SIZE}")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="where to write the model, default the app's asset path")
    arguments = parser.parse_args()
    output = export(arguments.weights, arguments.input_size, arguments.output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"wrote {output} ({size_mb:.1f} MB), {len(read_class_table(CLASSES_PATH))} classes, input {arguments.input_size}, grid {arguments.input_size // 4}")


if __name__ == "__main__":
    main()
