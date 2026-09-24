"""
SAM 2 assisted path labeling with a human yes/no review loop.

Each frame gets a JSON sidecar holding its click prompts, so corrections accumulate across runs
rather than starting over. Outputs per frame, all in the output folder:

    <stem>.json          click prompts and SAM's own score for the chosen mask
    <stem>_mask.png      binary path mask at original resolution, 255 = path
    <stem>_overlay.png   mask, its outline, the numbered clicks and a coordinate grid for review
"""

# Standard library imports
import json
from pathlib import Path

# Third party imports
import cv2
import numpy
import torch
from PIL import Image, ImageOps
from transformers import Sam2Model, Sam2Processor

MODEL_ID = "facebook/sam2.1-hiera-small"

# Walking with the head forward puts the path under the bottom center of the frame nearly every time.
DEFAULT_CLICK_X_FRACTION = 0.5
DEFAULT_CLICK_Y_FRACTION = 0.92

PINHOLE_MAX_PIXELS = 2000
GRID_STEP_PIXELS = 160
MASK_FILL_BGR = (255, 120, 0)
OUTLINE_BGR = (0, 255, 255)
POSITIVE_BGR = (0, 200, 0)
NEGATIVE_BGR = (0, 0, 255)


def empty_prompts() -> dict:
    return {"points": [], "labels": [], "score": None}


def default_prompts(image: Image.Image) -> dict:
    """One positive click at the bottom center, the usual first guess."""
    click = [int(image.width * DEFAULT_CLICK_X_FRACTION), int(image.height * DEFAULT_CLICK_Y_FRACTION)]
    return {"points": [click], "labels": [1], "score": None}


def load_prompts(sidecar_path: Path) -> dict:
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    return empty_prompts()


def save_prompts(sidecar_path: Path, prompts: dict) -> None:
    sidecar_path.write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_frame(image_path: Path) -> Image.Image:
    # Phones store rotation in EXIF, and the clicks are given in upright coordinates.
    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


class SamLabeler:
    """Holds the SAM 2 model so a batch of frames loads it once."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam2Processor.from_pretrained(MODEL_ID)
        self.model = Sam2Model.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    def predict_mask(
        self,
        image: Image.Image,
        points: list[list[int]],
        labels: list[int],
    ) -> tuple[numpy.ndarray, float]:
        """
        Run SAM 2 on one frame with a set of click prompts.

        :param image: RGB frame at original resolution.
        :param points: Click coordinates in original pixels.
        :param labels: 1 for a positive click, 0 for a negative click, one per point.
        :return: Boolean mask at original resolution and SAM's own quality score for it.
        :rtype: tuple[numpy.ndarray, float]
        """
        # A single click is ambiguous (one tile, the path, the whole ground), so ask for three
        # candidates and keep the best. With several clicks the intent is clear enough for one.
        multimask = len(points) == 1
        inputs = self.processor(
            images=image,
            input_points=[[points]],
            input_labels=[[labels]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.cpu()[0, 0]
        best_index = int(scores.argmax())
        return masks[0, best_index].numpy().astype(bool), float(scores[best_index])

    def label_frame(self, image_path: Path, out_dir: Path, prompts: dict) -> float:
        """
        Label one frame with the given clicks and write the mask, overlay and sidecar.

        :return: SAM's own score for the mask it chose.
        :rtype: float
        """
        image = load_frame(image_path)
        mask, score = self.predict_mask(image, prompts["points"], prompts["labels"])
        mask = clean_mask(mask, prompts["points"], prompts["labels"])
        prompts["score"] = score

        stem = image_path.stem
        save_prompts(out_dir / f"{stem}.json", prompts)
        Image.fromarray((mask * 255).astype(numpy.uint8)).save(out_dir / f"{stem}_mask.png")
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), render_overlay(image, mask, prompts["points"], prompts["labels"]))
        return score


def clean_mask(mask: numpy.ndarray, points: list[list[int]], labels: list[int]) -> numpy.ndarray:
    """
    Drop islands not touching a positive click, and fill pinhole gaps inside the path.

    SAM leaves scattered specks on textured paving next to the path. They are never path, and
    they would put false edges into anything that reads the mask boundary.

    :return: Cleaned boolean mask.
    :rtype: numpy.ndarray
    """
    _, component_ids = cv2.connectedComponents(mask.astype(numpy.uint8))
    kept_ids = {
        int(component_ids[y, x])
        for (x, y), label in zip(points, labels)
        if label == 1 and component_ids[y, x] != 0
    }
    cleaned = numpy.isin(component_ids, list(kept_ids))

    # Only tiny holes get filled. Large ones are manholes or tree pits, which the dataset's own
    # labels decide, not the noise filter.
    hole_count, hole_ids, hole_stats, _ = cv2.connectedComponentsWithStats((~cleaned).astype(numpy.uint8))
    for hole_id in range(1, hole_count):
        if hole_stats[hole_id, cv2.CC_STAT_AREA] < PINHOLE_MAX_PIXELS:
            cleaned[hole_ids == hole_id] = True
    return cleaned


def render_overlay(
    image: Image.Image,
    mask: numpy.ndarray,
    points: list[list[int]],
    labels: list[int],
) -> numpy.ndarray:
    """
    Draw the mask, its outline, the numbered clicks and a labeled grid for review.

    The grid lets a reviewer name a mistake by coordinates rather than by pointing, which is
    what the refine command takes.
    """
    canvas = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    height, width = mask.shape

    fill = canvas.copy()
    fill[mask] = MASK_FILL_BGR
    canvas = cv2.addWeighted(fill, 0.4, canvas, 0.6, 0)

    contours, _ = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, OUTLINE_BGR, 3)

    for grid_x in range(0, width, GRID_STEP_PIXELS):
        cv2.line(canvas, (grid_x, 0), (grid_x, height), (255, 255, 255), 1)
        cv2.putText(canvas, str(grid_x), (grid_x + 4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for grid_y in range(0, height, GRID_STEP_PIXELS):
        cv2.line(canvas, (0, grid_y), (width, grid_y), (255, 255, 255), 1)
        cv2.putText(canvas, str(grid_y), (4, grid_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for click_number, (point, label) in enumerate(zip(points, labels), start=1):
        color = POSITIVE_BGR if label == 1 else NEGATIVE_BGR
        cv2.circle(canvas, tuple(point), 14, color, -1)
        cv2.circle(canvas, tuple(point), 14, (0, 0, 0), 2)
        cv2.putText(canvas, str(click_number), (point[0] + 18, point[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return canvas
