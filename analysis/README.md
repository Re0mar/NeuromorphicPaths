# Analysis

Offline analysis for NeuromorphicPaths, in Python. It runs the app's path model on recorded
frames and scores it against hand-checked labels.

The detector here is a copy of the app's `PathDetector.kt`. It loads the same model file from
`app/src/main/assets/best_int8.tflite`, with the same preprocessing and decoding, so a score
measured here is a score for the model the app runs.

## Install

Python 3.12 or newer. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

That covers the detector, scoring and tests. Labeling also needs torch and SAM 2, a
multi-gigabyte install, so it's a separate extra:

```powershell
.venv\Scripts\python -m pip install -e ".[labeling]"
```

For labeling on an NVIDIA GPU, install the CUDA build of torch first. Otherwise pip installs a
CPU-only build, and each click takes several seconds instead of well under one.

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Run

Frames, recordings and label masks go in `data/`, which git ignores. Recordings show
passers-by, so none of it is committed.

**Label frames.** `propose` starts each frame from one click at the bottom center. `refine` adds
clicks. Coordinates are pixels of the original frame, read off the grid drawn on the overlay.

```powershell
.venv\Scripts\python -m neuromorphicpaths_analysis.labeling propose data\frames\*.jpg --out data\labels
.venv\Scripts\python -m neuromorphicpaths_analysis.labeling refine data\frames\00008.jpg --out data\labels --pos 1050,700 --neg 400,700
```

Each frame gets `<name>_mask.png` (the label), `<name>_overlay.png` (for review) and
`<name>.json` (its clicks, so refinements build on earlier ones). A negative click on its own
can shrink the mask to almost nothing. When excluding something, also click on what should stay.

**What counts as path.** The model has one class, `sidewalk`, learned from the
`flat-sidewalk` class of the `segments/sidewalk-semantic` dataset. Labels have to follow that
dataset's rules, or the score measures a difference in rules rather than a model error. Checked
against its labels:

| Path | Not path |
|---|---|
| All sidewalk paving, including differently colored strips and decorative cobbles | Curbs and gutters |
| Manholes, drain grates and lights set into the sidewalk | Crosswalks, including the paved approach to one |
| | Cycling lanes |
| | Driveways and parking bays, even when paved like the sidewalk |
| | Tree pits and their grates |
| | Road |

SAM follows what the surface looks like, not these rules. It merges a driveway paved like the
sidewalk into the path, and it stops at a color change inside the sidewalk. So when reviewing an
overlay, check each row of the right-hand column, and add clicks where a boundary is a rule
rather than a visible edge.

**Score the model.**

```powershell
.venv\Scripts\python -m neuromorphicpaths_analysis.scoring --frames data\frames --labels data\labels
```

**Run the tests.**

```powershell
.venv\Scripts\python -m pytest
```

## What the scores mean

Both scores compare on the model's own mask grid. The model takes a 320×320 input and outputs
its mask on an 80×80 grid, so one cell is 4 pixels of model input. How big a cell is in the
original frame depends on the camera:

| Source | Frame | Cells covering the image | One cell in the frame |
|---|---|---|---|
| Neon scene camera | 1600×1200 | 80×60 | 20 px |
| Belgian sidewalk dataset | 1920×1080 | 80×45 | 24 px |
| Meta glasses stream | 504×896 | 45×80 | 11 px |

Labels are shrunk onto that grid before comparing. A cell counts as path when at least half of
it is path in the label. That way the model isn't charged for detail it can't produce.

- **IoU** is the overlap between model and label divided by their union, counted in cells.
  1.0 is perfect. It's dominated by the large middle of the path, so it barely notices edge
  mistakes.
- **Edge error** is how far the model's left and right path edges sit from the label's, in
  path widths, on each row in the lower half of the frame. 0.05 means the edge is off by 5% of
  the path's width. This is the number that feeds the rider's position across the path.
  **Sides compared** counts the row edges measured. **Missed** counts label rows the model's
  outline never reached.

A side that runs off the edge of the frame is **clipped**. That edge is the frame border, not
the path's edge, so it's left out of the edge error. The app doesn't flag clipping yet. The
copy here does, with a one-cell margin, because the model's box usually stops just short of
the border.

## Layers

The package is split into layers with a fixed direction of dependency.
`tests/test_layers.py` checks the import statements, so a rule can't erode one import at a
time.

| Layer | Holds | May use | Must not use |
|---|---|---|---|
| `detector` | Copy of the app's detector | the model file, numpy, OpenCV, Pillow, LiteRT | anything else in the package |
| `scoring` | IoU and edge error | `detector` | `labeling`. Labels are plain mask files to it |
| `labeling` | SAM 2 assisted labeling | torch, transformers | `detector`, `scoring` |

Only `labeling` may import torch or transformers, which keeps the heavy install optional.

Planned, not built yet:

- `geometry` will estimate the vanishing point, heading and position across the path from the
  traced outline. It may use `detector`.
- `recordings` will read Neon exports: scene video, frame timestamps, IMU and the camera
  calibration. It will use no other layer.

A new layer needs a row in `tests/test_layers.py`. The test fails until it has one.

## Keeping the copy in step with the app

The constants at the top of `detector/path_detector.py` carry the same names as the companion
object in `PathDetector.kt`, in snake case. When one changes in the app, change it here too.
The one deliberate difference is `CLIP_MARGIN_CELLS`, which the app doesn't have yet.

Resizing isn't pixel-identical. The app shrinks by repeated halving and then one bilinear draw,
and this copy uses Pillow's antialiased bilinear resize. Both avoid the aliasing of a single
large downscale, but individual pixel values differ slightly.
