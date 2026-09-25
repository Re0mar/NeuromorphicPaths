# NeuromorphicPaths

An Android app that watches the ground ahead of a walking person, finds the obstacles in view,
and turns them into a suggested heading. Each obstacle pushes the walker away from itself. The
pushes add up to one direction, and that direction, frame after frame, is the path.

The app is a thin shell. All the work happens behind four contracts, so any one part can be
swapped without touching the others: where frames come from, what finds obstacles in a frame,
what turns obstacles into a heading, and what shows the result.

## Modules

```
app  --->  input   --->  core
     --->  model   --->  core
     --->  output  --->  core
     --->  math    --->  core
```

| Module | Kind | Holds |
|---|---|---|
| `core` | plain JVM | The data types and the four contracts, plus the pipeline that chains them. No Android imports. |
| `math` | plain JVM | Geometry and the guidance field. No Android imports, so every formula is unit-testable on a laptop. |
| `input` | Android library | Frame sources: the phone camera, a video file for replay, and the orientation sensor. |
| `model` | Android library | Obstacle detectors. The pretrained detector lands here. |
| `output` | Android library | Displays: the on-screen overlay and a log line per frame. |
| `app` | Android application | Wires one of each together and puts it on screen. |

Gradle enforces the arrows. `core` and `math` cannot see `android.*`, and no module below `app`
can see another module below `app`. A change that needs a new arrow is a design change, not a
dependency line.

## The four contracts

All of them live in `core`, one interface each.

| Layer | Interface | What it promises |
|---|---|---|
| Input | `FrameSource` | A stream of `Frame`s: RGBA pixels, timestamp, camera pose, camera field of view. |
| Model | `ObstacleDetector` | For one frame, the list of `Detection`s: class, confidence, box, optional track id. |
| Math | `ObstacleLocator` | Detections in a frame to `Obstacle`s: bearing and range from the walker. |
| Math | `GuidanceField` | Obstacles and the walker's state to `Guidance`: desired heading, overall surprise, and the push each obstacle contributed. |
| Output | `GuidanceDisplay` | Receives one `GuidanceUpdate` per processed frame and shows it however it likes. |

`GuidancePipeline` in `core` runs them in that order. It drops frames the detector cannot keep up
with rather than queueing them, so what the walker sees is never older than one detector call.

## What is implemented right now

- **Input.** `CameraXFrameSource` streams the back camera. `VideoFileFrameSource` steps through a
  recording at a fixed interval, which is how phone and glasses recordings are replayed.
  `SensorPoseProvider` reads camera pitch from the rotation vector sensor.
- **Model.** `EmptyObstacleDetector` returns nothing. `ScriptedObstacleDetector` returns a fixed
  list every frame, for exercising the display. `OnnxYoloWorldDetector` runs YOLO-World with
  this project's class list through ONNX Runtime. Its model file is not committed. See *The
  detector model* below.
- **Math.** `GroundPlaneObstacleLocator` gets bearing from the box center and the horizontal field
  of view, and range from where the box meets the ground, given camera height and pitch. When
  the box bottom is cut off it falls back to a typical height per class. `PushFieldGuidance` is
  the push field: per-obstacle surprise from miss distance and time to contact, summed over
  candidate headings, lowest sum wins. `docs/math/push_field.md` explains it. `NoGuidanceField`
  is the straight-ahead stand-in for tests.
- **Output.** `ScreenOverlayDisplay` plus the `GuidanceOverlay` composable draws the frame, the
  boxes, a heading arrow and the numbers. `LogcatDisplay` writes one line per frame.

## Running it

Open the repository root in Android Studio and run the `app` configuration, or from a shell:

```
./gradlew :app:assembleDebug
./gradlew :core:test :math:test
```

The app has two buttons. **Camera** asks for permission and starts the live pipeline. **Open
video** picks a recording and replays it. The **Scripted** switch swaps in the fixed detections
the next time a source starts.

## The detector model

The app expects `model/src/main/assets/yolo_world/yolo_world.onnx`, which is about 50 MB and
is built, not written, so it stays out of git. `model/tools/export_yolo_world.py` produces it
from the public YOLO-World weights and the prompt list in `vocabulary.tsv` beside it. No
training happens. From a Python environment with `ultralytics`, `onnx` and `onnxslim`:

```
python model/tools/export_yolo_world.py
```

Without the file the app still builds and runs, and the YOLO-World choice is grayed out.
To change what the detector looks for, edit `vocabulary.tsv` and export again. The app reads
the same file, so the class order cannot drift between the two.

## Adding a piece

- A new frame source implements `FrameSource` in `input`.
- A new detector implements `ObstacleDetector` in `model`.
- A new display implements `GuidanceDisplay` in `output` and is added to the list in `app`.
- New math implements `ObstacleLocator` or `GuidanceField` in `math`, with a unit test beside it.

Two placeholders in `app` need real numbers before range estimates mean anything: the camera
height above the ground and the horizontal field of view. Both are named constants in
`AppDefaults.kt`.
