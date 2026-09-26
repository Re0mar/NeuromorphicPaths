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
| Model | `SurfaceSegmenter` | For one frame, a coarse `SceneClassMap`: what each cell of a grid over the frame is, pavement, grass, wall, sky. |
| Math | `ObstacleLocator` | Detections in a frame to `Obstacle`s: bearing and range from the walker. |
| Math | `SceneLocator` | A scene map in a frame to a `LocatedScene`: the ground surfaces around the walker, and the feet of walls, buildings and stairs as obstacles. |
| Math | `GuidanceField` | Obstacles, the ground and the walker's state to `Guidance`: desired heading, overall surprise, and the push each obstacle contributed. |
| Output | `GuidanceDisplay` | Receives one `GuidanceUpdate` per processed frame and shows it however it likes. |

`GuidancePipeline` in `core` runs them in that order. It drops frames the detector cannot keep up
with rather than queuing them, so what the walker sees is never older than one detector call.
The segmenter is optional and runs on every third frame, and the scene it produces is held and
used on the frames between, since walls and verges move slowly in the frame.

## What is implemented right now

- **Input.** `CameraXFrameSource` streams the back camera. `VideoFileFrameSource` steps through a
  recording at a fixed interval, which is how phone and glasses recordings are replayed.
  `SensorPoseProvider` reads camera pitch and azimuth from the rotation vector sensor.
  `AccelerometerCadenceSpeed` counts the walker's steps off the accelerometer and turns the
  cadence into a speed, live, and `Recording` does the same from a Sensor Logger
  `TotalAcceleration.csv` on replay. `GpsGroundSpeed` smooths the speed from GPS fixes when
  location permission is granted, as the outdoor cross-check.
- **Model.** `EmptyObstacleDetector` returns nothing. `ScriptedObstacleDetector` returns a fixed
  list every frame, for exercising the display. `OnnxYoloWorldDetector` runs YOLO-World with
  this project's class list through ONNX Runtime. Its model file is not committed. See *The
  detector model* below. `OnnxSegFormerSegmenter` runs SegFormer-B0 trained on ADE20K the same
  way, at 256 by 256, and maps its 150 labels onto nine scene classes through
  `ade20k_classes.tsv`. Its model file is not committed either. See *The segmenter model*.
- **Math.** `GroundPlaneObstacleLocator` gets bearing from the box center and the horizontal field
  of view, and range from where the box meets the ground, given camera height and pitch. When
  the box bottom is cut off it falls back to a typical height per class. `PushFieldGuidance` is
  the push field: per obstacle, the probability of a collision on a candidate heading from
  miss distance, time to contact, detector confidence and what the class is, turned into a
  surprise and summed over candidate headings, lowest sum wins, with a small charge per meter
  for ground the walker would rather not cross once a segmenter says what the ground is. It
  also reports the entropy of its belief over headings. `docs/math/push_field.md` explains it,
  including the running heading wobble it measures and why that does not set the turn
  tolerance. `GroundPlaneSceneLocator` turns a scene map into the surfaces the field asks about,
  by projecting each asked-for ground point back into the frame, and into obstacles for walls,
  buildings and stairs, by walking the cells where a structure meets the ground, projecting
  each through the ground plane and keeping one sample per half meter.
  `NoGuidanceField` is the straight-ahead stand-in for tests.
- **Tracking.** `DetectionTracker` in `math` follows a box from frame to frame by overlap and
  gives it a track id. A box the detector misses for a frame or two is carried through at a
  fading confidence, so one missed detection does not flip the heading. `TrackedObstacleDetector`
  wraps any detector with it, and `TrackedObstacleLocator` wraps any locator so a track's range
  over the last two seconds gives its closing speed, which the field uses for time to contact.
- **Output.** `ScreenOverlayDisplay` plus the `GuidanceOverlay` composable draws the frame, the
  boxes, a heading arrow and the numbers. `LogcatDisplay` writes one line per frame.

## Running it

Open the repository root in Android Studio and run the `app` configuration, or from a shell:

```
./gradlew :app:assembleDebug
./gradlew :core:test :math:test
```

The app has two buttons. **Camera** asks for permission and starts the live pipeline. **Open
recording** picks a folder holding a video and, if Sensor Logger ran beside it, its
`Orientation.csv` and `TotalAcceleration.csv`. With the logs, replay uses the pitch the camera
really had at each frame, lined up through the video's own end time and duration, feeds the
logged azimuth to the wobble estimate, and takes the walker's speed from their steps. Without
them, replay takes the pose from the live sensor and the field's default speed. The detector row picks
what runs, YOLO-World by default when its model is bundled, and changing it restarts the source.

To replay from a shell without touching the screen, put the folder in the app's own storage and
name it in the launch intent:

```
adb push recording.mp4 /data/local/tmp/outdoor1/
adb push Orientation.csv /data/local/tmp/outdoor1/
adb push TotalAcceleration.csv /data/local/tmp/outdoor1/
adb shell run-as com.neuromorphicpaths mkdir -p files/recordings/outdoor1
adb shell run-as com.neuromorphicpaths cp /data/local/tmp/outdoor1/recording.mp4 files/recordings/outdoor1/
adb shell run-as com.neuromorphicpaths cp /data/local/tmp/outdoor1/Orientation.csv files/recordings/outdoor1/
adb shell run-as com.neuromorphicpaths cp /data/local/tmp/outdoor1/TotalAcceleration.csv files/recordings/outdoor1/
adb shell am start -n com.neuromorphicpaths/.app.MainActivity --es recording outdoor1
adb logcat -s Guidance:D
```

The log line per frame carries the detector time, the counts, the heading, the surprise, the
walker's speed, wobble and turn tolerance, the entropy of the field's belief over headings, the
segmenter's time on the frames it ran on (-1 on the others) and how many structure samples the
scene added, so a replay's numbers can be pulled out with `grep`. One indented line per obstacle
follows it, with the track id, class, confidence, range, bearing, closing speed and that
obstacle's surprise. Structure samples have track -1 and class BUILDING, WALL or STAIRS.

The segmenter runs whenever its model is bundled. The **Surfaces** switch turns it off, and so
does `--ez segmenter false` on the intent. `--ez xnnpack true` asks ONNX Runtime for the XNNPACK
provider for the segmenter, which is there for timing comparisons only.

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

The export is at 640 by default. The app reads the input size from the model, so a different
size is only a flag, and `--output` writes the model somewhere other than the assets folder
for comparing exports side by side:

```
python model/tools/export_yolo_world.py --input-size 320 --output /tmp/yolo_world_320.onnx
```

640 is the size to ship. On the Pixel 8, replaying the outdoor walk through ONNX Runtime
1.30.0 on the CPU, the detector takes a median 553 ms per frame at 640 and 112 ms at 320, but
the 320 model puts a box in 327 of the 918 frames against 625 at 640. A portrait frame
letterboxed into a 320 square leaves the model 180 pixels of width, and the obstacles that
matter most, a cone a meter ahead or a post beside the path, are the ones it loses.

## The segmenter model

The app expects `model/src/main/assets/segformer/segformer_ade20k.onnx`, about 15 MB, built
by `model/tools/export_segformer.py` from the public SegFormer-B0 checkpoint trained on ADE20K,
and gitignored for the same reason as the detector. From a Python environment with `torch`,
`transformers`, `onnx` and `onnxruntime`:

```
python model/tools/export_segformer.py
```

The first run downloads the checkpoint into the Hugging Face cache. The export folds the
ImageNet normalization and the argmax into the graph, so the app hands it RGB in 0..1 and gets
back one class index per cell of a 64 by 64 grid. `ade20k_classes.tsv` beside the model says
which of the nine scene classes each of the 150 ADE20K labels means, pavement, grass, dirt,
road, building, wall, stairs, sky or other. The export script checks that table against the
checkpoint's own label list, and the app reads the same file, so the two cannot drift. To
change what counts as pavement or as a wall, edit the third column. No export is needed for
that.

The default input is 256 square, which gives a map every third detector frame for about a fifth
of a detector call on the Pixel. `--input-size` and `--output` work as they do for the detector.

Without the file the app still builds and runs, with the Surfaces switch grayed out, and the
field charges nothing for the ground.

## Adding a piece

- A new frame source implements `FrameSource` in `input`.
- A new detector implements `ObstacleDetector` in `model`.
- A new display implements `GuidanceDisplay` in `output` and is added to the list in `app`.
- New math implements `ObstacleLocator` or `GuidanceField` in `math`, with a unit test beside it.

Two placeholders in `app` need real numbers before range estimates mean anything: the camera
height above the ground and the horizontal field of view. Both are named constants in
`AppDefaults.kt`.
