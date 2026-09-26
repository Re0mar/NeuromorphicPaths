# Materials notes

Rough list of everything we used, hardware, software, data and the numbers that go with them. Pull whatever you need for the materials section. All the timing numbers say what phone and settings they came from, keep that with them if you use them.

## Hardware

- test phone: Google Pixel 8, Android 17. all the timings below are from this phone
- Android emulator (x86_64) for stuff that doesn't need a real camera
- a laptop for the offline stuff (running the models on the video, rerunning the math)
- planned: Pupil Labs Neon glasses as the camera, so the phone can stay in your pocket. not hooked up live yet

## The recording

- one outdoor walk, 25 Sept 2026, campus paths
- ~4 minutes (229 s), video 1080×1920 portrait, 30 fps
- phone held at about upper chest height, tilted down about 18 degrees on average
- Sensor Logger app running at the same time at 100 Hz, we use `Orientation.csv` (which way the phone points) and `TotalAcceleration.csv` (for counting steps)
- video starts 4.61 s after the sensor log, worked out from the video file's own timestamps
- on purpose it has: something head-on, wide gaps, narrow gaps, close passes, open stretches
- the moments we check against: traffic cone around 184 s, bike rack around 206 to 219 s, a wall on the left around 20 to 25 s, a van around 204 s
- not in git because there are people in it
- still to record: a walk indoors (classroom), and one that walks into a dead end to test STOP

## Software

- Kotlin 2.2 with Jetpack Compose for the screen
- Android: minimum version 14 (API 34), built for API 37
- CameraX for the live camera
- ONNX Runtime 1.30.0 on the phone's CPU to run both models. we tried XNNPACK for the second model and it was slower (231 vs 165 ms), so CPU it is
- 1.22.0 is the oldest ONNX Runtime that doesn't trigger Android's 16 KB page warning, so don't go below that
- Python on the laptop for exporting the models: ultralytics, torch, transformers, onnx, onnxruntime, opencv, matplotlib
- laptop tools in the repo: `./gradlew :math:priorSweep` (scores the app against the walker's real turns) and `./gradlew :math:noPath` (checks the STOP number on a recorded walk)

## Models (nothing trained, all off the shelf)

- **obstacle detector**: YOLO-World, the yolov8s-worldv2 version
  - picture size 640×640
  - 35 prompts mapped to 11 obstacle kinds
  - exported to ONNX, 48 MB, made by `model/tools/export_yolo_world.py`
  - not in git, everyone runs the export once
- **ground and walls**: SegFormer-B0 trained on ADE20K
  - picture size 256×256, gives back a 64×64 grid of labels
  - 150 labels grouped into 9 kinds
  - picked ADE20K over Cityscapes because it has indoor stuff like floor, stairs and columns (Cityscapes is outdoor, from a car's point of view)
  - exported to ONNX, 14.5 MB, made by `model/tools/export_segformer.py`
  - runs every third frame

## Speed on the Pixel 8

All debug build, ONNX Runtime 1.30.0 on CPU, replaying the recorded walk unless it says live.

- detector at 640 alone, cool phone: 553 ms per frame (median), about 1.8 frames a second
- detector at 320: 112 ms, about 9 frames a second, but it misses too much
- detector live at 640 in daylight: 411 ms (older runtime 1.20.0)
- second model (SegFormer) at 256: 196 ms median
- detector with the second model running too: 661 ms median
- the phone heats up about 5 degrees per replay and the detector slows down, five replays in a row went 553, 605, 645, 751 ms
- playback on the phone runs about 4 times slower than real time, so the 4 minute walk takes around 15 minutes
- memory while replaying: 335 MB at 320, 451 MB at 640
- app download size: about 161 MB of actual stuff inside it with both models

## Main settings (the numbers the math uses right now)

- turn prior: 30 degrees
- directions checked: 60 left to 60 right, 1 degree apart
- space each obstacle needs: at least 0.5 m (a walker is about that wide and sways)
- how far ahead in time an obstacle starts to matter: 1.5 to 3 s depending on the object type
- how ok it is to bump into it: 0 to 0.5 depending on the object type
- detector score counted as certain from 0.5
- default walking speed 1.4 m/s, a stopped walker counts as 0.3 m/s
- step length for the speed estimate: 0.7 m
- ground cost per meter: pavement 0, grass 0.1 bits, dirt 0.15, road 0.6
- wall and building spots: one every 0.5 m
- projected path: 6 steps of 0.5 m, so 3 m ahead
- ribbon: a fifth of the screen wide at your feet, 20 px at the top, never less than 0.7 solid in the app, solid at 1 bit of information, red at 3 bits of surprise
- STOP: looks twice as far ahead, red from 3 bits, STOP at 4 bits held for 1 s
- overlay over other apps: ribbon at 0.5 normally, 0.8 when things are bad (0.8 is the max Android lets you do and still tap through)
- all the screen numbers are in `output/.../GuidanceDisplayTuning.kt` and marked TUNABLE, the table is in the README

## Results we already have (in case the materials section needs context)

- on the replay, the app turned where the walker turned: the cone at 2.65 bits and 15 degrees left, the bike rack at 2.1 bits and 34 degrees right
- with the ground and walls model on, alerts over 0.3 bits went from 68 frames to 203, the biggest from 4.6 to 12.15 bits, mostly from walls and stairs near the start
- the ground model gets some stuff wrong: stairs as pavement, an ivy wall as grass, red brick as road (road was the top label for the ground right in front on 149 of 861 frames)
- walking speed from steps on the walk: 1.24 m/s median
- the "how much did the view change the app's mind" number: median 0.02 bits, max 2.73, zero on open ground
