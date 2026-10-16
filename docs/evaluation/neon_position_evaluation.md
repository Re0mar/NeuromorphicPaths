# A position for the glasses: dense SLAM and odometry

The Neon glasses give the pipeline an orientation from their IMU and no position. Without one the
scene's 0.25 m grid moves with the walker, so a post changes cell every quarter meter walked, and
nothing keeps an obstacle's history while the walker passes it. Open3D's dense SLAM was the
recommended way to get a position. This page measures it, and frame-to-frame odometry beside it, on
a recorded walk, and says why the pipeline still runs without a position by default.

Both are in the pipeline now behind `--position {absent,odometry,dense_slam}`. `absent` is the default.

## The walk and the reference

The walk is `2026-10-14_17-18-02`, 44 s in the tiered classroom: up to a chair, back past it, up the
stairs and down, then at the whiteboard wall. 1227 scene frames at 30 a second, every IMU reading
usable.

Nothing measured where the glasses really were, so a reference was built. COLMAP, a
structure-from-motion program, reconstructed the camera path from the scene video at 5 frames a
second, straightened with the glasses' own calibration. It matched the first 18.4 s (93 frames, the
chair and the start of the stairs) into one piece. Every score below covers that stretch, where the
reference path is about 8 m long.

COLMAP's path has no scale of its own. Its scale was taken from the depth model: the model's depth at
each of COLMAP's 3D points over COLMAP's depth for the same point, median over 32,368 observations,
0.457 m per COLMAP unit, with the middle half between 0.440 and 0.479. That scale comes from neither
tracker, so it can't flatter one. Each tracker's path is laid over the reference by a rotation and a
shift only, no scaling.

The scores:

- **Position error**: how far the tracker's camera is from the reference camera at the same moment.
- **Drift over 0.5 s or 3 s**: how far the tracker's movement over that span differs from the
  reference's. 0.5 s is the window N is measured over, and 3 s is about how long the planner holds on
  to its previous plan. A quarter of a meter is one grid cell.
- **Fitness**: the share of a frame's pixels the tracker matched. Open3D reports it.

## What the depth gives a tracker to work with

Both trackers match depth, and the depth comes from a model guessing it from one picture. The model's
depth for the same 3D point changes between frames 0.2 s apart by 1.3 % at the median and 5.2 % for
the worst tenth, measured at COLMAP's points. In meters that's a median of 0.027 m and 0.138 m for the
worst tenth. A quarter of the point pairs move more than 0.07 m, which is Open3D's default limit for
"the same surface".

## Dense SLAM needs depth at 30 frames a second

Dense SLAM matches each frame against a volume fused from every frame before it. Run offline on the
model's depth for every frame of the walk, at 504 px, it tracked well at 30 frames a second and lost
the walk at any lower rate:

| Depth frames a second | Position error, median | Drift 0.5 s, worst tenth | Drift 3 s, worst tenth | Fitness, median |
|---|---|---|---|---|
| 30 | 0.18 m | 0.23 m | 0.43 m | 0.74 |
| 10 | 0.98 m | 0.40 m | 1.92 m | 0.55 |
| 5 | 0.87 m | 0.43 m | 1.80 m | 0.42 |
| 1.7 | 1.00 m | 0.44 m | 1.92 m | 0.28 |

The same code on Open3D's own Lounge sequence, recorded by a depth sensor at 30 frames a second, gave
a median error of 0.28 m, 0.30 m drift over 3 s and a fitness of 0.69, with no frame under 0.3. So the
code works, and at 30 frames a second it works on the glasses' depth too.

Depth for every frame isn't available as the walk happens. The depth model took 1.4 s a frame at
504 px for that offline run, and the pipeline plans 2.5 to 3.5 frames a second at 336 and 280 px. That
is 9 to 12 times short of 30.

Two things were tried to make it track at the lower rates, and neither helped. A looser match
tolerance, 0.15 or 0.3 m, kept more pixels matched but left the camera standing still more. Starting
each frame from the last measured speed and the IMU's turn ran away at 30 frames a second and made
every rate worse.

## Odometry tracks at the rate the pipeline runs

Odometry matches each frame against the one before it, keeps no map, and runs on the CPU. At 0.6 s
between frames a walker covers 0.2 to 0.3 m, past Open3D's 0.07 m limit, so the surfaces ahead went
unmatched and the path came out short. A 0.3 m limit fixed most of that. On the same offline depth:

| Depth frames a second | Limit | Position error, median | Drift 3 s, worst tenth |
|---|---|---|---|
| 1.7 | 0.07 m | 0.58 m | 1.36 m |
| 1.7 | 0.3 m | 0.42 m | 1.04 m |
| 3.3 | 0.3 m | 0.23 m | 0.55 m |
| 5 | 0.3 m | 0.17 m | 0.54 m |

## In the pipeline, as the walk plays

The walk was played through the glasses' route at its own pace, `--neon-replay` on a capture made
from the recording, once per setting. A slower setting plans fewer frames in the same 41 s rather
than taking longer. The laptop was otherwise idle, at 15 % CPU, on the Quadro T2000.

| Setting | Planned frames a second | Arrival to plan, median / slowest twentieth | Frames with a position | Position error, median | Drift 3 s, worst tenth |
|---|---|---|---|---|---|
| absent, 336 px | 2.99 | 411 / 471 ms | none | | |
| odometry, 336 px | 2.42 | 491 / 1315 ms | 95 % | 0.20 m | 0.32 m |
| dense SLAM, 336 px | 2.47 | 534 / 919 ms | 70 % | 1.08 m | 2.21 m |
| absent, 280 px | 3.45 | 424 / 544 ms | none | | |
| odometry, 280 px | 3.33 | 449 / 577 ms | 97 % | 0.33 m | 0.51 m |
| dense SLAM, 280 px | 3.28 | 425 / 550 ms | 69 % | 1.00 m | 1.55 m |

Odometry is the better of the two at every rate the laptop reaches. At 336 px it costs a fifth of the
planned frames and stretches the slowest delays past a second. At 280 px it costs 4 %.

**A position doesn't reach the planner yet.** N was measured on about 0 % of obstacles in every run,
with a position or without. The position keeps an obstacle in the same cell, but N needs 3 samples
inside its 0.5 s window, and at 2.4 to 3.5 planned frames a second the window holds 1 or 2. Obstacles
do get velocities, 43 to 66 % of them, which only the planner's motion prediction reads, and that is
off. So with either tracker on, the planner gets the same inputs as without one. Widening the window
to hold 3 samples is a choice about our input, separate from this.

## Dense SLAM hangs

Open3D's dense SLAM hung inside its own calls on about one run in three or four, on both the plain
build (CPU) and the CUDA build. On the GPU a hung call also held up the depth model, which shares the
card, and the pipeline froze. So `--position dense_slam` runs it in a separate process with a 2 s
limit per frame. A call past the limit kills the process and starts a new one, and the frames go on
without a position until it's ready. The new process starts its world at the camera, so its
positions are moved by the last position the old one reported, and the scene's world-fixed cells
stay on the same places. After three restarts dense SLAM is given up on for the run. In the six runs
above nothing hung. In the three runs before them, one did, and the pipeline carried on.

One dense SLAM volume holds about 2 GB of the T2000's 4 GB, so `dense_slam` is refused on a run that
can switch to a recording, where each side would hold its own. The plain Windows build of Open3D is
CPU-only, and dense SLAM needs the CUDA build. Installing it is in `server/README.md`.

## Why the pipeline runs without a position

- Dense SLAM works on the glasses' depth when it gets depth for every frame at 30 a second. The depth
  model on this laptop makes 2.5 to 3.5 a second, and at that rate dense SLAM's error is about 1 m,
  four grid cells.
- Odometry tracks to 0.20 to 0.33 m at the pipeline's rate. That's one cell, at a cost of 4 to 19 % of
  the planned frames.
- Neither changes what the planner sees today, because N's window holds too few frames.
- `absent` keeps the full frame rate and works on any laptop that can run the depth model.

## Rerunning it

The capture is made from the recording with `examples/recording_to_capture.py`. Each run, from `server/`:

```
.venv/Scripts/python -m nav --source neon_live --neon-replay frame_logs/captures/2026-10-14_17-18-02 --position odometry --process-resolution 336 --sink web --record-to frame_logs/position_final/attention_odometry_336
```

with `--position` and `--process-resolution` changed per row. The frames, positions and delays are
in each run's frame log and `timing.jsonl`. The COLMAP reference and the scoring were run outside the
pipeline with pycolmap 4.2.1, installed with the depth model, and aren't part of the repository.
