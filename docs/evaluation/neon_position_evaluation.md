# A position for the glasses: dense SLAM and odometry

The Neon glasses give the pipeline an orientation from their IMU and no position. Without one the
scene's 0.25 m grid moves with the walker, so a post changes cell every quarter meter walked, and
nothing keeps an obstacle's history while the walker passes it. Open3D's dense SLAM was the
recommended way to get a position. This page measures it, and frame-to-frame odometry beside it, on
a recorded walk, and says why the pipeline still runs without a position by default.

Both are in the pipeline now behind `--position {absent,odometry,dense_slam}`. `absent` is the default.

## The walk and the reference

The walk is `2026-10-14_17-18-02`, 44 s in the tiered classroom: up to a chair, back past it, up the
stairs and down, then at the whiteboard wall. 1227 scene frames at 30 a second. Every IMU reading
was usable.

Nothing measured where the glasses really were, so a reference was built. COLMAP, a
structure-from-motion program, reconstructed the camera path from the scene video at 5 frames a
second, straightened with the glasses' own calibration. It matched the first 18.4 s (93 frames,
the chair and the start of the stairs) into one piece. Every score below covers that stretch,
where the reference path is about 8 m long.

COLMAP's path has no scale of its own. Its scale was taken from the depth model: the model's depth
at each of COLMAP's 3D points over COLMAP's depth for the same point, median over 32,368
observations, 0.457 m per COLMAP unit, with the middle half between 0.440 and 0.479. That scale
comes from neither tracker, so it can't flatter one. Each tracker's path is laid over the
reference by a rotation and a shift only, no scaling.

The scores:

- **Position error**: how far the tracker's camera is from the reference camera at the same moment.
- **Drift over 0.5 s or 3 s**: how far the tracker's movement over that span differs from the
  reference's. 0.5 s was N's window when the evaluation was run, and 3 s is about how long the planner holds on
  to its previous plan. A quarter of a meter is one grid cell.
- **Fitness**: the share of a frame's pixels the tracker matched. Open3D reports it.

## What the depth gives a tracker to work with

Both trackers match depth, and the depth comes from a model guessing it from one picture. The
model's depth for the same 3D point changes between frames 0.2 s apart by 1.3 % at the median and
5.2 % for the worst tenth, measured at COLMAP's points. In meters that's a median of 0.027 m and
0.138 m for the worst tenth. A quarter of the point pairs move more than 0.07 m, which is Open3D's
default limit for "the same surface".

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
504 px for that offline run, and the pipeline plans 2.5 to 3.5 frames a second at 336 and 280 px.
That is 9 to 12 times short of 30.

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

**A position only partly reaches the planner.** It fixes the cells to the world, so the planner gets different cells. But N was measured on about 0 % of obstacles in every run,
with a position or without. The position keeps an obstacle in the same cell, but N needs 3 samples
inside its 0.5 s window, and at 2.4 to 3.5 planned frames a second the window holds 1 or 2. Obstacles
do get velocities, 43 to 66 % of them, which only the planner's motion prediction reads, and that is
off. Widening the window to hold 3 samples is a choice about our input, separate from this, and the
update below does it.

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
- Each changes the cells the planner gets, and nothing here shows that the plans get better.
- `absent` keeps the full frame rate and works on any laptop that can run the depth model.

## How the current state came about

The position had to come as the walk happens, from the glasses or from a recording played as a
stream. Work done afterwards on a finished recording didn't count, because the walker needs the
arrow at the time.

1. **Dense SLAM first, since it was the recommended route.** It tracked the walk to 0.18 m with depth
   for every frame at 30 a second, and lost it at every lower rate. The depth model makes 2.5 to 3.5
   frames a second on this laptop, so as the walk happens dense SLAM is about 1 m off. The same code
   on a sensor's depth (Open3D's Lounge sequence) tracked fine, so the code wasn't the problem.
2. **Helping dense SLAM at low rates didn't work.** A looser match tolerance left the camera standing
   still more. Starting each frame from the IMU's turn and the last measured speed made every rate
   worse and ran away at 30 frames a second.
3. **Odometry, the lighter relative.** Matching each frame to the one before, with no map, at first
   lost a third of the distance walked. Open3D's 0.07 m match limit was the cause, since the walker
   covers 0.2 to 0.3 m between planned frames. A 0.3 m limit brought it to 0.20 to 0.33 m as the walk
   plays, about one grid cell.
4. **A 30 frames a second feature tracker, to get around the depth rate.** It followed image
   features on every frame and took metric depth from the model every 0.5 s. Its best result offline
   was 0.30 m, with depth arriving 0.5 s late and the track smoothed over 1 s. With depth every 0.6 s,
   arriving 0.9 s late as in the pipeline, it was 0.78 m off and had a position on 52 % of frames, so
   it was dropped. Along the way, the glasses' IMU orientation turned out 16 % short of what the images
   show on turns, and about 1 degree off over 0.1 s at the median. That's too loose to hold the rotation fixed and
   solve only for the position. Solving rotation from the images is what any further tracker would need.
5. **A position on its own doesn't reach the planner.** Fixing the grid to the world was the point,
   so that N could be measured. N also needs 3 samples in its 0.5 s window, and the pipeline plans
   2.4 to 3.5 frames a second, so N stayed at its floor with a position and without.
6. **Dense SLAM hung.** About one run in three or four, inside Open3D, on both builds. Running it in a
   process that can be killed and replaced keeps the arrow going when it does.
7. **The walking speed stayed a constant.** The planner assumes 1.4 m/s. Nothing in the mapping uses
   that figure, but the field, the alarm, the contact term, the previous plan and the arrow's
   lookahead all do. On the walk the reference covers, the pace ranged from standing to about 0.85 m/s.
   Odometry's speed over 1 to 3 s came within about 0.15 m/s of the reference's at the pipeline's
   rate, where the constant was about 1 m/s off. The constant stays as an average for a proof of
   concept. Odometry's speed is the obvious replacement.

So `absent` is the default. `odometry` and `dense_slam` stay in the pipeline behind `--position`, so
the comparison above can be run on any recording.

## Update, 2026-11-05: N's window goes to 1 s

N's window was 0.5 s with no recorded reason, and that is what held N at its floor. It is now 1 s, the
window the professor's slides use for car following (College 5 PDF page 65). Position stays off by
default. The same walk at 336 px, as it plays, on an idle GPU:

| Setting | Runs | Planned a second | Arrival to plan, median | Arrival to plan, 95th percentile | Cells with a measured N | N, median | Arrow at its limit | Alarm frames |
|---|---|---|---|---|---|---|---|---|
| Off, 0.5 s | 2 | 2.80 to 2.96 | 413 to 480 ms | 562 to 644 ms | 0 % | | 8.7 to 9.8 % | 55 to 57 % |
| Off, 1 s | 2 | 2.80 to 2.93 | 447 to 482 ms | 564 to 682 ms | 46 to 49 % | 0.04 m | 6.8 to 11.6 % | 54 to 56 % |
| Odometry, 0.5 s | 3 | 2.27 to 2.61 | 520 to 606 ms | 928 to 1167 ms | about 0 % | | 5.1 to 5.6 % | 51 to 55 % |
| Odometry, 1 s | 3 | 2.31 to 2.85 | 435 to 605 ms | 524 to 990 ms | 42 to 50 % | 0.13 m | 21.9 to 37.6 % | 51 to 58 % |

The delays overlap between the two windows, so the wider window has no cost we can measure. The same
setting moved by up to about 170 ms at the median between runs, so the single-run delays higher up this
page are looser than they look.

With position off the grid moves with the walker, so N partly compares different spots. With odometry
each cell stays on one spot, N comes out about three times larger, and the arrow sits at its limit far
more often. Whether that is the gap really closing or tracker error isn't known, and nothing here scores
whether a plan is better, which is why odometry stays off by default.

## What could still be tried

None of this was built or measured unless it says so.

**More depth frames a second.** This is what dense SLAM is short of.
- A smaller depth model. Depth Anything 3's small model has about 25 million parameters against about
  330 million for the large one. Scaling from the large model's 286 ms at 280 px puts it near 22 ms,
  or 11 to 15 ms at half precision. That's an estimate, not a measurement. It also gives relative depth,
  not meters, so the scale would have to come from somewhere else: the floor height, or a large-model
  frame now and then.
- The large model at half precision, or compiled with TensorRT. Lowering the resolution alone can't
  get there: by the measured trend it would take about 110 x 80 px to reach 20 ms.
- Glasses or a phone with a depth sensor. A depth sensor gives depth at its own frame rate, and that
  is what dense SLAM tracked well on.

**A proper visual-inertial tracker.** The dropped feature tracker solved each frame on its own. Real
trackers refine each landmark over many frames and adjust recent poses together, which is several
days of work at least.
- Existing ones (ORB-SLAM3, OpenVINS, Basalt) are C++ and mostly built for Linux and ROS. Learned
  ones (DROID-SLAM, DPVO) need GPU time the depth model is already using.
- They want the raw gyroscope and accelerometer. The glasses' native recordings have both, but the
  live stream and the laptop's captures keep only the fused orientation.

**Dense SLAM with color.** Open3D's color-and-depth tracking stalled on its first frame with our data
every time, so it was never compared. The frame logs carry no color, so it would need the cached
depth-and-color frames or the live route.

**Depth Anything 3's own poses.** Some of its checkpoints estimate camera poses from several frames
at once. The metric one in use doesn't. The metric multi-view model is 6.8 GB and doesn't fit the
4 GB card.

**Getting a position to the planner.** Whatever tracker is used, N needs its window widened to about
1.5 s, or its samples taken from a tracker that runs faster than the planner. That's our input, not
the professor's formula.

**Measured walking speed.** Odometry's speed replacing the 1.4 m/s constant needs no new tracker. It
would change the field, the alarm and the contact term, so it would want its own evaluation first.

**Known weak spots in odometry.**
- Facing a single flat wall, a step along the wall isn't pinned down. On a synthetic room it reported
  a 0.41 m error at a fit of 0.69, which no current check catches. Refusing a match whose surfaces all
  face one way would.
- Its slowest twentieth of delays at 336 px went past a second on the walk, because the pipeline waits
  for each position before planning. Planning without waiting, and using the position a frame late,
  was not tried.

**A wider check.** The reference covers 18 s of one walk. COLMAP on the other recorded walks, such as
the hallway, would show whether these numbers hold elsewhere.

## Rerunning it

The capture is made from the recording with `examples/recording_to_capture.py`. Each run, from `server/`:

```
.venv/Scripts/python -m nav --source neon_live --neon-replay frame_logs/captures/2026-10-14_17-18-02 --position odometry --process-resolution 336 --sink web --record-to frame_logs/position_final/attention_odometry_336
```

with `--position` and `--process-resolution` changed per row. The frames, positions and delays are
in each run's frame log and `timing.jsonl`. The COLMAP reference and the scoring were run outside the
pipeline with pycolmap 4.2.1, installed with the depth model, and aren't part of the repository.
