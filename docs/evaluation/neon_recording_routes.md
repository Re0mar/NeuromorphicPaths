# Replaying a Neon recording

A walk the laptop didn't see live can still be replayed from the glasses' own recording. Until now
that went through Pupil Labs' Neon Player depth plugin, whose saved depth maps the pipeline read
directly. Two things about that were unchecked: whether those maps are in meters, and whether a
replay sees the walk the way the live route did. This page answers both, on two walks recorded on
2026-10-08.

How to rerun every number here is at the end.

## The walks

Both were recorded in a classroom on the school network, with the glasses worn and the laptop
starting and stopping each recording through the glasses' real-time interface.

| Walk | Length | Scene frames | What was recorded |
|---|---|---|---|
| A | 214 s | 6355 at 30 a second | The Companion app's own recording, and at the same time the laptop's capture of the live stream |
| B | 27 s | 737 at 30 a second | The Companion app's own recording only |

Walk A exists twice on purpose. The live route replays the laptop's capture, as if the glasses
were connected. The recording route reads the Companion app's recording. With both made from one
walk, any difference between them is the route's, not the walk's.

Walk B is short because Neon Player's plugin runs only on Apple's GPU or the CPU. On this laptop
that's about 38 s a frame, and B alone took 7 hours 43 minutes.

The capture of walk A had 21,403 IMU readings over 207 s, none of them empty, so no other program
was holding the IMU stream. Both recordings had every IMU sample usable.

All runs used the depth model DA3METRIC-LARGE at a processing resolution of 336 px, except the
plugin, which runs at its own 504.

## Are the plugin's depth maps in meters?

**Yes.** The plugin converts the model's output to meters once, before saving.

Depth Anything 3's metric model answers as if every camera had a focal length of 300 px. That's how
strongly a camera zooms, measured in pixels. To get meters you multiply by the real camera's focal
at the resolution the model ran at, and divide by 300. The plugin's source at commit `e6202a9`
does this:
- it reads the recording's focal, 890.754 px, the mean of its two focals
- it scales that to the model's 504 px width: 890.754 × 504 / 1600 = 280.6 px
- it multiplies by 280.6 / 300 = 0.9353

So a saved map should be 0.9353 times what the model returns.

That is what it holds. The same model was run on the same unstraightened frames at 504 px, and its
raw output was resized the way the plugin resizes. The saved maps were then divided by it, pixel by
pixel, on 10 frames spread through walk B:

| | Saved map over raw output, median per pixel |
|---|---|
| Measured, 10 frames | 0.9343 (0.9319 to 0.9375) |
| One conversion, as the source says | 0.9353 |
| No conversion | 1.0000 |
| Converted twice | 0.8748 |

The measurement sits within 0.1 percent of one conversion, and far from the other two.

The saved maps are 737 frames of 300 by 400 float32, one per scene frame, at a quarter of the scene
size. Every value is finite, running from 0.41 to 19.33 m with a median of 1.61 m.

A camera-height check alone couldn't have told these apart. Because the plugin runs at 504 px, a
missing conversion would make depth only 1 / 0.9353 = 1.069 times too large. A camera 1.41 m up
would then read 1.51 m, which is inside the normal spread of a walk. That's why the per-pixel
comparison was done.

## The camera's height on each route

`check_planner floor` reads a run's floor plane on every frame and reports how high the camera sat
above it. A frame that kept the previous frame's floor, or had no IMU reading near it, is counted
but left out of the median, because it carries an earlier height or no gravity to measure against.

| Run | Frames | Floor fitted | Median camera height | 10th to 90th percentile |
|---|---|---|---|---|
| Walk A, live route on the capture | 432 | 404 | 1.36 m | 0.68 to 1.57 m |
| Walk A, recording route | 424 | 392 | 1.38 m | 0.67 to 1.58 m |
| Walk B, recording route | 50 | 48 | 1.41 m | 1.04 to 1.51 m |
| Walk B, plugin's saved maps | 737 | 713 | 1.47 m | 1.28 to 1.59 m |

Every row printed the same text in separate processes. The plugin's row was also run with the floor
check's height limit lifted from 2.2 m to 10 m, and nothing changed. No frame had been refused for a
floor too far down.

**The live route against the recording route, on walk A:** 0.02 m apart. A recording replays the way
the live route saw the walk.

**The plugin against the recording route, on walk B's same 47 frames:**
- The plugin puts the camera a median 0.079 m higher, a ratio of 1.054.
- Between the 10th and 90th percentiles, the per-frame difference runs from +0.023 to +0.202 m.

The units are settled above, so this is the straightening. The plugin estimates depth on the picture
as the lens bends it, then places each pixel with the unbent camera's numbers. The floor sits at the
bottom of the picture, where the bend is strongest, so the plane fitted to it comes out shifted.

## What differs besides the route

- **The pose.** On walk A both routes pose each frame from the IMU reading nearest its capture, or
  leave it without one. The plugin route carries the last reading forward instead. 12 of its frames,
  before the first IMU reading, had no gravity to fit against.
- **Which frames are planned.** The live replay plans the newest frame at the capture's own pace.
  The recording route takes 2 frames per second of recording. The plugin route has a map for every
  frame. That's why medians are compared, not frames, except where the same frames are named.
- **Desk tops.** Walk A went between classroom desks. On both of its routes, about a third of the
  fitted floors are 0.6 to 0.9 m below the camera:

  | Band | Live | Recording |
  |---|---|---|
  | 0.6 to 0.9 m | 101 | 81 |
  | 1.4 to 1.5 m | 92 | 87 |
  | 1.5 to 1.6 m | 64 | 70 |

  Those are desk tops. A desk top is level and well inside the floor check's 0.3 to 2.2 m, so it
  passes for a floor. That pulls walk A's medians down from the true camera height of about 1.45 m.
  It happens on both routes equally, so it doesn't affect the comparison, but it's a defect in the
  floor fit.

## The recording route's speed

On walk A the recording route planned 424 frames in 5 minutes 20 seconds, about 1.3 a second. Per
kept frame:
- decoding the scene video takes a median of 10.2 ms
- straightening it takes 4.9 ms
- the depth model takes 0.48 s

The 10.2 ms is per kept frame, and it includes the frames the decoder steps through in between. The
video library can't jump straight to a frame, so a long recording costs decoding its whole length,
even at 2 frames a second. Only the conversion to color and everything after it are limited to the
kept frames.

## Why the plugin route is retired

The plugin's maps are in meters, but they were estimated on the bent picture, so its floor sits
0.08 m off the straightened routes on the same frames. The floor is at the bottom of the picture,
where the lens bends it most, and the sides are bent the same way, where an obstacle beside the
walker is. The recording route straightens the picture first, the same way the
live route does, and replays a walk to within 0.02 m of what the live route saw. So a recording no
longer needs Neon Player, and the plugin route is removed.

## Rerunning it

The recordings aren't committed, because they're large. They're shared as folders under
`server/frame_logs/`:
- `recordings/walk_2026_10_08_a` and `recordings/walk_2026_10_08_b`, the Companion app's exports
- `captures/walk_2026_10_08_a`, the laptop's capture of walk A

The plugin row needs Neon Player 6.0.14 with the depth plugin at commit `e6202a9` of
`github.com/pupil-labs/npp-depth-estimation`. Its maps go in walk B's `.neon_player/cache/`. The
`neon_plugin` source that read them is removed in the same change that adds this page. To rerun that
row, check out the commit before the removal. The per-pixel check below needs only the maps, and it
still runs.

From `server/`, with a venv holding the `glasses` extra and CUDA torch:

```
# Walk A, the live route on the capture
.venv/Scripts/python -m nav --source neon_live --neon-replay frame_logs/captures/walk_2026_10_08_a --process-resolution 336 --sink web --record-to frame_logs/walk_2026_10_08_a_live --verbose

# Walk A and walk B, the recording route
.venv/Scripts/python -m nav --source neon_recording --recording-dir frame_logs/recordings/walk_2026_10_08_a --process-resolution 336 --sink web --record-to frame_logs/walk_2026_10_08_a_recording --verbose
.venv/Scripts/python -m nav --source neon_recording --recording-dir frame_logs/recordings/walk_2026_10_08_b --process-resolution 336 --sink web --record-to frame_logs/walk_2026_10_08_b_recording --verbose

# Walk B, the plugin's maps, before the plugin route was removed
.venv/Scripts/python -m nav --source neon_plugin --recording-dir frame_logs/recordings/walk_2026_10_08_b --sink web --record-to frame_logs/walk_2026_10_08_b_plugin --verbose

# The camera heights, and the plugin's with the height limit lifted
.venv/Scripts/python -m nav.evaluation.check_planner floor frame_logs/walk_2026_10_08_a_live frame_logs/walk_2026_10_08_a_recording frame_logs/walk_2026_10_08_b_recording frame_logs/walk_2026_10_08_b_plugin
.venv/Scripts/python -m nav.evaluation.check_planner floor frame_logs/walk_2026_10_08_b_plugin --scene-set floor_max_offset_meters=10

# The plugin's saved maps against the model's raw output, per pixel. Needs only the saved maps, not the removed source
.venv/Scripts/python examples/compare_plugin_cache.py frame_logs/recordings/walk_2026_10_08_b
```

The same-frame comparison and the decode timing were measured with short scripts kept outside this
repository. Their method is in their sections above.
