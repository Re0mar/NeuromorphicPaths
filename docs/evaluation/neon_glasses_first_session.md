# The Neon glasses, first session

What the laptop pipeline did with the Pupil Labs Neon glasses on real hardware: how late each
arrow was, how many came per second, and how often the floor was found. All of it comes from one
session on 2026-10-05, in a classroom.

How to rerun every number here is at the end.

## Conditions

These apply to every figure below unless a section says otherwise.

| | |
|---|---|
| Laptop | Quadro T2000 with Max-Q, 4 GB, NVIDIA driver 581.95, on mains power |
| Software | Python 3.12.10, torch 2.14.1+cu126, the DA3METRIC-LARGE depth model, `--process-resolution 336` |
| Network | the school Wi-Fi, with the laptop and the Companion phone both on it. Address given with `--neon-address`, discovery not tried. Time Echo round trip 8 ms on the walk, 6 to 13 ms across the day |
| Clocks | the Neon's clock ran 1274 ms behind the laptop's on the walk, and 1274 to 1302 ms across the day's runs. Drift over a run was not measured, because the walk was stopped before its closing measurement |
| Background load | Zoom, Firefox, Discord and Task Manager took about 3.5 cores early in the session. They were asked to be closed, and whether they were before the walk was not recorded |
| Not recorded | the Companion phone's model and app version. The walk had no timed warm-up, but the pipeline had been run several times earlier in the session |

## How late each arrow was

A frame goes through four stamps on its way to an arrow. Each step between two stamps is called
a share.

```mermaid
flowchart LR
    capture["Captured<br/>on the glasses"]
    arrival["Arrived<br/>on the laptop"]
    depth["Depth ready"]
    plan["Plan done"]

    capture -- "155 ms" --> arrival
    arrival -- "431 ms" --> depth
    depth -- "275 ms" --> plan
```

The numbers on the arrows are medians from the live walk. Medians don't add up, so the median from
capture to plan done is its own figure, 905 ms.

The live walk ran about 5.8 minutes indoors. 570 frames were planned after leaving out the first
10 s, while the network and the GPU warmed up.

| Share | median | 95th percentile | worst |
|---|---|---|---|
| capture to arrival | 155 ms | 714 ms | 1.6 s |
| arrival to depth ready | 431 ms | 776 ms | 1.3 s |
| depth ready to plan done | 275 ms | 753 ms | 1.5 s |
| capture to plan done | 905 ms | 2.0 s | 2.9 s |

Capture to arrival has about ±8 ms of extra uncertainty. The glasses' clock offset is measured
with Python's `time_ns()`, which ticks in 15.6 ms steps on Windows.

The arrow on the phone's browser page looked another 0.5 to 1.5 s behind to the walker. That was
judged by eye, not measured.

## How many arrows per second

1.68 planned frames a second on the walk, with a longest gap of 1.4 s between two plans.

The decoder logged 15 backlog warnings. It warns at most once every 10 s, so that counts warnings,
not how many times it fell behind.

Two standing runs, before the walk, compared the process resolution:

| `--process-resolution` | Planned frames a second | Run length |
|---|---|---|
| 504 | 0.76 | 4.6 minutes |
| 280 | 2.48 | 1.5 minutes |

The depth model alone, timed on one frame at a time, tops out at about 3.5 a second at 280. The
sensor plan assumed about 10 frames a second at reduced resolution. The measured 1.68 is the figure
to plan with on this laptop.

## How often the floor was found

The live walk ran before the depth was converted from the model's 300 pixel focal length to
meters. Without that conversion the model's raw depth at 336 is about 1.9 times too large, so the
floor came out about 2.9 m below the camera, and the floor check refuses anything over 2.2 m.
That walk's figure, `fitted` on 243 of 570 frames (42.6 %), describes that bug, not the pipeline.

The second capture was replayed with the conversion through `--neon-replay`, 432 frames. `fitted`
on 395 (91 %), with the camera a median of 1.56 m above the floor, 1.41 to 1.66 m between the 10th
and 90th percentiles. The other 37 were walls, refused for leaning a median of 82.6 degrees. The
same 432 frames at the old scale fitted 13 (3 %).

## On a replay

`check_planner numbers` on that replay's frame log printed the same output three times running.
The arrow sat at its sidestep limit on 5.1 % of frames, and the alarm was on for 24.1 %.

On the `--neon-replay` run itself, the scene took a median of 125 ms a frame and the planner 131 ms,
with 95th percentiles of 422 and 371 ms.

A raw capture replayed through `--neon-replay` took a median of 86 ms on the laptop from a packet
being fed in to its decoded frame reaching the pipeline, and 137 ms at the 95th percentile.

## Rerunning it

The recordings aren't committed, because they're large. They're shared as folders under
`server/frame_logs/`: the live walk's frame log `neon_walk_1`, and the raw captures
`captures/neon_walk_1` and `captures/neon_walk_2`. From `server/`:

```
# The live walk's latency table, rate, gap and floor counts
.venv/Scripts/python examples/timing_report.py frame_logs/neon_walk_1

# The floor on the second capture, with the depth conversion
.venv/Scripts/python -m nav --source neon_live --neon-replay frame_logs/captures/neon_walk_2 --process-resolution 336 --sink web --record-to frame_logs/neon_walk_2_replay --verbose

# The planner figures on that replay, the same output every run
.venv/Scripts/python -m nav.evaluation.check_planner numbers frame_logs/neon_walk_2_replay
```

The scene and planner medians come from that replay's `--verbose` lines. The floor's refusal
reasons come from running the scene's own floor functions over each frame of the log.

The `neon_walk_1` frame log was recorded before the conversion existed, so its depth is in the
model's raw units. Replaying it with `--source logged` reads that as meters. Use the raw captures
for anything that depends on depth.
