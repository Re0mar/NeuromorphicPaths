# The planner's arrow against the walker's real turns

Does the arrow point where people actually go? Until now the arrow was judged only against itself:
how often it sat at the sidestep limit and how much it jumped between frames. This measures it
against what a walker did on three recorded walks.

How to rerun all of it is at the end, and in [the guide](../guides/score_the_arrow_against_turns.md).

## How the walker's turns are found

**Where the walker went, not where the phone pointed.** The phone is held by hand and points
wherever the hand does. On the classroom walk its own yaw swept more than 500 degrees in 93 s. So
the walker's heading is their direction of travel. That comes from the phone's tracked position
(ARCore's pose, recorded with every frame), projected onto the floor, put on a 0.1 s grid, and
smoothed over 1.1 s centered on each moment. Within half a second of the start or end of a stretch
of track there's no full 1.1 s to smooth over, so the heading there is left unknown. Below 0.3 m/s
the walker is standing, and the heading is left unknown rather than guessed.

**Where the tracking can't be trusted, the track breaks.** ARCore sometimes relocalizes, and the
recorded position leaps in a single frame. One walk jumped 17.48 m in 0.033 s. Any step faster than
3 m/s, any hole in the frames longer than 0.5 s, and any frame with no position breaks the track,
and nothing is smoothed across a break. The breaks are counted below.

**A turn is a heading change of more than 11 degrees within 2 s.** Overlapping 2 s windows that
turn the same way count as one turn. Each turn is then cut down to its own swing: from the lowest
point before it to the furthest it goes in its direction. No turn may start before the previous one
ended. Where a turn starts, its onset, is found by taking the turn's middle half, drawing a line
through it at its own rate, and seeing where that line meets the heading the walker had before. For
a turn at a steady rate this is exact. In the tests it lands within 0.1 s of the true start for a
60 degree turn over 1.5 s, and within 0.15 s for a 25 degree turn over 2 s. A turn that speeds up
gets an onset slightly early, so its lead time reads short. One that slows down reads long.

**The 11 degrees is measured, not chosen.** It's how much a walker's heading changes over 2 s when
they mean to go straight. That was measured on a separate walk, recorded on the phone for the
purpose: 65 m in one straight line, 62.4 s of straight walking. There, the change over 2 s had a
median of 1.7 degrees and a 95th percentile of 6.9. Its 99th percentile, 10.39 degrees, rounded up,
is the threshold. Anything less is ordinary wobble. The three scored walks couldn't have set it.
They loop around rooms, and between them hold under 17 s of straight walking. To decide what counts
as straight, the measurement skips any stretch near a heading change over 25 degrees in 2 s. The
99th percentile came out at 10.39 to 10.40 degrees for that cut-off anywhere from 15 to 40, so the
cut-off didn't decide the number.

**Each turn is tagged by whether something stood ahead of the walker beforehand.** For the 2 s
before onset, the tag looks at a strip 0.35 m either side of the walker's line, out to 3 m ahead.
A frame counts only if the camera could see that line, from 1 m to 3 m ahead. If a quarter or more
of those frames held an obstacle in the strip, the turn is "obstacle ahead". If no frame could see
the line, the tag is "unknown", never "open".

## How the arrow is scored

The arrow is the planner's own number, `first_heading_radians`, exactly as the server sends it to
the phone. Nothing here recomputes, corrects or filters it.

**The arrow is read the way the walker reads it.** It's drawn over the camera picture, so it means
"this many degrees from where the phone points". The direction it sends the walker is the phone's
direction plus the arrow. That direction is what gets compared with where the walker went.

- **Agreement.** Over the second before a turn's onset, the average of that direction says which
  side the arrow was pointing. It's an average of directions, so +179 and -179 degrees average to
  straight back, not straight on. Within 5 degrees of straight on counts as "carry on". The turn
  agrees when the arrow pointed to the side the walker then turned.
- **Lead time.** How long the arrow had held that side, unbroken, when the turn started. It counts
  back at most 5 s, and never past the end of the previous turn. A lead that runs into either
  limit, or back to a moment where the walker's heading isn't known, is reported as "at limit",
  meaning at least that long.
- **Sidesteps while walking straight.** Stretches where the arrow pointed more than 10 degrees to
  one side for at least 0.5 s, with no turn during them or within 2 s after. The rate is per minute
  of walking that had an arrow to read.
- **Lagged correlation.** The arrow now set against how fast the walker turns a moment later, for
  every delay from 0 to 3 s. The delay with the strongest match is a second estimate of how far
  ahead the arrow runs, one that needs no turn threshold. A segment with fewer than 100 pairs of
  samples gets no estimate.

## Results

These are today's planner, with the contact term and its kinetic weight of 6.5. Three cold replays
of each walk gave identical numbers, because the scene's floor fit is seeded.

| Walk | Arrow shown to the walker | Scored walking | Turns | Obstacle ahead | Agreed | Wrong side | Carry on | Median lead | Unknown tag: agreed of |
|---|---|---|---|---|---|---|---|---|---|
| `pixel_walk_3` | none | 91.2 s known heading in 3 segments (133 s) | 18 | 17 | 10 | 7 | 0 | 0.27 s, 6 known, 3 at limit | 0 of 1 |
| `wifi_run_2` | the 1 s lookahead arrow, the one scored here | 33.3 s known heading in 1 segment (68.7 s) | 6 | 4 | 4 | 0 | 0 | 0.40 s, 3 known, 1 at limit | 1 of 2 |
| `pixel_display_run` | the older three-value arrow | 59.0 s known heading in 1 segment (92.7 s) | 9 | 4 | 3 | 0 | 1 | 0.05 s, 2 known, 2 at limit | 3 of 5 |

No turn on any walk was tagged "open ahead". These are walks around rooms, where a wall or desk is
nearly always within 3 m of the walker's line. So the split between turns made for an obstacle and
turns made for the route can't be drawn from these walks.

| Walk | Sidesteps while straight | Lagged correlation, strongest delay and match | Phone pointing off the walking direction |
|---|---|---|---|
| `pixel_walk_3` | 9, 5.9 a minute | one per segment: none (97 pairs), 0.5 s at 0.05 (381 pairs), 0.4 s at 0.19 (400) | median -0.9 deg, middle half spans 14.8 deg, over 30 deg on 17.8 % of samples |
| `wifi_run_2` | 12, 21.6 a minute | 2.6 s at 0.18 (303 pairs) | median -2.1 deg, middle half spans 44.6 deg, over 30 deg on 39.3 % |
| `pixel_display_run` | 10, 10.2 a minute | 0.5 s at 0.29 (568 pairs) | median +1.1 deg, middle half spans 18.0 deg, over 30 deg on 19.5 % |

### Before and after the contact term

The same walks scored with the planner's settings from before the contact term: the professor's
kinetic weight of 0.055 and the contact term switched off, which is how the contact term's own
change compared itself. The turns and their tags are the same in both, because they come from the
pose and the scene, never the planner. Only the arrow differs.

| Walk | Planner | Obstacle ahead: agreed, wrong side, carry on | Median lead | Sidesteps a minute | Lagged correlation |
|---|---|---|---|---|---|
| `pixel_walk_3` | before | 12, 5, 0 of 17 | 0.38 s, 10 known, 7 at limit | 5.9 | 0.7 s at 0.16, 0.3 s at 0.28 |
| | today's | 10, 7, 0 of 17 | 0.27 s, 6 known, 3 at limit | 5.9 | 0.5 s at 0.05, 0.4 s at 0.19 |
| `wifi_run_2` | before | 3, 1, 0 of 4 | 0.40 s, 3 known, 1 at limit | 23.4 | 2.5 s at 0.22 |
| | today's | 4, 0, 0 of 4 | 0.40 s, 3 known, 1 at limit | 21.6 | 2.6 s at 0.18 |
| `pixel_display_run` | before | 2, 0, 2 of 4 | 0.05 s, 2 known, 2 at limit | 7.1 | 0.5 s at 0.27 |
| | today's | 3, 0, 1 of 4 | 0.05 s, 2 known, 2 at limit | 10.2 | 0.5 s at 0.29 |

On `pixel_walk_3`, the one walk where nobody saw an arrow, the contact term moved two turns from
agreeing to the wrong side, and both its correlations got weaker. On the two walks where the walker
saw an arrow, it moved one turn each the other way. Two turns is within what a walk this size can
show by chance (see below), so this says the contact term didn't make the arrow a better predictor
of these turns. It doesn't say it made it worse.

### How much the threshold matters

The same walks at 8 and 14 degrees, either side of the measured 11:

| Walk | Threshold | Turns | Obstacle ahead: agreed, wrong side, carry on | Median lead |
|---|---|---|---|---|
| `pixel_walk_3` | 8 | 18 | 10, 7, 0 of 17 | 0.27 s, 6 known |
| | 11 | 18 | 10, 7, 0 of 17 | 0.27 s, 6 known |
| | 14 | 16 | 10, 6, 0 of 16 | 0.30 s, 7 known |
| `wifi_run_2` | 8 | 9 | 5, 2, 0 of 7 | 0.27 s, 4 known |
| | 11 | 6 | 4, 0, 0 of 4 | 0.40 s, 3 known |
| | 14 | 6 | 3, 0, 1 of 4 | 0.80 s, 2 known |
| `pixel_display_run` | 8 | 9 | 3, 0, 1 of 4 | 0.10 s, 3 known |
| | 11 | 9 | 3, 0, 1 of 4 | 0.05 s, 2 known |
| | 14 | 10 | 4, 0, 1 of 5 | 0.10 s, 3 known |

The `pixel_walk_3` figures hardly move. The two shorter walks move by a turn or two, because they
have few turns to move.

## What these numbers can and can't carry

**Few turns.** On `pixel_walk_3`, the one walk where the walker saw no arrow, the arrow pointed the
right way before 10 of 17 obstacle-ahead turns. Pointing at random would average 8.5 of 17, and a
fair coin gives 10 or more out of 17 about 31 % of the time. The settings from before the contact
term got 12 of 17, which a coin matches about 7 % of the time. So at this size neither is evidence
that the arrow predicts where people turn, and neither is evidence against it. One turn moves a
share of 17 by 6 points, and a share of 4 by 25. Every share here should be read with its count.

**Seeing the arrow.** On `wifi_run_2` and `pixel_display_run` the walker had an arrow in front of
them, so agreement there partly measures whether they followed it. On `pixel_display_run` the arrow
they saw was the older one, which could only point straight or fully to one side. The arrow scored
here is today's. Only `pixel_walk_3` measures whether the planner predicts a free choice.

**The phone's own pointing.** The arrow is read relative to the phone, so where the phone leads a
turn, the arrow's direction leads with it. Part of any agreement may be the phone's rather than the
planner's, and these numbers can't separate the two. The phone pointed more than 30 degrees off the
walking direction on 18 to 39 % of samples. The planner plans along the phone's direction rather
than the walker's, and changing that is work on the planner, not on this measurement.

**Lead times.** The arrow's lead before a turn is a fraction of a second where it's known at all:
medians of 0.05 to 0.40 s, over 2 to 6 turns per walk. Some of those are "at limit" because the
walker's heading went unknown before the arrow let go, so they're lower bounds. Even so, the arrow
reads the plan 1 s ahead, and a lead well under a second means the arrow mostly turned with the
walker rather than before them. The lagged correlation agrees: its strongest matches are weak,
0.05 to 0.29, and one segment has too few pairs for any.

**Sidesteps.** On `wifi_run_2` the arrow asked for a sidestep 21.6 times a minute while the walker
went straight, against 5.9 and 10.2 a minute on the others.

**Known headings.** Much of each walk has no known heading, because the walker stood or shuffled:
`wifi_run_2` has a known heading on 33.3 s of its 68.7 s. The track broke at a tracker jump 12
times inside `pixel_walk_3`'s scored segments, 3 times in `pixel_display_run` and twice in
`wifi_run_2`. `pixel_display_run` also broke once at a hole in its frames.

**Frames the scene refused.** The scene plans nothing for a frame where it finds no floor and has
no earlier floor to fall back on, so those frames have no arrow. `wifi_run_2` lost 130 of its 2144
frames this way and `pixel_display_run` 93 of 2675. `pixel_walk_3` lost none. The report lists
each reason with its count.

**The walker's own notes.** `pixel_walk_3` came with notes of two desks and a cabinet in its first
25 s. That stretch is in a 12.8 s segment too short to score, or in a 51 s gap with no frames, so the
notes can't be checked against the scored turns.

**The replay is not the live run.** The live run dropped frames it couldn't keep up with. The replay
plans every frame. The replay also repeats the live loop's call to the planner rather than sharing
its code, and a test pins that call so a change to it is caught.

**Old recordings, new settings.** `pixel_walk_3` and `pixel_display_run` were recorded before the
phone sent its gravity flag, so the decoder supplies it. Every recording predates the seeded floor
fit. `pixel_walk_3` and `wifi_run_2` get `floor_ransac_seed=0` and
`floor_ransac_success_probability=0.99999999` by flag, and `pixel_display_run` gets the same two
from `--scene-defaults`. Those are the defaults, and the probability is Open3D's own. `pixel_walk_3`
predates the floor's camera-height check, so `floor_max_offset_meters=inf` turns that check off, as
its live run had none. `pixel_display_run` has no record of its scene settings, so it runs on
today's defaults plus `floor_max_tilt_degrees=50`, the tilt in the documented live command.

## Situation

- **Code:** commit `b0d5e28`, the contact term merged in, plus a comment-only change. Nav code hash
  `d674dc142065`. The commit that adds this version of the document also groups the report's
  refusal reasons, which changes no score.
- **Planner:** today's defaults, with the contact term, a kinetic weight of 6.5 and the 1 s
  lookahead. The comparison rows set `lateral_kinetic_weight=0.055` and `contact_term_enabled=false`.
- **Replays:** three cold passes per walk, all identical. The comparison and threshold rows used one
  cached pass each, which the cold passes match.
- **Segments:** only segments of 20 s or more are scored. `pixel_walk_3`'s are 39.3, 46.2 and
  47.4 s. `wifi_run_2`'s is 68.7 s. `pixel_display_run`'s is 92.7 s.
- **Threshold:** from `straight_walk_4`, 2048 frames over 68.3 s, recorded on the phone and
  replayed to the laptop.

## Reproduce

From `server/`, with the recordings in `frame_logs/`:

```
python -m nav.evaluation frame_logs/pixel_walk_3 --scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999 --scene-set floor_max_offset_meters=inf
python -m nav.evaluation frame_logs/wifi_run_2 --scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999
python -m nav.evaluation frame_logs/pixel_display_run --scene-defaults --scene-set floor_max_tilt_degrees=50
python -m nav.evaluation --spread-only frame_logs/straight_walk_4
```

Add `--eval-set turn_threshold_degrees=8` or `14` for the threshold rows, and
`--set lateral_kinetic_weight=0.055 --set contact_term_enabled=false` for the comparison rows. Each
walk takes 3 to 7 minutes, three passes included.
