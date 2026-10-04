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
smoothed over 1.1 s centered on each moment. Below 0.3 m/s the walker is standing, and the heading
is left unknown rather than guessed.

**Where the tracking can't be trusted, the track breaks.** ARCore sometimes relocalizes, and the
recorded position leaps in a single frame. One walk jumped 17.48 m in 0.033 s. Any step faster than
3 m/s, any hole in the frames longer than 0.5 s, and any frame with no position breaks the track,
and nothing is smoothed across a break. The breaks are counted below.

**A turn is a heading change of more than 11 degrees within 2 s.** Overlapping 2 s windows that
turn the same way count as one turn. Where a turn starts, its onset, is found by taking the turn's
middle half, drawing a line through it at its own rate, and seeing where that line meets the
heading the walker had before. For a turn at a steady rate this is exact. In the tests it lands
within 0.1 s of the true start for a 60 degree turn over 1.5 s, and within 0.15 s for a 25 degree
turn over 2 s. A turn that speeds up gets an onset slightly early, so its lead time reads short.
One that slows down reads long.

**The 11 degrees is measured, not chosen.** It's how much a walker's heading changes over 2 s when
they mean to go straight. That was measured on a separate walk, recorded on the phone for the
purpose: 65 m in one straight line, 62.5 s of straight walking. There, the change over 2 s had a
median of 1.7 degrees and a 95th percentile of 7.0. Its 99th percentile, 10.39 degrees, rounded up,
is the threshold. Anything less is ordinary wobble. The three scored walks couldn't have set it.
They loop around rooms, and between them hold under 41 s of straight walking. To decide what counts
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

- **Agreement.** Over the second before a turn's onset, the mean of that direction says which side
  the arrow was pointing. Within 5 degrees of straight on counts as "carry on". The turn agrees
  when the arrow pointed to the side the walker then turned.
- **Lead time.** How long the arrow had held that side, unbroken, when the turn started. It counts
  back at most 5 s, and never past the end of the previous turn. A lead that runs into either limit
  is reported as "at limit", meaning at least that long.
- **Sidesteps while walking straight.** Stretches where the arrow pointed more than 10 degrees to
  one side for at least 0.5 s, with no turn during them or within 2 s after.
- **Lagged correlation.** The arrow now set against how fast the walker turns a moment later, for
  every delay from 0 to 3 s. The delay with the strongest match is a second estimate of how far
  ahead the arrow runs, one that needs no turn threshold.

## Results

Three cold replays of each walk gave identical numbers, because the scene's floor fit is now seeded.

| Walk | Arrow shown to the walker | Scored walking | Turns | Obstacle ahead | Agreed | Wrong side | Carry on | Median lead | Unknown tag: agreed of |
|---|---|---|---|---|---|---|---|---|---|
| `pixel_walk_3` | none | 98.1 s known heading in 3 segments (133 s) | 21 | 20 | 13 | 7 | 0 | 0.43 s, 7 known, 3 at limit | 0 of 1 |
| `wifi_run_2` | the 1 s lookahead arrow, the one scored here | 35.7 s known heading in 1 segment (68.7 s) | 8 | 6 | 4 | 2 | 0 | 0.70 s, 3 known | 1 of 2 |
| `pixel_display_run` | the older three-value arrow | 60.6 s known heading in 1 segment (92.7 s) | 9 | 5 | 2 | 1 | 2 | 0.00 s, 2 known | 4 of 4 |

No turn on any walk was tagged "open ahead". These are walks around rooms, where a wall or desk is
nearly always within 3 m of the walker's line. So the split between turns made for an obstacle and
turns made for the route can't be drawn from these walks.

| Walk | Sidesteps while straight | Lagged correlation, strongest delay and match | Phone pointing off the walking direction |
|---|---|---|---|
| `pixel_walk_3` | 7, 4.3 a minute | 0.0 s at -0.13 (105 pairs), 0.7 s at 0.14 (417), 0.3 s at 0.28 (418), one per segment | median -1.3 deg, middle half spans 15.7 deg, over 30 deg on 17.7 % of samples |
| `wifi_run_2` | 11, 18.5 a minute | 2.5 s at 0.22 (327 pairs) | median -1.0 deg, middle half spans 26.3 deg, over 30 deg on 36.7 % |
| `pixel_display_run` | 6, 5.9 a minute | 0.4 s at 0.26 (580 pairs) | median +1.1 deg, middle half spans 19.1 deg, over 30 deg on 19.5 % |

### How much the threshold matters

The same walks at 8 and 14 degrees, either side of the measured 11:

| Walk | Threshold | Turns | Obstacle ahead: agreed, wrong side, carry on | Median lead |
|---|---|---|---|---|
| `pixel_walk_3` | 8 | 21 | 14, 6, 0 of 20 | 0.40 s, 10 known |
| | 11 | 21 | 13, 7, 0 of 20 | 0.43 s, 7 known |
| | 14 | 20 | 13, 7, 0 of 20 | 0.46 s, 9 known |
| `wifi_run_2` | 8 | 11 | 5, 4, 0 of 9 | 0.18 s, 4 known |
| | 11 | 8 | 4, 2, 0 of 6 | 0.70 s, 3 known |
| | 14 | 8 | 3, 2, 1 of 6 | 0.41 s, 3 known |
| `pixel_display_run` | 8 | 9 | 2, 2, 1 of 5 | 0.00 s, 2 known |
| | 11 | 9 | 2, 1, 2 of 5 | 0.00 s, 2 known |
| | 14 | 10 | 2, 1, 3 of 6 | 0.20 s, 2 known |

The `pixel_walk_3` figures hardly move. The two shorter walks move by a turn or two, because they
have few turns to move.

## What these numbers can and can't carry

**Few turns.** On `pixel_walk_3`, the one walk where the walker saw no arrow, the arrow pointed the
right way before 13 of 20 obstacle-ahead turns. Pointing at random would average 10 of 20, and a
fair coin gives 13 or more out of 20 about 13 % of the time. So at this size the agreement isn't
evidence that the arrow predicts where people turn, and it isn't evidence against it either. One
turn moves a share of 20 by 5 points, and a share of 6 by 17. Every share here should be read with
its count.

**Seeing the arrow.** On `wifi_run_2` and `pixel_display_run` the walker had an arrow in front of
them, so agreement there partly measures whether they followed it. On `pixel_display_run` the arrow
they saw was the older one, which could only point straight or fully to one side. The arrow scored
here is today's. Only `pixel_walk_3` measures whether the planner predicts a free choice.

**The phone's own pointing.** The arrow is read relative to the phone, so where the phone leads a
turn, the arrow's direction leads with it. Part of any agreement may be the phone's rather than the
planner's, and these numbers can't separate the two. The phone pointed more than 30 degrees off the
walking direction on 18 to 37 % of samples. The planner plans along the phone's direction rather
than the walker's, and changing that is work on the planner, not on this measurement.

**Lead times.** The arrow's lead before a turn is a fraction of a second where it's known at all:
medians of 0.0 to 0.7 s, over 2 to 7 turns per walk. The arrow reads the plan 1 s ahead, so a lead
under a second means the arrow mostly turned with the walker rather than before them. The lagged
correlation agrees: its strongest matches are weak, 0.14 to 0.28, with one segment negative.

**Sidesteps.** On `wifi_run_2` the arrow asked for a sidestep 18.5 times a minute while the walker
went straight, against 4 to 6 a minute on the others.

**Known headings.** Much of each walk has no known heading, because the walker stood or shuffled:
`wifi_run_2` has a known heading on 35.7 s of its 68.7 s. The tracker jumped 12 times inside
`pixel_walk_3`'s scored segments, 3 times in `pixel_display_run` and twice in `wifi_run_2`.

**Turn sizes in the lists.** The replay lists each turn with the heading change across its whole
interval, which can come out smaller than any 2 s window inside it, and once on `pixel_walk_3` even
reads `right -3`. That turn sits at the very end of a segment. Its side comes from its 2 s windows,
and the scores use the side, not the listed change.

**The walker's own notes.** `pixel_walk_3` came with notes of two desks and a cabinet in its first
25 s. That stretch is in a 12.8 s segment too short to score, or in a 51 s gap with no frames, so the
notes can't be checked against the scored turns.

**The replay is not the live run.** The live run dropped frames it couldn't keep up with. The replay
plans every frame. The replay also repeats the live loop's call to the planner rather than sharing
its code, and a test pins that call so a change to it is caught.

**Old recordings, new settings.** `pixel_walk_3` and `pixel_display_run` were recorded before the
phone sent its gravity flag, so the decoder supplies it. Every recording predates the seeded floor
fit. `pixel_walk_3` and `wifi_run_2` get `floor_ransac_seed=0` and
`floor_ransac_success_probability=0.99999999` by flag, and `pixel_display_run` gets the same two from
`--scene-defaults`. Those are the defaults, and the probability is Open3D's own. `pixel_walk_3` predates the floor's
camera-height check, so `floor_max_offset_meters=inf` turns that check off, as its live run had none.
`pixel_display_run` has no record of its scene settings, so it runs on today's defaults plus
`floor_max_tilt_degrees=50`, the tilt in the documented live command.

## Situation

- **Code:** commit `da9c670` (nav code hash `3b4a86b95ad5`), on top of the seeded floor fit.
- **Planner:** today's defaults, including the professor's 0.055 kinetic weight and the 1 s lookahead.
- **Replays:** three cold passes per walk, all identical.
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

Add `--eval-set turn_threshold_degrees=8` or `14` for the threshold rows. Each walk takes 3 to 8
minutes, three passes included.
