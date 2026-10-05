# The arrow's side flips, and what holds it at its limit

Two questions about the laptop planner's arrow, measured on recorded walks. Does the arrow swing from
one side of an obstacle to the other between frames? And when the nearest thing in the walker's way
is 3 to 5.32 m ahead, what keeps the arrow pointing as far sideways as it can?

How to rerun every number here is at the end.

## Terms used below

**The arrow** is the direction the planner tells the walker to go, read off its planned path one
second ahead. It can point at most 35.5 degrees to either side: that is a full sideways step, 1.0 m/s,
taken at walking pace, 1.4 m/s. An arrow within half a degree of that is **at its limit**.

**A full swing** is the arrow at its limit on one side in one frame and at its limit on the other side
in the next frame, 33 ms later.

**Plan disagreement** asks how far two consecutive plans differ once both are placed on the floor. The
first plan is moved into the world through where the walker stood, then compared with the second at the
same place on the floor. The figure is the mean sideways gap in meters. Its **90th percentile** is the
gap that only the worst tenth of frame pairs exceed.

**The band** is the frames where the nearest thing in a 0.5 m corridor either side of the walker's line
is 3 to 5.32 m ahead. 5.32 m is as far as the plan looks: 3.8 s at walking pace. The planner had been
asked to sit at its limit on at most half of those frames, because something that far away rarely needs
a full-speed sidestep.

## How it was measured

Three walks recorded with the Pixel app, replayed on the laptop through the same scene and planner the
live run uses. One scene runs across each whole walk, and the walk's last stretch without a pause of
more than 5 s is planned frame by frame.

| Walk | Where | Planned frames |
|---|---|---|
| `contact_walk_1` | indoors, the first walk with the contact term | 1202 |
| `pixel_walk_3` | outdoors | 1391 |
| `pixel_display_run` | the classroom, between tables and chairs | 2582 |

The floor fit is seeded, so a replay gives the same numbers every time on one machine. Two planner
settings are compared throughout: the planner as it now ships, and the same planner without the change
this document describes.

## The flips

### What the big disagreements were

Before the change, the 90th percentile of plan disagreement was 1.18 m on `contact_walk_1`, 1.10 m on
`pixel_walk_3` and 1.30 m in the classroom. Each frame pair at or above it was sorted into one of five
kinds, checked in this order:

1. **The tracker jumped.** The phone's tracked position moved faster than 3 m/s between the frames.
2. **The phone turned.** The plan is drawn along the phone's forward direction, so a phone turning
   between frames swings an unchanged plan through the world. A pair counts here when lining up the two
   frames' directions removes at least half the gap.
3. **Something new came into view.** The plans pass an obstacle on opposite sides, and the first frame
   hadn't seen it.
4. **A side flip.** The plans pass an obstacle both frames saw on opposite sides.
5. **A shift on the same side.** Neither of the above, but the plans still differ.

Obstacles are matched between frames by where they sit on the floor, not by the scene's group number,
because a group number names a patch of the walker's view rather than an object.

| Walk | Pairs at or above the 90th percentile | Side flips | New obstacle | Phone turned | Same side | Tracker jump |
|---|---|---|---|---|---|---|
| `contact_walk_1` | 121 | 109 (90.1 %) | 2 | 10 | 0 | 0 |
| `pixel_walk_3` | 139 | 103 (74.1 %) | 22 | 10 | 4 | 0 |
| Classroom | 259 | 220 (84.9 %) | 25 | 11 | 2 | 1 |

Most of the large disagreements were side flips: an obstacle centered ahead, a table or a cluster of
chairs, with the plan passing it on the left in one frame and on the right in the next. Counted at a
fixed bar rather than a percentile, the arrow made a full swing 77 times a minute on `contact_walk_1`,
76 on `pixel_walk_3` and 105 in the classroom.

### Why

The planner chooses its path by adding up a cost for every place the walker might be over the next
3.8 s and taking the cheapest route, the formulation from the course. When going left and going right
around a centered obstacle cost nearly the same, any small change between frames tips the choice. The
planner keeps nothing from one frame to the next, so nothing resists the tip.

The sideways weight had been suspected, because raising it from 0.055 to 6.5 earlier raised the 90th
percentile. That isn't the cause. At the course's weight of 0.055 the arrow swung more often on two of
the three walks: 100 times a minute on `pixel_walk_3` and 127 in the classroom.

### What was changed

The planner now remembers the plan it made one frame earlier and prices any path by how far it strays
from that plan. In the course's own form it is a prior surprise: half of (distance from the previous
plan, over a spread) squared, added to the cost beside the obstacle terms, the same shape as the goal
already had. It covers only the rows the walker reaches in the first second, about as long as they need
to act on the arrow, so the plan further out stays free to change as new things come into view.

The previous plan is held relative to the walker's own direction, not fixed in the world. Between frames
it is moved on the way the planner already assumes the walker moves: forward at walking pace, and
sideways along the plan itself. It is dropped after a gap of more than half a second, after the clock goes
backwards, and when the goal moves more than 1.5 m, so a wearer who looks toward the other of two gaps is
steered there.

**How it behaves in use.** Two versions were considered: the previous plan fixed in the world, or held
relative to the walker's own direction.

An extreme change of direction:

- World version. The old plan is placed in the world and re-expressed in the new frame.
  - A U-turn or a 90 degree turn puts it behind or beside the walker, out of the new plan's rows. The
    prior then simply vanishes, which is fine.
  - The bad case is a deliberate turn of 20 to 45 degrees. The old plan now lies off to the old side,
    and the prior pulls the new arrow back toward the direction just abandoned, harder the bigger the
    turn. The goal follows the camera (ahead), but the prior is fixed to the world, so they fight.
- Walker-frame version. The prior means "keep the same offsets relative to my forward". It never
  resists a turn. It only remembers "I was passing on the left".

Standing still:

- World version: the plan is held in place, which is harmless while the scene is static. If the person
  stands and turns the phone, it pulls toward the old world direction, the same problem as above.
- Walker-frame version: the old plan is advanced by one frame of assumed walking, about 5 cm at
  1.4 m/s and 30 Hz. Standing makes that 5 cm wrong, which is negligible.

How long it holds: there is no timer. A side holds until the other side becomes cheaper by more than
the cost of switching, and that cost grows with the square of the change. Small corrections cost almost
nothing, and only a swing to the other side is expensive. A side that becomes blocked is left at once.

The world version also needs the walker's position, which the Neon glasses don't provide, so it would
have done nothing on them. The walker-frame version was built.

**The options, rated** from 0 to 5 on how easily each extends, how well it stays in one place, how
cleanly it states what it means, and how well it holds up:

| Option | Extensibility | Modularity | Abstraction | Robustness | Fits the course's formulation |
|---|---|---|---|---|---|
| A prior toward the previous plan, held relative to the walker | 4 | 4 | 3 | 3 | Yes, a prior beside the obstacle terms |
| A prior toward the previous plan, fixed in the world | 4 | 3 | 4 | 2 | Yes |
| Keep the previous side whenever both sides' costs are within a margin | 2 | 3 | 2 | 2 | No, a rule bolted onto the choice |
| Smooth the arrow over frames after the plan is chosen | 2 | 4 | 1 | 1 | No. The arrow stops being the plan's |
| Leave it | 3 | 5 | 5 | 1 | Unchanged |

Smoothing scores 1 on robustness because averaging +35.5 and -35.5 degrees gives straight ahead, into
the obstacle the flips are around. Leaving it scores 1 because one to two full swings a second is an
arrow nobody can follow.

**The spread** was set by replaying all three walks at spreads from 0.125 to 4 m, over the first second
or over the whole 3.8 s. Two checks had to hold at every setting. First, a side that becomes blocked
must still be left in time: with a post 1.5 m ahead the walker has 1.07 s, and getting the body past it
takes 0.32 s, so the plan may lag at most 0.75 s. It switched within one frame at every setting tried.
Second, once nothing is in the way the arrow must come back to straight within the 1 s it reads ahead.
It did, within two frames, at every setting chosen from. Covering the whole 3.8 s with a spread under
1 m held the arrow at its limit on 12 to 17 % of the classroom's frames with nothing in the corridor,
over the 10 % allowed, so the prior covers the first second only. 0.25 m ships, with room on both sides:
the arrow stops coming back to straight at 0.05 m, and the swings return from about 1 m.

### Result

| | `contact_walk_1` | `pixel_walk_3` | Classroom |
|---|---|---|---|
| Full swings a minute, before | 77.1 | 75.9 | 105.4 |
| Full swings a minute, now | 0.0 | 1.3 | 0.0 |
| 90th percentile of plan disagreement, before | 1.176 m | 1.102 m | 1.303 m |
| 90th percentile of plan disagreement, now | 0.254 m | 0.449 m | 0.273 m |

The alarm is unchanged on every walk, because it doesn't read the plan.

## What holds the arrow at its limit 3 to 5.32 m out

### The band before and after

| | `pixel_walk_3` | Classroom |
|---|---|---|
| Band frames | 122 | 195 |
| At the limit, before | 58.2 % | 59.0 % |
| At the limit, now | 1.6 % | 30.3 % |

The change above brought the band under its target on both walks. The arrow reads the plan one second
ahead, the last row the prior covers, so a sidestep for something 3 to 5 m away now builds up over a few
frames instead of jumping straight to the limit. A side that becomes blocked close in is still left at
once.

### What the remaining frames are

Each band frame with the arrow at its limit was re-planned four times, each time without one suspected
cause, to see whether it would still sit at the limit:

- **The phone pointing off the walker's line**, tested by turning the obstacles into the walker's
  direction of travel.
- **A wall costing once per 0.25 m patch**, so that it pushes harder than one obstacle would, tested by
  joining touching wall patches into one.
- **The goal acting only on the plan's last row**, tested by spreading the same goal over every row.
- **A full sidestep simply being right**, tested by geometry: whether the slowest sidestep that gets the
  body past the obstacle in time would still read as the limit.

| | `pixel_walk_3`, now | `pixel_walk_3`, before | Classroom, now | Classroom, before |
|---|---|---|---|---|
| Band frames at the limit | 2 | 71 | 59 | 115 |
| Explained by the phone pointing off the line | 0 | 15 | 27 | 36 |
| Explained by the goal on the last row | 0 | 22 | 0 | 10 |
| Explained by walls | 0 | 0 | 0 | 0 |
| A full sidestep is right | 0 | 0 | 0 | 0 |
| Explained by none of the four | 2 | 46 | 30 | 73 |

A frame can be explained by more than one cause, so the rows can add up to more than the total. Two
classroom frames couldn't be tested for the phone pointing, before or after, because the walker's
direction of travel wasn't known there: standing, or close to a gap in the tracking.

- The prior took out every frame the goal test explained.
- The phone pointing explains 27 of the classroom's 59 remaining frames. In those the phone pointed a
  median 23.9 degrees off the walking direction, between 7.9 and 36.3.
- Walls can't be judged on these walks. `pixel_walk_3` has no wall in any band frame, and the classroom
  has 36 wall points out of 12,313, in 4 frames. The zero says nothing either way.
- A steady sidestep for anything 3 to 5.32 m ahead reads well under the limit, so a full sidestep is
  never simply right there.

Most of the frames none of the four explains have something nearer than 3 m just beside the corridor,
0.5 to 1.0 m to the side, where the band doesn't look: both of `pixel_walk_3`'s, and 19 of the
classroom's 30. They are desk clusters about 2 m wide, and an obstacle shaped like a U whose arms reach
to about 2 m ahead.

### The band, restated

So the band now also asks that nothing nearer than 3 m sits within 1.0 m either side of the walker's
line. A frame with something 2 m away just outside the corridor no longer counts as "3 to 5.32 m ahead".
This changes only how frames are counted, never what the planner does. It also lowers the band share
partly by relabeling frames, which is why the earlier figure is kept beside it.

| | `pixel_walk_3` | Classroom |
|---|---|---|
| Restated band frames | 58 | 80 |
| At the limit, before | 26 (44.8 %) | 39 (48.8 %) |
| At the limit, now | 0 | 17 (21.2 %) |

A share over fewer than 100 frames moves by more than a whole percentage point per frame, so it isn't
taken as a result. Both walks are under that, and the restated band is reported but not judged until a
longer walk adds frames. On the earlier definition the target is met on both.

## What is left open

- **The plan is made along the phone's direction, not the walker's.** It explains 27 of the classroom's
  remaining band frames, and 4 to 8 % of the large plan disagreements. Planning along the direction of
  travel would remove it.
- **A wall counted once rather than once per patch.** These walks have almost no walls in the band, so a
  walk along a corridor is what would show whether it matters.
- **The restated band needs more frames.** A walk with at least 100 such frames would let it be judged.

## Rerun

From `server/`, with its venv, and the recordings passed by path. `pixel_walk_3`'s recording predates
three floor settings, so they are set to today's defaults, and the classroom recording has no settings
file at all.

```
python -m nav.evaluation.check_planner numbers frame_logs/contact_walk_1
python -m nav.evaluation.check_planner numbers frame_logs/pixel_walk_3 --scene-set floor_max_offset_meters=2.2 --scene-set floor_ransac_seed=0 --scene-set floor_ransac_success_probability=0.99999999
python -m nav.evaluation.check_planner numbers frame_logs/pixel_display_run --scene-defaults --scene-set floor_max_tilt_degrees=50
```

`flips` and `band` take the same arguments in place of `numbers`. Add
`--set previous_plan_prior_enabled=false` for the figures before the change. `--cached` keeps the scene
pass between runs, which takes minutes cold.
