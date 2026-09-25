# The push field

How a list of obstacles becomes one heading and one surprise number. The code is
`math/src/main/kotlin/com/neuromorphicpaths/math/PushFieldGuidance.kt`, the per-class numbers
are in `ObstacleProfile.kt` beside it, and the tests beside those pin every claim below.

## The picture

Stand where the walker stands and look ahead. Every object in view sits some distance off the
line you are about to walk, and some distance along it. Imagine trying each possible heading in
turn, a few degrees left, straight, a few degrees right, and asking of each one: how likely is
it that this line hits something? Objects close to that line and close to you make a hit
likely. Objects far off the line or far away barely count. Turning hard is a little unlikely by
itself, because people mostly walk where they are facing. The best heading is the one where a
hit is least likely. How much less likely the best line is than the line the walker is actually
on is how surprised the system is at what the walker is doing.

Every quantity in this document is a probability first. The surprise of a probability p is
minus log2 of p, in bits, so probabilities that multiply become surprises that add, and the
numbers mean the same thing everywhere in this project.

## One object, one heading

Take an object at bearing θ (theta, the angle from the camera's forward axis, positive to the
right) and range r (meters along the ground). For a candidate heading φ (phi, same convention),
the object sits at a relative angle θ − φ from that line. Two distances follow:

- how far along the line the walker draws level with it, s = r cos(θ − φ). Negative means the
  object is behind the walker, and it is ignored.
- how far it sits off the line at that moment, the miss distance d = r sin(θ − φ).

The time to contact is τ (tau) = s / v, where v is the walker's speed, or the object's closing
speed when something has measured one. That is the same Tau as in car following. It is floored
at 0.2 s so a very close object gives a large number rather than infinity.

Four things have to be true at once for this object, on this heading, to be a collision. Each
is a probability, and they multiply:

1. **It is in the path.** A Gaussian in the miss distance, exp(−d² / 2w²), with w the
   clearance the object's class needs. One clearance off the line and it is in the path with
   probability 0.61. Two clearances, 0.14.
2. **Contact comes soon.** 1 − exp(−½ (T / τ)²), with T the class's horizon in seconds. At τ
   equal to the horizon this is 0.39. At half the horizon, 0.86.
3. **It exists.** The detector's score for the box, calibrated. A detector's score is not a
   probability, and the first replay with it taken literally showed why: a cone a meter ahead
   scored 0.45 and could never be worth more than a bit, so the field nudged where the walker
   swerved. A score of 0.5 or more counts as certain, and below that the probability falls
   linearly, so the detector's own 0.25 threshold reads as an even chance.
4. **Touching it would matter.** 1 − a, with a the class's acceptability of contact. A bin
   brushed in passing is nothing much. A car is not.

The probability of a collision with this object is the product, and the object's surprise is

    U = −log2(1 − P(in path) · P(contact soon) · P(exists) · P(matters))

in bits. For an object on the line, certain, and with no acceptability, the term inside the
log is exp(−½ (T / τ)²), so the surprise is ½ (T / τ)² / ln 2. That is the time-to-contact
surprise the rest of this project uses, so an object dead ahead at exactly the horizon is 0.72
bits and at half the horizon 2.9 bits, as before.

The product form fixes one real fault of an earlier version, which multiplied that Tau surprise
by the in-path Gaussian directly. An object passed 0.8 m away at arm's length, 0.3 m ahead,
scored about 22 bits that way, enough to swerve from something already being cleared. As a
probability it is in the path with 0.28 and contact is certain, so the collision probability
is 0.28 and the surprise is 0.47 bits.

## What each class means

Three numbers per class, each entering the formula as the thing it names. None is a weight.

| Class | Clearance w, m | Horizon T, s | Acceptability a | Reading |
|---|---|---|---|---|
| Tree, barrier, hole, pole, unknown | 0.5 | 2.0 | 0.0 | Fixed and hard. Give it a normal berth and never touch it. |
| Person | 0.6 | 2.0 | 0.3 | Wider berth, but a brush is a bump, not an injury, and they step aside. |
| Animal | 0.6 | 2.0 | 0.2 | As a person, less predictable. |
| Car | 1.0 | 3.0 | 0.0 | Big, possibly moving, the widest berth and the longest look ahead. |
| Bike | 0.5 | 2.0 | 0.1 | Parked, mostly. Pedals catch. |
| Table | 0.5 | 1.5 | 0.1 | Small and still, counts late. |
| Chair | 0.5 | 1.5 | 0.2 | Small, still, and a shin can nudge it. |
| Trash can | 0.5 | 1.5 | 0.5 | Brushed past half the time without anyone minding. |

No clearance is under 0.5 m. A walker is about that wide and sways, so the chance that
something half a meter off the line is in the path is 61 percent whatever it is. The values
are arguable in meters and seconds, which is the point. Whether a chair is nudged or avoided is
a question the team can answer without reading code.

## All objects, all headings

Objects are taken as independent, so the probability of missing all of them is the product of
the individual misses, and the surprise of a heading is the sum of the object surprises. On top
of that sits a prior on the heading itself, a Gaussian with spread σ (sigma) of 30 degrees,
whose surprise relative to straight ahead is ½ (φ / σ)² / ln 2. That last term is what makes
the walker go straight when nothing is around. The cost of a heading is the two together.

The code scores every heading from 60 degrees left to 60 degrees right in 1 degree steps.

## What comes out

Three numbers, from that grid of costs.

**The desired heading** is the cheapest heading found. Ties break toward the smaller turn, and
toward the right when the two sides are exactly equal. Two to the minus cost, normalized over
the grid, is a posterior distribution over headings, and the cheapest heading is its mode.
Choosing the heading with the lowest expected surprise is one step of active inference, which
is what ties this field to the lectures.

**The overall surprise**, the alert, is the cost of the walker's actual heading minus the cost
of the desired one. It is exactly zero when the walker is already doing the best available
thing, whether that is because nothing is there or because they have already turned. It is not
minus log2 of the posterior at the walker's heading. In an open scene the prior spreads over
121 candidates and puts about a percent on straight ahead, which would read as 6 bits with
nothing in view. The posterior is used for two things only, its mode and its entropy.

**The heading entropy** is the entropy of that posterior, in bits: minus the sum over headings
of p log2 p. It says how spread the field's belief is. One clearly best heading gives a number
near zero. An open scene gives the entropy of the prior alone, several bits. A row of trees
with one gap sharpens the belief onto the gap, and the entropy drops. The overlay that draws
the projected path fades with this number, and the alert reddens with the surprise. They are
different things and are kept apart on purpose.

The per-object push is still reported. It is the slope of that object's surprise against
heading at the walker's current heading, positive when turning right would help, plus a forward
component that is minus the surprise, since an object only ever argues for slowing down. That
is for the display and the log, not for steering.

## Why a search and not a sum of forces

Adding one push vector per object is the obvious way to do this, and it gets three cases wrong.
Two objects flanking the line cancel and leave the walker aimed at the gap even when the gap is
too narrow. One object dead ahead has no sideways component at all. And nothing in view gives
no force, which is fine, but nothing to say about how confident that is either.

Evaluating the same per-object probabilities over a range of headings fixes all three without
a special case. The narrow gap costs more than either side, so a side wins. The dead-ahead
object costs more than a turn, so a turn wins, and the tie rule picks which. Nothing in view
costs zero everywhere, the prior picks straight, and the entropy says the field is not sure.

## Regions count once

A wall, a building face or a grass verge will arrive as many boundary samples, not one box.
Fifty samples along one wall are not fifty independent objects, and treating them so would
make a wall outweigh a person fifty to one. Until region input exists, the rule is written
here so it is not forgotten: a region contributes one sample per clearance width along its
boundary, or only its nearest sample, and never one per pixel.

## Closing speed and memory between frames

The field itself has no memory. What it remembers arrives in the obstacles. A tracker follows
each box from frame to frame by overlap and gives it an id, and a box the detector misses is
carried through for two frames at a fading confidence, so the "it exists" term shrinks rather
than dropping to zero and the heading does not flip on one missed detection. The range of a
track over its last two seconds, fitted with a straight line, gives the closing speed that
sets τ. A standing object closes at the walker's own speed. A person walking toward the
walker closes faster and counts for more. A person walking away has a closing speed at or
below zero, and then contact never comes and the object costs nothing. Fewer than three fresh
ranges in the window leave the closing speed unknown, and the walker's own speed stands in.

The range behind that estimate comes from where a box meets the ground, which moves with
every degree of camera pitch, so the closing speed is a noisy number at two or three frames
a second. How noisy, on the real walk, is written in the project record beside the replay.

## What is not in it yet

The walker's own speed comes from GPS when the app has location permission and is outdoors,
smoothed over one second, and is 1.4 m/s otherwise. Surfaces, the second half of what each
class means, wait on region input.

## The walker's own wobble

The field keeps a running spread of the walker's heading over the ground, from the phone's
rotation vector, weighted so the last second counts most. That number is shown on screen and
logged. It is meant to set the prior's spread σ from data, as the wobble times 2.07, which is
the band that holds 96 percent of a walker's headings.

That switch is off. On the first outdoor recording the one-second wobble has a median of 4
degrees, and it only passes 20 degrees through corners. As a tolerance, 4 degrees times 2.07
is about 8 degrees, and a 20 degree turn would then cost about 4 bits, against 0.7 bits for an
obstacle two seconds ahead. The field would hold its line into most things. Something in the
mapping is off, the window, the ratio, or the idea that sway and tolerance are the same
quantity, and that is a decision to make with the team rather than by tuning until it looks
right. The switch is `turnToleranceFromWobble` in the parameters, with a floor of 5 degrees
and the 60 degree search range as the ceiling.
