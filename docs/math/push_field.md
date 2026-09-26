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
with one gap sharpens the belief onto the gap, and the entropy drops. On the outdoor walk it
barely moved, 6.73 bits against a ceiling of 6.92, because a few bits of cost over a few
degrees cannot sharpen a belief that a 30 degree prior has spread over 121 candidates. It is
reported and logged, and nothing on screen depends on it.

**The heading information** is how far the scene moved the belief away from the prior, in
bits: the sum over headings of p log2 (p / q), with p the posterior and q the prior at the
same heading. That is the divergence of the posterior from the prior, which the literature
calls Bayesian surprise, and it is the information-gain half of the expected free energy that
active inference minimizes, so it is the one number here with a name in the course's family.
Nothing in view gives exactly zero, however spread the belief is, since the posterior then is
the prior. A barrier a meter ahead gives about half a bit, two meters ahead under a tenth, a
wall along the path about one bit, and walls on both sides that leave only straight ahead
about two. The numbers are small because the dent one object makes is shallow against a prior
spread over 121 headings, and what matters is that they order the scenes the right way. This
is what the projected path fades by, below.

The per-object push is still reported. It is the slope of that object's surprise against
heading at the walker's current heading, positive when turning right would help, plus a forward
component that is minus the surprise, since an object only ever argues for slowing down. That
is for the display and the log, not for steering.

## The projected path

One heading says which way to go. It does not say how the next few meters bend around
several things at once, which is what the picture on screen is for. So the field is rolled
forward. From where the walker stands, the cheapest heading is taken and a virtual walker
moves half a meter along it. Every obstacle is placed again relative to that new position,
the ground term is read from there, and the field is asked again. Six steps of that reach 3 m,
about two seconds of walking, the same distance the ground term looks ahead. The prior stays
centered on the walker's real heading the whole way, because it says where the walker wants
to go, not where the last step happened to point. The chain of positions, projected through
the ground plane back into the frame, is the curve. With nothing in view it is a straight
line dead ahead. With a wall along the right it bends left, and the test that says so pins it.

On screen the curve is a ribbon, not a line, since it is the thing the walker is meant to
see. It is a fifth of the frame's width where it leaves the bottom edge and narrows with
height to about twenty pixels at the top, the way a strip of ground does in perspective, and
the heading arrow sits on top of it. Whatever else is drawn on it, the ribbon fades with
distance from full at the walker's feet to nothing at its far end, so it is always seen
disappearing rather than stopping at an edge.

Two things are drawn on the ribbon, and they are kept on two channels on purpose.

**Opacity is information.** Each step carries the heading information at that step, the
divergence above, and the ribbon is drawn solid where the scene shaped the choice and less
so where it did not. One bit, a wall along the path or a barrier a meter ahead doubling the
odds of the chosen direction, is fully solid. An open scene gives zero everywhere, and since
the ribbon is meant to be prominent whatever the scene says, the floor is high: seven tenths
at zero information, solid at one bit. The reading is: a solid ribbon is a path the scene
decided, a slightly paler one is the walker's own line with nothing to say about it, and the
fade with distance runs on top of both.

**Color is surprise.** The whole curve takes one color, blue when there is nothing to worry
about and red from three bits up, the cone at its worst on the outdoor walk. The number it
follows is the higher of two surprises: the overall surprise, which says the walker's own line
is worse than the best one, and the lowest surprise ahead, covered below, which says even the
best line is heading into something. So the ribbon turns red when the walker is off the
good line, and also when there is no good line left, and it turns red before any STOP is
shown. That is the alert. It is not how sure the field is, and a solid blue curve and a faint
red one are both possible: the first is a scene that decided a path the walker is on, the
second an open line the walker has wandered off.

The entropy was the first candidate for the opacity, and it does not work here. On the whole
outdoor walk it sat within two tenths of a bit of the prior's own entropy, because the prior
spreads over 121 candidates and a few bits of cost over a few of them barely move the sum. It
also cannot tell a belief sharpened onto one gap from one flattened over two equally good
sides, which is the case a walker most needs to see. The divergence is zero for the open
scene, grows with what the scene did, and is a named quantity in the course's own vocabulary.

### What the ribbon is telling you, in plain words

| What you see | What the field is saying |
|---|---|
| A blue ribbon going straight up the middle | Nothing ahead is surprising. Keep walking where you are walking. |
| A blue ribbon bending to one side | Something is ahead, and stepping around it the way the ribbon bends is not surprising at all. |
| A solid ribbon | What is in view decided this path. A wall, a gap or an obstacle pulled the choice away from plain straight ahead. |
| A slightly paler ribbon | Nothing in view had anything to say about the path. It is just your own line drawn out. The app's own screen shows this. Over other apps the ribbon keeps one strength, set by the window. |
| The ribbon turning purple, then red | Surprise is climbing. Either your line is heading into something the best line avoids, or every line is heading into something. Red means three bits or more, a collision is now likely enough to act on. Over other apps the ribbon also gets stronger then, from half to 0.8. |
| The far end fading to nothing | Always there. The path is a guess about the next few meters, so it is drawn trailing off rather than ending in a hard edge. |

## No way through

The ribbon always goes somewhere. The field picks the least surprising heading, and there is
always a least surprising one, even when every heading ahead runs into a wall. So a second
question is asked on every frame: how surprising is the least surprising heading on offer?
That is the lowest surprise ahead. When it is low, some heading leads clear. When it is high,
every heading from hard left to hard right is expected to hit something, and there is no way
through.

Asked with the same two second horizons the arrow uses, that question cannot tell a dead end
from a squeeze. A wall straight across the path only reaches 3 bits about 2 m out, and the
outdoor walk had six moments where the walker slipped past a post or a wall sample 0.6 m away
at 3 bits or more. So it is asked with every horizon doubled, "how surprised would you be over
the next four seconds rather than the next two". With that, a wall across the path reads 2.5
bits at 4 m, 3.6 at 3 m and 7.1 at 2 m, while a gap between two posts 0.6 m either side of the
line reads 1.7 at 2 m. The arrow and the ribbon keep the shipped horizons. Only this number
uses the stretch, and the stretch is `noWayThroughHorizonStretch` in `PushFieldParameters`.

The floating overlay turns the number into three states. The thresholds are in
`GuidanceDisplayTuning`, each one labeled.

- **Red from 3 bits.** The same line where the ribbon's color reaches full red. It comes first,
  so the walker sees red before a STOP.
- **STOP from 4 bits, held for a second.** A wall across the path crosses 4 bits about 2.8 m
  out, and a second of walking later the STOP is up, about 1.5 m from it. On the outdoor walk
  the lowest surprise ahead reached 4 bits three times, never for longer than half a second,
  so the rule would have raised no STOP there. The walk has no real dead end in it, so the
  scale on the dead-end side is from the synthetic wall only.
- **U-turn once stopped.** When the walker's measured speed reaches zero while STOP is up, the
  symbol becomes a U-turn and stays one until they walk again. It has to latch. The field
  treats a standing walker as barely moving, and standing things fade at that speed, so the
  surprise ahead drops the moment the walker stops. When they set off again the field is asked
  afresh, and the STOP clears once the surprise ahead is back under 3 bits.

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

A wall, a building face or a flight of stairs arrives from the segmenter as a region of cells,
not a box. The cells where such a region touches ground are its foot, and each foot cell is
projected through the ground plane to a point on the ground, the same geometry that places a
box by its bottom edge. Fifty foot points along one wall are not fifty independent objects,
and treating them so would make a wall outweigh a person fifty to one. So the points are
thinned, nearest first, to one per half meter of the same class, half a meter being the
smallest clearance any class gets, and each surviving point enters the field as an obstacle:
class WALL or BUILDING, certain, no track, closing at the walker's own speed. A wall
that runs beside the path for ten meters is then about twenty samples, each with the ordinary
in-path Gaussian, and the ones near the walker do the pushing.

Stairs are on the map but not in the field for now. On the outdoor recording the stairs were
the walker's route, and treating them as a hard obstacle put a 12 bit alert on the flight the
walker was climbing. Whether indoor stairs need a berth, a charge per meter, or nothing waits
for the classroom recording, and the locator takes the set of classes it places as a
parameter so that is one line to change.

The ground under a structure is not also charged as a surface. The field asks what the ground
is at a point, the point projects into a wall cell, and the answer is unknown, which costs
nothing. The wall's cost is already in its samples.

## Closing speed and memory between frames

The field itself has no memory. What it remembers arrives in the obstacles. A tracker follows
each box from frame to frame by overlap and gives it an id, and a box the detector misses is
carried through for two frames at a fading confidence, so the "it exists" term shrinks rather
than dropping to zero and the heading does not flip on one missed detection. The range of a
track over its last two seconds, fitted with a straight line, gives a closing speed. A
standing object closes at the walker's own speed. A person walking toward the walker closes
faster and counts for more. Fewer than three fresh ranges in the window leave the closing
speed unknown, and the walker's own speed stands in.

The range behind that estimate comes from where a box meets the ground, which moves with
every degree of camera pitch, so the closing speed is a noisy number at two or three frames a
second. On the outdoor walk the estimate for things that do not move has a median of 0.63 m/s
where the walker was doing 1.24, and a fifth of them read as not closing at all. So the
measured closing speed is only allowed to raise the urgency, never to lower it below the
walker's own speed. A person walking away therefore still counts as a standing person until
the estimate is good enough to tell the two apart, which is the safe side to err on.

## The walker's own speed

The walker's speed comes from their steps. Each step lands as a bump in the accelerometer,
the bumps are counted, the cadence is one over the mean of the last three step intervals, and
the speed is cadence times a 0.7 m step, smoothed over a second. No step for a second and a
half means the walker has stopped, and the speed decays to zero. The field floors a measured
speed at 0.3 m/s, so a walker standing still is treated as barely moving: a bin 2.5 m away is
then over eight seconds off and worth almost nothing, while a person walking toward the walker
keeps their own closing speed and still counts, and the arrow still points somewhere on the
first step. GPS speed, outdoors and with permission, is kept as a cross-check. Live and replay
run the same estimator, live off the phone's accelerometer and replay off the recording's
acceleration log. The 0.7 m step is a constant, and a taller or hurrying walker takes a longer
one. The GPS cross-check is what would calibrate it.

## The ground ahead

A segmenter says what the ground is. SegFormer-B0 trained on ADE20K reads every third frame
at 256 by 256 and answers with a 64 by 64 grid of labels, which are mapped onto pavement,
grass, dirt, road, the three structure classes, sky and other. The grid is stretched over the
frame, so asking what the ground is at a point ahead of the walker means projecting that point
into the frame and reading the cell it lands in. The map is held until the next one, so on the
two frames between it is a little stale, which for verges and walls does not matter.

Each candidate heading is then charged for the ground it crosses. The line is sampled every
0.25 m out to the distance the walker covers in 2 s, and each sample is charged the per-meter
cost of the surface it lands on: nothing for pavement, 0.1 bits per meter for grass, 0.15 for
dirt, 0.6 for a road. Read as a probability, each meter of grass is fine with probability 0.93.
Ground outside the frame, above the horizon, under a structure or under something the map
calls other costs nothing, so a pipeline without a segmenter behaves as if the term were not
there.

The effect is a preference, not a rule. A walker with a wall along the pavement's edge crosses
the grass to get away from it, since a few tenths of a bit of grass are cheaper than the wall's
berth, and a walker with nothing in the way stays on the pavement, since any turn onto the
grass costs more than staying put. Given the choice, grass beats a road by a wide margin.

## The walker's own wobble

The field keeps a running spread of the walker's heading over the ground, from the phone's
rotation vector, weighted so the last second counts most. That number is shown on screen and
logged as a measurement of the walker. It does not set the prior's spread σ, and that was
decided on the first outdoor recording rather than by argument.

The idea was that σ should be the walker's sway times 2.07, the band that holds 96 percent of
a walker's headings. The one-second sway on that walk has a median of 4 degrees, so that gives
about 8 degrees. Re-running the field over the logged walk with σ at 8 degrees, it made no
calls at all and missed the cone the walker swerved around. Fixed values of 15, 20, 30 and 45
degrees made 15, 36, 59 and 88 calls, of which 4, 6, 12 and 15 agreed with the walker's next
two seconds, and every mapping from the measured sway with a floor landed between the fixed
values with more noise. Sway is how much a walker wanders while holding a line. The prior's
spread is how far they are willing to turn to avoid something. They are different quantities,
and σ stays a parameter at 30 degrees.
