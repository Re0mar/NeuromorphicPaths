# The push field

How a list of obstacles becomes one heading and one surprise number. The code is
`math/src/main/kotlin/com/neuromorphicpaths/math/PushFieldGuidance.kt`, and the tests beside it
pin every claim below.

## The picture

Stand where the walker stands and look ahead. Every object in view sits some distance off the
line you are about to walk, and some distance along it. Imagine trying each possible heading in
turn, a few degrees left, straight, a few degrees right, and asking of each one: how bad would
this line be? Objects close to that line and close to you make it bad. Objects far off the line
or far away barely count. Turning hard is a little bad by itself. The best heading is the one
where the total badness is lowest. That total, for the line the walker is actually on, minus
the total for the best line, is how surprised the system is at what the walker is doing.

"Badness" here is surprise in bits, so the numbers add up and mean the same thing everywhere in
this project.

## One object, one heading

Take an object at bearing θ (theta, the angle from the camera's forward axis, positive to the
right) and range r (meters along the ground). For a candidate heading φ (phi, same convention),
the object sits at a relative angle θ − φ from that line. Two distances follow:

- how far along the line the walker draws level with it, s = r cos(θ − φ). Negative means the
  object is behind the walker, and it is ignored.
- how far it sits off the line at that moment, the miss distance d = r sin(θ − φ).

The time to contact is τ (tau) = s / v, where v is the walker's speed, or the object's closing
speed when something has measured one. That is the same Tau as in car following.

The object's surprise, in bits, is

    U = ½ · (T₀ / τ)² · exp(−d² / 2w²) / ln 2

with T₀ the reference time (2 s) and w the clearance (0.5 m). The shape is half the square of a
signal-to-noise ratio, converted to bits, the same as every other surprise here. The ratio is
how urgent the object is, T₀ / τ, scaled down by how far off the line it sits. An object dead
ahead at exactly T₀ is 0.72 bits. At half that time it is 2.9 bits. One clearance off the line
takes 40 percent off. Two clearances take 86 percent off.

τ is floored at 0.2 s so a very close object gives a large number rather than infinity.

## All objects, all headings

The cost of a heading is the sum of every object's surprise at that heading, plus a small cost
for the turn itself, ½ (φ / σ)² / ln 2 with σ (sigma) at 30 degrees. That last term is what
makes the walker go straight when nothing is around.

The code tries every heading from 60 degrees left to 60 degrees right in 1 degree steps and
keeps the cheapest. Ties break toward the smaller turn, and toward the right when the two sides
are exactly equal.

## Why a search and not a sum of forces

Adding one push vector per object is the obvious way to do this, and it gets three cases wrong.
Two objects flanking the line cancel and leave the walker aimed at the gap even when the gap is
too narrow. One object dead ahead has no sideways component at all. And nothing in view gives
no force, which is fine, but nothing to say about how confident that is either.

Evaluating the same per-object surprises over a range of headings fixes all three without a
special case. The narrow gap costs more than either side, so a side wins. The dead-ahead object
costs more than a turn, so a turn wins, and the tie rule picks which. Nothing in view costs zero
everywhere, and the turn term picks straight.

The per-object push is still reported. It is the slope of that object's surprise against
heading at the walker's current heading, positive when turning right would help, plus a forward
component that is minus the surprise, since an object only ever argues for slowing down. That
is for the display and the log, not for steering.

## The two numbers that come out

The desired heading is the cheapest one found. The overall surprise is the cost of the walker's
actual heading minus the cost of the desired one. It is exactly zero when the walker is already
doing the best available thing, whether that is because nothing is there or because they have
already turned.

## What is not in it yet

Closing speed per object is passed through but nothing produces it, so every object closes at
walking speed. The walker's own speed comes from GPS when the app has location permission and
is outdoors, smoothed over one second, and is 1.4 m/s otherwise. There is no memory between
frames in the field itself, so the heading can flicker when a detection does. The screen holds
the last boxes and arrow for 0.8 s over an empty frame, which hides the flicker without fixing it.

## The walker's own wobble

The field keeps a running spread of the walker's heading over the ground, from the phone's
rotation vector, weighted so the last second counts most. That number is shown on screen and
logged. It is meant to set the turn tolerance σ from data, as the wobble times 2.07, which is
the band that holds 96 percent of a walker's headings.

That switch is off. On the first outdoor recording the one-second wobble has a median of 4
degrees, and it only passes 20 degrees through corners. As a tolerance, 4 degrees times 2.07
is about 8 degrees, and a 20 degree turn would then cost about 4 bits, against 0.7 bits for an
obstacle two seconds ahead. The field would hold its line into most things. Something in the
mapping is off, the window, the ratio, or the idea that sway and tolerance are the same
quantity, and that is a decision to make with the team rather than by tuning until it looks
right. The switch is `turnToleranceFromWobble` in the parameters, with a floor of 5 degrees
and the 60 degree search range as the ceiling.
