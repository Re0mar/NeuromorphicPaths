# Surprise and Tau for keeping a rider on a path

Draft math for the Group 9 project, NWI-IMC077. Version 14-09-2026.

This is a proposal, not a result. Every equation is either taken from a course source or derived by
us, and the last two sections say which. The lecture of 14-09 covers relative entropy, so notation
may shift after it.

---

## General Concept

A rider on a footpath or bike path has two things to aim for, each mapped to its own surprise number:

- **Goal L, the line.** The line they would like to hold, usually the middle. (Possibly refined to
  "the middle of the right half of the path.") Surprise grows with the square of the distance from
  the line. This is the target-hitting model in Vertegaal et al.
- **Goal E, the edge.** The edge they must not cross. Surprise grows as the gap to the edge closes
  at the current rate. This is the car-following model in Vertegaal et al., with the edge in place
  of the lead car. Written out, it is a time-to-contact quantity Tau.

Surprise measures how far the rider's current position sits from what they are aiming for, counted
in units of their own noise rather than in meters. A rider who wobbles 10 cm side to side is not
surprised by a 10 cm error. A rider who wobbles 2 cm is.

The two are computed, displayed and analyzed separately. They are not independent (both depend on x),
and a sum hides which channel drove a correction. That comparison is the thing the study needs.

"Rider" covers walking and cycling. The math is the same for both. What changes is the speed range,
roughly 1.4 m/s walking and 3 to 6 m/s cycling, and with it the size of the wobble.

---

## Symbols

| Symbol | Plain meaning | Unit |
|---|---|---|
| x | sideways position of the rider, 0 at the middle of the path | m |
| v | sideways speed, positive toward the right edge | m/s |
| w | usable half-width: half the path width minus half the rider's own width | m |
| μₜ | where the ideal line is, the mean of the goal distribution | m |
| σₜ | how much the line tolerates, the spread of the goal distribution | m |
| z | how many σₜ fit between the line and the edge | none |
| μₚ, σₚ | the rider's recent average position and wobble, the progress distribution | m |
| Δt | window length, what counts as one unit of time | s |
| δt | time between two analyzed camera frames | s |
| α | per-frame smoothing weight | none |
| d | gap between the rider and the edge they are moving toward | m |
| τ | Tau, time until that gap closes at the current rate | s |
| Uₗ, Uₑ | surprise for the line and for the edge | bits |
| IDₑ | index of difficulty for the edge, the logarithmic form | bits |
| T | lookahead time for the display | s |
| B | band around the line that counts as recovered | m |
| a, b | intercept and slope of the recovery-time fit | s, s/bit |
| W | path width | m |
| p | where the camera stands across the path, 0 at the left edge and 1 at the right | none |
| ψ | heading: angle between where the camera points and where the path goes, positive when the path heads off to the right | rad |
| u | forward speed along the direction of travel | m/s |

**Choosing z.** Welford's convention, on the week 2 slides, counts 96% of hits as inside the target.
For a normal distribution that is about ±2.07 standard deviations, the same figure MacKenzie uses
for effective width (4.133 σ across). So z = 2.07 and σₜ = w / 2.07.

---

## Camera input

This section is a draft. Nothing below has been run on a real recording yet.

The inputs come from a head-worn camera, with the head kept pointing forward. In each frame the
path detector marks the path, and the analysis fits a straight line along each of its two edges.
The code is in `analysis/`, and its README explains the fit. Three numbers come out of those lines.

- **p**, where the camera stands across the path. It comes from the two edges' slopes alone,
  plus a small correction when the camera is both tilted down and turned, which needs the focal
  length.
- **ψ**, the heading. It needs the camera's focal length, which the Neon ships with.
- **W**, the path width in meters. It needs the camera's height above the ground, which for a
  head-worn camera is the participant's eye height, measured once.

Forward speed **u** comes from outside the frame: GPS in the app, and for Neon recordings the
speed the ground moves past the camera. That second source is not built yet.

The rider is treated as a point, so the usable half-width is half the path width.

    w  = W / 2
    x  = (p − 0.5) · W
    v  = −u · sin ψ                     from one frame
    v  ≈ (x − x_previous) / δt          from two frames of video

The minus sign follows from the heading's direction. Positive ψ means the path heads off to the
right of where the camera points, so a rider moving where the camera points drifts left.

The one-frame form needs forward speed but works on a single image. The two-frame form needs no
speed but amplifies the frame-to-frame noise in x, so smooth it with the same α as below before
using it. Where both are available, their disagreement is a check on each other.

**Units.** Every quantity the surprise channels read is a ratio: x over σₜ, and the gap d over
the closing speed |v|. So p, x, w and v can all be counted in path widths (W = 1) without
changing any surprise value or threshold. Meters only matter when comparing wobble between paths
of different widths.

**Frames to skip.** The analysis marks a frame unreliable when an edge isn't straight enough to
trust: a curve, something covering the edge, or a path running off the side of the frame. It
also drops frames where the gyro shows the head turned away from the direction of travel. A
skipped frame leaves the running values where they were, and time moves on.

**Two-tone sidewalks.** Many Dutch sidewalks are a band of tiles beside a strip of brick. The
detector often marks only one of them, and then p and W describe that band. A p below 0 or above
1 then means standing beside the marked band, not off the sidewalk. ψ is unaffected, since every
line running along the path gives the same heading.

---

## Measuring the rider

With a camera, x and v are estimates rather than exact values, from the section above. What has
to be estimated on top of them is the rider's recent average and wobble. Both are kept as running
values so they can be updated every frame.

    α    = 1 − exp(−δt / Δt)
    e    = x − μₚ
    μₚ  ← μₚ + α·e
    σₚ² ← (1 − α)·(σₚ² + α·e²)

Deriving α from the window rather than fixing it per frame gives the same numbers at 60 and at 144
frames per second. Start with Δt = 1 s, since Vertegaal et al. measured spread over 1-second
windows.

Expect σₚ to grow with forward speed. The week 2 slides cite signal-dependent motor noise (Harris
and Wolpert 1998), and the college 3 readings include Schmidt et al. (1979), the older statement of
the same effect. Measure it per speed band rather than assuming one value.

<details>
<summary>Worked scenarios: does the estimator track the rider</summary>

For the estimator the question is not whether the rider is on the line but whether the running
values follow what the rider does. Two cases matter: a steady wobble, where the estimate should
settle onto the true wobble size, and a sudden push, where the mean lags and the wobble estimate
spikes.

Settings shared by every scenario in this document:

| Parameter | Value | Notes |
|---|---|---|
| path width | 2.0 m | bike path |
| rider width | 0.6 m | bike |
| w | 0.7 m | usable half-width |
| σₜ | 0.338 m | 0.7 / 2.07 |
| Δt | 1 s | window length |
| δt (worked examples) | 0.1 s | α = 1 − exp(−0.1) = 0.0952 |
| δt (60 frames per second) | 1/60 s | α = 0.0165, same arithmetic |
| δt (Neon scene camera) | 1/30 s | α = 0.0328, when every frame is analyzed |

**Scenario 1: steady wobble.** The rider swings 5 cm either side of the middle with a 1.2 s period,
x = 0.05·sin(2π·t / 1.2). The true spread of that signal is 0.05 / √2 = 0.0354 m.

*Snapshot.* One update from a settled state μₚ = 0.010 m, σₚ² = 0.00125 m² (σₚ = 0.0354 m), new
sample x = 0.045 m.

    e    = 0.045 − 0.010                    = 0.035
    μₚ  ← 0.010 + 0.0952·0.035              = 0.0133
    σₚ² ← 0.9048·(0.00125 + 0.0952·0.035²)  = 0.001237
    σₚ   = √0.001237                        = 0.0352

One sample moves the mean by a tenth of the error and leaves the wobble estimate where it was.

*Streaming.* From μₚ = 0, σₚ² = 0 at t = 0.

| t (s) | x (m) | e (m) | μₚ (m) | σₚ (m) |
|---|---|---|---|---|
| 0.1 | 0.0250 | 0.0250 | 0.0024 | 0.0073 |
| 0.2 | 0.0433 | 0.0409 | 0.0063 | 0.0139 |
| 0.3 | 0.0500 | 0.0437 | 0.0104 | 0.0184 |
| 0.4 | 0.0433 | 0.0329 | 0.0136 | 0.0200 |
| 0.5 | 0.0250 | 0.0114 | 0.0147 | 0.0193 |
| 0.6 | 0.0000 | −0.0147 | 0.0133 | 0.0189 |
| 1.0 | −0.0433 | −0.0427 | −0.0047 | 0.0303 |
| 2.0 | −0.0433 | −0.0499 | 0.0019 | 0.0322 |
| 3.0 | 0.0000 | −0.0099 | 0.0090 | 0.0322 |

The mean stays within 1.5 cm of the middle and the wobble estimate settles by about 2 s. It settles
at 0.032 m rather than 0.035 m because the 1 s window is close to the 1.2 s wobble period, so part of
each swing is read as the mean moving. A longer window would close that gap and lag a push more.

*Frame rate.* The same 3 s wobble sampled at 60 frames per second, with α = 0.0165, ends at
μₚ = 0.0096 m and σₚ = 0.0328 m against 0.0090 m and 0.0322 m at 10 samples per second. Deriving α
from the window is what makes those agree.

**Scenario 2: sudden push.** From the settled state of scenario 1 (μₚ = 0, σₚ = 0.0354 m), a push
moves the rider right at 0.5 m/s for 0.6 s, to x = 0.30 m, where they stay.

*Snapshot.* One update at t = 0.5 s from μₚ = 0.0433 m, σₚ² = 0.0050 m² (σₚ = 0.0707 m), new sample
x = 0.25 m.

    e    = 0.25 − 0.0433                    = 0.2067
    μₚ  ← 0.0433 + 0.0952·0.2067            = 0.0630
    σₚ² ← 0.9048·(0.0050 + 0.0952·0.2067²)  = 0.00820
    σₚ   = √0.00820                         = 0.0906

The mean moves 2 cm while the rider moved 5 cm, and the wobble estimate jumps by a third. A big
error inflates σₚ before μₚ has caught up.

*Streaming.*

| t (s) | x (m) | e (m) | μₚ (m) | σₚ (m) |
|---|---|---|---|---|
| 0.1 | 0.05 | 0.0500 | 0.0048 | 0.0367 |
| 0.2 | 0.10 | 0.0952 | 0.0138 | 0.0447 |
| 0.3 | 0.15 | 0.1362 | 0.0268 | 0.0584 |
| 0.4 | 0.20 | 0.1732 | 0.0433 | 0.0753 |
| 0.5 | 0.25 | 0.2067 | 0.0629 | 0.0938 |
| 0.6 | 0.30 | 0.2371 | 0.0855 | 0.1132 |
| 1.0 | 0.30 | 0.1589 | 0.1562 | 0.1369 |
| 2.0 | 0.30 | 0.0585 | 0.2471 | 0.1082 |
| 3.0 | 0.30 | 0.0215 | 0.2805 | 0.0704 |

The mean reaches 0.156 m at 1 s, about half of the 0.30 m, and 0.28 m by 3 s. The wobble estimate
peaks at 0.137 m around 1 s and decays back as the rider holds still. So a push shows up first as a
spike in σₚ and only later as a shift in μₚ. The analysis has to keep that in mind when it reads
σₚ as the rider's precision.

</details>

---

## Goal L: the line

How many noise units the rider is from where they should be. The distance from the line divided by
the tolerance is a signal-to-noise ratio, and surprise is half its square, converted to bits.

    sₗ = (μₚ − μₜ) / σₜ                        signal-to-noise ratio, Vertegaal Eq 9
    Uₗ = sₗ² / (2 ln 2)                         bits, Vertegaal Eq 15, equal spreads

Eq 15 assumes the rider's wobble equals the tolerance. The full form is the relative entropy between
two normal distributions, Vertegaal Eq 31. It also charges a rider who is centered but wobbly, which
is where precision enters on its own.

    Uₗ = log₂(σₜ / σₚ) + ((μₚ − μₜ)² + σₚ²) / (2 ln 2 · σₜ²) − 1 / (2 ln 2)

With σₚ = σₜ the extra terms cancel and Eq 15 comes back.

**Lookahead for the display.** Murray-Smith et al. (section 3.4.2) point out that an active
inference agent acts on predictions rather than on the current state. For what the rider sees,
compare the predicted position instead of the current one.

    x̂ = x + v·T
    sₗ = (x̂ − μₜ) / σₜ

Use μₚ for analysis and x̂ for the display.

<details>
<summary>Worked scenarios: holding the line, leaving the line</summary>

Same setting as the estimator scenarios: w = 0.7 m, μₜ = 0, σₜ = 0.338 m, Δt = 1 s, samples every
0.1 s. T = 0.5 s for the lookahead, as an example value only.

**Scenario 1: holding the line.** The settled wobble from the estimator's scenario 1.

*Snapshot.* μₚ = 0.012 m, σₚ = 0.035 m.

    sₗ = 0.012 / 0.338                      = 0.0355
    Uₗ = 0.0355² / 1.386                    = 0.0009 bits

A rider this close to the line carries no surprise on the line goal. The same rider through the full
form:

    Uₗ = log₂(0.338 / 0.035) + (0.012² + 0.035²) / (1.386 · 0.338²) − 0.721
       = 3.27 + 0.009 − 0.721               = 2.56 bits

That number is large because the log term rewards a wobble equal to the tolerance and charges a
rider whose wobble is smaller. Whether that is the intended direction of Eq 31 is an open check,
see *What still has to be checked*. With σₚ = σₜ the full form gives the same 0.0009 bits as Eq 15.

*Lookahead.* At x = 0.045 m moving right at v = 0.15 m/s, x̂ = 0.045 + 0.15·0.5 = 0.12 m, so
sₗ = 0.355 and Uₗ = 0.091 bits. The display sees a little more than the analysis does, because it
is looking half a second ahead.

*Streaming.* Six samples from the settled wobble, μₚ as the estimator produced it.

| t (s) | x (m) | μₚ (m) | sₗ | Uₗ (bits) |
|---|---|---|---|---|
| 2.1 | −0.0500 | −0.0030 | −0.009 | 0.0001 |
| 2.2 | −0.0433 | −0.0069 | −0.020 | 0.0003 |
| 2.3 | −0.0250 | −0.0086 | −0.025 | 0.0005 |
| 2.4 | 0.0000 | −0.0078 | −0.023 | 0.0004 |
| 2.5 | 0.0250 | −0.0047 | −0.014 | 0.0001 |
| 2.6 | 0.0433 | −0.0001 | 0.000 | 0.0000 |

The wobble itself never reaches the analysis. It is averaged away in μₚ, and Uₗ stays at zero to
four decimals.

**Scenario 2: leaving the line.** The push from the estimator's scenario 2, 0.5 m/s to the right for
0.6 s, then holding at x = 0.30 m.

*Snapshot.* At t = 2 s, μₚ = 0.20 m and σₚ = 0.15 m, the wobble estimate still inflated by the push.

    sₗ = 0.20 / 0.338                       = 0.591
    Uₗ = 0.591² / 1.386                     = 0.252 bits

Full form: log₂(0.338 / 0.15) + (0.20² + 0.15²) / (1.386 · 0.338²) − 0.721 = 1.17 + 0.39 − 0.72
= 0.85 bits.

*Lookahead.* At the end of the push, x = 0.30 m and v = 0.5 m/s, so x̂ = 0.30 + 0.25 = 0.55 m.
That is 1.63 tolerances out and Uₗ = 1.91 bits. The rider is still inside the usable width of
0.7 m, but the display already shows someone heading for the edge. The analysis number at the same
instant, from μₚ = 0.0855 m, is 0.046 bits. The two disagree by a factor of forty because μₚ lags
and x̂ leads.

*Streaming.*

| t (s) | x (m) | μₚ (m) | sₗ | Uₗ (bits) |
|---|---|---|---|---|
| 0.1 | 0.05 | 0.0048 | 0.014 | 0.0001 |
| 0.2 | 0.10 | 0.0138 | 0.041 | 0.0012 |
| 0.3 | 0.15 | 0.0268 | 0.079 | 0.0045 |
| 0.4 | 0.20 | 0.0433 | 0.128 | 0.0118 |
| 0.5 | 0.25 | 0.0629 | 0.186 | 0.0250 |
| 0.6 | 0.30 | 0.0855 | 0.253 | 0.0461 |
| 1.0 | 0.30 | 0.1562 | 0.462 | 0.154 |
| 2.0 | 0.30 | 0.2471 | 0.731 | 0.385 |
| 3.0 | 0.30 | 0.2805 | 0.830 | 0.497 |

Line surprise climbs for three seconds after the push has stopped, because μₚ is still catching up.
That is the right behavior for a goal that says where you should be on average. It is the wrong
behavior for a warning, which is the edge goal's job.

</details>

---

## Goal E: the edge

Vertegaal's car-following surprise (Eq 32) uses S for the gap to the lead car and N for how much the
gap closed in one window.

    U = (N / S)² / (2 ln 2)

For a path, the gap is to the edge the rider is moving toward.

    d    = w − x·sign(v)                     d ≤ 0 means already outside the tolerance
    1/τ  = |v| / d                           closing rate over gap, zero when moving parallel
    Uₑ   = (Δt / τ)² / (2 ln 2)              bits
    IDₑ  = log₂(Δt / τ + 1)                  bits, Vertegaal Eq 38 form

Closing distance over one window is |v|·Δt, and dividing by the gap gives Δt / τ. So Eq 32 is a Tau
equation. Working in 1/τ rather than τ matters. τ is infinite whenever the rider moves parallel to
the edge, and a heavy-tailed quantity breaks the squared cost. 1/τ is zero there.

Uₑ is zero whenever the rider is not moving toward an edge. Sitting still next to the edge is Goal
L's job.

**Two numbers that line up.**

- The week 1 danger rule, gap over change in gap below 1, is τ < Δt. That is IDₑ = 1 bit, or
  Uₑ ≈ 0.72 bits.
- In Vertegaal's car-following data, drivers ran out of capacity near an ID of 1.1 bits.

The path task will have its own value. Finding it is part of the study.

**Tau-dot.** The rate of change of Tau, written τ̇, is Lee's control variable. With τ = gap / closing
rate, a constant deceleration that stops exactly at the edge gives τ̇ = −0.5. Above −0.5 the edge is
crossed with speed left over, below it the correction stops short. Sign conventions differ between
papers. The 0.5 is the part that stays fixed.

<details>
<summary>Worked scenarios: staying clear of the edge, heading for it</summary>

Same setting: w = 0.7 m, Δt = 1 s, samples every 0.1 s. Positive v is toward the right edge, so for
v > 0 the gap is d = 0.7 − x and for v < 0 it is d = 0.7 + x. The edge goal reads x and v directly,
not μₚ, so nothing here lags.

**Scenario 1: staying clear.** The settled wobble, with v from the wobble itself.

*Snapshot.* x = 0.0433 m, v = 0.131 m/s to the right.

    d    = 0.7 − 0.0433                     = 0.657 m
    1/τ  = 0.131 / 0.657                    = 0.199 per second, so τ = 5.0 s
    Uₑ   = (1 · 0.199)² / 1.386             = 0.029 bits
    IDₑ  = log₂(0.199 + 1)                  = 0.262 bits

Five seconds from the edge at the current rate. Far below the 0.72 bit alarm.

*Streaming.*

| t (s) | x (m) | v (m/s) | d (m) | τ (s) | Uₑ (bits) | IDₑ (bits) |
|---|---|---|---|---|---|---|
| 2.1 | −0.0500 | 0.000 | 0.650 | ∞ | 0.000 | 0.000 |
| 2.2 | −0.0433 | 0.131 | 0.743 | 5.7 | 0.022 | 0.234 |
| 2.3 | −0.0250 | 0.227 | 0.725 | 3.2 | 0.071 | 0.393 |
| 2.4 | 0.0000 | 0.262 | 0.700 | 2.7 | 0.101 | 0.458 |
| 2.5 | 0.0250 | 0.227 | 0.675 | 3.0 | 0.081 | 0.418 |
| 2.6 | 0.0433 | 0.131 | 0.657 | 5.0 | 0.029 | 0.262 |

At the turning point of the swing v is zero, so 1/τ is zero and Uₑ is zero, even though the rider is
5 cm off center. That is the parallel case, and it is why the math works in 1/τ. Uₑ peaks mid-swing
at 0.10 bits, where the rider is moving fastest, and never comes near the alarm.

**Scenario 2: heading for the edge.** The push, 0.5 m/s to the right from x = 0.

*Snapshot.* x = 0.25 m, v = 0.5 m/s.

    d    = 0.7 − 0.25                       = 0.45 m
    1/τ  = 0.5 / 0.45                       = 1.11 per second, so τ = 0.9 s
    Uₑ   = (1 · 1.11)² / 1.386              = 0.891 bits
    IDₑ  = log₂(1.11 + 1)                   = 1.08 bits

Under a second from the edge. Above the 0.72 bit alarm threshold, and close to the 1.1 bits where
Vertegaal's drivers ran out of capacity.

*Streaming.* θ₁ = 0.72 bits switches the alarm on.

| t (s) | x (m) | v (m/s) | d (m) | τ (s) | Uₑ (bits) | IDₑ (bits) | alarm |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.05 | 0.5 | 0.65 | 1.30 | 0.427 | 0.823 | off |
| 0.2 | 0.10 | 0.5 | 0.60 | 1.20 | 0.501 | 0.875 | off |
| 0.3 | 0.15 | 0.5 | 0.55 | 1.10 | 0.596 | 0.933 | off |
| 0.4 | 0.20 | 0.5 | 0.50 | 1.00 | 0.721 | 1.000 | at threshold |
| 0.5 | 0.25 | 0.5 | 0.45 | 0.90 | 0.891 | 1.078 | on |
| 0.6 | 0.30 | 0.5 | 0.40 | 0.80 | 1.127 | 1.170 | on |

The alarm fires 0.4 to 0.5 s into the push, with the rider still 45 cm inside the usable width. The
line goal at the same instant reads 0.025 bits, because its mean is still catching up. That gap is
the whole argument for two channels. Once the push stops and v drops to zero, Uₑ drops to zero with
it, and holding still at x = 0.30 m becomes the line goal's problem.

</details>

---

## Why the two are not summed

Surprise in bits adds for independent events, so Uₗ + Uₑ is tempting. These two are not
independent, since both depend on x, and a sum hides which channel drove a correction. That
comparison is the thing the study needs.

- Goal L says where you should be. Smooth, and active whether or not you are moving.
- Goal E says how soon it goes wrong. Silent while parallel, sharp as an edge approaches.

---

## From surprise to what the rider sees

Neither paper derives a display. Vertegaal et al. (section 6.5) used a fixed-width feedback bar with
no display equations, and Murray-Smith et al. (section 5.2) note that users struggle with displays
of uncertainty. The mapping below is ours.

Perception is logarithmic (week 2), so a brightness driven directly by the quadratic U would look
flat and then jump. The continuous line therefore uses the logarithmic form, and the alarm fires on
U itself.

    line intensity = clamp( log₂(|sₗ| + 1) / log₂(z + 1), 0, 1 )      0 on the line, 1 at the edge

    alarm on   when Uₑ > θ₁      start θ₁ = 0.72 bits, τ = Δt
    alarm off  when Uₑ < θ₀      start θ₀ = 0.32 bits, τ = 1.5·Δt

The gap between θ₁ and θ₀ stops the alarm flickering. Both are starting values to tune.

---

## What the study can test

**Correction onset.** Riders start correcting at a consistent Uₑ or IDₑ, not at a consistent gap d.
Fit both and compare.

**Recovery time.** Week 2 and Vertegaal Eq 21 and 22 model a correction as removing a fixed fraction
of the remaining normalized error per unit time, which makes recovery time logarithmic in the
starting error.

    tᵣ = a + b · log₂( |x₀ − μₜ| / B + 1 )

This is Fitts' law applied to getting back on the line. b is seconds per bit, and 1/b is an index
of performance in bits per second.

**Predictability over size.** With scripted sideways pushes, vary push size and push probability
independently. Correction latency should follow how improbable the push was, not how large it was.

**Dependence on the display.** Remove the line after exposure and check whether performance falls
below the rider's own baseline.

---

## What still has to be checked

- With a camera, σₚ measures the rider's wobble plus the camera's own noise: the mask's jitter
  from frame to frame and the head's movement. Measure that noise floor once, for example
  standing still on a straight path, and subtract its variance before reading σₚ as the rider's
  precision.
- v = −u · sin ψ assumes the camera points where the rider is going. Head turns break that,
  which is why those frames are dropped. The gyro threshold for dropping them is a starting
  value, 4° walking and 2° cycling, still to tune.
- Forward speed from phone GPS is noisy at walking pace, because the position wobbles by a few
  meters between one-second fixes. Android's own speed value, derived from the satellite signals,
  is steadier. The Neon records no GPS, so recordings need speed from the video instead.
- The lookahead time T could be measured rather than chosen: where the rider's gaze lands on the
  path, divided by forward speed. The Neon records gaze.

- Every Vertegaal equation number above was read from the arXiv HTML through a summary, not from
  the PDF. Check each form and number against the PDF before it goes into the report or the
  presentation.
- Reading Eq 32's N as closing distance over one window is our reading, not the paper's wording.
- The gap d, the lookahead x̂, both alarm thresholds and the line intensity mapping are ours.
- z = 2.07 is a target-hitting convention applied to a path. It may not be the right tolerance for
  a path edge.
- τ̇ = −0.5 comes from constant-deceleration algebra, not from a course source.
- Eq 31 is the relative entropy between two normal distributions, which the 14-09 lecture covers.
  Check the notation against the lecture form after it.
- The argument order of Eq 31. As written, the log₂(σₜ / σₚ) term charges a rider whose wobble is
  smaller than the tolerance, and the worked example gives 2.6 bits for a tight rider sitting on the
  line. Relative entropy is not symmetric, and the other order charges a wide rider instead. Which
  order the paper uses, and which one the study wants, both need settling before Eq 31 is used for
  anything beyond Eq 15.

---

## Sources

- Vertegaal, R., Merritt, T., Greenberg, S., Tarun, A. P., Li, Z. and Fountas, Z. (2025).
  Interactive inference: A neuromorphic theory of human-computer interaction. arXiv 2502.05935.
  Equations 9, 15, 21, 22, 31, 32 and 38. Sections 6.2 and 6.5.
- Murray-Smith, R., Williamson, J. H. and Stein, S. (2025). Active inference and human-computer
  interaction. arXiv 2412.14741. Sections 3.4.2 and 5.2.
- Course slides, weeks 1 and 2: surprise as negative log probability, trueness and precision, Tau
  and the danger rule, surprisal of a normal distribution, relaxation and Fitts' law, Welford's 96%
  convention, signal-dependent noise.
- Fitts (1954), Welford (1968), Shannon (1949), Schmidt et al. (1979), Harris and Wolpert (1998),
  MacKenzie (1992). Named through the slides and the college 3 reading list. Not yet read for this
  draft.
- Lee's Tau papers, for τ̇. Not a course reading, not yet read.
