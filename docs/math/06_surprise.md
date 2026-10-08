[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [5. Following things over time](05_following_over_time.md) · Next: [7. The field](07_the_field.md)

# 6. Surprise

Surprise is what the planner tries to keep low. Every obstacle point adds some, depending on where
the walker would be standing. Two terms do the pricing. The professor's term compares how much a
reading wobbles with how much room is left. Our contact term asks how likely the body is to
actually touch the thing. Walls get extra wobble before either term sees them. Section 7 then adds
everything up over time and sideways position.

## His surprise potential

A reading is surprising when its wobble is large compared with the room left. A post 2 m away
that wobbles a centimeter barely matters. One that wobbles half a meter does, and the planner
steers around the wobble. This is the professor's **surprise potential**, written
$`{\color{purple}{U}}`$.

In its general form it's "how far off you are, in units of noise, squared and halved". When the
thing is something to stay away from, the professor flips the fraction. The gap goes on the bottom
and its wobble goes on top. A big, steady gap means low surprise. A small gap that's wobbling
means high surprise. That flipped version is the **avoidance form**, and it's the one the planner
uses.

Think of carrying a full cup of coffee through a doorway. Wide doorway and steady hands, no
problem. Narrow doorway or shaky hands, and the stress climbs fast. What matters is the shake
compared with the room. The comparison stops working at a post that's perfectly still. You'd still
worry about walking into it, but this term barely does, because its shake is tiny. That's the gap
our contact term fills, further down.

> [!NOTE]
> **Ingredients**
> - d, the center distance from a candidate sideways position to an obstacle point, at one
>   future moment (section 7).
> - The footprint radius r = 0.35 m (`server/nav/walker.py`), so $`{\color{teal}{S}} = d - r`$.
> - $`{\color{orange}{N}}`$ for the point's group (section 5), tripled for walls (below).
> - The floor under $`{\color{teal}{S}}`$, $`\varepsilon_S = 0.06`$ m, the floor under
>   $`{\color{orange}{N}}`$, $`\varepsilon_N = 10^{-6}`$ m, and the cap $`U_{\max} = 2.0 \times 10^4`$
>   (`server/nav/planner/config.py`).

The general form, from the paper:

```math
{\color{purple}{U}} = \frac{1}{2}\left(\frac{{\color{teal}{S}}}{{\color{orange}{N}}}\right)^2
```

The avoidance form, from the car-following lecture, with $`\Delta{\color{teal}{S}}`$ the spread of
the gap over a 1 s window:

```math
{\color{purple}{U}} = \frac{1}{2}\left(\frac{\Delta{\color{teal}{S}}}{{\color{teal}{S}}}\right)^2
```

What the planner computes, per second, for one point:

```math
{\color{purple}{U}} = \min\!\left(\frac{1}{2}\left(\frac{\max({\color{orange}{N}},\ \varepsilon_N)}{\max({\color{teal}{S}},\ \varepsilon_S)}\right)^2,\ U_{\max}\right), \qquad {\color{teal}{S}} = d - r
```

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| d | Center distance from a candidate position to a point | m | walker | `server/nav/planner/field.py` |
| r | Footprint radius, 0.35 | m | walker | `server/nav/walker.py` |
| $`{\color{teal}{S}}`$ | Clearance from the footprint's edge, negative before the floor | m | walker | `server/nav/planner/surprise.py` |
| $`\Delta{\color{teal}{S}}`$ | In the lecture, the spread of the gap over 1 s | m | none | none, the code uses $`{\color{orange}{N}}`$ |
| $`{\color{orange}{N}}`$ | Noise scale, how much the clearance wobbles | m | none | `server/nav/scene/history.py` |
| $`\varepsilon_S`$ | Floor under $`{\color{teal}{S}}`$, 0.06 | m | none | `server/nav/planner/config.py` |
| $`\varepsilon_N`$ | Floor under $`{\color{orange}{N}}`$, $`10^{-6}`$ | m | none | `server/nav/planner/config.py` |
| $`U_{\max}`$ | Cap on one point's surprise, $`2.0 \times 10^4`$ | per second | none | `server/nav/planner/config.py` |
| $`{\color{purple}{U}}`$ | Surprise of one point, as a rate | natural-log units per second | none | `server/nav/planner/surprise.py` |

| Step | Expression | Why |
|---|---|---|
| 1 | $`p(x) \propto e^{-{\color{purple}{U}}}`$, $`{\color{purple}{U}} = \dfrac{(x - \mu)^2}{2\sigma^2}`$ | A bell curve is e to the minus something. Minus its log hands back that something, with the constant dropped |
| 2 | $`{\color{teal}{S}} = x - \mu`$, $`{\color{orange}{N}} = \sigma`$, so $`{\color{purple}{U}} = \tfrac12({\color{teal}{S}}/{\color{orange}{N}})^2`$ | The error is the signal and the spread is the noise |
| 3 | $`{\color{purple}{U}} = \tfrac12(\Delta{\color{teal}{S}}/{\color{teal}{S}})^2`$ | Avoiding a target swaps signal and noise. The gap is what keeps you safe, so it goes on the bottom |
| 4 | $`\Delta{\color{teal}{S}} \to {\color{orange}{N}}`$, $`{\color{teal}{S}} \to d - r`$ | Our wobble over 0.5 s, and the gap from each candidate position |
| 5 | $`\max({\color{teal}{S}}, 0.06)`$, $`\max({\color{orange}{N}}, 10^{-6})`$ | The $`{\color{teal}{S}}`$ floor keeps the fraction finite when the walker touches the point. The $`{\color{orange}{N}}`$ floor keeps a zero $`{\color{orange}{N}}`$ from reading as exactly zero. Live, the scene already floors $`{\color{orange}{N}}`$ at 0.01 m |
| 6 | $`\min(\cdot, 2.0 \times 10^4)`$ | An upper limit on one point's surprise |
| 7 | multiply by $`\Delta t = 0.1`$ s in the path search | It's a rate per second (section 7) |

**Units.** This is a natural log, so the values are nats. The planner adds every term up in nats
and divides the path's total by $`\ln 2`$ once, at the end of the path search (section 8), so the
cost it sends is in bits. One bit is $`\ln 2 = 0.693`$ nats. The figures on this page are nats.

Some real numbers, per second, with the per-step charge at $`\Delta t = 0.1`$ s:

| Situation | Working | $`{\color{purple}{U}}`$ per second | Per 0.1 s step |
|---|---|---|---|
| $`{\color{teal}{S}} = 1`$ m, $`{\color{orange}{N}} = 0.1`$ m | $`\tfrac12(0.1/1)^2`$ | 0.005 | 0.0005 |
| $`{\color{teal}{S}} = {\color{orange}{N}} = 0.2`$ m | $`\tfrac12(1)^2`$ | 0.5 | 0.05 |
| Steady post, $`{\color{orange}{N}} = 0.0337`$ m, $`{\color{teal}{S}} = 0.5`$ m | $`\tfrac12(0.0674)^2`$ | 0.00227 | 0.000227 |
| Same post at contact, $`d = 0.20`$, so $`{\color{teal}{S}} = -0.15`$, floored to 0.06 | $`\tfrac12(0.0337/0.06)^2 = \tfrac12(0.5617)^2`$ | 0.158 | 0.0158 |
| Same post, walking straight through it, $`d = 0`$, $`{\color{teal}{S}} = -0.35`$, floored to 0.06 | the same | 0.158 | 0.0158 |
| Cap | $`\tfrac12 \cdot 200^2 = 20{,}000`$, reached at $`{\color{orange}{N}}/{\color{teal}{S}} = 200`$, so $`{\color{orange}{N}} = 12`$ m at the floor | 20,000 | 2,000 |

The 0.0337 m is the median noise of groups in the corridor within 2 m on the classroom walk
(`server/tests/test_planner_pipeline.py`). On `pixel_walk_3` that median is 0.0958 m. The last two rows are the
problem the contact term fixes. Once the gap hits the floor, brushing past a steady post 6 cm away
and walking straight into it cost exactly the same, about 0.16 per second.

**The avoidance form elsewhere.** The alarm's number in section 9 also uses the avoidance form. It
puts one second over the time to contact, $`{\color{red}{\tau}}`$, in place of
$`\Delta{\color{teal}{S}}/{\color{teal}{S}}`$, and converts to bits:
$`{\color{purple}{U}} = (1\text{ s}/{\color{red}{\tau}})^2 / (2\ln 2)`$. At 1 s to contact that's
$`1/1.3863 = 0.72`$ bits. Reading $`\Delta{\color{teal}{S}}/{\color{teal}{S}}`$ as 1 s over the time to
contact is our reading of the deck, not his wording.

> [!WARNING]
> - **Everything inside the floor costs the same.** Below $`{\color{teal}{S}} = 0.06`$ m, including
>   any overlap, the surprise stops growing. A post with a small $`{\color{orange}{N}}`$ is cheap to
>   walk into.
> - **Only one point per group counts**, the one nearest the walker (section 4).
> - **$`{\color{teal}{S}}`$ uses the footprint**, r = 0.35 m, not the body.
> - **No closing speed enters.** The only motion is the walker's own advance (section 7).
> - **On "1 s over time to contact".** The deck calls $`\Delta{\color{teal}{S}}/{\color{teal}{S}}`$
>   "tau" (College 5, PDF page 71 notes) and says $`\Delta{\color{teal}{S}}`$ shows how fast the
>   cars move toward or away from each other (PDF page 65 notes). Reading
>   $`\Delta{\color{teal}{S}}`$ as a closing speed over 1 s gives 1 s over the time to contact. Two
>   catches. $`\Delta{\color{teal}{S}}`$ is a standard deviation, so it grows when the gap opens as
>   well as when it closes. And the usual tau in time-to-contact work is gap over closing speed,
>   the inverse of the deck's ratio.

> [!IMPORTANT]
> The surprise potential is the professor's. The Stationary Interaction paper builds it from the
> bell curve (§3.1, Eq. 1, p. 4), gives the bits conversion (§3.2, Eq. 2, p. 4) and writes it as
> half the squared ratio of signal to noise (§4.1, Eq. 8, pp. 7 to 8). College 4, PDF page 12, has
> the same $`\tfrac12 s^2`$ with $`s = {\color{teal}{S}}/{\color{orange}{N}}`$. The avoidance form,
> with signal and noise swapped, comes from the car-following study on College 5, PDF pages 61 to
> 71, and the formula itself is on PDF pages 62 and 65. The floor under $`{\color{teal}{S}}`$, the floor under $`{\color{orange}{N}}`$, the cap,
> and the 1 s in the alarm's version: the code marks these as his.

## Walls get three times the noise

A wall or a tree trunk is worth avoiding from further out than a bollard. So before any term sees
a point, a wall's $`{\color{orange}{N}}`$ is multiplied by 3. The planner then reads the wall as less
certain and gives it more room.

It's like the extra space you leave a parked truck compared with a fire hydrant, even when both
stand still. The comparison stops at size. The wall flag only looks at height (section 4), so a
tall person standing still gets the same treatment.

> [!NOTE]
> **Ingredients**
> - The wall flag from section 4, set when a square's tallest point is 1.5 m or more.
> - $`{\color{orange}{N}}`$ from section 5.
> - The multiplier $`\kappa = 3.0`$ (`server/nav/planner/config.py`).

```math
{\color{orange}{N}}_{\text{eff}} = \begin{cases} \kappa\,{\color{orange}{N}} & \text{wall} \\ {\color{orange}{N}} & \text{otherwise} \end{cases}
```

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`\kappa`$ | Wall multiplier, 3.0 | none | none | `server/nav/planner/config.py` |
| $`{\color{orange}{N}}_{\text{eff}}`$ | The noise every term reads | m | none | `server/nav/planner/field.py` |

With $`{\color{orange}{N}} = 0.0337`$ m, a wall's $`{\color{orange}{N}}_{\text{eff}}`$ is 0.1011 m.

1. His term at $`{\color{teal}{S}} = 0.5`$ m goes from $`\tfrac12(0.0674)^2 = 0.00227`$ to
   $`\tfrac12(0.2022)^2 = 0.0204`$ per second. That's $`\kappa^2 = 9`$ times.
2. The contact term's spread, below, goes from $`\sqrt{0.0337^2 + 0.10^2} = 0.1055`$ m to
   $`\sqrt{0.1011^2 + 0.10^2} = 0.1422`$ m.
3. A wall at the floor, $`{\color{orange}{N}} = 0.01`$ m, gets 0.03 m.

> [!WARNING]
> The multiplier acts on $`{\color{orange}{N}}`$, not on the cost. In his term it's squared, so the
> cost goes up 9 times. In the contact term it widens the spread, which lowers the cost right at
> the wall and raises it further out.

> [!TIP]
> Ours. The code's comment: "A wall is worth avoiding further out than a post."

## The contact term

His term prices wobble. It doesn't ask the obvious question: would the body hit the thing? The
contact term asks exactly that. Two things make the answer uncertain. The reading wobbles
($`{\color{orange}{N}}`$), and a person drifts a little to the side of the line the arrow asks for,
which we call **sway**. Together they blur where the body will really be, so there's some
**contact probability** p. The cost is the surprise of getting past cleanly, minus the log of the
chance of not touching.

Think of squeezing past someone in a narrow hallway with a backpack on. You know roughly where
you'll be, but you drift a bit, so there's a chance you brush them, and it gets big fast as the
gap closes. The comparison stops at how people actually walk. A real person corrects as they go,
and here the drift is a fixed 0.10 m guess.

> [!NOTE]
> **Ingredients**
> - d, the center distance from a candidate position to a point, at one future moment (section 7).
> - The body half-width h = 0.30 m, so the body's gap is $`{\color{teal}{S}}_b = d - h`$
>   (`server/nav/planner/config.py`).
> - $`{\color{orange}{N}}_{\text{eff}}`$, with the wall multiplier.
> - The sway s = 0.10 m. It's an assumption, not a measurement. `wifi_run_2` (section 12) is the
>   one recorded walk with the arrow shown, six scorable turns, and the sway hasn't been measured
>   from it yet.
> - The walking speed $`v_w = 1.4`$ m/s and the cap $`C_{\max} = 50`$.

```math
{\color{teal}{S}}_b = d - h, \qquad \sigma = \sqrt{{\color{orange}{N}}_{\text{eff}}^2 + s^2}, \qquad p = \Phi\!\left(-\frac{{\color{teal}{S}}_b}{\sigma}\right)
```

```math
{\color{purple}{U}}_c = \frac{\min\!\left(-\ln \Phi\!\left(\dfrac{{\color{teal}{S}}_b}{\sigma}\right),\ C_{\max}\right)}{t_{\text{pass}}}, \qquad t_{\text{pass}} = \frac{2h}{v_w} = \frac{0.60}{1.4} = 0.4286\text{ s}
```

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| h | Body half-width, 0.30 | m | walker | `server/nav/planner/config.py` |
| $`{\color{teal}{S}}_b`$ | Gap from the body's edge, negative on overlap | m | walker | `server/nav/planner/contact.py` |
| s | Sway, how far a walker drifts sideways, one standard deviation, 0.10 | m | walker | `server/nav/planner/config.py` |
| $`\sigma`$ | Total blur on the gap | m | walker | `server/nav/planner/contact.py` |
| $`\Phi`$ | Bell-curve area to the left of a value | none | none | `server/nav/planner/contact.py` |
| p | Chance the body touches the point | probability | none | not stored, see below |
| $`C_{\max}`$ | Cap on one point's surprise, 50 | nats | none | `server/nav/planner/config.py` |
| $`t_{\text{pass}}`$ | Time the body takes to walk past an obstacle | s | none | `server/nav/planner/contact.py` |
| $`{\color{purple}{U}}_c`$ | Contact surprise of one point, as a rate | nats per second | none | `server/nav/planner/contact.py` |

| Step | Expression | Why |
|---|---|---|
| 1 | $`{\color{teal}{S}}_b = d - h`$ | The question is about the body, not the steering footprint |
| 2 | $`\sigma^2 = {\color{orange}{N}}_{\text{eff}}^2 + s^2`$ | Two independent wobbles add as squares |
| 3 | $`p = \Phi(-{\color{teal}{S}}_b/\sigma)`$ | The chance the blurred gap is below zero |
| 4 | $`-\ln(1 - p) = -\ln\Phi({\color{teal}{S}}_b/\sigma)`$ | The surprise of getting past without touching |
| 5 | $`\min(\cdot, 50)`$ | A cap. At s = 0.10 m it never binds |
| 6 | divide by $`t_{\text{pass}} = 2h/v_w`$ | Makes it a rate. The path search multiplies by $`\Delta t`$, so walking past one obstacle adds up to about one surprise whatever $`\Delta t`$ is |
| 7 | add beside his term, weight 1 | It's separate evidence about the same scene |

The code never stores p. It goes straight to the surprise with `scipy.special.log_ndtr`, which
stays accurate far out in the tail.

With $`{\color{orange}{N}} = 0.0337`$ m the blur is $`\sigma = \sqrt{0.001136 + 0.01} = 0.10553`$ m.
Some real numbers:

| Situation | $`\frac{{\color{teal}{S}}_b}{\sigma}`$ | $`-\ln\Phi`$ | $`{\color{purple}{U}}_c`$ per second | His $`{\color{purple}{U}}`$ per second |
|---|---|---|---|---|
| Just touching, $`{\color{teal}{S}}_b = 0`$ | 0 | $`\ln 2 = 0.6931`$ | $`0.6931/0.4286 = 1.617`$ | |
| One spread clear, $`{\color{orange}{N}} = 0`$, $`{\color{teal}{S}}_b = 0.10`$ | 1 | $`-\ln 0.841345 = 0.1728`$ | 0.403 | |
| Steady post, 10 cm overlap, $`d = 0.20`$ | $`-0.10/0.10553 = -0.948`$ | $`-\ln 0.1717 = 1.762`$ | 4.11 | 0.158 |
| Walking straight through it, $`d = 0`$ | $`-0.30/0.10553 = -2.843`$ | $`-\ln 0.002236 = 6.10`$ | 14.2 | 0.158 |
| Deepest overlap, $`{\color{orange}{N}} = 0`$, $`d = 0`$ | $`-3`$ | $`-\ln\Phi(-3) = 6.608`$ | 15.42 | |

In the 10 cm overlap row the chance of touching is $`p = \Phi(0.948) = 0.828`$. In the walk-through
row the contact term charges $`14.2 \cdot 0.1 = 1.42`$ per 0.1 s step against his 0.0158, about 90
times more. The most any one point can reach at s = 0.10 m is 6.61 before the division and 15.4
per second after it, far under the cap of 50. That's the bound at $`{\color{orange}{N}} = 0`$. The
scene never sends an $`{\color{orange}{N}}`$ below 0.01 m, so live the most is
$`-\ln\Phi(-0.30/\sqrt{0.01^2 + 0.10^2}) = -\ln\Phi(-2.985) = 6.56`$, or 15.3 per second. The cap only binds for a sway under about 0.031 m.

> [!WARNING]
> - **The blur is a bell curve**, and the reading's wobble and the walker's sway are treated as
>   independent.
> - **The sway is a guess**, 0.10 m, until someone walks by the arrow and we can measure it.
> - **One point is one possible contact.** Consecutive time slices that overlap the same obstacle
>   are added as separate contacts.
> - **Groups add** (section 7), and a wall is many 0.25 m squares. Walking beside a wall costs
>   about 1.4 to 1.5 times one square at the same gap. His term overcounts a wall more, 2 to 4
>   times.
> - **Units.** This is a natural log, like his term. The planner converts the summed path cost to
>   bits once, in section 8.

> [!TIP]
> Ours, added beside his term. Under his term a steady post costs almost nothing, about 0.16 per
> second even at contact, because its $`{\color{orange}{N}}`$ is tiny and its gap is floored at
> 0.06 m. Walking straight into it was barely worse than passing close by. The contact term costs
> 14.2 per second there. It enters the field at weight 1. The 6.5 that shows up elsewhere is the
> sideways effort weight (section 8), not a weight on this term. The lecture measures crash risk
> in a similar shape, as the surprise of not crashing, $`-\log_2(1 - p)`$ (College 5, PDF page 68),
> but the contact term itself is ours.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [5. Following things over time](05_following_over_time.md) · Next: [7. The field](07_the_field.md)
