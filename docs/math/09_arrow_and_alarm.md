[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [8. Planning a path](08_planning_a_path.md) · Next: [10. How much the scene shaped the plan](10_scene_shaped_the_plan.md)

# 9. The arrow and the alarm

The planner hands the walker two things. The **arrow** says which way to go: it points from where
the walker is now to where the plan has them one second from now. The **alarm** says something is
about to be hit: it looks only at the strip straight ahead that the body would sweep, works out how
long until the walker reaches the nearest thing in it at walking pace, and goes off under 0.7 s.
That time is the **time to contact**, $`{\color{red}{\tau}}`$. The two are separate on purpose. The
alarm never reads the plan.

Riding a bike, you steer by looking a few meters down the road, not at your front wheel. The arrow
works the same way, aiming at the plan one second out. The alarm is more like a car's parking
sensor, which beeps when something is close in front. Where it stops working: a parking sensor
measures how fast the gap is actually closing, and ours assumes the walker is moving at 1.4 m/s even
when they're standing still.

> [!NOTE]
> **Ingredients**
> - The plan's sideways offsets $`o_k`$, one per slice, from section 8.
> - The lookahead $`T_h`$ = 1.0 s, the time step $`\Delta t`$ = 0.1 s and the walking speed $`v_w`$ =
>   1.4 m/s, from `server/nav/planner/config.py`.
> - The obstacle points' positions on the floor, $`x_i`$ sideways and $`y_i`$ forward, and each one's
>   clearance $`{\color{teal}{S}}_i`$ measured from the 0.35 m footprint. All from the scene, section 4.
> - The body half-width $`h`$ = 0.30 m, the alarm threshold $`T_a`$ = 0.7 s and the hold $`H`$ = 0.5 s,
>   from `server/nav/planner/config.py`.
> - Frame timestamps, for the hold.

## The formulas

**The arrow.** The lookahead slice is $`k_h`$. The arrow's angle compares how far sideways the plan
goes by then with how far forward the walker gets. Positive is right.

```math
k_h = \mathrm{round}(T_h / \Delta t) = 10, \qquad \theta = \mathrm{atan2}\big(o_{k_h} - o_0,\ k_h\,\Delta t\,v_w\big)
```

**The alarm corridor.** Only points ahead and within the body's half-width to either side count.

```math
\mathcal{C} = \lbrace\, i : y_i > 0,\ \lvert x_i \rvert \le h \,\rbrace
```

**Time to contact and the raise rule.** Divide the smallest clearance in the corridor by walking
speed. Ask for the alarm when that's under the threshold.

```math
{\color{red}{\tau}}_c = \frac{\min_{i \in \mathcal{C}} {\color{teal}{S}}_i}{v_w}, \qquad \text{raise} = \big(\mathcal{C} \ne \emptyset\big) \wedge \big({\color{red}{\tau}}_c < T_a\big)
```

**The hold.** Once raised at time $`t_r`$, the alarm stays shown for at least $`H`$.

```math
\text{shown}(t) = \text{raise}(t) \ \vee \ \big(t_r \text{ is set} \wedge t - t_r < H\big)
```

$`t_r`$ is set on the first raising frame. It's cleared on a quiet frame once the hold has run out,
or by a timestamp earlier than $`t_r`$. Only frame timestamps count, never the laptop's clock.

**Avoidance surprise in bits.** The course's way of saying how soon the walker reaches the nearest
thing in the way, as a surprise. One second over the time to contact, squared, halved, and turned
into bits. It colors the path (section 14). The alarm doesn't use it, but the path's full red is
this same formula at the alarm's threshold, so the two line up.

```math
{\color{red}{\tau}} = \frac{\max\big(\min_{i \in \mathcal{C}} {\color{teal}{S}}_i,\ \varepsilon_S\big)}{v_w}, \qquad {\color{purple}{U}} = \frac{(1\ \text{s} / {\color{red}{\tau}})^2}{2 \ln 2}
```

$`{\color{purple}{U}}`$ is 0 when the corridor is empty. Writing it as $`\tfrac12 (1\ \text{s}/{\color{red}{\tau}})^2`$
natural-log units divided by $`\ln 2`$ gives the same number, and that's how the code computes it,
through the planner's one conversion to bits. The path's summed cost goes through the same
conversion (section 8). The scene information in section 10 is in bits too, because its divergence
uses $`\log_2`$.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`o_k`$ | plan's sideways offset on slice $`k`$, positive right | m | walker ground frame | `server/nav/planner/heading.py` |
| $`T_h`$ | lookahead, 1.0 s | s | time | `server/nav/planner/config.py` |
| $`k_h`$ | lookahead slice, 10 | none | time | `server/nav/planner/heading.py` |
| $`\theta`$ | arrow angle, sent as `lookahead_heading_radians` | rad | walker ground frame | `server/nav/planner/heading.py` |
| $`v_w`$ | walking speed, 1.4 m/s | m/s | walker ground frame | `server/nav/planner/config.py` |
| $`x_i, y_i`$ | point's sideways and forward position | m | walker ground frame | `server/nav/planner/alarm.py` |
| $`h`$ | body half-width, 0.30 m | m | walker ground frame | `server/nav/planner/config.py` |
| $`\mathcal{C}`$ | points in the corridor | none | walker ground frame | `server/nav/planner/alarm.py` |
| $`{\color{teal}{S}}_i`$ | clearance, gap from the 0.35 m footprint's edge to point $`i`$ | m | walker ground frame | `server/nav/scene/grouping.py` |
| $`{\color{red}{\tau}}_c`$ | time to contact at walking pace | s | time | `server/nav/planner/alarm.py` |
| $`T_a`$ | alarm threshold, 0.7 s | s | time | `server/nav/planner/config.py` |
| $`H`$ | alarm hold, 0.5 s | s | time | `server/nav/planner/config.py` |
| $`t_r`$ | time the alarm was raised | s | frame timestamps | `server/nav/planner/alarm.py` |
| $`\varepsilon_S`$ | his floor under clearance, 0.06 m | m | none | `server/nav/planner/config.py` |
| $`{\color{purple}{U}}`$ | avoidance surprise | bits | none | `server/nav/planner/alarm.py` |

## Building the avoidance surprise

| Step | Expression | Why |
|---|---|---|
| 1 | $`\min_{i \in \mathcal{C}} {\color{teal}{S}}_i`$ | The nearest thing in the strip the body sweeps |
| 2 | $`\max(\cdot,\ 0.06)`$ | His floor, so a touching point doesn't divide by zero |
| 3 | $`{\color{red}{\tau}} = \cdot / 1.4`$ | Seconds until the walker gets there at walking pace |
| 4 | $`1\ \text{s} / {\color{red}{\tau}}`$ | His avoidance ratio, read as one second over the time to contact |
| 5 | $`\tfrac12 (\cdot)^2`$ | His half-squared surprise, in natural-log units |
| 6 | $`\div \ln 2`$ | Natural-log units to bits. Steps 5 and 6 together are the $`1/(2\ln 2)`$ |

## Worked examples

**The arrow.**
1. $`k_h = \mathrm{round}(1.0 / 0.1) = 10`$. Forward by then: $`10 \times 0.1 \times 1.4 = 1.4`$ m.
2. One cell per step, 10 steps: the plan is at most 1.0 m to the side.
3. The widest arrow is $`\mathrm{atan2}(1.0, 1.4) = 0.6202`$ rad $`= 35.54°`$. That's a full sidestep,
   1.0 m/s, against walking pace, 1.4 m/s. It equals $`\mathrm{atan2}(v_{\text{lat}}, v_w)`$ only
   because one step reaches exactly one cell.
4. A plan 0.9 m across gives $`\mathrm{atan2}(0.9, 1.4) = 0.5713`$ rad, 32.7°.

**Why not the first step.** The arrow used to be the angle of the plan's first 0.1 s step. One step
moves −0.1, 0 or +0.1 m against 0.14 m forward, so that angle was −35.54°, 0° or +35.54° and
nothing between: three values. At the 1 s lookahead the plan can be any of 21 offsets. On the
classroom walk the arrow takes 20 different values instead of 3, and sits at its sideways limit on
75.8 % of frames instead of 87.6 %.

**The alarm.**
1. Raise when $`{\color{teal}{S}} < 0.7 \times 1.4 = 0.98`$ m. Clearance is measured from the 0.35 m
   footprint, so for a point dead ahead that's its center closer than $`0.98 + 0.35 = 1.33`$ m.
2. A post 0.9 m ahead: $`{\color{teal}{S}} = 0.55`$ m, $`{\color{red}{\tau}}_c = 0.55 / 1.4 = 0.39`$ s. Raise.
3. A post 1.4 m ahead: $`{\color{teal}{S}} = 1.05`$ m, $`{\color{red}{\tau}}_c = 0.75`$ s. No raise. Exactly at
   0.7 s doesn't raise either, since the test is strictly less than.
4. A point 0.29 m to the side and 0.8 m ahead is in the corridor. At 0.31 m it's out.
5. Raised at 0.0 s and quiet after: still shown at 0.1, 0.3 and 0.49 s, cleared at 0.5 s. That's 15
   frames at 30 frames a second. The hold runs from the first raise, so an alarm raised for longer
   than 0.5 s clears on its first quiet frame.

**The avoidance surprise.**
1. One second to contact, $`{\color{teal}{S}}`$ = 1.4 m: $`1 / (2 \ln 2) = 1 / 1.3863 = 0.72`$ bits.
2. 0.7 s, the alarm threshold, $`{\color{teal}{S}}`$ = 0.98 m: $`(1/0.7)^2 / 1.3863 = 2.0408 / 1.3863 = 1.47`$ bits.
3. At the floor, $`{\color{teal}{S}}`$ = 0.06 m: $`{\color{red}{\tau}} = 0.04286`$ s, $`23.33^2 / 1.3863 = 392.7`$ bits.

> [!WARNING]
> - **A swerve and return inside the first second reads as straight.** The arrow only compares now
>   with one second out. Its forward distance also assumes 1.4 m/s.
> - **The time to contact is on the short side.** The corridor uses the 0.30 m body, but the
>   clearance inside it comes from the scene, measured from the 0.35 m footprint. The footprint is
>   0.05 m wider, which at 1.4 m/s is about 0.036 s.
> - **The alarm assumes walking pace.** Something coming toward the walker faster isn't treated as
>   more urgent, and a walker standing still in front of something still gets the alarm. The scene's
>   measured closing rate is a two-frame difference that reads 1 cm of jitter as about a third of a
>   meter per second, so it isn't used.
> - **The corridor goes straight on**, whatever the plan says.
> - **The hold delays clearing** by up to 0.5 s.
> - **The path turns fully red when the alarm raises**, at 0.7 s to contact, which is 1.47 bits.
>   Both come from the alarm's one threshold, so changing it moves both. Section 14 has the path's
>   colors.

> [!IMPORTANT]
> **The avoidance form is his.** When the thing in question is something to keep away from, the
> course swaps signal and noise: the gap goes in the denominator and its 1 s spread on top,
> $`{\color{purple}{U}} = \tfrac12 (\Delta {\color{teal}{S}} / {\color{teal}{S}})^2`$ (the car-following study on
> College 5, PDF pages 61 to 71, with the formula on PDF pages 62 and 65). The conversion to bits, $`1/(2 \ln 2)`$, is in the Stationary
> Interaction paper, Eq. 2. The 1 s reference is marked his in the code
> (`AVOIDANCE_REFERENCE_SECONDS` in `server/nav/planner/alarm.py`). Reading
> $`\Delta {\color{teal}{S}} / {\color{teal}{S}}`$ as one second over the time to contact is ours. The deck calls the ratio "tau"
> and says $`\Delta {\color{teal}{S}}`$ shows how fast the cars approach or separate. If that spread
> stands for the distance closed in 1 s, the ratio is 1 s divided by (gap over closing speed),
> which is 1 s over the time to contact. Two
> cautions: a spread grows when the gap opens too, and the usual tau in the time-to-contact
> literature is gap over closing speed, the inverse of the deck's ratio.

> [!TIP]
> **Ours, with why.**
> - **The arrow at the 1 s lookahead.** The first step only ever gave three angles, as worked out
>   above.
> - **The corridor at the body's 0.30 m.** At the old 0.35 m plus 0.15 m, a doorway about 0.9 m wide
>   on the 2026-10-03 apartment walk raised the alarm on 93 % of its frames. At 0.30 m it's 36 %,
>   and things the walker stood in front of still raise it on 84 %.
> - **Time to contact at walking pace, under 0.7 s.** 0.7 s at 1.4 m/s is 0.98 m, which stays under
>   the 1 m at which something counts as close. Replayed on the classroom walk with the corridor and
>   the hold, the alarm is on for 60.0 % of frames with 66 changes, down from 76.0 % and 408.
> - **The 0.5 s hold**, so the alarm doesn't flicker. It takes the classroom walk from 108 changes to
>   66, and the last stretch of `pixel_walk_3` from 55 to 35.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [8. Planning a path](08_planning_a_path.md) · Next: [10. How much the scene shaped the plan](10_scene_shaped_the_plan.md)
