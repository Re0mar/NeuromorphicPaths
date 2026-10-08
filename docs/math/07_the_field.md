[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [6. Surprise](06_surprise.md) · Next: [8. Planning a path](08_planning_a_path.md)

# 7. The field

Sections 4 to 6 price one obstacle point from one spot at one moment. The planner needs that
price everywhere it might go. So it builds a table. Each row is a moment in the future, every
0.1 s out to 3.8 s. Each column is a place the walker could be sideways, every 0.1 m from 3 m left
to 3 m right. Each entry says how costly it would be, per second, to be at that sideways spot at
that moment, given everything in view. That table is the **field**, and section 8 finds the
cheapest path through it.

Think of a spreadsheet for a wide sidewalk, with the lanes as columns and the coming seconds as
rows. You shade each box by how bad that lane will be at that moment, then pick a route down the
sheet that stays in the light boxes. The comparison stops at the walking itself. The field assumes
you keep walking straight ahead at 1.4 m/s in every column, and nobody slows down or stops.

![The field as time slices by lateral cells](../diagrams/math_field_grid.svg)

In the diagram, each row is one 0.1 s slice and each column one 0.1 m sideways position. Look at
where one obstacle's cost lands: in the columns nearest it, and in the slices around the moment
the walker draws level with it.

> [!NOTE]
> **Ingredients**
> - The obstacle points, one per group, with each one's $`{\color{orange}{N}}`$ and wall flag
>   (sections 4 and 5).
> - The cost terms from section 6. His surprise is always on. The contact term is on by default
>   (`contact_term_enabled`).
> - The time step $`\Delta t = 0.1`$ s and the horizon 3.8 s (`server/nav/planner/config.py`).
> - The walking speed $`v_w = 1.4`$ m/s.
> - The planner's sideways grid: $`\pm 3.0`$ m at $`\Delta g = 0.1`$ m. This is a different grid from
>   the scene's 0.25 m squares.

## The formulas

**How many slices.** Counting now:

```math
K = \mathrm{round}\!\left(\frac{3.8}{0.1}\right) + 1 = 39, \qquad t_k = k\,\Delta t, \quad k = 0, \dots, 38
```

**How many sideways positions.**

```math
M = \mathrm{round}\!\left(\frac{2 \cdot 3.0}{0.1}\right) + 1 = 61, \qquad g_c = -3.0,\ -2.9,\ \dots,\ 2.9,\ 3.0
```

**Where each point is at slice k.** The walker walks forward, so every point comes toward them:

```math
x_j(k) = x_j, \qquad y_j(k) = y_j - v_w\,t_k
```

**How far it is from each candidate position.**

```math
d_{c,j}(k) = \sqrt{\bigl(g_c - x_j(k)\bigr)^2 + y_j(k)^2}
```

**Adding it up.** For each term, each group counts its worst point. The groups add, then the terms
add, each at weight 1:

```math
F[k, c] = \sum_{\text{terms}}\ \sum_{G}\ \max_{j \in G}\ {\color{purple}{U}}_{\text{term}}\!\left(d_{c,j}(k),\ {\color{orange}{N}}_{\text{eff},j}\right)
```

**What a path pays.** Every entry is a rate per second. The path search in section 8 charges
$`F[k, c]\,\Delta t`$ for each slice a path passes through.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`\Delta t`$ | Time step, 0.1 | s | none | `server/nav/planner/config.py` |
| K | Number of slices, now included, 39 | count | none | `server/nav/planner/field.py` |
| $`t_k`$ | Time of slice k | s | none | `server/nav/planner/field.py` |
| $`\Delta g`$ | Spacing of the sideways positions, 0.1 | m | walker | `server/nav/planner/config.py` |
| M | Number of sideways positions, 61 | count | none | `server/nav/planner/field.py` |
| $`g_c`$ | Sideways position c, positive right | m | walker | `server/nav/planner/field.py` |
| $`v_w`$ | Walking speed, 1.4 | m/s | walker | `server/nav/planner/config.py` |
| $`x_j`$, $`y_j`$ | Point j now, right and ahead of the walker | m | walker | `server/nav/scene/pipeline.py` |
| $`x_j(k)`$, $`y_j(k)`$ | Point j at slice k, relative to where the walker will be | m | walker, this frame | `server/nav/planner/field.py` |
| $`d_{c,j}(k)`$ | Center distance from position c to point j at slice k | m | walker | `server/nav/planner/field.py` |
| G | A group of points, one 0.25 m scene square | none | none | `server/nav/scene/grouping.py` |
| $`{\color{purple}{U}}_{\text{term}}`$ | One term's surprise rate for one point (section 6) | per second | none | `server/nav/planner/surprise.py`, `server/nav/planner/contact.py` |
| $`F[k, c]`$ | The field, cost per second at slice k and position c, shape 39 by 61 | per second | walker | `server/nav/planner/field.py` |

| Step | Expression | Why |
|---|---|---|
| 1 | $`g_c`$ from $`-3.0`$ to $`3.0`$ in 61 steps | The places the walker could be sideways |
| 2 | $`K = \mathrm{round}(3.8/0.1) + 1 = 39`$ | The moments the plan looks at, 38 steps of 0.1 s after now |
| 3 | $`y_j(k) = y_j - 1.4\,t_k`$ | The walker covers $`k \cdot 0.1 \cdot 1.4`$ m by slice k |
| 4 | $`d_{c,j}(k)`$ | The gap every term prices |
| 5 | $`{\color{purple}{U}}_{\text{term}}(d, {\color{orange}{N}}_{\text{eff}})`$ | Section 6, once per term |
| 6 | $`\max_{j \in G}`$ | Within a group, only the costliest point counts |
| 7 | $`\sum_G`$ | Separate groups add up |
| 8 | $`\sum_{\text{terms}}`$, weight 1 | Each term is separate evidence about the same scene |
| 9 | $`F[k, c]\,\Delta t`$ in the path search | A rate times a duration gives a cost |

**Why round.** $`3.8/0.1`$ in floating point comes out just under 38, because neither 3.8 nor 0.1 is
stored exactly. `round` gives 38, so K = 39 and the last slice is at 3.8 s. A plain `int()` would
cut it to 37 and give 38 slices, losing the last one.

## Worked example

A post 0.5 m to the right and 2.0 m ahead, $`{\color{orange}{N}} = 0.0337`$ m, not a wall.

1. **Where it goes.** By slice 10, $`t = 1.0`$ s, the walker has covered 1.4 m, so the post is at
   $`(0.5, 0.6)`$. By slice 14 it's at $`(0.5,\ 2.0 - 1.96) = (0.5, 0.04)`$. The walker draws level
   at $`2.0/1.4 = 1.43`$ s. By slice 20 it's at $`(0.5, -0.8)`$, behind the walker.
2. **Distance at slice 14.** From $`g_c = 0`$, $`d = \sqrt{0.25 + 0.0016} = 0.5016`$ m. From
   $`g_c = -0.3`$, $`d = \sqrt{0.64 + 0.0016} = 0.8010`$ m.
3. **His term from $`g_c = 0`$.** $`{\color{teal}{S}} = 0.5016 - 0.35 = 0.1516`$, so
   $`{\color{orange}{N}}/{\color{teal}{S}} = 0.2223`$. Squared that's 0.04942, and half is 0.0247 per
   second.
4. **Contact term from $`g_c = 0`$.** $`{\color{teal}{S}}_b = 0.5016 - 0.30 = 0.2016`$ and
   $`\sigma = 0.10553`$, so $`{\color{teal}{S}}_b/\sigma = 1.910`$. $`\Phi(1.910) = 0.97196`$ and
   $`-\ln 0.97196 = 0.02844`$. Divided by 0.4286 s that's 0.0664 per second.
5. **The entry.** $`F[14, g_c{=}0] = 0.0247 + 0.0664 = 0.0911`$ per second. A path through it pays
   $`0.0911 \cdot 0.1 = 0.00911`$ for that step.
6. **Three cells to the left.** From $`g_c = -0.3`$, $`{\color{teal}{S}} = 0.8010 - 0.35 = 0.4510`$, so
   his term is $`\tfrac12(0.0337/0.4510)^2 = \tfrac12(0.0747)^2 = 0.0028`$ per second. The contact
   gap is $`0.5010/0.10553 = 4.75`$ spreads, so the contact term is about 0.000002 per second,
   effectively zero. Standing 0.3 m further left cuts the entry from 0.0911 to about 0.0028 per
   second, roughly 33 times less.

**The combining rule on its own.** Say group 1 has two points costing 0.020 and 0.050 per second,
and group 2 has one point costing 0.010. That term adds $`0.050 + 0.010 = 0.060`$ to the entry.

**Motion prediction.** With it on, a group with a measured velocity also slides along that
velocity at each slice. It's off by default (`predict_motion = False`), because the scene's group
velocities aren't trusted yet. The details are in the Planned section.

> [!WARNING]
> - **Straight ahead at 1.4 m/s, in every column.** Every candidate position is assumed to keep
>   the same forward pace. A sidestep doesn't slow the walker down, and a walker who has stopped is
>   still modeled as walking. The horizon covers $`1.4 \cdot 3.8 = 5.32`$ m, inside the scene's 6 m
>   reach.
> - **Things behind the walker still cost.** The distance squares the forward gap, so a post
>   0.8 m behind counts the same as one 0.8 m ahead.
> - **The worst point per group never actually chooses.** The scene sends one point per group,
>   the nearest to the walker (section 4), so in the live pipeline the field is a plain sum over
>   groups. The maximum only does work when a caller hands over several points per group, as the
>   tests do.
> - **A group is a 0.25 m square, not an object.** A wall or a long table is many groups, all
>   added, so it costs more than one post at the same gap.
> - **These are rates.** A printed table of field values without the factor of 0.1 is in
>   different units from a path cost.
> - **Two widths of 3 m, kept equal by hand.** The planner's $`\pm 3.0`$ m and the scene's 3 m reach
>   are separate settings. Nothing beyond 3 m to the side is ever seen, so the edge columns look
>   emptier than the world may be.

> [!IMPORTANT]
> The time step of 0.1 s, the 3.8 s horizon, and the combining rule, worst point within a group
> and then the sum across groups: the code marks these as his. The code also records his walking
> speed as 5.0 m/s, for a cyclist. Adding separate terms follows his rule that independent
> likelihood terms add (College 5, PDF page 54).

> [!TIP]
> These are ours:
> - **1.4 m/s**, an average walking pace in place of his cyclist's 5.0 m/s.
> - **61 sideways positions 0.1 m apart**, matching the scene's 3 m reach.
> - **Rounding the slice count**, so 3.8 s isn't lost to floating point.
> - **Running his combining rule once per term**, because one term's worst point need not be
>   another's.
> - **Motion prediction**, which is off.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [6. Surprise](06_surprise.md) · Next: [8. Planning a path](08_planning_a_path.md)
