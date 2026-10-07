[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [9. The arrow and the alarm](09_arrow_and_alarm.md) · Next: [11. The walker's response](11_walker_response.md)

# 10. How much the scene shaped the plan

Did what the camera saw actually change where the plan goes? To answer that, the planner runs twice
more each frame. Once with everything, and once as a planner that sees nothing: same goal, same
memory of the last plan, no obstacles. For each run it asks where the walker is likely to be one
second from now, as a spread of chances over the sideways cells. Then it measures how different the
two spreads are. That difference is the **Kullback-Leibler divergence**, KL for short, of the
**posterior** (with the scene) from the **prior** (without it), in bits. It's 0 when the scene
changed nothing, with nothing in view for example. An empty corridor isn't enough, because a post off
to the side still enters the posterior. It's not a confidence score.

Think of a weather forecast against the usual weather for that day of the year. If today's forecast
looks like the usual, it told you nothing new. If it says snow in July, it told you a lot. Where it
stops working: the KL says how far the plan moved away from what the planner would do blind, not
whether moving was the right call.

> [!NOTE]
> **Ingredients**
> - The forward costs $`J`$ and backward costs $`B`$ from section 8, at the lookahead slice
>   $`k_h`$ = 10, one second out.
> - Two fields to run them on. The **posterior field** is the planner's own field. The **prior field**
>   is the same with the obstacle terms taken out, collision and contact, and the goal term and
>   previous-plan prior kept. Built in `server/nav/planner/pipeline.py`.
> - The effort term $`{\color{blue}{T}}`$ lives inside the dynamic program, so both runs have it.
> - Cost: for each field, a forward pass up to $`k_h`$ and a backward pass from it. Each runs over
>   its half of the field, so the two together cost about one full pass. The tests bound plan plus
>   information under 10 ms (`server/tests/test_dynamic_programming.py`).

## The formulas

**Cheapest whole path through each cell.** At the lookahead slice, forward plus backward is the
cost of the best complete path through that cell, counted once (section 8). It's $`\infty`$ where a
cell can't be reached.

```math
c_j = J_{k_h}(j) + B_{k_h}(j)
```

**From costs to chances, base e.** The costs are natural logs, so $`e^{-c}`$ turns them back into
relative chances. Every extra $`\ln 2 = 0.693`$ of cost halves a cell's chance. This is a
**softmin**: the cheapest cell gets the biggest share, and cells that cost $`\infty`$ get none.

```math
p_j = \frac{e^{-(c_j - \min c)}}{\sum_{j'} e^{-(c_{j'} - \min c)}}
```

$`q_j`$ is built the same way from the prior field.

**The divergence.** Sum over cells the chance with the scene, times log2 of how much the scene
raised or lowered that cell's chance.

```math
I = \max\Big(0,\ \sum_{j:\ p_j > 0} p_j \log_2 \frac{p_j}{q_j}\Big)
```

A cell with $`p_j > 0`$ and $`q_j = 0`$ raises an error instead of returning infinity. Both runs share
the same reachable cells, so it shouldn't come up.

**The ceiling.** When the scene leaves only one sensible cell, $`p`$ puts everything on it and
$`I = -\log_2 q_j`$ for that cell. The most that can ever come out is

```math
I \le -\log_2 \min_{j:\ q_j > 0} q_j
```

The minimum runs over the reachable cells only. A cell that can't be reached has $`q_j = 0`$.

The bound moves with the prior. Against the goal alone it's 7.3 bits, worked out below. With the
pull toward the previous plan in the prior, which is how the planner runs, the prior can be much
narrower, and the bound with it. On every planned frame of `pixel_walk_3` the highest $`I`$ was
22.2 bits.

That's **not** $`\log_2`$ of the number of cells, 21 here. $`\log_2 21`$ would be the ceiling against a
prior that gives every cell the same chance. This prior favors the center, because every sideways
cell costs effort, so an edge cell has a smaller chance than $`1/21`$ and pinning the plan there reads
higher.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`k_h`$ | lookahead slice, 10, one second out | none | time | `server/nav/planner/heading.py` |
| $`j`$ | sideways cell, center 30, 21 reachable at $`k_h`$ | none | walker ground frame | `server/nav/planner/information.py` |
| $`J_{k_h}(j)`$ | cheapest cost from the start to cell $`j`$, own charge included | cost | none | `server/nav/planner/dynamic_programming.py` |
| $`B_{k_h}(j)`$ | cheapest cost from cell $`j`$ to the end, own charge left out | cost | none | `server/nav/planner/dynamic_programming.py` |
| $`c_j`$ | cheapest whole path through cell $`j`$ | cost, natural-log units | none | `server/nav/planner/information.py` |
| $`p_j`$ | chance of being at cell $`j`$ one second out, with the scene | none | walker ground frame | `server/nav/planner/information.py` |
| $`q_j`$ | the same, for the planner that sees nothing | none | walker ground frame | `server/nav/planner/information.py` |
| $`I`$ | scene information, KL of $`p`$ from $`q`$ | bits | none | `server/nav/planner/information.py` |

## Building it

| Step | Expression | Why |
|---|---|---|
| 1 | $`c_j = J_{k_h}(j) + B_{k_h}(j)`$ | Cheapest whole path through each cell at one second out |
| 2 | $`c_j - \min c`$ | Shift so the cheapest cell is 0. The shift cancels in the division |
| 3 | $`e^{-(\cdot)}`$, then divide by the sum | Costs to chances. The costs are natural logs, so base e undoes them |
| 4 | Same steps on the prior field | What the planner would expect with nothing in view |
| 5 | $`\sum p_j \log_2 (p_j / q_j)`$ | KL divergence, in bits because of the $`\log_2`$ |
| 6 | $`\max(0, \cdot)`$ | KL can't be negative. This only clears rounding |

## Worked example

The prior field with the goal straight ahead and no previous plan. Every number comes from section 8.

1. Reaching $`n`$ cells off center by slice 10 costs $`0.325\,\lvert n \rvert`$. The rest of the path
   costs the goal term, $`0.0222\,(0.1 n)^2 = 0.000222\,n^2`$. So
   $`c_n = 0.325\,\lvert n \rvert + 0.000222\,n^2`$ for $`n`$ from −10 to 10.
2. The cheapest is the center, $`c_0 = 0`$. Each cell out multiplies the weight by
   $`e^{-0.325} = 0.72253`$, and the goal term trims a little more:
   weight $`= 0.72253^{\lvert n \rvert} \times e^{-0.000222\,n^2}`$.
3. The weights for $`n`$ = 1 to 10 are 0.72237, 0.52158, 0.37644, 0.27157, 0.19582, 0.14114,
   0.10168, 0.07323, 0.05271 and 0.03792. Both sides plus the center:
   $`Z = 1 + 2 \times 2.49446 = 5.9889`$.
4. Prior chance of the center: $`q = 1 / 5.9889 = 0.1670`$. Prior chance of an edge cell, a full
   1.0 m sidestep by one second: $`q = 0.03792 / 5.9889 = 0.00633`$.
5. **Plan pinned to the center.** Everything on the center cell gives
   $`I = -\log_2 0.1670 = \log_2 5.9889 = 2.582`$ bits. That's not 0, even though the center is the
   prior's own favorite: the prior spread its chances over 21 cells and the scene narrowed that
   to one.
6. **Plan pinned to the edge.** $`I = -\log_2 0.00633 = 7.303`$ bits. That's the ceiling for this
   prior.
7. Compare $`\log_2 21 = 4.392`$ bits. The edge case goes over it, which is why that isn't the
   ceiling.
8. **Empty scene.** The two fields are identical, so $`p = q`$ and $`I = 0`$.
9. **The softmin by itself.** Costs (0, $`\ln 2`$, $`\infty`$) give chances (2/3, 1/3, 0).

The planner's own functions give the same 0.1670, 0.00633, 2.582 and 7.303 on this prior.

Before 2026-10-07 the softmin used $`2^{-c}`$ on these natural-log costs, which spread the chances
more evenly: the center read 3.011 bits and the edge 6.283.

> [!WARNING]
> - **Two bases, on purpose.** $`c`$ is a natural-log cost, so the chances use $`e^{-c}`$. The
>   divergence between the chances is taken with $`\log_2`$, so it comes out in bits.
> - **One slice only.** It reads the plan one second out and nothing else.
> - **The goal and the last plan are held fixed, not removed.** The prior keeps both, so $`I`$ is what
>   the scene added given the goal and the memory, not everything that shaped the plan.

> [!IMPORTANT]
> **KL as Bayesian surprise is his.** For two bell curves with the same width, the KL divergence
> between them is half the squared gap between their centers over the variance. That's exactly his
> surprise potential, which is why the Stationary Interaction paper calls it Bayesian surprise
> (§4.1, Eq. 8) and expresses it in bits (§5). The module's first docstring line calls this measure
> "the professor's Bayesian surprise of the plan" (`server/nav/planner/information.py`). That one
> line is the whole claim. Here the KL is taken between two spreads over 21 cells, not two bell
> curves, so his formula is the special case and this is the general one.

> [!TIP]
> **The construction is ours.** Forward plus backward per cell, the base-e softmin, reading it at
> the 1 s lookahead, and what goes into the prior. None of it is marked his. The aim is to say how
> much this frame's view changed the plan, apart from the goal and the memory.
> - **The prior keeps the goal**, or the scene would get credit for a turn the gaze caused.
> - **The prior keeps the previous-plan prior**, since that's the walker's memory and not something
>   seen this frame. Holding a side doesn't count as scene information.
> - **The backward pass** from section 8 exists for this. The planner alone never needs it.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [9. The arrow and the alarm](09_arrow_and_alarm.md) · Next: [11. The walker's response](11_walker_response.md)
