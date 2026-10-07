[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [4. Obstacles and the walker's body](04_obstacles_and_body.md) · Next: [6. Surprise](06_surprise.md)

# 5. Following things over time

One frame tells us how far away each group is. It doesn't tell us how much to trust that number.
So the scene keeps the last half second of clearances for every group and asks how much they've
been wobbling. A post whose distance jitters by a centimeter is steady. One whose distance jumps
around by half a meter isn't. That wobble is the **noise scale**, written
$`{\color{orange}{N}}`$, and it's the plain sample standard deviation of the group's recent
clearances $`{\color{teal}{S}}`$.

The same record gives two more numbers for free: how fast a group's clearance shrank between the
last two readings, the **closing rate**, and how fast its middle moved across the floor, its
**velocity**.

Think of reading a bathroom scale while you shift your weight. The number wobbles, and the spread
of the last few readings tells you how much to trust any one of them. The comparison stops working
when things move on purpose. You don't walk toward a scale, but a walker does walk toward a post,
and $`{\color{orange}{N}}`$ counts that steady change as wobble too.

> [!NOTE]
> **Ingredients**
> - Each group's clearance $`{\color{teal}{S}}`$, every frame (section 4).
> - A group id that stays on the same patch of floor from frame to frame. That needs the world
>   ids from section 4, which need a tracked position, so only the Pixel has them.
> - Each frame's timestamp, on the laptop's clock.
> - The window, 0.5 s, the minimum of 3 samples, and the floor of 0.01 m under
>   $`{\color{orange}{N}}`$ (`server/nav/scene/config.py`).
> - For the velocity, each group's middle on floor axes fixed to the world (section 4).

## The formulas

**The window.** At time t, a group seen this frame keeps only samples with $`t_i \ge t - 0.5`$. A
sample exactly 0.5 s old stays.

**The noise scale.** With n samples in the window:

```math
{\color{orange}{N}} = \begin{cases} N_{\text{floor}} & n < 3 \\[4pt] \max\!\left(\sqrt{\dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}\left({\color{teal}{S}}_i - \bar{{\color{teal}{S}}}\right)^2},\ N_{\text{floor}}\right) & n \ge 3 \end{cases}
```

Below 3 samples it returns the floor, 0.01 m. The $`n - 1`$ makes it the sample standard deviation
(`np.std` with `ddof=1`).

**Closing rate.** The newest two samples only, positive when the gap is shrinking:

```math
\dot{c} = \frac{{\color{teal}{S}}_{n-1} - {\color{teal}{S}}_n}{t_n - t_{n-1}}
```

It's empty with fewer than 2 samples or a gap of zero or less.

**Velocity.** The newest two positions of the group's middle, on world floor axes:

```math
\mathbf{v} = \frac{\mathbf{c}_n - \mathbf{c}_{n-1}}{t_n - t_{n-1}}
```

It's turned from world axes into the walker's axes before the planner sees it. Without a tracked
position there's no velocity at all.

**Forgetting.** After each frame, a group whose newest sample is older than $`t - 0.5`$ is deleted,
so ids from long ago don't pile up.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| t | This frame's timestamp | s | laptop clock | `server/nav/scene/history.py` |
| $`t_i`$ | A stored sample's timestamp | s | laptop clock | `server/nav/scene/history.py` |
| $`{\color{teal}{S}}_i`$ | One stored clearance of the group | m | walker | `server/nav/scene/history.py` |
| $`\bar{{\color{teal}{S}}}`$ | Their average | m | walker | `server/nav/scene/history.py` |
| n | Samples in the window | count | none | `server/nav/scene/history.py` |
| $`N_{\text{floor}}`$ | The lowest $`{\color{orange}{N}}`$ ever reported, 0.01 | m | none | `server/nav/scene/config.py` |
| $`{\color{orange}{N}}`$ | Noise scale, how much $`{\color{teal}{S}}`$ wobbled over the window | m | none, it's a spread of a distance | `server/nav/scene/history.py` |
| $`\dot{c}`$ | Closing rate, how fast the gap shrank | m/s | walker | `server/nav/scene/history.py` |
| $`\mathbf{c}_n`$ | The group's middle at the newest sample | m | world ground | `server/nav/scene/pipeline.py` |
| $`\mathbf{v}`$ | The group's velocity | m/s | world ground, then walker | `server/nav/scene/history.py` |

## Building the noise scale

| Step | Expression | Why |
|---|---|---|
| 1 | append $`({\color{teal}{S}}, t)`$ to the group | One sample per frame the group is seen |
| 2 | drop samples with $`t_i < t - 0.5`$ | Only the last half second counts |
| 3 | if $`n < 3`$, return 0.01 | Below 3 samples a standard deviation says nothing |
| 4 | $`\bar{{\color{teal}{S}}} = \frac{1}{n}\sum {\color{teal}{S}}_i`$ | The average gap |
| 5 | $`\sum ({\color{teal}{S}}_i - \bar{{\color{teal}{S}}})^2`$ | How far each sample sits from it, squared |
| 6 | divide by $`n - 1`$, take the root | The sample standard deviation |
| 7 | $`\max(\cdot, 0.01)`$ | $`{\color{orange}{N}}`$ never reads below 1 cm |

## Worked examples

**Steady readings.** Three samples, 2.00, 2.01 and 1.99 m. The average is 2.00. The squared gaps
are 0, 0.0001 and 0.0001, which sum to 0.0002. Divided by 2 that's 0.0001, and the root is 0.01.
So $`{\color{orange}{N}} = \max(0.01, 0.01) = 0.01`$ m.

**Shakier readings.** Four samples, 2.00, 2.05, 1.95 and 2.10 m. The average is 2.025. The squared
gaps are 0.000625, 0.000625, 0.005625 and 0.005625, summing to 0.0125. Divided by 3 that's
0.004167, and $`{\color{orange}{N}} = \sqrt{0.004167} = 0.0645`$ m.

**Too few.** Two samples, whatever their values. $`n < 3`$, so $`{\color{orange}{N}} = 0.01`$ m.

**Walking straight at a post.** The Pixel at 30 frames a second, walking at 1.4 m/s toward a post
that doesn't move, with world ids so the post keeps its id. The clearance drops by
$`1.4/30 = 0.0467`$ m every frame. A half-second window at 30 frames a second holds 15 or 16
samples. For evenly spaced values the sample standard deviation is the spacing times
$`\sqrt{n(n+1)/12}`$:

| Samples | $`\sqrt{n(n+1)/12}`$ | $`{\color{orange}{N}}`$ |
|---|---|---|
| 15 | $`\sqrt{15 \cdot 16/12} = \sqrt{20} = 4.472`$ | $`0.0467 \cdot 4.472 = 0.209`$ m |
| 16 | $`\sqrt{16 \cdot 17/12} = 4.761`$ | $`0.0467 \cdot 4.761 = 0.222`$ m |

That's about 0.21 m from the approach alone, with no sensor noise at all. For scale, the median noise of
groups in the corridor within 2 m on the classroom walk was 0.0337 m
(`server/tests/test_planner_pipeline.py`). On `pixel_walk_3` the same median is 0.0958 m.

**The Neon at 1.68 planned frames a second.** On average, frames arrive $`1/1.68 = 0.595`$ s apart,
which is longer than the window. At that pace the previous sample is usually gone by the time the
next one arrives, so the window holds 1 sample and the group gets $`{\color{orange}{N}} = 0.01`$ m.
To measure anything, 3 samples have to fit inside 0.5 s, which means two gaps of 0.25 s or less.
Evenly spaced, that takes 4 frames a second or more. This is worked out from the session's average
rate. No log of that session was read to check it frame by frame.

**Closing rate.** A group reads 1.80 m and then 1.75 m, 1/30 s apart. That's
$`(1.80 - 1.75) \cdot 30 = 1.5`$ m/s. A jitter of just 0.01 m between two frames reads as
$`0.01 \cdot 30 = 0.3`$ m/s, which is why the alarm doesn't use it.

**Velocity.** A group's middle goes from $`(1.00, 5.00)`$ to $`(1.02, 4.96)`$ m in 1/30 s. That's
$`(0.02 \cdot 30,\ -0.04 \cdot 30) = (0.6, -1.2)`$ m/s on the world's floor axes. With the walker
turned 30° to the right of the world's forward axis, a group moving $`(0, 1.0)`$ m/s in the world moves
$`(-0.5, 0.866)`$ m/s for the walker, drifting left and pulling away.

**Forgetting.** A group last seen at $`t = 10.0`$ s is kept at $`t = 10.4`$, where the cutoff is 9.9,
and deleted at $`t = 10.6`$, where the cutoff is 10.1.

> [!WARNING]
> - **$`{\color{orange}{N}}`$ includes the walker's own approach.** The code takes the plain sample standard
>   deviation of $`{\color{teal}{S}}`$ with no trend removed. So the walker's own approach shows up
>   in it, about 0.21 m at 30 frames a second, as worked out above. That's six times the 0.0337 m
>   median noise of corridor groups within 2 m on the classroom walk. The docstring in `server/nav/scene/history.py` describes jitter.
>   This is what the code measures.
> - **On the Neon, $`{\color{orange}{N}}`$ is approximate.** The glasses have no tracked position,
>   so the grid moves with the walker and the same id is a different patch of floor each frame.
>   At the 2026-10-05 glasses session's average of 1.68 frames a second it's nearly always the
>   floor, 0.01 m, so in practice it isn't measured.
> - **Below 4 evenly spaced frames a second, $`{\color{orange}{N}}`$ is a constant.** The window
>   can't hold 3 samples, so it's 0.01 m.
> - **The floor doesn't stop a division.** The config comment says the floor stops surprise
>   dividing by almost zero. In the professor's term $`{\color{orange}{N}}`$ sits on top of the
>   fraction (section 6), and the contact term's spread never drops below the 0.10 m sway. What
>   the floor does is keep $`{\color{orange}{N}}`$ from reading as zero.
> - **The planner doesn't read the closing rate.** It's worked out and attached to every
>   obstacle, but only the evaluation fixture code (`server/nav/evaluation/fixture.py`) reads it.
>   A comment in `server/nav/scene/history.py` says time to contact divides by it. It doesn't.
>   Time to contact uses the walking speed, 1.4 m/s (section 9).
> - **A velocity can come from points, not motion.** The middle of a square's points moves when
>   points enter or leave the square. A still object that's half hidden one frame and fully
>   visible the next gets a velocity. That's why motion prediction is off (section 7).
> - **Old samples linger a little.** A group seen within the window but not this frame keeps
>   samples older than the window, because only groups seen this frame get trimmed. Nothing reads
>   $`{\color{orange}{N}}`$ for an unseen group, and the extras are trimmed before anything reads
>   it again, so the plan never sees them.

> [!IMPORTANT]
> $`{\color{orange}{N}}`$ is the professor's noise scale. In the Stationary Interaction paper it's
> the spread of the measured quantity, in the same units as $`{\color{teal}{S}}`$ (§3.1, Eq. 1, and
> §4.1, Eq. 8). In his car-following example the spread is taken over a 1 s moving window
> (College 5, PDF page 65).

> [!TIP]
> These are ours:
> - **A 0.5 s window** instead of his 1 s, which was for car following. Half a second is about
>   0.7 m of walking.
> - **The sample form**, dividing by $`n - 1`$.
> - **At least 3 samples**, because below that a standard deviation says nothing.
> - **The 0.01 m floor.**
> - **The closing rate, the velocity and the forgetting rule.** The closing rate is unused. The
>   velocity only feeds motion prediction, which is off.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [4. Obstacles and the walker's body](04_obstacles_and_body.md) · Next: [6. Surprise](06_surprise.md)
