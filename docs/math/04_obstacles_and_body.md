[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [3. Finding the floor](03_finding_the_floor.md) · Next: [5. Following things over time](05_following_over_time.md)

# 4. Obstacles and the walker's body

After section 3 we have a cloud of points in meters, and we know where the floor is. The planner
can't reason about thousands of loose points every frame, so this step boils them down. It lays
every point flat on the floor as "this far to the right, this far ahead" of the walker. It sorts
the points into 0.25 m squares on the floor, throws out squares with too little in them, and
shrinks each square that's left to the one point nearest the walker. Then it measures how much
room is left between the edge of the walker and that point.

In the code a square is a **cell**, the points in one cell form a **group**, and the room left is
the **clearance**, written $`{\color{teal}{S}}`$. A group is the planner's idea of "one obstacle".

Think of a parking garage with painted bays. You don't describe every car part by part. You note
which bays are taken and, for each taken bay, the corner of the car nearest you. The comparison
stops working at the bay itself. A garage bay holds one car, but our squares know nothing about
objects, so a long wall fills a whole row of squares and counts as many obstacles.

> [!NOTE]
> **Ingredients**
> - Points in meters, already thinned to one point per 5 cm cube (section 1).
> - The floor plane and the up direction (sections 2 and 3). They give each point's height and
>   the two directions lying flat on the floor, forward and lateral.
> - Only points between 0.20 m and 2.00 m above the floor (section 3). Lower is floor, higher is
>   something the walker passes under.
> - The walker's position. On the Pixel it's the camera's position as ARCore tracks it. The Neon
>   glasses have no position, so the walker sits at zero in the camera's own frame.
> - The footprint radius r = 0.35 m (`server/nav/walker.py`) and the body half-width h = 0.30 m
>   (`server/nav/planner/config.py`).

## The formulas

**Where a point is on the floor.** Take the point's offset from the walker and measure it along
the two floor directions:

```math
x_i = (\mathbf{p}_i - \mathbf{p}_w)\cdot\hat{l}, \qquad y_i = (\mathbf{p}_i - \mathbf{p}_w)\cdot\hat{f}
```

**Which square it lands in, body frame.** The grid reaches 3 m to each side and 6 m ahead, and
moves with the walker:

```math
C = \left\lceil \frac{2W}{c} \right\rceil = 24, \qquad \text{col} = \left\lfloor \frac{x + W}{c} \right\rfloor, \qquad \text{row} = \left\lfloor \frac{y}{c} \right\rfloor, \qquad \text{id} = \text{row}\cdot C + \text{col}
```

for points with $`-W \le x < W`$ and $`0 \le y < F`$. Anything else gets id $`-1`$ and is dropped.

**Which square it lands in, world frame.** With a tracked position the squares are pinned to the
world instead, so a post keeps its id while the walker goes past it:

```math
\text{col} = \left\lfloor \frac{X}{c} \right\rfloor + 2^{20}, \qquad \text{row} = \left\lfloor \frac{Y}{c} \right\rfloor + 2^{20}, \qquad \text{id} = \text{row}\cdot 2^{21} + \text{col}
```

Points outside the same 6 m by 6 m window in front of the walker are still dropped.

**Keep a square only with enough in it**, $`n_g \ge 2`$.

**Shrink a group to its nearest point, and keep its middle:**

```math
(x_n, y_n) = \arg\min_{j \in g} \sqrt{x_j^2 + y_j^2}, \qquad (\bar{x}, \bar{y}) = \frac{1}{n_g}\sum_{j \in g} (x_j, y_j)
```

**Wall flag.** A group is a wall when its tallest point reaches 1.5 m:

```math
\text{wall}_g = \left[\, e_{\max,g} \ge 1.5 \,\right]
```

**Clearance.** The walker is a circle of radius r around their own position. The clearance is the
gap from the edge of that circle to the group's nearest point, never below zero:

```math
{\color{teal}{S}} = \max\!\left(0,\ \sqrt{x_n^2 + y_n^2} - r\right)
```

**The body, a second width.** The contact term (section 6) asks whether the body itself would
touch something. It measures from the body half-width h instead:

```math
{\color{teal}{S}}_b = d - h
```

where d is a center distance. $`{\color{teal}{S}}_b`$ is allowed to go negative, which means
overlap. The two widths differ by $`r - h = 0.35 - 0.30 = 0.05`$ m. That 0.05 m is room to steer,
and no constant holds it. The alarm (section 9) uses h only for how wide a corridor it watches.
Its time to contact divides the clearance $`{\color{teal}{S}}`$ below, measured from r = 0.35 m.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`\mathbf{p}_i`$ | One point of the cloud | m | camera, or world with a position | `server/nav/scene/pipeline.py` |
| $`\mathbf{p}_w`$ | The walker's position, zero without tracking | m | camera, or world | `server/nav/scene/pipeline.py` |
| $`\hat{f}`$, $`\hat{l}`$ | Forward and lateral directions on the floor. Forward is where the camera looks, tilt removed. Lateral points right | none | same as $`\mathbf{p}_i`$ | `server/nav/scene/floor.py` |
| $`x_i`$, $`y_i`$ | How far right and how far ahead a point is | m | walker | `server/nav/scene/pipeline.py` |
| $`X`$, $`Y`$ | The same point on floor axes fixed to the world | m | world ground | `server/nav/scene/grouping.py` |
| $`c`$ | Edge of one square, 0.25 | m | floor | `server/nav/scene/config.py` |
| $`W`$ | How far the grid reaches to each side, 3.0 | m | walker | `server/nav/scene/config.py` |
| $`F`$ | How far the grid reaches ahead, 6.0 | m | walker | `server/nav/scene/config.py` |
| $`C`$ | Columns in the body grid, 24 | count | walker | `server/nav/scene/grouping.py` |
| $`n_g`$ | Points in group g, counted as 5 cm cubes | count | none | `server/nav/scene/grouping.py` |
| $`(x_n, y_n)`$ | The group's point nearest the walker | m | walker | `server/nav/scene/grouping.py` |
| $`(\bar{x}, \bar{y})`$ | The average of the group's points | m | walker | `server/nav/scene/grouping.py` |
| $`e_{\max,g}`$ | Height of the group's tallest point above the floor | m | floor | `server/nav/scene/grouping.py` |
| r | Footprint radius, 0.35 | m | walker | `server/nav/walker.py` |
| h | Body half-width, 0.30 | m | walker | `server/nav/planner/config.py` |
| $`{\color{teal}{S}}`$ | Clearance, the gap from the footprint's edge to the nearest point | m | walker | `server/nav/scene/grouping.py` |
| $`{\color{teal}{S}}_b`$ | Gap from the body's edge, negative on overlap | m | walker | `server/nav/planner/contact.py` |
| d | Center distance from the walker, or a candidate position, to a point | m | walker | `server/nav/planner/field.py` |

## From points to one clearance per group

| Step | Expression | Why |
|---|---|---|
| 1 | keep $`0.20 < e < 2.00`$, height above the floor | Below is floor, above is overhead (section 3) |
| 2 | $`x_i`$, $`y_i`$ along $`\hat{l}`$, $`\hat{f}`$ | The planner works in "right of me" and "ahead of me" |
| 3 | id from $`\lfloor (x+W)/c \rfloor`$ and $`\lfloor y/c \rfloor`$, or the world version | One 0.25 m square is one group |
| 4 | keep when $`n_g \ge 2`$ | A single point is as likely to be depth noise as an object |
| 5 | $`(x_n, y_n)`$ = nearest member | One point stands for the group from here on |
| 6 | $`(\bar{x}, \bar{y})`$ = average member | Section 5 follows the group's middle to get a velocity |
| 7 | wall when $`e_{\max,g} \ge 1.5`$ | Walls and tree trunks get a wider berth (section 6) |
| 8 | $`{\color{teal}{S}} = \max(0, \sqrt{x_n^2 + y_n^2} - r)`$ | The room left, measured from the footprint's edge |

## Worked example

Three points of a table edge land on the floor at $`(0.55, 2.10)`$, $`(0.60, 2.05)`$ and
$`(0.70, 2.20)`$ m, in the body frame.

1. Columns: $`(0.55 + 3)/0.25 = 14.2`$, $`(0.60 + 3)/0.25 = 14.4`$ and $`(0.70 + 3)/0.25 = 14.8`$. All
   floor to 14.
2. Rows: $`2.10/0.25 = 8.4`$, $`2.05/0.25 = 8.2`$ and $`2.20/0.25 = 8.8`$. All floor to 8.
3. id $`= 8 \cdot 24 + 14 = 206`$. The body grid has $`24 \cdot 24 = 576`$ squares in total.
4. Three points, at least 2, so the group stays.
5. Distances from the walker: $`\sqrt{0.3025 + 4.41} = \sqrt{4.7125} = 2.1708`$,
   $`\sqrt{0.36 + 4.2025} = \sqrt{4.5625} = 2.1360`$ and $`\sqrt{0.49 + 4.84} = \sqrt{5.33} = 2.3087`$ m.
   The nearest point is $`(0.60, 2.05)`$.
6. Middle: $`(1.85/3,\ 6.35/3) = (0.6167, 2.1167)`$ m.
7. A table top at 0.75 m is not a wall. One point at 1.80 m in the same square would make it one.
8. $`{\color{teal}{S}} = 2.1360 - 0.35 = 1.7860`$ m. Measured from the body it would be
   $`2.1360 - 0.30 = 1.8360`$ m.

Two points much closer show the floor at zero and the second width:

| Nearest point | Center distance | $`{\color{teal}{S}}`$ from r = 0.35 | $`{\color{teal}{S}}_b`$ from h = 0.30 |
|---|---|---|---|
| $`(0.20, 0.30)`$ | $`\sqrt{0.13} = 0.3606`$ | 0.0106 | 0.0606 |
| $`(0.10, 0.20)`$ | $`\sqrt{0.05} = 0.2236`$ | 0, floored | $`-0.0764`$, overlap |

With a tracked position, a point at world $`X = -1.3`$, $`Y = 12.6`$ m gets column
$`\lfloor -5.2 \rfloor + 1{,}048{,}576 = 1{,}048{,}570`$ and row
$`\lfloor 50.4 \rfloor + 1{,}048{,}576 = 1{,}048{,}626`$. Its id is
$`1{,}048{,}626 \cdot 2{,}097{,}152 + 1{,}048{,}570 = 2{,}199{,}129{,}161{,}722`$. The world grid
reaches $`2^{20} \cdot 0.25 = 262{,}144`$ m to each side of where the run started.

> [!WARNING]
> - **A square isn't an object.** A 2 m wall along the path covers 8 squares, so it's 8 groups.
>   Later sections add groups up, so a wall costs more than a post at the same distance.
> - **The body grid moves with the walker.** Without a position, a post changes id every 0.25 m
>   the walker covers. Its history in section 5 starts over each time. Only the Pixel has a
>   tracked position, so only the Pixel gets world ids.
> - **World squares follow the world's axes.** A post standing on a square's border splits into
>   two groups, each with its own history.
> - **Height is dropped.** A table edge at 0.75 m and a chair leg below it land on the same spot
>   of floor.
> - **Nearest to the walker, not nearest to a sidestep.** The planner later measures from
>   candidate positions to the side (section 7). The point nearest the walker isn't always the
>   point nearest a position 1 m over. The error is under one square's diagonal,
>   $`0.25\sqrt{2} = 0.354`$ m.
> - **The wall flag is height only.** A person 1.8 m tall standing still counts as a wall.
> - **The planner's field doesn't read this $`{\color{teal}{S}}`$.** It recomputes the gap from
>   every candidate position at every future moment (section 7). The $`{\color{teal}{S}}`$ worked
>   out here feeds the noise in section 5 and the alarm in section 9.

> [!IMPORTANT]
> $`{\color{teal}{S}}`$ is the professor's symbol. In his avoidance form it's the gap you want to
> keep, the distance to the car in front (College 5, PDF pages 61 to 71). In the Stationary
> Interaction paper it's the signal, the distance from where you are to where you expected to be
> (§4.1, Eq. 8). We fill it in with the gap from the walker to an obstacle.

> [!TIP]
> Everything else in this section is ours:
> - **The 0.25 m squares.** The professor's surprise speaks of points. A depth camera gives tens
>   of thousands of them, so we group them first. A square of 0.25 m is coarse enough that one
>   square is roughly one obstacle.
> - **The 2-point minimum.** One point is as likely to be depth noise as something real.
> - **The nearest point per square.** It's one point per group, chosen once per frame.
> - **The wall flag at 1.5 m.** A wall or a tree trunk deserves a wider berth than a bollard.
>   Section 6 says how.
> - **The round footprint, r = 0.35 m.** A shoulder half-width plus a margin. It can be changed
>   from the command line with `--walker-radius`.
> - **The body, h = 0.30 m.** A shoulder half-width with room for arm swing, narrower than the
>   footprint. The contact term asks whether the body would touch, and the alarm uses h for its
>   corridor width. The footprint's extra 0.05 m is room to steer.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [3. Finding the floor](03_finding_the_floor.md) · Next: [5. Following things over time](05_following_over_time.md)
