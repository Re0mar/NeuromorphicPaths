[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [12. Scoring the arrow](12_scoring_the_arrow.md) · Next: [14. How the path looks](14_how_the_path_looks.md)

# 13. Gaze on the floor

**The plain idea.** The Neon glasses report where in the image the walker is looking. Draw a line from
the camera through that pixel, see where it hits the floor, and read that spot as so far to the side
and so far ahead. That's the gaze point on the floor, and in gaze mode it becomes the planner's goal.

**An analogy.** A laser pointer strapped to your glasses, aimed wherever your eyes go. The red dot on
the pavement is the goal. The analogy stops working when you look up: a real pointer would light up a
wall or nothing, while this calculation only knows about the floor plane and gives up on anything that
doesn't come down to it.

> [!NOTE]
> **Ingredients**
> - The gaze pixel from the Neon, already straightened for lens distortion and scaled into the depth
>   image's pixels (`server/nav/sources/neon_live.py`, `server/nav/sources/estimated_depth.py`).
> - The depth image's camera numbers: focal lengths $`f_x, f_y`$ and image center $`c_x, c_y`$, in pixels.
> - The scene's last floor plane in the camera frame, from section 3.
> - Gaze mode switched on. It isn't the default. In the default mode the goal is straight ahead.

The ray through the pixel, and where it meets the floor:

```math
\mathbf r = \Big(\frac{u - c_x}{f_x},\ \frac{v - c_y}{f_y},\ 1\Big), \qquad \mathbf n\cdot\mathbf x + d = 0 \ \text{on the floor}
```

```math
t = -\frac{d}{\mathbf r\cdot\mathbf n}, \qquad \mathbf P = t\,\mathbf r, \qquad (\text{lateral},\ \text{forward}) = \big(\mathbf P\cdot\hat{\mathbf l},\ \mathbf P\cdot\hat{\mathbf f}\big)
```

The floor axes come from the plane. Forward is the camera's forward with its up-and-down part removed,
and right is forward crossed with the floor's up:

```math
\hat{\mathbf f} = \frac{\hat{\mathbf z}_c - (\hat{\mathbf z}_c\cdot\mathbf n)\,\mathbf n}{\lVert\hat{\mathbf z}_c - (\hat{\mathbf z}_c\cdot\mathbf n)\,\mathbf n\rVert}, \qquad \hat{\mathbf l} = \hat{\mathbf f}\times\mathbf n
```

It returns nothing when there's no gaze pixel, when there's no floor plane yet, when the ray is level
or pointing up ($`\mathbf r\cdot\mathbf n \ge 0`$), or when the hit distance $`t`$ isn't positive. With
nothing, the goal sits straight ahead, at lateral 0. With a point, the goal clips it to lateral
$`-3`$ to $`3`$ m and forward $`0`$ to $`4`$ m. Only the lateral value steers. The goal keeps a forward value,
4 m when there's no point, but nothing downstream reads it.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`(u, v)`$ | gaze pixel, column and row | pixels | depth image | `server/nav/runtime/loop.py` |
| $`f_x, f_y`$ | focal lengths | pixels | depth image | `server/nav/runtime/loop.py` |
| $`c_x, c_y`$ | image center | pixels | depth image | `server/nav/runtime/loop.py` |
| $`\mathbf r`$ | ray through the pixel, one unit deep | none | camera: x right, y down, z forward | `server/nav/runtime/loop.py` |
| $`\mathbf n`$ | floor normal, unit length, pointing up | none | camera | `server/nav/scene/floor.py` |
| $`d`$ | floor offset. The camera's height when $`\mathbf n`$ points up | m | camera | `server/nav/scene/floor.py` |
| $`t`$ | how far along the ray the floor is. The ray is one unit deep, so $`t`$ is the hit's depth | m | camera | `server/nav/runtime/loop.py` |
| $`\mathbf P`$ | the hit point | m | camera | `server/nav/runtime/loop.py` |
| $`\hat{\mathbf z}_c`$ | camera forward, $`(0, 0, 1)`$ | none | camera | `server/nav/scene/floor.py` |
| $`\hat{\mathbf f}`$, $`\hat{\mathbf l}`$ | floor forward and floor right | none | camera | `server/nav/scene/floor.py` (`ground_axes`) |
| lateral, forward | the gaze point on the floor | m | ground: right positive, forward away from the walker | `server/nav/runtime/loop.py` |

| Step | Expression | Why |
|---|---|---|
| 1 | $`\mathbf r = ((u - c_x)/f_x,\ (v - c_y)/f_y,\ 1)`$ | Pinhole camera run backwards. Every point on this ray lands on pixel $`(u, v)`$ |
| 2 | $`\mathbf n\cdot(t\,\mathbf r) + d = 0`$ | Ask which point $`t\,\mathbf r`$ on the ray lies on the floor |
| 3 | $`t = -d / (\mathbf r\cdot\mathbf n)`$ | Solve for $`t`$. Needs $`\mathbf r\cdot\mathbf n < 0`$, a ray heading down |
| 4 | $`\mathbf P = t\,\mathbf r`$ | The hit point in the camera frame |
| 5 | lateral $`= \mathbf P\cdot\hat{\mathbf l}`$, forward $`= \mathbf P\cdot\hat{\mathbf f}`$ | Measure it along the floor axes. Both run parallel to the floor, so the walker's foot below the camera reads as $`(0, 0)`$ |

Here's one with round numbers. Focal lengths 300 px, image center $`(168, 168)`$, gaze at
pixel $`(200, 260)`$. The camera is 1.56 m up, the 2026-10-05 glasses session's median, and pitched 10°
down, so $`\mathbf n = (0, -\cos 10°, -\sin 10°) = (0, -0.98481, -0.17365)`$ and $`d = 1.56`$.

1. $`\mathbf r = (32/300,\ 92/300,\ 1) = (0.10667,\ 0.30667,\ 1)`$.
2. $`\mathbf r\cdot\mathbf n = 0.30667 \times (-0.98481) + 1 \times (-0.17365) = -0.47566`$. Negative, so
   the ray heads down.
3. $`t = 1.56 / 0.47566 = 3.2797`$, and $`\mathbf P = (0.3498,\ 1.0058,\ 3.2797)`$. Check:
   $`\mathbf n\cdot\mathbf P + d \approx 0`$.
4. $`\hat{\mathbf z}_c\cdot\mathbf n = -0.17365`$, so the unnormalized forward is
   $`(0, -0.17101, 0.96985)`$ with length 0.98481, giving $`\hat{\mathbf f} = (0, -0.17365, 0.98481)`$ and
   $`\hat{\mathbf l} = (1, 0, 0)`$.
5. Lateral $`= 0.350`$ m. Forward $`= 1.0058 \times (-0.17365) + 3.2797 \times 0.98481 = -0.1747 + 3.2298 = 3.055`$ m.

The goal is $`(0.35, 3.06)`$, a bit to the right and about 3 m ahead.

**A ray that lands behind.** Same camera and height, but the head pitched 70° down and the gaze near
the bottom of the image, pixel $`(168, 318)`$. Then $`\mathbf n = (0, -0.34202, -0.93969)`$,
$`\hat{\mathbf f} = (0, -0.93969, 0.34202)`$ and $`\mathbf r = (0, 0.5, 1)`$.

1. $`\mathbf r\cdot\mathbf n = 0.5 \times (-0.34202) - 0.93969 = -1.11070`$. Heading down, so it's
   accepted.
2. $`t = 1.56 / 1.11070 = 1.40452`$, $`\mathbf P = (0, 0.70226, 1.40452)`$.
3. Forward $`= 0.70226 \times (-0.93969) + 1.40452 \times 0.34202 = -0.65991 + 0.48038 = -0.180`$ m.

That's 0.18 m behind the walker's foot. The code returns it anyway. The goal clips forward to 0, but
nothing reads the goal's forward value, so the clip changes nothing. The lateral value, 0 here, still
steers the plan as if the walker were looking at a point straight ahead.

> [!WARNING]
> - **No check for behind.** The code comment says it returns nothing when the ray doesn't reach the
>   floor ahead. It actually returns a point for any downward ray, even one landing behind the walker,
>   as in the second example. The goal's clip to forward 0 doesn't catch it, because nothing reads
>   the goal's forward value. A ray landing behind still steers by its lateral value.
> - **Pinhole only.** The gaze pixel is straightened once upstream. This step assumes a perfect pinhole.
> - **"Ahead" is where the camera points.** On the glasses that's where the head faces, not where the
>   body is walking.
> - **The goal's pull is weak.** In an empty scene the goal term can't move the plan by itself: one
>   sidestep costs 0.325 against at most 0.8 the goal can save over the whole path. Section 8 has the
>   numbers.

> [!TIP]
> Ours. A plain ray and plane intersection, with no attribution in the code. It's there so the planner
> can aim where the walker looks, which only the Neon measures.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [12. Scoring the arrow](12_scoring_the_arrow.md) · Next: [14. How the path looks](14_how_the_path_looks.md)
