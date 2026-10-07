[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [13. Gaze on the floor](13_gaze_on_the_floor.md) · Next: [How we stick to the professor's math](15_professors_math.md)

# 14. How the path looks

**The plain idea.** The planned path is drawn on the depth view and on the web page. Its color says how
close something in the walker's way is: blue when nothing is near, fully red once the alarm raises,
when something is 0.7 s away at walking pace. How solid the fill is says how much what the camera saw bent the plan: 70 % opaque
when the scene changed nothing, fully solid once the scene information from section 10 reaches 1 bit.

**An analogy.** A kettle with a color-changing base: blue when cold, red when it's hot enough to burn.
The analogy stops working at the top end. A kettle keeps getting hotter past red, while the path stops changing
once it's red, so it can't show the difference between 0.7 s and 0.3 s to contact.

> [!NOTE]
> **Ingredients**
> - The avoidance surprise $`{\color{purple}{U}}`$ in bits, from section 9. It comes from the time to
>   contact $`{\color{red}{\tau}}`$, the clearance $`{\color{teal}{S}}`$ of the nearest thing in the 0.30 m
>   half-width corridor divided by walking speed 1.4 m/s.
> - The scene information $`I`$, from section 10.
> - Two end colors, blue (47, 111, 255) and red (235, 48, 48), and the OKLab conversion published by
>   Björn Ottosson.
> - Nothing else. The colors of obstacles and walls are fixed.

## Color

```math
{\color{purple}{U}} = \frac{(1\ \text{s} / {\color{red}{\tau}})^2}{2\ln 2}, \qquad \phi = \min\Big(1,\ \frac{\color{purple}{U}}{U_{\text{red}}}\Big), \qquad \text{color} = \mathrm{OKLab}^{-1}\big(L_0 + \phi\,(L_1 - L_0)\big)
```

$`U_{\text{red}}`$ is the same formula at the alarm's threshold, $`{\color{red}{\tau}} = 0.7`$ s:
$`(1/0.7)^2 / (2\ln 2) = 2.0408 / 1.3863 = 1.47`$ bits. The planner works it out from the threshold
and hands it to both displays, so changing the threshold moves the red with it.

OKLab is a color space built so that equal steps look like equal changes to the eye. Blending there
keeps the middle of the ramp from going muddy, which a plain RGB average tends to do.

## Fill opacity

```math
\text{opacity} = 0.7 + 0.3\,\min\Big(1,\ \frac{I}{1\ \text{bit}}\Big), \qquad \text{border opacity} = 0.6
```

The border stays at 0.6 so the path's direction is visible even when the fill is faint.

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`{\color{purple}{U}}`$ | avoidance surprise, `avoidance_surprise_bits` | bits | none | `server/nav/planner/alarm.py` |
| $`{\color{red}{\tau}}`$ | time to contact at walking pace | s | none | `server/nav/planner/alarm.py` |
| $`{\color{teal}{S}}`$ | clearance of the nearest thing in the corridor, from the 0.35 m footprint | m | ground | `server/nav/planner/alarm.py` |
| $`\phi`$ | how far along the blue-to-red ramp, 0 to 1 | none | none | `server/nav/sinks/path_style.py` |
| $`U_{\text{red}}`$ | surprise at which the path is fully red, the alarm's threshold in bits, 1.47 (`path_red_from_bits`) | bits | none | `server/nav/planner/alarm.py` |
| $`L_0`$, $`L_1`$ | blue and red in OKLab | none | none | `server/nav/sinks/path_style.py` |
| $`I`$ | scene information, `scene_information_bits` | bits | none | `server/nav/planner/information.py` |
| 0.7, 1, 0.6 | fill floor, fill full at 1 bit, border | none | none | `server/nav/sinks/path_style.py` |

| Step | Expression | Why |
|---|---|---|
| 1 | $`{\color{red}{\tau}} = {\color{teal}{S}} / 1.4`$ | Time to reach the nearest thing at walking pace |
| 2 | $`{\color{purple}{U}} = (1 / {\color{red}{\tau}})^2 / (2\ln 2)`$ | His avoidance form in bits, section 9 |
| 3 | $`\phi = \min(1, {\color{purple}{U}} / U_{\text{red}})`$ | $`U_{\text{red}}`$ is $`{\color{red}{\tau}} = 0.7`$ s, where the alarm raises: $`2.0408 / 1.3863 = 1.4721`$ |
| 4 | Convert both end colors to OKLab | So the blend looks even |
| 5 | $`L_0 + \phi\,(L_1 - L_0)`$ | Straight-line blend between them |
| 6 | Back to sRGB, round to whole numbers | What the screen takes |

The end colors in OKLab are $`L_0 = (0.5881, -0.0267, -0.2227)`$ and
$`L_1 = (0.6114, 0.1976, 0.0999)`$. Walking toward a single post in the corridor gives this:

| Clearance $`{\color{teal}{S}}`$ | $`{\color{red}{\tau}} = {\color{teal}{S}} / 1.4`$ | $`{\color{purple}{U}}`$ | $`\phi`$ | Color |
|---|---|---|---|---|
| nothing in the corridor | none | 0 | 0 | (47, 111, 255), blue |
| 2.80 m | 2.00 s | $`0.25 / 1.3863 = 0.180`$ | 0.12 | (85, 113, 233) |
| 1.96 m | 1.40 s | $`0.5102 / 1.3863 = 0.368`$ | 0.25 | (114, 113, 210) |
| 1.39 m | 0.99 s | $`1.0204 / 1.3863 = 0.736`$ | 0.5 | (159, 105, 163) |
| 0.98 m | 0.70 s | $`2.0408 / 1.3863 = 1.472`$ | 1 | (235, 48, 48), red, and the alarm raises below this |

These are what the code's own functions return for each row.

At $`\phi = 0.5`$ the OKLab blend is $`(0.5998, 0.0854, -0.0614)`$, which converts back to
(159.2, 105.5, 163.3) and rounds to (159, 105, 163). Its lightness 0.5998 sits halfway between the two
ends. A plain RGB average would give $`((47 + 235)/2, (111 + 48)/2, (255 + 48)/2) = (141, 80, 152)`$
instead.

For the fill, take $`I = 0.37`$ bits. The opacity is $`0.7 + 0.3 \times 0.37 = 0.811`$. At $`I = 0.5`$ it's
$`0.7 + 0.15 = 0.85`$, and anything from 1 bit up is fully solid.

The 1 bit hasn't moved, but on 2026-10-07 the scene information started being computed from
chances in base e (section 10), which reads higher than the old base-2 version on the same scene.
So the fill is solid more often than it was. On every planned frame of `pixel_walk_3` it went from
44.8 % solid to 52.9 %, and on `contact_walk_1` from 55.7 % to 67.5 %.

## Fixed colors and sizes

| Thing | Drawn as | Where in the code |
|---|---|---|
| Obstacle groups | amber (255, 200, 0) | `server/nav/sinks/path_style.py` |
| Walls | magenta (255, 0, 255) | `server/nav/sinks/path_style.py` |
| Path border | the path's color at opacity 0.6 | `server/nav/sinks/path_style.py` |
| Path width | twice the body half-width, $`2 \times 0.30 = 0.60`$ m | `server/nav/sinks/web_messages.py` |
| Field cells marked as seen | cells whose floor spot projects inside the depth image | `server/nav/sinks/floor_geometry.py` |

The depth view and the web page call the same style functions, so they always agree.

> [!WARNING]
> - **The ramp saturates.** Everything closer than 0.98 m looks the same red.
> - **Red and the alarm line up, but the alarm holds.** The path reaches full red at the same 0.7 s
>   the alarm raises at. The alarm then stays up for at least 0.5 s, while the color follows each
>   frame, so the color can fall back before the alarm clears.
> - **Before 2026-10-07 the path went red at 0.72 bits**, one second to contact, ahead of the alarm.
>   From 1.40 m down to 0.98 m of clearance the path was fully red with the alarm still off.
> - **Drawn narrower than it's planned.** The path is drawn 0.60 m wide, but the clearance it keeps is
>   measured from the 0.35 m footprint radius, 0.70 m across.
> - **Seen means in view.** A cell behind an obstacle still counts as seen, because only the field of
>   view is checked.

> [!TIP]
> Ours. The ramps, their end points and the fixed colors are project choices. The inputs aren't: the
> color reads his avoidance form from section 9, and the opacity reads the scene information from
> section 10, which the code describes as the professor's Bayesian surprise of the plan. The OKLab
> matrices are Björn Ottosson's published ones.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [13. Gaze on the floor](13_gaze_on_the_floor.md) · Next: [How we stick to the professor's math](15_professors_math.md)
