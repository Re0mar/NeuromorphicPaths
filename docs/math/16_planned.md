[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [How we stick to the professor's math](15_professors_math.md)

# Planned, and off by default

Two pieces of math aren't part of the planner as it runs today. One is in the code and switched off.
The other is planned. Both will be switched on and off live, and both stay off unless someone turns
them on. Each gets a full section here once it ships.

## Motion prediction

Right now the planner treats everything as standing still. At each future slice it moves every
obstacle toward the walker at walking pace and nothing else. Motion prediction also slides an
obstacle along its own measured velocity, so someone walking toward you is planned for where
they'll be, not where they are. It's in the code as `predict_motion` and is off.

```math
y(t_k) = y + v_y\, t_k - v_w\, t_k, \qquad x(t_k) = x + v_x\, t_k
```

| Symbol | Plain English | Units | Frame | Where in the code |
|---|---|---|---|---|
| $`x`$, $`y`$ | The obstacle's sideways and forward position now | m | walker's ground frame, right and ahead positive | `server/nav/planner/field.py` |
| $`v_x`$, $`v_y`$ | The obstacle's measured velocity, 0 when prediction is off or there's no velocity | m/s | walker's ground frame | `server/nav/scene/history.py`, `server/nav/scene/pipeline.py` |
| $`v_w`$ | Walking speed, 1.4 | m/s | | `server/nav/planner/config.py:41` |
| $`t_k`$ | Time of slice $`k`$, $`k \cdot 0.1`$ s | s | | `server/nav/planner/field.py` |

A person 4.0 m ahead walks toward the walker at 1.0 m/s, so $`v_y = -1.0`$. Two seconds out, at
slice 20:

| Step | Expression | Result |
|---|---|---|
| Prediction off | $`4.0 - 1.4 \times 2.0`$ | 1.2 m ahead |
| Prediction on | $`4.0 - 1.0 \times 2.0 - 1.4 \times 2.0`$ | 0.8 m behind, already passed |

So with prediction off, the plan thinks there's still 1.2 m to go when the two have already met.

**Why it's off.** The velocity is how far the middle of a 0.25 m scene cell's points moved between
the last two frames. Those points change when part of something steps into or out of view, even if
it doesn't move. So a standing object that's partly hidden on one frame and fully visible on the next
gets a velocity it doesn't have. It stays off until the measured velocities have been checked on a
recorded walk with someone walking toward the walker.

**Planned:** an on and off button on the web page, so it can be switched during a walk. The switch
will be logged with the frames, so a replay of the walk switches it at the same moments and gives
the same plans.

> [!WARNING]
> A velocity needs a position, because the cells have to stay fixed in the world between frames.
> Only the Pixel route has a position. On the Neon glasses no obstacle ever gets a velocity, so
> switching prediction on changes nothing there.

> [!TIP]
> Ours. The professor's avoidance example has one lead car whose gap is measured directly. Sliding
> obstacles by a measured velocity is our addition, and it changes where the points are, not his
> surprise formula.

## A cost for leaving the path

The planner currently doesn't know where the path is. A sidewalk, the grass beside it and a flower
bed all count the same, as long as nothing is standing on them. The plan is to add a cost for
stepping off the walkable path, or onto ground that's less preferred, as one more observation the
planner weighs when it works out its path.

The camera picture would be read by the path model from the retired phone app, which marks which
part of a picture is walkable path. That model is kept in git history, and the root README's
Retired section says how to get it back. The cost would go into the field next to the other terms,
so it's one more likelihood term that adds, the way independent observations add in his
posterior. Like motion prediction, it'll be switched live from the web page, off by default, and
logged so replays match.

> [!TIP]
> Ours. The professor's public material says nothing about terrain, path edges, preferred surfaces
> or risk areas. The only spatial cost it models is keeping clear of one lead car. So an off-path
> cost is our extension. How it's computed isn't designed yet. Built as one more term that adds to
> the others, the way his likelihood terms add (College 5, PDF page 54), it would fit his math, and
> it should be presented as ours.

---

[Contents](README.md#contents) · [The colors](README.md#the-colors) · Previous: [How we stick to the professor's math](15_professors_math.md)
