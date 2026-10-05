"""
What the planner needs to build its surprise field and run the dynamic program over it.

Six of these are the professor's values from the lecture and the paper, and are marked as his.
The rest are walking values we chose. The ones measured against recorded walks carry the numbers in
their comments.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum


class GoalMode(Enum):
    """What the planner aims at when nothing is in the way. Values are the command line spellings."""

    AHEAD = "ahead"
    GAZE = "gaze"


@dataclass(frozen=True)
class PlannerConfig:
    """The field, the dynamic program, the arrow's lookahead, and the alarm's corridor, threshold and hold."""

    time_step_seconds: float = 0.1  # His dt.
    horizon_seconds: float = 3.8  # His horizon.
    clearance_epsilon_meters: float = 0.06  # His epsilon, the floor under S.
    noise_epsilon_meters: float = 1.0e-6  # His numeric floor under N, so a zero N is a zero surprise, not a NaN.
    # Ours, raised from his 0.055 so the plan stops sidestepping at full speed whenever anything is
    # ahead. His surprise alone can't afford it: a steadily measured post costs less to hit than to
    # dodge from 0.35. With the contact term the post safety test's posts are cleared through 7 and
    # hit at 8. 6.5 is the highest weight that still clears them with the sway anywhere from 0.05 to
    # 0.20 m and the near noise at either recorded walk's median. The worst of those, 0.20 m of sway
    # at 0.0337 m of noise, hits at 7. The safety tests in test_planner_pipeline.py pin all of it.
    # Replayed on the seeded floor fit, against his weight without the contact term: with something
    # 3 to 5.32 m ahead the arrow sits at the sidestep limit on 59.0 % of classroom frames (was 94.9)
    # and 58.2 % on pixel_walk_3 (was 81.1), still over the 50 % aimed for. With nothing in the way
    # it never does on either walk (was 9.1 and 15.3).
    lateral_kinetic_weight: float = 6.5
    surprise_cap: float = 2.0e4  # His cap on a single point's surprise.
    walking_speed_mps: float = 1.4  # Ours. Average walking pace, against his 5.0 for a cyclist.
    grid_half_width_meters: float = 3.0  # Ours. Matches the scene grid.
    grid_spacing_meters: float = 0.1  # Ours.
    max_lateral_speed_mps: float = 1.0  # Ours. How fast a walker can sidestep.
    wall_noise_multiplier: float = 3.0  # Ours. A wall is worth avoiding further out than a post.
    predict_motion: bool = False  # Off until the scene's group velocities are trusted.
    # Ours. Clearance at walking pace, not over a closing rate. Times walking speed it must stay
    # under the 1 m at which something counts as close, so 0.7 s at 1.4 m/s is 0.98 m. Replayed with
    # the corridor and hold below: alarm on 60.0 % with 66 changes on the classroom walk, was 76.0 % and 408.
    alarm_time_to_contact_seconds: float = 0.7
    # Ours. The body, a shoulder half-width with room for arm swing, narrower than the footprint the
    # planner steers with. Two things read it. The alarm looks this far to each side, and the contact
    # term measures overlap against it. At the planner's 0.35 m plus 0.15 m, a doorway about 0.9 m
    # wide on the 2026-10-03 apartment walk raised the alarm on 93 % of its frames. At 0.30 m it is
    # 36 %, and things the walker stood in front of still raise it on 84 %.
    body_half_width_meters: float = 0.30
    # Ours. Adds the surprise of the body touching something beside his surprise. Off only to compare
    # against the planner without it.
    contact_term_enabled: bool = True
    # Ours, and an assumption rather than a measurement. How far a walker drifts sideways from the
    # line the arrow asks for, one standard deviation. No recorded walk had anyone steering by the
    # arrow, so it cannot be measured from them yet.
    walker_sway_meters: float = 0.10
    # Ours. The most one point's contact surprise may reach. With the 0.30 m half-width it binds only
    # for a sway under about 0.031 m. At 0.10 m no point can cost more than 6.61.
    contact_surprise_cap: float = 50.0
    # Ours. How long a raised alarm stays up before it may clear. Takes the classroom walk from 108
    # state changes to 66, and the last segment of pixel_walk_3 from 55 to 35.
    alarm_hold_seconds: float = 0.5
    # Ours. The arrow points at where the path is this far ahead. On the classroom walk the arrow
    # takes 20 values instead of 3, and sits at the sidestep limit on 75.8 % of frames instead of 87.6 %.
    heading_lookahead_seconds: float = 1.0
    # Ours. A prior toward the plan the walker was already following, held in the walker's own frame.
    # His dynamic program keeps nothing between frames, and on the recorded walks the arrow swung from
    # one sidestep limit to the other 65 to 127 times a minute, at his weight and at ours alike.
    # Off only to compare against the planner without memory.
    previous_plan_prior_enabled: bool = True
    # Ours. How far from the previous plan a position may be before the prior counts it as surprising,
    # one standard deviation. Swept 2026-10-05 on contact_walk_1, pixel_walk_3 and the classroom walk,
    # last segments. Over the first second, every spread from 0.125 to 0.5 m held every limit and cut the
    # full swings from 77, 76 and 105 a minute to 0 to 6.5. At 0.25 m: 0, 1.3 and 0 a minute, the 90th
    # percentile of plan disagreement 0.254, 0.449 and 0.273 m (was 1.176, 1.102, 1.303), and the arrow at
    # its limit with something 3 to 5.32 m ahead on 1.6 % of pixel_walk_3's band frames and 30.3 % of the
    # classroom's (was 58.2 and 59.0). 0.25 rather than the sharpest 0.125, for margin: the arrow stops
    # coming back to straight once the obstacle is gone at 0.05 m, and the swings climb again from 1.0 m.
    previous_plan_spread_meters: float = 0.25
    # Ours. The prior covers the rows the walker reaches in this long, about as long as they need to act
    # on the arrow. Rows beyond it stay free to replan. Over the whole 3.8 s horizon, spreads of 0.125 to
    # 0.5 m held the arrow at its limit on 12 to 17 % of the classroom's clear-corridor frames, over the
    # 10 % allowed, because the far rows were committed to before anything new came into view.
    previous_plan_prior_seconds: float = 1.0
    # Ours. A previous plan older than this says nothing about the next one. Half a second is 0.7 m of
    # walking, and about fifteen frames dropped at 30 Hz.
    previous_plan_max_gap_seconds: float = 0.5
    goal_distance_meters: float = 4.0  # Old file's goal_dist.
    goal_tolerance_meters: float = 1.5  # Half a typical hallway's width, so the goal term picks among safe paths.
