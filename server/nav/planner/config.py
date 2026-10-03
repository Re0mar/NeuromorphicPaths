"""
What the planner needs to build its surprise field and run the dynamic program over it.

Four of these are the professor's values from the lecture and the paper, and are marked as his.
The rest are walking values we chose and expect to tune once there are real recordings.
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
    # His weight on the lateral kinetic term, kept on purpose. Raising it would stop the arrow
    # pinning, but a steadily measured post then costs less to hit than to dodge. At 0.35 the
    # planner walks into a post 1.5 m ahead, which the safety tests in test_planner_pipeline.py catch.
    lateral_kinetic_weight: float = 0.055
    surprise_cap: float = 2.0e4  # His cap on a single point's surprise.
    walking_speed_mps: float = 1.4  # Ours. Average walking pace, against his 5.0 for a cyclist.
    grid_half_width_meters: float = 3.0  # Ours. Matches the scene grid.
    grid_spacing_meters: float = 0.1  # Ours.
    max_lateral_speed_mps: float = 1.0  # Ours. How fast a walker can sidestep.
    wall_noise_multiplier: float = 3.0  # Ours. A wall is worth avoiding further out than a post.
    predict_motion: bool = False  # Off until the scene's group velocities are trusted.
    # Ours. Clearance at walking pace, not over a closing rate. Times walking speed it must stay
    # under the 1 m at which something counts as close, so 0.7 s at 1.4 m/s is 0.98 m. Replayed with
    # the corridor and hold below: alarm on 61.7 % with 66 changes on the classroom walk, was 76.0 % and 408.
    alarm_time_to_contact_seconds: float = 0.7
    # Ours. Added to the footprint radius on each side, so the corridor is 0.50 m wide either way.
    # On the classroom walk the group raising the alarm sits a median 0.15 m to the side, none past the edge.
    corridor_margin_meters: float = 0.15
    # Ours. How long a raised alarm stays up before it may clear. Takes the classroom walk from 100
    # state changes to 66, and pixel_walk_3 from 207 to 129 over all six of its segments.
    alarm_hold_seconds: float = 0.5
    # Ours. The arrow points at where the path is this far ahead. On the classroom walk the arrow
    # takes 20 values instead of 3, and sits at the sidestep limit on 75.8 % of frames instead of 87.6 %.
    heading_lookahead_seconds: float = 1.0
    goal_distance_meters: float = 4.0  # Old file's goal_dist.
    goal_tolerance_meters: float = 1.5  # Half a typical hallway's width, so the goal term picks among safe paths.
