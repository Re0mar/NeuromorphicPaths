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
    """The field, the dynamic program, and the alarm threshold."""

    time_step_seconds: float = 0.1  # His dt.
    horizon_seconds: float = 3.8  # His horizon.
    clearance_epsilon_meters: float = 0.06  # His epsilon, the floor under S.
    noise_epsilon_meters: float = 1.0e-6  # His numeric floor under N, so a zero N is a zero surprise, not a NaN.
    lateral_kinetic_weight: float = 0.055  # His weight on the lateral kinetic term.
    surprise_cap: float = 2.0e4  # His cap on a single point's surprise.
    walking_speed_mps: float = 1.4  # Ours. Average walking pace, against his 5.0 for a cyclist.
    grid_half_width_meters: float = 3.0  # Ours. Matches the scene grid.
    grid_spacing_meters: float = 0.1  # Ours.
    max_lateral_speed_mps: float = 1.0  # Ours. How fast a walker can sidestep.
    wall_noise_multiplier: float = 3.0  # Ours. A wall is worth avoiding further out than a post.
    predict_motion: bool = False  # Off until the scene's group velocities are trusted.
    alarm_time_to_contact_seconds: float = 1.0  # Under a second to contact turns the display red.
    goal_distance_meters: float = 4.0  # Old file's goal_dist.
    goal_tolerance_meters: float = 1.5  # Half the corridor width, so the goal term picks among safe paths.
