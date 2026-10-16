"""
Which position source a run uses, and the settings each one reads.

The glasses give orientation from their IMU and no position. These are the two ways a position can
be measured from the depth the pipeline already makes, both from Open3D's RGB-D tracking. Off by
default. At the 2.5 to 3.5 depth frames a second this laptop makes as a walk plays, dense SLAM is
about 1 m off and odometry 0.2 to 0.3 m, and neither changes the planner's inputs yet.
docs/evaluation/neon_position_evaluation.md has the measurements.
"""

# Standard library imports
import math
from dataclasses import dataclass
from enum import Enum


class PositionSource(Enum):
    """Where a glasses frame's position comes from. Values are the spellings the command line accepts."""

    ABSENT = "absent"  # IMU orientation only, no position. What the pipeline has always done.
    DENSE_SLAM = "dense_slam"  # Open3D's dense SLAM: every frame tracked against a fused volume of the ones before.
    ODOMETRY = "odometry"  # Open3D's RGB-D odometry: every frame tracked against the one before, no map.


@dataclass(frozen=True)
class PositionConfig:
    """Settings for both trackers. Each value is the one the evaluation ran with."""

    source: PositionSource = PositionSource.ABSENT
    # Depth past this is left out of tracking. The evaluation's 6 m covers a room and its far wall.
    depth_max_meters: float = 6.0
    # How far a frame's depth may sit from the fused volume's and still count as the same surface.
    # Open3D's default. Dense SLAM tracked worse at 0.15 and 0.3 m on the evaluation walk.
    dense_slam_depth_difference_meters: float = 0.07
    # The same cutoff for odometry, between one frame and the one before. At the 0.6 s between planned
    # frames a walker covers 0.2 to 0.3 m, and Open3D's 0.07 m default dropped the surfaces ahead.
    # 0.3 m cut the evaluation walk's median error from 0.58 to 0.42 m at 1.7 frames a second.
    odometry_depth_difference_meters: float = 0.3
    # The fused volume's voxel. 5 cm is a fifth of the scene's 0.25 m cell.
    voxel_size_meters: float = 0.05
    # Room for about 40,000 blocks of 16 cubed voxels, Open3D's example figure, enough for a building floor.
    voxel_block_count: int = 40000
    # Under this share of matched pixels a frame gets no position. The frame is still fused, because
    # refusing poor frames left the camera stuck in the evaluation.
    min_fitness: float = 0.1
    # A step faster than this between two frames is refused. Faster than any walker, the same 3 m/s
    # the evaluation's track uses to spot a tracker jumping.
    max_step_speed_mps: float = 3.0
    # Dense SLAM's process is replaced this many times at most. Past that it's given up on for the
    # run, so a process that dies on every start doesn't restart on every frame.
    max_restarts: int = 3
    # A tracker call longer than this is treated as a stall. Open3D's tracking hung on some runs in
    # the evaluation, and the walker shouldn't lose the arrow over it.
    stall_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name in ("depth_max_meters", "dense_slam_depth_difference_meters", "odometry_depth_difference_meters", "voxel_size_meters", "max_step_speed_mps", "stall_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number, got {value}")
        if self.max_restarts < 0:
            raise ValueError(f"max_restarts must be zero or more, got {self.max_restarts}")
        if self.voxel_block_count <= 0:
            raise ValueError(f"voxel_block_count must be positive, got {self.voxel_block_count}")
        if not 0.0 <= self.min_fitness < 1.0:
            raise ValueError(f"min_fitness must be in [0, 1), got {self.min_fitness}")
