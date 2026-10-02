"""
What the scene layer needs to turn depth pixels into grouped ground obstacles.

Owned here rather than in nav/config.py so that adding a knob touches the layer that reads it.
The command line composes this into RunConfig, and that is the only thing nav/config.py knows
about it.
"""

# Standard library imports
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    """Defaults carried over from the old monolith wherever it had one."""

    depth_stride: int = 2  # Old file's stride.
    voxel_size_meters: float = 0.05  # Old file's voxel.
    min_depth_meters: float = 0.1  # Old file's depth_to_cloud range. Closer than this is a smudged lens.
    max_depth_meters: float = 30.0  # Old file's depth_to_cloud range. Further than this is sky.
    ankle_height_meters: float = 0.20  # Old file's h_min. Below this is floor, not obstacle.
    head_height_meters: float = 2.00  # Old file's h_max. Above this the walker passes under it.
    cell_size_meters: float = 0.25  # Coarser than the old file's 0.10 grid, because a cell is now one obstacle.
    grid_half_width_meters: float = 3.0  # Old file's x_half.
    grid_forward_meters: float = 6.0  # Old file's z_max.
    min_points_per_cell: int = 2  # One point is as likely to be depth noise as an object.
    noise_window_seconds: float = 0.5  # Half a second of history is what N is measured over.
    min_history_samples: int = 3  # Below this a standard deviation says nothing.
    noise_floor_meters: float = 0.01  # N never goes below this, or surprise divides by almost zero.
    floor_candidate_min_below_camera_meters: float = 0.5  # Old file: only points this far below the camera vote for floor.
    floor_min_candidate_points: int = 200  # Old file: fewer candidates than this and the previous plane is kept.
    floor_ransac_distance_meters: float = 0.05  # Old file's RANSAC inlier distance.
    floor_ransac_iterations: int = 300  # Old file's RANSAC iteration count.
    floor_max_tilt_degrees: float = 35.0  # Old file's floor sanity check.
    floor_min_offset_meters: float = 0.3  # Old file's floor sanity check.
    floor_max_offset_meters: float = 2.2  # A head-worn or hand-held camera is under about two meters. The first Pixel walk's false plane put it at 2.3.
    wall_cell_min_height_meters: float = 1.5  # A cell with points this tall is treated as a wall.
