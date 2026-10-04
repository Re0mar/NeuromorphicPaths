"""
One planned frame as the evaluation reads it.

The replay builds these from a recording. A test builds them by hand. Either way the arrow is the
planner's own number, and the rest is what the scene knew about where the camera was and what it saw.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.types import ObstacleSet, Plane


@dataclass(frozen=True)
class PlannedFrame:
    """One frame the planner planned. Fields are None where the frame had no world pose."""

    timestamp_seconds: float
    # PlannedPath.first_heading_radians, exactly as the server returned it.
    arrow_radians: float
    # The phone's forward on the floor, in the world. The arrow is measured from it.
    forward_axis_world: np.ndarray | None
    camera_position_world: np.ndarray | None
    # 3 by 3, camera axes (x right, y down, z forward) into the world.
    camera_rotation_world: np.ndarray | None
    floor_world: Plane | None
    # 3 by 3, of the depth image.
    intrinsics: np.ndarray
    image_width_pixels: int
    image_height_pixels: int
    # Lateral and forward relative to the phone's forward axis on the floor, as the scene reports them.
    obstacles: ObstacleSet
