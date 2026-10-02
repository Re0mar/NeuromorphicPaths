"""
The shapes that cross layer boundaries, and the two protocols that are those boundaries.

Camera frame convention, the same one OpenCV and Depth Anything use: x right, y down, z forward.
Ground frame, derived from the fitted floor plane: lateral positive to the right, forward away
from the walker.

Nothing in this module may import a device, a model or a server. Every layer imports it, so an
import added here reaches all of them.
"""

# Standard library imports
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

# Third party imports
import numpy as np


@dataclass(frozen=True)
class Plane:
    """A flat surface, normally the ground, written so that normal . point + offset == 0 on it."""

    normal: np.ndarray
    offset_meters: float


class FloorSource(Enum):
    """Where the scene's floor for a frame came from. Values are the words the debug line prints."""

    SUPPLIED = "supplied"  # The source sent a plane and it passed the gate.
    FITTED = "fitted"  # Fitted from this frame's cloud, because no plane came or the one that came was refused.
    PREVIOUS = "previous"  # Neither of the above produced a floor, so the last frame's stands.


# Up in the world frame a positioned pose describes. ARCore's world, the only one a source supplies
# today, has y up, and docs/arcore_wire_format.md states it so the next source can match. The
# floor fit measures "level" and "below the camera" against this once a pose places the camera.
WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Pose:
    """Where the camera is pointing, and where it is when anything knows that.

    The orientation rotates camera-frame vectors into the world. When has_position is true that
    world is one with WORLD_UP up, which is how the scene knows which way gravity points on a
    phone held sideways.
    """

    orientation: np.ndarray
    position: np.ndarray | None
    has_position: bool

    def __post_init__(self) -> None:
        # has_position is what every downstream branch reads to decide between the body frame and
        # the world frame. An inconsistent pair would silently pick the wrong one.
        if self.has_position != (self.position is not None):
            raise ValueError(
                f"has_position is {self.has_position} but position is "
                f"{'present' if self.position is not None else 'None'}"
            )


@dataclass(frozen=True)
class DepthFrame:
    """One frame of depth in meters, with everything the scene layer needs to place it in space."""

    timestamp_seconds: float
    depth_meters: np.ndarray
    intrinsics: np.ndarray
    pose: Pose
    ground_plane: Plane | None
    gaze_pixel: np.ndarray | None

    def __post_init__(self) -> None:
        # Every source builds one of these, and a wrongly shaped array from any of them would
        # otherwise surface as a broadcasting error deep inside unprojection.
        if self.depth_meters.ndim != 2:
            raise ValueError(f"depth_meters must be (height, width), got shape {self.depth_meters.shape}")
        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3, 3), got shape {self.intrinsics.shape}")
        if self.gaze_pixel is not None and self.gaze_pixel.shape != (2,):
            raise ValueError(f"gaze_pixel must be (2,) when present, got shape {self.gaze_pixel.shape}")


@dataclass(frozen=True)
class ObstaclePoint:
    """One surviving ground point, carrying what the planner needs to score it.

    S is the clearance from the edge of the walker's footprint, not from the walker's center.
    N is how much that clearance has been wobbling, which is what makes a reading surprising
    rather than merely close.
    """

    lateral_meters: float
    forward_meters: float
    group_id: int
    clearance_meters: float
    noise_scale_meters: float
    closing_rate_mps: float | None
    velocity_mps: np.ndarray | None
    is_wall: bool
    # The nearest point as the camera saw it, (3,) camera frame. For the depth view only. The
    # planner never reads it, and a test that builds a point by hand gives it zeros.
    camera_point: np.ndarray


@dataclass(frozen=True)
class ObstacleSet:
    """Every group in view this frame."""

    timestamp_seconds: float
    points: tuple[ObstaclePoint, ...]
    groups_in_view: int


@dataclass(frozen=True)
class DebugView:
    """
    What a person tuning the planner needs to see beside the path: the planner's input.

    The frame the path was planned for, the obstacles the scene found in it, the floor the scene
    used and where it came from, and the walking speed the planner assumed. The floor is here
    because a renderer lays the path on it, and the speed because a path is offsets against time
    and the floor is meters.
    """

    frame: DepthFrame
    obstacles: ObstacleSet
    floor: Plane
    floor_source: FloorSource
    walking_speed_mps: float


@dataclass(frozen=True)
class PlannedPath:
    """The path the planner chose, in lateral offsets against time, plus what a display needs."""

    timestamp_seconds: float
    times_seconds: np.ndarray
    lateral_offsets_meters: np.ndarray
    first_heading_radians: float
    alarm: bool
    cumulative_cost_bits: float

    def __post_init__(self) -> None:
        # This is the last shape before a sink serializes it, so a non-finite value caught here
        # is one that never reaches a JSON encoder that would have written a bare NaN.
        if self.times_seconds.shape != self.lateral_offsets_meters.shape:
            raise ValueError(
                f"times_seconds {self.times_seconds.shape} and lateral_offsets_meters "
                f"{self.lateral_offsets_meters.shape} must have the same shape"
            )
        if not np.all(np.isfinite(self.times_seconds)):
            raise ValueError("times_seconds contains a non-finite value")
        if not np.all(np.isfinite(self.lateral_offsets_meters)):
            raise ValueError("lateral_offsets_meters contains a non-finite value")
        if not np.isfinite(self.first_heading_radians):
            raise ValueError(f"first_heading_radians must be finite, got {self.first_heading_radians}")
        if not np.isfinite(self.cumulative_cost_bits):
            raise ValueError(f"cumulative_cost_bits must be finite, got {self.cumulative_cost_bits}")


class DepthFrameSource(Protocol):
    """What every source is, seen from the loop. The only thing the loop knows about a device."""

    def frames(self) -> Iterator[DepthFrame]: ...

    def close(self) -> None: ...


class PathSink(Protocol):
    """What every display is, seen from the loop."""

    def publish(self, path: PlannedPath) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class DebugSink(PathSink, Protocol):
    """A sink that can also draw the surprise field and the planner's input.

    Separate from PathSink so the field, which is a grid the size of the planner's horizon, and
    the view, which carries a whole depth frame, only go where a person is looking at them. A
    phone over TCP gets the path alone.
    """

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None: ...
