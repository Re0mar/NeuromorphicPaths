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

    The orientation rotates camera-frame vectors into the world. When orientation_is_gravity_aligned
    is true that world has WORLD_UP up, which is how the scene knows which way gravity points on a
    phone held sideways. That needs no position: the Neon's IMU gives gravity and no position at all.
    has_position decides something else, whether points are placed in the world or stay in the
    camera frame.
    """

    orientation: np.ndarray
    position: np.ndarray | None
    has_position: bool
    # Whether the orientation rotates into a world whose up is WORLD_UP, which is what lets the
    # scene read gravity from it. True for a device that tracks against gravity, by its own pose
    # or by an IMU. False is the conservative answer, and it means the scene falls back to the
    # image's own up, so a source that does not say gets what a plain video file gets.
    orientation_is_gravity_aligned: bool = False

    def __post_init__(self) -> None:
        # has_position is what every downstream branch reads to decide between the body frame and
        # the world frame. An inconsistent pair would silently pick the wrong one.
        if self.has_position != (self.position is not None):
            raise ValueError(
                f"has_position is {self.has_position} but position is "
                f"{'present' if self.position is not None else 'None'}"
            )


@dataclass(frozen=True)
class FrameTiming:
    """When one frame was captured and how far along the laptop it has got, on the laptop clock.

    Seconds since the Unix epoch, from nav.clock. Capture is the sensor's own stamp moved onto that
    clock, None when the offset between the two clocks is unknown. Depth ready is None for a source
    whose frames arrive with depth already in them.
    """

    capture_seconds: float | None
    arrival_seconds: float
    depth_ready_seconds: float | None

    def __post_init__(self) -> None:
        # Every share in the timing log is measured from arrival, so a record without one has
        # nothing to measure from. The type says float, and nothing else would stop a None.
        if self.arrival_seconds is None:
            raise ValueError("arrival_seconds is required, got None")
        for field_name in ("capture_seconds", "arrival_seconds", "depth_ready_seconds"):
            value = getattr(self, field_name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
        # Both of these are read off the same monotonic laptop clock, so the order cannot break
        # unless the code stamped them in the wrong order. Capture is not checked against arrival,
        # because a clock offset a few milliseconds off can legitimately put capture after arrival.
        if self.depth_ready_seconds is not None and self.depth_ready_seconds < self.arrival_seconds:
            raise ValueError(
                f"depth_ready_seconds {self.depth_ready_seconds} is before arrival_seconds {self.arrival_seconds}"
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
    # Measurement only. Nothing in the scene or the planner reads it, and a frame without it is a
    # complete frame.
    timing: FrameTiming | None = None

    def __post_init__(self) -> None:
        # Every source builds one of these, and a wrongly shaped array from any of them would
        # otherwise surface as a broadcasting error deep inside unprojection.
        if not np.isfinite(self.timestamp_seconds):
            # The clearance history, the planner's previous plan and the timing log all order
            # frames by this. A NaN breaks all three without an error from any of them.
            raise ValueError(f"timestamp_seconds must be finite, got {self.timestamp_seconds}")
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
    used and where it came from, the walking speed the planner assumed, the body's half-width, and
    the avoidance surprise at which the path turns fully red. The floor is here because a renderer
    lays the path on it, the speed because a path is offsets against time and the floor is meters,
    the half-width because the path is drawn as wide as the body that walks it, and the red point
    because it is the alarm's threshold in bits, which only the planner's config knows.
    """

    frame: DepthFrame
    obstacles: ObstacleSet
    floor: Plane
    floor_source: FloorSource
    walking_speed_mps: float
    body_half_width_meters: float
    path_red_from_bits: float


@dataclass(frozen=True)
class PlannedPath:
    """The path the planner chose, in lateral offsets against time, plus what a display needs."""

    timestamp_seconds: float
    times_seconds: np.ndarray
    lateral_offsets_meters: np.ndarray
    # The heading from the walker to where the path is heading_lookahead_seconds ahead, positive right.
    # Not the path's first step, which moves too little to point anywhere useful.
    lookahead_heading_radians: float
    alarm: bool
    # The chosen path's summed cost, in bits. The planner sums natural logs and converts once, so this
    # is never a natural-log value.
    cumulative_cost_bits: float
    # How far what the camera saw moved the plan from what it would do with nothing in view, at the
    # arrow's lookahead. Not a confidence: an empty corridor gives a sure plan and 0 bits.
    scene_information_bits: float
    # How soon the walker reaches the nearest thing in its way, in the course's avoidance form.
    # 0 with nothing in the way, 0.72 at one second to contact.
    avoidance_surprise_bits: float
    # The stereo cue, decided by the planner's audio module so a display only applies it. Where the
    # danger is while the alarm is up, -1 left to +1 right, None while it is down. Then how loud each
    # ear should be, from the cue's floor to 1. Full volume in both ears is no cue, which is what a
    # path from before the cue existed reads back as.
    alarm_pan: float | None = None
    ear_gain_left: float = 1.0
    ear_gain_right: float = 1.0

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
        if not np.isfinite(self.lookahead_heading_radians):
            raise ValueError(f"lookahead_heading_radians must be finite, got {self.lookahead_heading_radians}")
        if not np.isfinite(self.cumulative_cost_bits):
            raise ValueError(f"cumulative_cost_bits must be finite, got {self.cumulative_cost_bits}")
        for field_name in ("scene_information_bits", "avoidance_surprise_bits"):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and zero or more, got {value}")
        if self.alarm_pan is not None and (not np.isfinite(self.alarm_pan) or not -1.0 <= self.alarm_pan <= 1.0):
            raise ValueError(f"alarm_pan must be from -1 to 1, got {self.alarm_pan}")
        if self.alarm_pan is not None and not self.alarm:
            # A side with no alarm would pan a danger tone that is not sounding.
            raise ValueError(f"alarm_pan must be None while the alarm is down, got {self.alarm_pan}")
        for field_name in ("ear_gain_left", "ear_gain_right"):
            value = getattr(self, field_name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be from 0 to 1, got {value}")


class DepthFrameSource(Protocol):
    """What every source is, seen from the loop. The only thing the loop knows about a device."""

    def frames(self) -> Iterator[DepthFrame]: ...

    def close(self) -> None: ...


class PathSink(Protocol):
    """What every display is, seen from the loop.

    start is called once, before the source yields its first frame, so a sink that listens is
    listening from the start of the run. A page or a phone that arrives before the first planned
    frame would otherwise find nothing to connect to, and on a still phone the first frame can
    be minutes away.
    """

    def start(self) -> None: ...

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
