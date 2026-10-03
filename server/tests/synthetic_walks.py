"""
Synthetic walks and phone frames for the evaluation tests.

A walk is integrated from a heading function at the phone's frame rate, in a y-up world. Heading zero
faces -z and positive turns toward +x, the convention `floor_heading_radians` uses, so a test can state
its expected sides and angles from the walk's own construction.

A phone frame is held upright, the way the walks were recorded: the depth image's long side runs
vertically, so the image spans about 69 degrees up and down and about 42 side to side.
"""

# Standard library imports
import dataclasses
import json
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.frames import PlannedFrame
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.types import WORLD_UP, DepthFrame, ObstaclePoint, ObstacleSet, Plane, Pose
from synthetic_depth import CAMERA_HEIGHT_METERS as SYNTHETIC_CAMERA_HEIGHT_METERS
from synthetic_depth import PITCH_DEGREES as SYNTHETIC_PITCH_DEGREES
from synthetic_depth import clean_scene, degrade_without_floor

FRAME_RATE_HZ = 30.0
WALKING_SPEED_MPS = 1.4
CAMERA_HEIGHT_METERS = 1.2
STRIDE_RATE_HZ = 1.8
# Measured on the three recorded walks: the frame-to-frame second difference of the ARCore position
# has a median of 1.1 to 1.5 mm. The larger tail is the phone moving with the stride, which
# sway_meters models separately.
POSE_JITTER_METERS = 0.002
STRIDE_SWAY_METERS = 0.03
FLOOR = Plane(normal=WORLD_UP.copy(), offset_meters=0.0)
# The recorded depth image: 160 by 90, focal length about 116 pixels.
IMAGE_WIDTH_PIXELS = 160
IMAGE_HEIGHT_PIXELS = 90
INTRINSICS = np.array([[116.0, 0.0, 80.0], [0.0, 116.0, 45.0], [0.0, 0.0, 1.0]])
# Pointing at the floor ahead, as the walks were recorded.
PITCH_DEGREES = 35.0


def direction(heading_radians: float) -> np.ndarray:
    """
    The horizontal unit vector a heading points along, written out rather than taken from the
    code under test, so a sign error there can't move the fixtures with it.
    """
    return np.array([np.sin(heading_radians), 0.0, -np.cos(heading_radians)])


def walk(
    heading_degrees_at,
    duration_seconds: float,
    speed_mps_at=None,
    noise_meters: float = 0.0,
    sway_meters: float = 0.0,
    seed: int = 0,
):
    """
    Integrate a walk from a heading function, at the frame rate a phone records at.

    :param noise_meters: White noise per frame on each axis.
    :param sway_meters: Side-to-side sway of the phone with each stride, at STRIDE_RATE_HZ.
    :return: (times, positions), positions in a y-up world starting at the origin, camera height up.
    """
    times = np.arange(0.0, duration_seconds, 1.0 / FRAME_RATE_HZ)
    headings = np.radians([heading_degrees_at(time) for time in times])
    speeds = np.array([WALKING_SPEED_MPS if speed_mps_at is None else speed_mps_at(time) for time in times])
    directions = np.array([direction(heading) for heading in headings])
    steps = directions * (speeds / FRAME_RATE_HZ)[:, None]
    positions = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)[:-1]])
    positions[:, 1] = CAMERA_HEIGHT_METERS
    if sway_meters:
        rights = np.cross(directions, WORLD_UP)
        positions = positions + rights * (sway_meters * np.sin(2.0 * np.pi * STRIDE_RATE_HZ * times))[:, None]
    if noise_meters:
        positions = positions + np.random.default_rng(seed).normal(0.0, noise_meters, positions.shape)
    return times, positions


def ramp(start_seconds: float, turn_seconds: float, total_degrees: float):
    """A heading that holds zero, turns at a constant rate, then holds the new heading."""
    def heading_at(time: float) -> float:
        progress = min(max((time - start_seconds) / turn_seconds, 0.0), 1.0)
        return total_degrees * progress
    return heading_at


def turns_in_sequence(*turns: tuple[float, float, float]):
    """Several ramps added up. Each is (start seconds, turn seconds, degrees)."""
    ramps = [ramp(*turn) for turn in turns]
    return lambda time: sum(each(time) for each in ramps)


def obstacle(lateral_meters: float, forward_meters: float, group: int = 1) -> ObstaclePoint:
    """A group as the scene reports it, relative to the phone's forward axis on the floor."""
    return ObstaclePoint(lateral_meters, forward_meters, group, 1.0, 0.1, None, None, False, np.zeros(3))


def upright_camera_rotation(camera_heading_radians: float, pitch_degrees: float = PITCH_DEGREES) -> np.ndarray:
    """
    Camera axes (x right in the image, y down, z forward) into the world, for a phone held upright.

    Upright, the image's columns run up the world and its rows run to the walker's right.
    """
    pitch = np.radians(pitch_degrees)
    level = direction(camera_heading_radians)
    forward = np.cos(pitch) * level - np.sin(pitch) * WORLD_UP
    rows_axis = np.cross(level, WORLD_UP)
    columns_axis = np.cross(rows_axis, forward)
    return np.column_stack([columns_axis, rows_axis, forward])


def phone_frame(
    time_seconds: float,
    position_world: np.ndarray,
    camera_heading_degrees: float,
    points: tuple[ObstaclePoint, ...] = (),
    arrow_degrees: float = 0.0,
    with_world_pose: bool = True,
) -> PlannedFrame:
    """A planned frame from a phone at a position, pointed along a heading and pitched at the floor."""
    heading = np.radians(camera_heading_degrees)
    obstacles = ObstacleSet(timestamp_seconds=time_seconds, points=points, groups_in_view=len(points))
    if not with_world_pose:
        return PlannedFrame(
            time_seconds, np.radians(arrow_degrees), None, None, None, None,
            INTRINSICS, IMAGE_WIDTH_PIXELS, IMAGE_HEIGHT_PIXELS, obstacles,
        )
    return PlannedFrame(
        timestamp_seconds=time_seconds,
        arrow_radians=float(np.radians(arrow_degrees)),
        forward_axis_world=direction(heading),
        camera_position_world=np.asarray(position_world, dtype=np.float64),
        camera_rotation_world=upright_camera_rotation(heading),
        floor_world=FLOOR,
        intrinsics=INTRINSICS,
        image_width_pixels=IMAGE_WIDTH_PIXELS,
        image_height_pixels=IMAGE_HEIGHT_PIXELS,
        obstacles=obstacles,
    )


# *******************************************
# Recordings
# *******************************************
# A recording is written through the real RecordingTap, so a test reads the same format the walks
# are in. Its depth is synthetic_depth's clean scene, seen from a phone held sideways (landscape)
# and pitched at the floor, the camera synthetic_depth renders for. The scene is the same relative
# to the phone on every frame, and the frame supplies its floor, so the scene never fits one by
# RANSAC and a replay repeats exactly.

RECORDING_RATE_HZ = 10.0


def quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of a rotation matrix. The inverse of rotation_matrix_from_quaternion_wxyz."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        quaternion = [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale]
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        quaternion = [(rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale]
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        quaternion = [(rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale]
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def landscape_camera_rotation(camera_heading_radians: float, pitch_degrees: float) -> np.ndarray:
    """Camera axes (x right, y down, z forward) into the world, for a phone held sideways and pitched down."""
    pitch = np.radians(pitch_degrees)
    level = direction(camera_heading_radians)
    forward = np.cos(pitch) * level - np.sin(pitch) * WORLD_UP
    right = np.cross(level, WORLD_UP)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def walking_pose(position_world: np.ndarray, camera_heading_radians: float, pitch_degrees: float, has_position: bool = True) -> Pose:
    """A gravity-aligned pose for a phone at a position, looking along a heading, pitched at the floor."""
    rotation = landscape_camera_rotation(camera_heading_radians, pitch_degrees)
    return Pose(
        orientation=quaternion_wxyz(rotation),
        position=np.asarray(position_world, dtype=np.float64) if has_position else None,
        has_position=has_position,
        orientation_is_gravity_aligned=True,
    )


class ListSource:
    """A depth source over a list of frames, for wrapping in a RecordingTap."""

    def __init__(self, frames: list[DepthFrame]) -> None:
        self._frames = frames

    def frames(self):
        yield from self._frames

    def close(self) -> None:
        """Nothing to release."""


def write_recording(
    log_dir: Path,
    heading_degrees_at,
    duration_seconds: float,
    run_config_scene: dict | None = None,
    write_run_config: bool = True,
    gravity_aligned_at=lambda time: True,
    floorless_until_seconds: float = 0.0,
) -> Path:
    """
    Record a synthetic walk at RECORDING_RATE_HZ through the real tap, with a run_config.json beside it.

    :param run_config_scene: The scene block to write. Every SceneConfig field at today's values by default.
    :param floorless_until_seconds: Frames before this have no floor in view and supply none, which
        the scene refuses when it has no earlier plane to fall back on.
    :return: The log directory.
    """
    scene = clean_scene(box_lateral_meters=0.3, box_forward_meters=2.5)
    floorless = degrade_without_floor(scene.depth_meters, scene.floor_plane_camera, scene.intrinsics)
    times = np.arange(0.0, duration_seconds, 1.0 / RECORDING_RATE_HZ)
    headings = np.radians([heading_degrees_at(time) for time in times])
    steps = np.array([direction(heading) for heading in headings]) * (WALKING_SPEED_MPS / RECORDING_RATE_HZ)
    positions = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)[:-1]])
    positions[:, 1] = SYNTHETIC_CAMERA_HEIGHT_METERS
    frames = []
    for time, heading, position in zip(times, headings, positions):
        pose = walking_pose(position, heading, SYNTHETIC_PITCH_DEGREES)
        if not gravity_aligned_at(time):
            pose = Pose(pose.orientation, pose.position, pose.has_position, orientation_is_gravity_aligned=False)
        if time < floorless_until_seconds:
            frames.append(DepthFrame(float(time), floorless, scene.intrinsics, pose, None, None))
        else:
            frames.append(DepthFrame(float(time), scene.depth_meters, scene.intrinsics, pose, scene.floor_plane_camera, None))
    for _ in RecordingTap(ListSource(frames), log_dir).frames():
        pass
    if write_run_config:
        block = dataclasses.asdict(SceneConfig()) if run_config_scene is None else run_config_scene
        content = {"source_kind": "arcore_tcp", "goal_mode": "ahead", "scene": block, "walker": {"radius_meters": 0.35}}
        (Path(log_dir) / "run_config.json").write_bytes((json.dumps(content, indent=2) + "\n").encode("utf-8"))
    return Path(log_dir)
