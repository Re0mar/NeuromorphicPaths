"""
A DepthFrame in, an ObstacleSet out. The whole scene layer, in order.

The planner's grid is always in the body frame at the current frame: forward is where the camera
points, lateral is across it, and the walker is at the origin. That holds whether or not the pose
has a position. What a position buys is identity. With one, points are placed in the world before
grouping, so a post keeps its group id as the walker advances past it and the clearance history
is measuring the same thing frame to frame. Without one, the grid moves with the walker and N is
approximate, which is a known limit rather than a defect.
"""

# Standard library imports
import logging
import time

# Third party imports
import numpy as np

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.floor import (
    CAMERA_UP,
    FloorRefusal,
    fit_floor_with_refusal,
    ground_axes,
    height_above_floor,
    normalize_plane,
    plane_is_a_floor,
)
from nav.scene.grouping import (
    GroupSummary,
    assign_groups,
    assign_world_groups,
    clearance,
    filter_height_band,
    summarize_groups,
    within_planning_window,
)
from nav.scene.history import ClearanceHistory
from nav.scene.transform import camera_to_world_plane, camera_to_world_points, rotation_matrix_from_quaternion_wxyz
from nav.scene.unproject import downsample, unproject_depth
from nav.types import WORLD_UP, DepthFrame, FloorSource, ObstaclePoint, ObstacleSet, Plane
from nav.walker import WalkerConfig

log = logging.getLogger(__name__)

CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])


class ScenePipeline:
    """Holds the state that carries between frames: the last floor and the clearance history."""

    def __init__(self, config: SceneConfig, walker: WalkerConfig) -> None:
        self._config = config
        self._walker = walker
        self._previous_plane: Plane | None = None
        self._last_floor_source: FloorSource | None = None
        self._last_floor_refusal: FloorRefusal | None = None
        self._history = ClearanceHistory(
            window_seconds=config.noise_window_seconds,
            min_samples=config.min_history_samples,
            noise_floor_meters=config.noise_floor_meters,
        )

    @property
    def previous_plane(self) -> Plane | None:
        """The last floor used, in the camera frame. The runtime reads it to put the gaze on the ground."""
        return self._previous_plane

    @property
    def last_floor_source(self) -> FloorSource | None:
        """Where this frame's floor came from. None before the first frame, and after a frame the scene refused."""
        return self._last_floor_source

    @property
    def last_floor_refusal(self) -> FloorRefusal | None:
        """Why this frame's fit gave nothing, when its floor is the previous one. None otherwise."""
        return self._last_floor_refusal

    def process(self, frame: DepthFrame) -> ObstacleSet:
        """
        Turn one depth frame into the obstacles the planner scores.

        :return: One ObstaclePoint per group in view, at the group's nearest point.
        :rtype: ObstacleSet
        """
        config = self._config
        started = time.perf_counter()
        # Cleared first, so a frame refused below reads as no floor rather than the last frame's.
        # The runtime's timing log reads this after a failed frame to say whether it had a floor.
        self._last_floor_source = None
        self._last_floor_refusal = None

        points = unproject_depth(frame.depth_meters, frame.intrinsics, config.depth_stride, config)
        points = downsample(points, config.voxel_size_meters)
        after_cloud = time.perf_counter()

        plane_camera, floor_source = self._choose_floor(frame, points)
        self._previous_plane = plane_camera
        self._last_floor_source = floor_source
        after_floor = time.perf_counter()
        # The offset is the camera's height above the floor. On a real walk it should sit near eye
        # height, and this line is how that gets checked against a metric depth model.
        log.debug(
            "floor: camera %.2f m above it, normal %.1f deg from up, %s",
            plane_camera.offset_meters,
            np.degrees(np.arccos(np.clip(plane_camera.normal @ self._up_in_camera_frame(frame), -1.0, 1.0))),
            floor_source.value,
        )

        pose = frame.pose
        # The camera-frame copy outlives the world transform so each group's nearest point can be
        # handed to the depth view where the camera saw it. Same rows as points, kept in step.
        points_camera = points
        if pose.has_position:
            points = camera_to_world_points(points, pose)
            plane = camera_to_world_plane(plane_camera, pose)
            walker_position = np.asarray(pose.position, dtype=np.float64)
            forward_hint = rotation_matrix_from_quaternion_wxyz(pose.orientation) @ CAMERA_FORWARD
        else:
            plane = plane_camera
            walker_position = np.zeros(3)
            forward_hint = CAMERA_FORWARD

        heights = height_above_floor(points, plane)
        points, heights, in_band = filter_height_band(points, heights, config)
        points_camera = points_camera[in_band]

        # Walker-relative ground coordinates, which is what the planner and the clearances use.
        lateral_axis, forward_axis = ground_axes(plane, forward_hint)
        relative_to_walker = points - walker_position
        relative_ground = np.column_stack((relative_to_walker @ lateral_axis, relative_to_walker @ forward_axis))

        if pose.has_position:
            # Ids from a grid fixed to the world, so they survive the walker moving. Points
            # outside the planner's window still get dropped, by id, below.
            world_lateral_axis, world_forward_axis = ground_axes(plane)
            world_ground = np.column_stack((points @ world_lateral_axis, points @ world_forward_axis))
            group_ids = assign_world_groups(world_ground, config)
            group_ids[~within_planning_window(relative_ground, config)] = -1
            # The history measures velocity on the fixed world axes. The planner slides a group in
            # the walker's axes. The two agree only when the walker faces world forward, so every
            # velocity is rotated through this before it leaves the scene.
            world_to_walker = np.array(
                [
                    [world_lateral_axis @ lateral_axis, world_forward_axis @ lateral_axis],
                    [world_lateral_axis @ forward_axis, world_forward_axis @ forward_axis],
                ]
            )
        else:
            world_ground = None
            world_to_walker = None
            group_ids = assign_groups(relative_ground, config)

        summaries = summarize_groups(relative_ground, heights, group_ids, config)
        after_groups = time.perf_counter()

        clearances = [clearance(summary, self._walker) for summary in summaries]
        centroids = [self._history_centroid(summary, world_ground, group_ids) for summary in summaries]
        self._history.update(frame.timestamp_seconds, [summary.group_id for summary in summaries], clearances, centroids)
        self._history.forget_unseen(frame.timestamp_seconds)

        obstacle_points = tuple(
            self._obstacle_point(summary, clearance_meters, world_to_walker, points_camera)
            for summary, clearance_meters in zip(summaries, clearances, strict=True)
        )
        finished = time.perf_counter()

        log.debug(
            "scene %.1f ms: cloud %.1f, floor %.1f, groups %.1f, history %.1f, %d groups from %d points",
            (finished - started) * 1000,
            (after_cloud - started) * 1000,
            (after_floor - after_cloud) * 1000,
            (after_groups - after_floor) * 1000,
            (finished - after_groups) * 1000,
            len(obstacle_points),
            len(points),
        )

        return ObstacleSet(
            timestamp_seconds=frame.timestamp_seconds,
            points=obstacle_points,
            groups_in_view=len(obstacle_points),
        )

    @staticmethod
    def _up_in_camera_frame(frame: DepthFrame) -> np.ndarray:
        # Gravity, whenever the source says its orientation is aligned to it. Image-up otherwise,
        # which assumes the camera is held roughly level and is all a plain video file can offer.
        # The Pixel in portrait sends its depth image sideways, and measured against image-up its
        # floor leaned 89 degrees on every frame of the first walk.
        #
        # The question is about the orientation, not the position. The glasses report a
        # gravity-aligned orientation and no position at all, and asking for a position threw
        # their gravity away and gave them the very defect this gate was built to fix.
        if not frame.pose.orientation_is_gravity_aligned:
            return CAMERA_UP
        return rotation_matrix_from_quaternion_wxyz(frame.pose.orientation).T @ WORLD_UP

    def _choose_floor(self, frame: DepthFrame, points: np.ndarray) -> tuple[Plane, FloorSource]:
        # A source that knows the ground says so, but it is not believed on its word. The first
        # Pixel walk sent a plane a meter below the real floor on every frame, and the fit is the
        # second opinion. A refused plane takes the path a frame with no plane takes.
        up_camera = self._up_in_camera_frame(frame)
        if frame.ground_plane is not None:
            supplied = normalize_plane(frame.ground_plane, up_camera)
            refusal = plane_is_a_floor(supplied, self._config, up_camera)
            if refusal is None:
                return supplied, FloorSource.SUPPLIED
            log.debug("supplied floor refused: %s, fitting instead", refusal)
        fitted, refusal = fit_floor_with_refusal(points, self._previous_plane, self._config, up_camera)
        # The fit hands back the previous object itself when it falls back, so identity is the test.
        if fitted is self._previous_plane:
            self._last_floor_refusal = refusal
            return fitted, FloorSource.PREVIOUS
        return fitted, FloorSource.FITTED

    def _history_centroid(self, summary: GroupSummary, world_ground: np.ndarray | None, group_ids: np.ndarray) -> np.ndarray:
        # In the world frame the centroid for velocity has to be in world coordinates, or every
        # group would appear to move at the walker's speed. In the body frame it is only used to
        # keep the sample shape uniform, and velocity is never asked for.
        if world_ground is None:
            return np.array([summary.centroid_lateral_meters, summary.centroid_forward_meters])
        members = world_ground[group_ids == summary.group_id]
        return members.mean(axis=0)

    def _obstacle_point(
        self,
        summary: GroupSummary,
        clearance_meters: float,
        world_to_walker: np.ndarray | None,
        points_camera: np.ndarray,
    ) -> ObstaclePoint:
        velocity = None
        if world_to_walker is not None:
            # Only meaningful when the grid did not move with the walker, and only in the
            # walker's axes once it leaves here.
            world_velocity = self._history.velocity(summary.group_id)
            if world_velocity is not None:
                velocity = world_to_walker @ world_velocity
        return ObstaclePoint(
            lateral_meters=summary.nearest_lateral_meters,
            forward_meters=summary.nearest_forward_meters,
            group_id=summary.group_id,
            clearance_meters=clearance_meters,
            noise_scale_meters=self._history.noise_scale(summary.group_id),
            closing_rate_mps=self._history.closing_rate(summary.group_id),
            velocity_mps=velocity,
            is_wall=summary.is_wall,
            camera_point=points_camera[summary.nearest_index],
        )
