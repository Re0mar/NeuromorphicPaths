"""
Camera positions measured from the depth frames themselves, with Open3D's RGB-D tracking.

Two trackers behind one protocol, and both report only where the camera is. The frame keeps its IMU
orientation, which stays the source of up for the floor fit whatever measured the position.

Dense SLAM tracks each frame against a fused volume of every frame before it. It starts its world
turned the way the IMU says on its first frame, and after that follows its own rotation.
Odometry tracks each frame against the one before it and keeps no map. It takes the IMU's turn
between the two frames as its starting guess.

The world is the IMU's world after the mount, so y is up, with the origin wherever the camera was on
the first frame tracked. A tracker that has to start over carries its last position into the new
world, so the scene's world-fixed cells stay where they were.
"""

# Standard library imports
import logging
from typing import Protocol

# Third party imports
import numpy as np
import open3d

# Local package imports
from nav.pose.config import PositionConfig
from nav.scene.transform import rotation_matrix_from_quaternion_wxyz
from nav.types import DepthFrame

log = logging.getLogger(__name__)

# Depth arrives in meters, so Open3D's millimeter default would shrink the world a thousand times.
DEPTH_SCALE_METERS = 1.0
# The fused volume's truncation, in voxels. Open3D's example value.
TRUNCATION_VOXELS = 8.0
# Nearest depth the raycast keeps. Open3D's example value.
RAYCAST_DEPTH_MIN_METERS = 0.1
# Width of odometry's robust loss over its outlier cutoff. Open3D's own defaults are 0.05 and 0.07 m,
# so a looser cutoff keeps the same shape of loss.
HUBER_TO_CUTOFF_RATIO = 0.05 / 0.07


class TrackingOutcome:
    """One frame's result: the position, or None, and how much of the frame matched."""

    __slots__ = ("position", "fitness")

    def __init__(self, position: np.ndarray | None, fitness: float) -> None:
        self.position = position
        self.fitness = fitness


class PositionTracker(Protocol):
    """Gives each frame in turn a position in a world that stays put, or None when it can't."""

    def track(self, frame: DepthFrame) -> TrackingOutcome: ...

    def close(self) -> None:
        """Free what the tracker holds, on the thread that made it."""
        ...


def dense_slam_is_available() -> bool:
    """Whether this Open3D build can run dense SLAM. Only the CUDA build can, the CPU build stalls."""
    return open3d.core.cuda.is_available()


# The install that makes dense_slam_is_available true, named in every refusal so nobody has to search.
DENSE_SLAM_INSTALL = (
    "pip uninstall -y open3d && pip install "
    "https://github.com/isl-org/Open3D/releases/download/v0.20.0/open3d_cuda-0.20.0-cp312-cp312-win_amd64.whl"
)


def _depth_image(depth_meters: np.ndarray, depth_max_meters: float, device: open3d.core.Device) -> open3d.t.geometry.Image:
    """The frame's depth as Open3D takes it: invalid and too-far pixels zero, which Open3D reads as missing."""
    depth = np.nan_to_num(depth_meters, nan=0.0).astype(np.float32)
    depth[depth > depth_max_meters] = 0.0
    return open3d.t.geometry.Image(open3d.core.Tensor(depth)).to(device)


def _blank_color(shape: tuple[int, int], device: open3d.core.Device) -> open3d.t.geometry.Image:
    # A DepthFrame carries no color, and point-to-plane tracking doesn't read it. The volume still
    # wants a color channel to integrate.
    return open3d.t.geometry.Image(open3d.core.Tensor(np.zeros((*shape, 3), np.float32))).to(device)


def _world_from_camera(frame: DepthFrame) -> np.ndarray:
    return rotation_matrix_from_quaternion_wxyz(frame.pose.orientation)


def _is_plausible_step(previous_position: np.ndarray, position: np.ndarray, elapsed_seconds: float, config: PositionConfig) -> bool:
    """Whether a walker could have covered the step in the time between the two frames."""
    if elapsed_seconds <= 0.0:
        return bool(np.allclose(previous_position, position))
    return float(np.linalg.norm(position - previous_position)) / elapsed_seconds <= config.max_step_speed_mps


class DenseSlamTracker:
    """Open3D's dense SLAM on the GPU: track against the fused volume, then fuse the frame into it."""

    def __init__(self, config: PositionConfig) -> None:
        self._config = config
        self._device = open3d.core.Device("CUDA:0")
        # Where the camera last was, carried into a new world when the model has to start over.
        self._last_position = np.zeros(3)
        self._last_timestamp: float | None = None
        self._clear()

    def _clear(self) -> None:
        self._model = None
        self._input = None
        self._raycast = None
        self._shape: tuple[int, int] | None = None
        self._world_from_camera: np.ndarray | None = None
        self._frame_index = 0

    def close(self) -> None:
        # The volume and the frames are GPU memory. Left to interpreter shutdown, freeing them raced
        # torch's own CUDA teardown and the process hung on exit after a run that had finished.
        self._clear()
        open3d.core.cuda.release_cache()

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        shape = frame.depth_meters.shape
        if self._model is None or shape != self._shape:
            return self._start(frame)

        self._input.set_data_from_image("depth", _depth_image(frame.depth_meters, self._config.depth_max_meters, self._device))
        self._input.set_data_from_image("color", _blank_color(shape, self._device))
        position = None
        fitness = 0.0
        try:
            result = self._model.track_frame_to_model(
                self._input,
                self._raycast,
                DEPTH_SCALE_METERS,
                self._config.depth_max_meters,
                self._config.dense_slam_depth_difference_meters,
            )
            fitness = float(result.fitness)
            tracked = self._world_from_camera @ result.transformation.numpy()
            elapsed = frame.timestamp_seconds - self._last_timestamp
            if _is_plausible_step(self._world_from_camera[:3, 3], tracked[:3, 3], elapsed, self._config):
                self._world_from_camera = tracked
                if fitness >= self._config.min_fitness:
                    position = tracked[:3, 3].copy()
            else:
                # Faster than anyone walks, so the fit locked onto the wrong surface. The pose stays
                # where it was, and this frame is fused from there.
                log.debug("dense SLAM step refused as faster than %.1f m/s", self._config.max_step_speed_mps)
        except RuntimeError as tracking_failure:
            # Open3D raises when the 6x6 system is singular: a blank wall, or nothing in range. The
            # frame has no position, and the volume is fused from the last pose.
            log.debug("dense SLAM couldn't track this frame (caught %s, expected): %s", type(tracking_failure).__name__, tracking_failure)
        self._fuse()
        self._last_timestamp = frame.timestamp_seconds
        if position is not None:
            self._last_position = position
        return TrackingOutcome(position, fitness)

    def _start(self, frame: DepthFrame) -> TrackingOutcome:
        shape = frame.depth_meters.shape
        intrinsics = open3d.core.Tensor(np.asarray(frame.intrinsics, dtype=np.float64))
        # The world is turned the way the IMU says, so its y is up, and starts at the camera's last
        # position. On the first frame that's the origin.
        self._world_from_camera = np.eye(4)
        self._world_from_camera[:3, :3] = _world_from_camera(frame)
        self._world_from_camera[:3, 3] = self._last_position
        self._last_timestamp = frame.timestamp_seconds
        self._model = open3d.t.pipelines.slam.Model(
            self._config.voxel_size_meters,
            16,
            self._config.voxel_block_count,
            open3d.core.Tensor(self._world_from_camera),
            self._device,
        )
        self._input = open3d.t.pipelines.slam.Frame(shape[0], shape[1], intrinsics, self._device)
        self._raycast = open3d.t.pipelines.slam.Frame(shape[0], shape[1], intrinsics, self._device)
        self._shape = shape
        self._frame_index = 0
        self._input.set_data_from_image("depth", _depth_image(frame.depth_meters, self._config.depth_max_meters, self._device))
        self._input.set_data_from_image("color", _blank_color(shape, self._device))
        self._fuse()
        # Open3D sizes its raycast's range buffer from the call before, so the first raycast comes out
        # partial ("could not generate full range map") and the second frame would track against a
        # third of the picture. A second raycast of the same model, with the grown buffer, fills it.
        self._raycast_model()
        return TrackingOutcome(self._world_from_camera[:3, 3].copy(), 1.0)

    def _fuse(self) -> None:
        self._model.update_frame_pose(self._frame_index, open3d.core.Tensor(self._world_from_camera))
        self._model.integrate(self._input, DEPTH_SCALE_METERS, self._config.depth_max_meters, TRUNCATION_VOXELS)
        self._raycast_model()
        self._frame_index += 1

    def _raycast_model(self) -> None:
        """Render the fused volume from the current pose, which is what the next frame tracks against."""
        self._model.synthesize_model_frame(
            self._raycast, DEPTH_SCALE_METERS, RAYCAST_DEPTH_MIN_METERS, self._config.depth_max_meters, TRUNCATION_VOXELS, False, -1.0,
        )


class FrameToFrameOdometry:
    """Open3D's RGB-D odometry on the CPU: each frame against the one before, rotation from the IMU."""

    def __init__(self, config: PositionConfig) -> None:
        self._config = config
        self._device = open3d.core.Device("CPU:0")
        self._previous: DepthFrame | None = None
        self._previous_image = None
        self._position = np.zeros(3)

    def close(self) -> None:
        self._previous = None
        self._previous_image = None

    def track(self, frame: DepthFrame) -> TrackingOutcome:
        shape = frame.depth_meters.shape
        image = open3d.t.geometry.RGBDImage(
            _blank_color(shape, self._device),
            _depth_image(frame.depth_meters, self._config.depth_max_meters, self._device),
        )
        previous, previous_image = self._previous, self._previous_image
        self._previous, self._previous_image = frame, image
        if previous is None or previous.depth_meters.shape != shape:
            # Nothing to match against. The position holds, which on the first frame is the origin.
            return TrackingOutcome(self._position.copy(), 1.0)

        # Odometry's transform takes the previous camera's points into this one's. The IMU says how
        # the camera turned in between, which is the starting guess.
        world_from_previous = _world_from_camera(previous)
        start = np.eye(4)
        start[:3, :3] = _world_from_camera(frame).T @ world_from_previous
        try:
            result = open3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                previous_image,
                image,
                open3d.core.Tensor(np.asarray(frame.intrinsics, dtype=np.float64)),
                open3d.core.Tensor(start),
                DEPTH_SCALE_METERS,
                self._config.depth_max_meters,
                method=open3d.t.pipelines.odometry.Method.PointToPlane,
                params=open3d.t.pipelines.odometry.OdometryLossParams(
                    depth_outlier_trunc=self._config.odometry_depth_difference_meters,
                    # Open3D's own ratio of the robust loss's width to the cutoff, 0.05 to 0.07.
                    depth_huber_delta=self._config.odometry_depth_difference_meters * HUBER_TO_CUTOFF_RATIO,
                ),
            )
        except RuntimeError as tracking_failure:
            # A singular system, as for dense SLAM. No step is taken, so the position holds.
            log.debug("odometry couldn't track this frame (caught %s, expected): %s", type(tracking_failure).__name__, tracking_failure)
            return TrackingOutcome(None, 0.0)

        fitness = float(result.fitness)
        current_from_previous = result.transformation.numpy()
        # This camera's center, seen from the previous camera, then turned into the world.
        center_in_previous = -current_from_previous[:3, :3].T @ current_from_previous[:3, 3]
        stepped = self._position + world_from_previous @ center_in_previous
        if not _is_plausible_step(self._position, stepped, frame.timestamp_seconds - previous.timestamp_seconds, self._config):
            # Faster than anyone walks, so the fit locked onto the wrong surface. No step is taken.
            log.debug("odometry step refused as faster than %.1f m/s", self._config.max_step_speed_mps)
            return TrackingOutcome(None, fitness)
        self._position = stepped
        position = self._position.copy() if fitness >= self._config.min_fitness else None
        return TrackingOutcome(position, fitness)
