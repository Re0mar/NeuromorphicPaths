"""
Depth frames from a Neon recording that the Neon Player depth plugin has already run over.

The plugin stores one stacked array of depth maps at a quarter of the scene resolution, index
aligned to the scene video's frames. This reads that array back and pairs each map with the
recording's IMU orientation and gaze at the same timestamp, so a recorded walk feeds the planner
with no estimator on this laptop.

The recording itself is read through a small protocol rather than directly, because the native
recording format is binary and undocumented and the tests cannot build one. The plugin's array
they can build. The native reader wrapping pupil_labs.neon_recording is the one shipping
implementation, and the only other file allowed to import that package.
"""

# Standard library imports
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

# Third party imports
import numpy as np

# Local package imports
from nav.pose.imu_orientation import identity_pose, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import scale_intrinsics
from nav.sources.config import NeonPluginConfig, NeonPluginModel
from nav.types import DepthFrame

log = logging.getLogger(__name__)

# Where Neon Player keeps a plugin's cache, relative to the recording, keyed by the plugin's class
# name. Read from neon-player's plugin base class, which builds it as
# recording_dir / ".neon_player" / "cache" / type(plugin).__name__.
PLUGIN_CACHE_RELATIVE_DIR = Path(".neon_player") / "cache" / "DepthEstimationPlugin"
DEPTH_VALUES_FILENAME_TEMPLATE = "depth_values_{stem}.npy"
NANOSECONDS_PER_SECOND = 1.0e9


class NeonRecordingReader(Protocol):
    """What this source needs from a recording. Times are the recording's own, in nanoseconds."""

    def scene_times_ns(self) -> np.ndarray: ...

    def scene_size(self) -> tuple[int, int]: ...

    def scene_camera_matrix(self) -> np.ndarray: ...

    def imu_quaternions_wxyz_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None: ...

    def gaze_points_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None: ...


def xyzw_to_wxyz(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """
    Reorder quaternion columns from the recording's (x, y, z, w) to the (w, x, y, z) every Pose uses.

    :param quaternions_xyzw: (N, 4) array in the recording's order.
    :return: (N, 4) array in Pose order.
    :rtype: np.ndarray
    """
    if quaternions_xyzw.ndim != 2 or quaternions_xyzw.shape[1] != 4:
        raise ValueError(f"expected (N, 4) quaternions, got shape {quaternions_xyzw.shape}")
    return quaternions_xyzw[:, [3, 0, 1, 2]]


class NeonPluginDepthFrameSource:
    """Yields the plugin's cached depth maps as DepthFrame objects, in recording order."""

    def __init__(
        self,
        config: NeonPluginConfig,
        reader_factory: Callable[[Path], NeonRecordingReader],
    ) -> None:
        self._config = config
        self._recording_dir = Path(config.recording_dir)
        self._reader_factory = reader_factory

    def _cache_file(self) -> Path:
        model = self._config.model
        cache_dir = self._recording_dir / PLUGIN_CACHE_RELATIVE_DIR
        cache_file = cache_dir / DEPTH_VALUES_FILENAME_TEMPLATE.format(stem=model.cache_stem)
        if not cache_file.is_file():
            raise FileNotFoundError(
                f"no depth cache at {cache_file}. Open this recording in Neon Player, run the "
                f"Depth Estimation plugin with the {model.value} model, and let it finish"
            )
        return cache_file

    def frames(self) -> Iterator[DepthFrame]:
        if not self._recording_dir.is_dir():
            raise FileNotFoundError(f"{self._recording_dir} is not a directory")

        model = self._config.model
        if not model.is_metric:
            # The planner's clearance S is meters. Relative depth would make every surprise value
            # wrong by an unknown scale, the same defect the old script's default model had.
            raise ValueError(f"{model.value} caches relative depth, the planner needs meters. Use {NeonPluginModel.METRIC_LARGE.value}")

        cache_file = self._cache_file()
        depth_stack = np.load(cache_file, mmap_mode="r")
        if depth_stack.ndim != 3:
            raise ValueError(f"{cache_file.name} should be (frames, rows, columns), got shape {depth_stack.shape}")
        frame_count, depth_height, depth_width = depth_stack.shape

        reader = self._reader_factory(self._recording_dir)
        times_ns = np.asarray(reader.scene_times_ns(), dtype=np.int64)
        if times_ns.ndim != 1:
            raise ValueError(f"scene times should be one dimensional, got shape {times_ns.shape}")

        usable = min(frame_count, len(times_ns))
        if usable != frame_count or usable != len(times_ns):
            # The plugin zips frames and maps without strict, so an off-by-one at the end of a
            # recording is possible. More than that means the cache is from another recording.
            log.warning(
                "%d depth maps but %d scene frames in %s, using the first %d",
                frame_count, len(times_ns), self._recording_dir.name, usable,
            )
        if usable == 0:
            raise ValueError(f"{cache_file.name} holds no frames")
        times_ns = times_ns[:usable]

        intrinsics = scale_intrinsics(reader.scene_camera_matrix(), reader.scene_size(), (depth_height, depth_width))
        tolerance_ns = int(self._config.sample_tolerance_seconds * NANOSECONDS_PER_SECOND)

        orientations = reader.imu_quaternions_wxyz_at(times_ns, tolerance_ns)
        if orientations is None:
            log.warning("%s has no IMU data, poses will be identity", self._recording_dir.name)
        gaze_points = reader.gaze_points_at(times_ns, tolerance_ns)

        scene_height, scene_width = reader.scene_size()
        gaze_scale = np.array([depth_width / scene_width, depth_height / scene_height])

        log.info("replaying %d plugin depth maps of %dx%d from %s", usable, depth_width, depth_height, self._recording_dir.name)

        for index in range(usable):
            if orientations is not None and np.all(np.isfinite(orientations[index])):
                # The same IMU as the live stream, so the same mount. The xyzw reorder above is the
                # recording's own column order and has already happened.
                pose = pose_from_imu(np.asarray(orientations[index], dtype=np.float64), NEON_IMU_MOUNT)
            else:
                pose = identity_pose()

            gaze_pixel = None
            if gaze_points is not None and np.all(np.isfinite(gaze_points[index])):
                gaze_pixel = np.asarray(gaze_points[index], dtype=np.float64) * gaze_scale

            yield DepthFrame(
                timestamp_seconds=float(times_ns[index]) / NANOSECONDS_PER_SECOND,
                depth_meters=np.ascontiguousarray(depth_stack[index], dtype=np.float32),
                intrinsics=intrinsics,
                pose=pose,
                ground_plane=None,
                gaze_pixel=gaze_pixel,
            )

    def close(self) -> None:
        """Nothing held open between frames. Present because the loop closes every source."""


class NativeNeonRecordingReader:
    """Reads a native Neon recording through pupil_labs.neon_recording. The shipping reader."""

    def __init__(self, recording_dir: Path) -> None:
        # Optional dependency, present only with the glasses extra. The plugin cache can be read
        # on a laptop that never installed it, which is why the recording reader is injectable.
        from pupil_labs.neon_recording import NeonRecording

        self._recording = NeonRecording(recording_dir)

    def scene_times_ns(self) -> np.ndarray:
        return np.asarray(self._recording.scene.time, dtype=np.int64)

    def scene_size(self) -> tuple[int, int]:
        scene = self._recording.scene
        if scene.height is None or scene.width is None:
            raise ValueError("scene video reports no size, the recording may be missing its video")
        return int(scene.height), int(scene.width)

    def scene_camera_matrix(self) -> np.ndarray:
        calibration = self._recording.calibration
        if calibration is None:
            raise FileNotFoundError("recording has no calibration.bin, so the scene camera intrinsics are unknown")
        return np.asarray(calibration.scene_camera_matrix, dtype=np.float64)

    def imu_quaternions_wxyz_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None:
        try:
            imu = self._recording.imu
        except Exception as missing:  # noqa: BLE001, the library raises its own SensorError subclass
            log.info("no IMU stream (caught %s, expected on some recordings): %s", type(missing).__name__, missing)
            return None
        sampled = imu.sample(times_ns, method="nearest", tolerance=tolerance_ns)
        # The recording stores x, y, z, w. Every Pose is w, x, y, z. Reordering by name here is
        # what keeps pitch from flipping silently.
        return xyzw_to_wxyz(np.asarray(sampled.rotation, dtype=np.float64))

    def gaze_points_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None:
        try:
            gaze = self._recording.gaze
        except Exception as missing:  # noqa: BLE001, same as above
            log.info("no gaze stream (caught %s, expected on some recordings): %s", type(missing).__name__, missing)
            return None
        sampled = gaze.sample(times_ns, method="nearest", tolerance=tolerance_ns)
        return np.asarray(sampled.point, dtype=np.float64)
