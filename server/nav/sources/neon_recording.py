"""
A native Neon recording, replayed through the same straightened camera as the live glasses.

The recording's scene video is decoded here, a frame at a time, and each frame is straightened with
the recording's own calibration before the depth estimator sees it. The frame carries the
straightened camera matrix on, so the estimator's focal-over-300 conversion uses the focal the
picture actually has. From the frame onward this is the live route exactly: same straightener, same
estimator, same IMU rule. A recorded walk and a live one differ in where the frames come from, and
in which frames get planned: the live route takes the newest frame, this one a set rate of recording.

Neon Player's depth plugin cache is not read. Its values are meters. The plugin at
pupil-labs/npp-depth-estimation e6202a9 multiplies the model's output by the recording's focal
scaled to the 504 px it ran at, over 300, before saving. On walk_2026_10_08_b its saved maps came to
0.9343 times the model's raw output, against 0.9353 predicted for that one conversion
(docs/evaluation/neon_recording_routes.md). But it was estimated on the picture before
straightening, so on the same 47 frames its fitted floor put the camera 0.079 m higher than this
source does. The floor sits at the bottom of the picture, where the lens bends it most. So a
recording is replayed here instead, and the route that read the cache was removed.

The recording is read through a small protocol, because the native format is binary and undocumented
and the tests can't build one. This file and neon_stream.py are the only two allowed to import
pupil_labs.
"""

# Standard library imports
import logging
from collections.abc import Callable, Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

# Third party imports
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.pose.imu_orientation import orientation_at, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.sources.camera_model import CameraModelError, undistorter_for
from nav.sources.config import NeonRecordingConfig
from nav.sources.neon_camera import NEON_SCENE_SIZE
from nav.sources.rgb import RgbFrame
from nav.types import FrameTiming

if TYPE_CHECKING:
    # For the annotation only. At run time the package is imported inside the reader, because it
    # comes with the glasses extra.
    from pupil_labs.neon_recording.calib import Calibration

log = logging.getLogger(__name__)

NANOSECONDS_PER_SECOND = 1_000_000_000
# Every Companion export has this beside its streams. Its absence means the folder isn't a recording.
NEON_RECORDING_INFO_FILENAME = "info.json"
# How far a gaze sample may sit from a scene frame and still be that frame's gaze. A scene frame is
# 33 ms and gaze runs far faster, the same reasoning as the IMU's tolerance beside its rule.
GAZE_SAMPLE_TOLERANCE_NANOSECONDS = 50_000_000
# How often, in recording time, the running count of frames with no orientation is logged.
POSE_MISS_LOG_INTERVAL_NANOSECONDS = 5 * NANOSECONDS_PER_SECOND


class RecordingStream(Enum):
    """The recording's streams the reader reads besides the scene video, by the library's attribute names."""

    IMU = "imu"
    GAZE = "gaze"


class NeonRecordingCalibrationError(ValueError):
    """The recording's calibration can't describe its scene camera, so nothing in it can be placed."""


class NeonRecordingReader(Protocol):
    """What a source needs from a recording. Times are the recording's own, int64 nanoseconds."""

    def scene_times_ns(self) -> np.ndarray: ...

    def scene_size(self) -> tuple[int, int]: ...

    def scene_camera_matrix(self) -> np.ndarray: ...

    def scene_distortion_coefficients(self) -> np.ndarray: ...

    def scene_frame_rgb(self, index: int) -> np.ndarray: ...

    def imu_samples(self) -> tuple[np.ndarray, np.ndarray]: ...

    def gaze_points_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None: ...

    def close(self) -> None: ...


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


def sample_nearest(sample_times_ns: np.ndarray, values: np.ndarray, target_times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray:
    """
    For each target time, the value of the sample nearest it, or NaN when none is within the tolerance.

    The library's own sampling refuses the whole request when one target has no match, so a single
    gap in a stream would end the replay. Here a gap costs only the frames inside it.

    :param sample_times_ns: (N,) int64, ascending.
    :param values: (N, K) one row per sample.
    :param target_times_ns: (M,) int64.
    :param tolerance_ns: How far the chosen sample may be from its target.
    :return: (M, K) float64, NaN rows where nothing was near enough.
    :rtype: np.ndarray
    :raises ValueError: When the sample times aren't ascending or don't pair with the values.
    """
    sample_times = np.asarray(sample_times_ns, dtype=np.int64).reshape(-1)
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    targets = np.asarray(target_times_ns, dtype=np.int64).reshape(-1)
    if rows.shape[0] != sample_times.size:
        raise ValueError(f"{sample_times.size} sample times for {rows.shape[0]} values")
    if sample_times.size > 1 and np.any(np.diff(sample_times) < 0):
        raise ValueError("sample times must be ascending")
    result = np.full((targets.size, rows.shape[1]), np.nan)
    if sample_times.size == 0:
        return result
    after = np.clip(np.searchsorted(sample_times, targets), 0, sample_times.size - 1)
    before = np.clip(after - 1, 0, sample_times.size - 1)
    # Int64 throughout. As float seconds these stamps, around 1.7e18 ns, lose their last digits.
    gap_after = np.abs(sample_times[after] - targets)
    gap_before = np.abs(sample_times[before] - targets)
    nearest = np.where(gap_before <= gap_after, before, after)
    near_enough = np.minimum(gap_before, gap_after) <= tolerance_ns
    result[near_enough] = rows[nearest[near_enough]]
    return result


def thinned_indices(times_ns: np.ndarray, frames_per_second: float) -> np.ndarray:
    """
    Which frames to keep so the replay runs at about the given rate in recording time.

    A grid of ticks one period apart, anchored at the first frame. The first frame at or after each
    tick is kept, and at most one per tick. After a gap the next frame is kept, and the next tick is the
    grid's first one after it, so a gap never turns into a burst of frames catching up and the grid
    keeps the phase it started with.

    :param times_ns: (N,) int64 frame times, strictly increasing.
    :param frames_per_second: The rate wanted. Above the recording's own rate every frame is kept.
    :return: Indices into times_ns, ascending.
    :rtype: np.ndarray
    :raises ValueError: For times that aren't strictly increasing, or a rate that isn't a finite positive number.
    """
    if not np.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError(f"frames_per_second must be a finite positive number, got {frames_per_second}")
    times = np.asarray(times_ns, dtype=np.int64).reshape(-1)
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("frame times must be strictly increasing")
    # At least 1 ns, so a rate above a billion a second keeps every frame instead of dividing by zero.
    period_ns = max(1, int(round(NANOSECONDS_PER_SECOND / frames_per_second)))
    kept: list[int] = []
    next_tick = None
    for index, time in enumerate(times.tolist()):
        if next_tick is None or time >= next_tick:
            kept.append(index)
            # The next tick is the first grid tick after this frame, so one late frame doesn't
            # leave the ticks it skipped to be filled by the frames right behind it.
            ticks_passed = 0 if next_tick is None else (time - next_tick) // period_ns
            next_tick = (time if next_tick is None else next_tick + ticks_passed * period_ns) + period_ns
    return np.asarray(kept, dtype=np.int64)


class NeonRecordingRgbSource:
    """Yields a recording's scene frames, straightened, with their gaze and the pose from the recording's IMU."""

    def __init__(self, config: NeonRecordingConfig, reader_factory: Callable[[Path], NeonRecordingReader]) -> None:
        self._config = config
        self._recording_dir = Path(config.recording_dir)
        self._reader_factory = reader_factory
        self._reader: NeonRecordingReader | None = None

    def frames(self) -> Iterator[RgbFrame]:
        if not self._recording_dir.is_dir():
            raise FileNotFoundError(f"{self._recording_dir} is not a directory")
        reader = self._reader_factory(self._recording_dir)
        self._reader = reader
        name = self._recording_dir.name

        times_ns = np.asarray(reader.scene_times_ns(), dtype=np.int64).reshape(-1)
        scene_size = reader.scene_size()
        try:
            camera_matrix = reader.scene_camera_matrix()
            distortion_coefficients = reader.scene_distortion_coefficients()
        except FileNotFoundError as missing:
            raise NeonRecordingCalibrationError(f"{name} has no scene camera calibration, so nothing in it can be placed: {missing}") from missing
        try:
            undistorter = undistorter_for(camera_matrix, distortion_coefficients, NEON_SCENE_SIZE, scene_size)
        except CameraModelError as unusable:
            # A guessed field of view instead would put every obstacle at the edges in the wrong
            # place sideways, which is the defect this source exists to avoid.
            raise NeonRecordingCalibrationError(f"the calibration in {name} cannot describe its scene camera: {unusable}") from unusable
        log.info(
            "straightening %s with its own calibration: %.1f deg wide before, %.1f deg after the crop",
            name,
            undistorter.field_of_view_before_degrees,
            undistorter.field_of_view_after_degrees,
        )

        kept = thinned_indices(times_ns, self._config.frames_per_second)
        kept_times_ns = times_ns[kept]
        log.info("replaying %d of %d scene frames from %s, at %.2f a second of recording", kept.size, times_ns.size, name, self._config.frames_per_second)

        imu_times_ns, imu_orientations = reader.imu_samples()
        imu_times_seconds = np.asarray(imu_times_ns, dtype=np.int64) / NANOSECONDS_PER_SECOND
        gaze_points = reader.gaze_points_at(kept_times_ns, GAZE_SAMPLE_TOLERANCE_NANOSECONDS)
        # Gaze comes in the recording's native pixels. Decoded frames at another size need it scaled first.
        gaze_scale = np.array([scene_size[1] / NEON_SCENE_SIZE[1], scene_size[0] / NEON_SCENE_SIZE[0]])

        without_pose = 0
        next_miss_log_ns = int(kept_times_ns[0]) + POSE_MISS_LOG_INTERVAL_NANOSECONDS if kept.size else 0
        for position, index in enumerate(kept.tolist()):
            image_rgb = np.asarray(reader.scene_frame_rgb(index), dtype=np.uint8)
            if image_rgb.shape[:2] != scene_size:
                raise NeonRecordingCalibrationError(
                    f"frame {index} of {name} is {image_rgb.shape[1]}x{image_rgb.shape[0]}, "
                    f"the straightener was built for {scene_size[1]}x{scene_size[0]}"
                )
            time_ns = int(times_ns[index])

            # The same rule as the live glasses: this frame's own nearest usable reading, or nothing.
            # An older orientation carried forward would put the floor off by however far the head turned.
            orientation = orientation_at(imu_times_seconds, imu_orientations, time_ns / NANOSECONDS_PER_SECOND)
            pose = None if orientation is None else pose_from_imu(orientation, NEON_IMU_MOUNT)
            if pose is None:
                without_pose += 1
            if time_ns >= next_miss_log_ns:
                if without_pose:
                    log.warning("%d of %d frames from %s so far had no usable IMU reading near them", without_pose, position + 1, name)
                next_miss_log_ns = time_ns + POSE_MISS_LOG_INTERVAL_NANOSECONDS

            gaze_pixel = None
            if gaze_points is not None and np.all(np.isfinite(gaze_points[position])):
                gaze_pixel = undistorter.undistort_pixel(gaze_points[position] * gaze_scale)

            yield RgbFrame(
                timestamp_seconds=time_ns / NANOSECONDS_PER_SECOND,
                image_rgb=undistorter.undistort_image(image_rgb),
                gaze_pixel=gaze_pixel,
                pose=pose,
                camera_matrix=undistorter.camera_matrix,
                # The recording's clock is the Neon's, so there is no laptop capture time to give.
                timing=FrameTiming(capture_seconds=None, arrival_seconds=laptop_time_seconds(), depth_ready_seconds=None),
            )

        log.info("%d of %d frames from %s had no usable IMU reading near them", without_pose, kept.size, name)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None


class NativeNeonRecordingReader:
    """Reads a native Neon recording through pupil_labs.neon_recording. The shipping reader."""

    def __init__(self, recording_dir: Path) -> None:
        # Optional dependency, present only with the glasses extra. The tests fake this reader, and a
        # laptop without the extra can still import every module that names it.
        from pupil_labs.neon_recording import NeonRecording

        self._recording = NeonRecording(recording_dir)
        # The library raises its own nested class when a stream won't load, whether its files are
        # missing or damaged. The original error is on its __cause__.
        self._stream_error = NeonRecording.SensorError

    def scene_times_ns(self) -> np.ndarray:
        return np.asarray(self._recording.scene.time, dtype=np.int64)

    def scene_size(self) -> tuple[int, int]:
        scene = self._recording.scene
        if scene.height is None or scene.width is None:
            raise ValueError("scene video reports no size, the recording may be missing its video")
        return int(scene.height), int(scene.width)

    def scene_camera_matrix(self) -> np.ndarray:
        return np.asarray(self._calibration().scene_camera_matrix, dtype=np.float64)

    def scene_distortion_coefficients(self) -> np.ndarray:
        return np.asarray(self._calibration().scene_distortion_coefficients, dtype=np.float64).reshape(-1)

    def scene_frame_rgb(self, index: int) -> np.ndarray:
        # Only the kept frame is converted to RGB. The video decoder still steps through the frames
        # between this one and the last, so decode cost follows the recording's length.
        return np.asarray(self._recording.scene[index].rgb, dtype=np.uint8)

    def imu_samples(self) -> tuple[np.ndarray, np.ndarray]:
        imu = self._stream(RecordingStream.IMU)
        if imu is None:
            return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float64)
        # The recording stores x, y, z, w. Every Pose is w, x, y, z. Reordering by name here is what
        # keeps pitch from flipping silently.
        rotations = np.asarray(imu.rotation, dtype=np.float64).reshape(-1, 4)
        return np.asarray(imu.time, dtype=np.int64), xyzw_to_wxyz(rotations)

    def gaze_points_at(self, times_ns: np.ndarray, tolerance_ns: int) -> np.ndarray | None:
        samples = self.gaze_samples()
        if samples is None:
            return None
        gaze_times_ns, points = samples
        return sample_nearest(gaze_times_ns, points, times_ns, tolerance_ns)

    def gaze_samples(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Every gaze sample as recorded: (N,) int64 times and (N, 2) scene pixels. None when the stream won't load."""
        gaze = self._stream(RecordingStream.GAZE)
        if gaze is None:
            return None
        return np.asarray(gaze.time, dtype=np.int64), np.asarray(gaze.point, dtype=np.float64).reshape(-1, 2)

    def close(self) -> None:
        self._recording.close()

    def _calibration(self) -> "Calibration":
        calibration = self._recording.calibration
        if calibration is None:
            raise FileNotFoundError("recording has no calibration.bin, so the scene camera intrinsics are unknown")
        return calibration

    def _stream(self, stream: "RecordingStream") -> object | None:
        """
        A stream, or None when it won't load.

        A Companion export always carries both streams, so a load failure is never routine. It's logged
        as a warning with what actually went wrong, and the replay carries on without that stream.
        """
        try:
            return getattr(self._recording, stream.value)
        except self._stream_error as failure:
            log.warning(
                "the %s stream would not load (caught %s: %s, caused by %r), so frames get none",
                stream.value,
                type(failure).__name__,
                failure,
                failure.__cause__,
            )
            return None
