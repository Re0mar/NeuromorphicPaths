"""
How far the floor leans from up, on the same frames posed two ways: as the run posed them, and from
the IMU reading nearest each frame's capture.

A replay of a glasses capture plans whichever frames the laptop was ready for, so two replays plan
different frames and their floors can't be compared frame for frame. This refits one replay's own
frames instead. Each frame goes through two scenes: one with the pose the run logged, one with the
pose the capture's IMU had when the frame was taken. On a flat floor the right pose leaves the
fitted floor level, so how far it leans from that pose's up measures the pose's error.

A replay moves the capture's stamps onto today's clock by one constant, and it logs that constant as
"stamps shifted by". It has to be given, not fitted. The capture's frames are 33.3333 or 33.3445 ms
apart, so stamps slid one frame along still match to within 11 us and a fit can't tell that from the
right answer. Every logged stamp minus the shift must land within MAX_STAMP_RESIDUAL_SECONDS of one of the
capture's own, or nothing is reported.
"""

# Standard library imports
import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.pose.imu_orientation import identity_pose, orientation_at, pose_from_imu
from nav.pose.neon_mount import NEON_IMU_MOUNT
from nav.runtime.timing import FrameOutcome, read_timing_log
from nav.scene.config import SceneConfig
from nav.scene.pipeline import ScenePipeline
from nav.sources.logged import LoggedDepthFrameSource
from nav.sources.neon_stream import IMU_FIELDS, IMU_FILENAME, SCENE_PACKETS_FILENAME, read_capture_packets, read_capture_samples
from nav.types import DepthFrame, FloorSource
from nav.walker import WalkerConfig

# The right shift leaves under a microsecond, float64 at these stamps being about 0.25 us. A shift
# one frame off leaves about 11 us, which this has to refuse.
MAX_STAMP_RESIDUAL_SECONDS = 2e-6


class StampsDoNotMatch(ValueError):
    """The frame log's stamps, less the given shift, are not the capture's. Wrong shift or wrong capture."""


class NothingToCompare(ValueError):
    """The run dropped every frame it logged before its scene saw it, so there is no floor on either side."""


@dataclass(frozen=True)
class PoseFloors:
    """One way of posing the frames, and the floors it gave."""

    frames: int
    fitted: int
    # Degrees from that pose's up, one per fitted floor.
    leans_degrees: np.ndarray
    frames_without_pose: int


@dataclass(frozen=True)
class FloorLean:
    """The same frames posed as logged and posed from the capture's IMU."""

    logged: PoseFloors
    capture: PoseFloors
    largest_stamp_residual_seconds: float
    # Frames where refitting with the logged pose did not give the floor source the run recorded.
    # Zero is what makes the comparison trustworthy, since the logged side should be the run itself.
    # None when the log has no timing log to check against.
    logged_mismatches: int | None
    # Frames the run logged and dropped for a newer one before its scene saw them. Left out of both sides.
    dropped_by_the_run: int


@dataclass
class _Side:
    scene: ScenePipeline
    fitted: int = 0
    frames_without_pose: int = 0
    leans: list[float] = dataclasses.field(default_factory=list)


def floor_lean(log_dir: Path, capture_dir: Path, replay_shift_seconds: float, scene_config: SceneConfig) -> FloorLean:
    """
    Refit one replay's frames with the logged pose and with the capture-time pose, and compare the floors.

    :param log_dir: A frame log written by `--record-to` during a `--neon-replay` run.
    :param capture_dir: The capture that run played back.
    :param replay_shift_seconds: The shift the run logged as "stamps shifted by".
    :param scene_config: The run's scene settings.
    :return: Both sides' floors, the largest stamp residual, and the reproduction check.
    :rtype: FloorLean
    :raises StampsDoNotMatch: When a logged stamp, less the shift, is no capture stamp.
    :raises NothingToCompare: When the run dropped every frame, so neither side has one.
    :raises FrameDecodeError: When the frame log is empty or unreadable, from the log's own reader.
    """
    capture_stamps = np.unique(np.array([stamp for stamp, _ in read_capture_packets(Path(capture_dir) / SCENE_PACKETS_FILENAME)]))
    imu = np.array(read_capture_samples(Path(capture_dir) / IMU_FILENAME, IMU_FIELDS), dtype=np.float64).reshape(-1, len(IMU_FIELDS))
    imu = imu[np.argsort(imu[:, 0], kind="stable")]
    try:
        recorded = {record.timestamp_seconds: record for record in read_timing_log(Path(log_dir))}
    except FileNotFoundError:
        # An older log, recorded before runs kept a timing log. The comparison still stands, it
        # just can't be checked against what the run itself fitted.
        recorded = None

    walker = WalkerConfig()
    # One plane fitted without trusting the pose, because the pose is what's being measured. The
    # level-surface choice refuses a floor leaning from the pose's up, so a wrong pose would show
    # as no floor instead of as a lean.
    scene_config = dataclasses.replace(scene_config, floor_from_level_surfaces=False)
    logged = _Side(ScenePipeline(scene_config, walker))
    capture = _Side(ScenePipeline(scene_config, walker))
    frames = 0
    dropped = 0
    largest_residual = 0.0
    mismatches = 0
    for frame in _frames(log_dir):
        record = None if recorded is None else recorded.get(frame.timestamp_seconds)
        if record is not None and record.outcome is FrameOutcome.DROPPED:
            # The run logged it and a newer frame replaced it before its scene saw it. Refitting it
            # would give both scenes a frame the run's scene never had, and its floor carries on.
            dropped += 1
            continue
        frames += 1
        capture_stamp = frame.timestamp_seconds - replay_shift_seconds
        residual = _residual(capture_stamps, capture_stamp)
        if residual > MAX_STAMP_RESIDUAL_SECONDS:
            raise StampsDoNotMatch(
                f"the frame at {frame.timestamp_seconds!r} less the shift {replay_shift_seconds!r} is {residual * 1e6:.1f} us "
                f"from the nearest frame in {Path(capture_dir).name}, over the {MAX_STAMP_RESIDUAL_SECONDS * 1e6:.0f} us allowed. "
                f"Copy the shift from the run's own 'stamps shifted by' line, and check it is that run's capture"
            )
        largest_residual = max(largest_residual, residual)

        orientation = orientation_at(imu[:, 0], imu[:, 1:], capture_stamp)
        posed_at_capture = dataclasses.replace(
            frame,
            pose=identity_pose() if orientation is None else pose_from_imu(orientation, NEON_IMU_MOUNT),
        )
        logged_source = _fit(logged, frame)
        _fit(capture, posed_at_capture)
        if record is not None and record.floor_source is not logged_source:
            mismatches += 1
    if frames == 0:
        # The log's reader has already refused a log with no frames. This is a log with frames the
        # run never fitted, which a table of zeros would pass off as a comparison.
        raise NothingToCompare(f"the run dropped all {dropped} frames in {Path(log_dir).name} before its scene saw them, so there is no floor to compare")
    return FloorLean(
        logged=_summary(logged, frames),
        capture=_summary(capture, frames),
        largest_stamp_residual_seconds=largest_residual,
        logged_mismatches=None if recorded is None else mismatches,
        dropped_by_the_run=dropped,
    )


def format_floor_lean(name: str, result: FloorLean) -> str:
    """The two sides as a table, with the checks that say whether to trust it."""
    lines = [
        f"=== {name}",
        f"stamps matched to the capture within {result.largest_stamp_residual_seconds * 1e6:.2f} us",
        _mismatch_line(result.logged_mismatches),
        f"{result.dropped_by_the_run} frames the run dropped for a newer one were left out, as its scene never saw them",
        "",
        f"{'pose':<10}{'frames':>8}{'no pose':>9}{'fitted':>8}{'share':>8}{'lean median':>13}{'p90':>8}{'max':>8}",
    ]
    for label, side in (("logged", result.logged), ("capture", result.capture)):
        if side.leans_degrees.size:
            median, p90, largest = (float(np.percentile(side.leans_degrees, q)) for q in (50, 90, 100))
            leans = f"{median:>13.2f}{p90:>8.2f}{largest:>8.2f}"
        else:
            leans = f"{'-':>13}{'-':>8}{'-':>8}"
        # A side with no frames has no share. floor_lean refuses that case, but the table shouldn't divide by it.
        share = f"{side.fitted / side.frames:>8.1%}" if side.frames else f"{'-':>8}"
        lines.append(f"{label:<10}{side.frames:>8}{side.frames_without_pose:>9}{side.fitted:>8}{share}{leans}")
    lines.append("Lean is degrees between the fitted floor's normal and that pose's up. On a flat floor a right pose gives near zero.")
    return "\n".join(lines) + "\n"


def _mismatch_line(mismatches: int | None) -> str:
    if mismatches is None:
        return "no timing log, so the logged side was not checked against the floors the run fitted"
    if mismatches == 0:
        return "refitting with the logged pose gave the run's own floor source on every frame"
    return f"refitting with the logged pose gave a different floor source than the run on {mismatches} frames, so read the comparison with care"


def _frames(log_dir: Path) -> Iterator[DepthFrame]:
    source = LoggedDepthFrameSource(Path(log_dir))
    try:
        yield from source.frames()
    finally:
        source.close()


def _residual(sorted_stamps: np.ndarray, stamp: float) -> float:
    index = int(np.searchsorted(sorted_stamps, stamp))
    neighbors = sorted_stamps[max(index - 1, 0) : index + 1]
    return float(np.min(np.abs(neighbors - stamp)))


def _fit(side: _Side, frame: DepthFrame) -> FloorSource | None:
    """Run the frame through this side's scene and keep the floor's lean when it was fitted."""
    if not frame.pose.orientation_is_gravity_aligned:
        side.frames_without_pose += 1
    try:
        side.scene.process(frame)
    except ValueError:
        # A frame the scene refused, as the live worker and the replay both allow. No floor for it.
        return None
    source = side.scene.last_floor_source
    if source is FloorSource.FITTED:
        side.fitted += 1
        up = ScenePipeline.up_in_camera_frame(frame)
        side.leans.append(float(np.degrees(np.arccos(np.clip(side.scene.previous_plane.normal @ up, -1.0, 1.0)))))
    return source


def _summary(side: _Side, frames: int) -> PoseFloors:
    return PoseFloors(
        frames=frames,
        fitted=side.fitted,
        leans_degrees=np.asarray(side.leans, dtype=np.float64),
        frames_without_pose=side.frames_without_pose,
    )
