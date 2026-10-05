"""
The frame loop. Source to sink, with the scene, the planner and the user model in between.

This is the one place every layer is named together. It builds them from the configuration,
feeds frames to the worker, publishes the newest result, and closes everything in the right order
when the source ends or the user interrupts.
"""

# Standard library imports
import json
import logging
import time
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.config import RunConfig, build_sink, build_source
from nav.planner.pipeline import PlannerPipeline
from nav.runtime.textio import append_text_lf, write_text_lf
from nav.runtime.worker import NewestFrameWorker
from nav.scene.floor import ground_axes
from nav.scene.pipeline import ScenePipeline
from nav.scene.transform import rotation_matrix_from_quaternion_wxyz
from nav.types import DebugSink, DebugView, DepthFrame, PlannedPath
from nav.usermodel.work import WorkMeter

log = logging.getLogger(__name__)

EPISODES_FILENAME = "episodes.jsonl"
RUN_CONFIG_FILENAME = "run_config.json"
# The observed heading is measured against a slowly moving baseline, so a long curve in the
# path does not read as one permanent turn. Five seconds is a few strides.
HEADING_BASELINE_SECONDS = 5.0
CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class FrameResult:
    """What one frame produced, kept together so the debug sink draws a field, a view and their own path."""

    path: PlannedPath
    field: np.ndarray
    grid: np.ndarray
    view: DebugView


def yaw_from_quaternion(orientation_wxyz: np.ndarray) -> float:
    """
    The heading the camera's forward axis has about the vertical, in radians. Positive is right.

    Vertical here is camera up before any rotation, so for a device that reports orientation
    relative to gravity this is the compass-free yaw.

    :param orientation_wxyz: Unit quaternion, (w, x, y, z).
    :return: Yaw in radians, in (-pi, pi].
    :rtype: float
    """
    forward_world = rotation_matrix_from_quaternion_wxyz(orientation_wxyz) @ CAMERA_FORWARD
    # Camera y points down, so the ground plane is x and z. Yaw is the angle of forward in it.
    return float(np.arctan2(forward_world[0], forward_world[2]))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class HeadingBaseline:
    """A slow running average of yaw, so the observed heading is the departure from it."""

    def __init__(self, time_constant_seconds: float) -> None:
        self._time_constant = time_constant_seconds
        self._baseline: float | None = None
        self._last_time: float | None = None

    def observed(self, yaw: float, timestamp_seconds: float) -> float:
        if self._baseline is None or self._last_time is None:
            self._baseline = yaw
            self._last_time = timestamp_seconds
            return 0.0
        dt = max(0.0, timestamp_seconds - self._last_time)
        self._last_time = timestamp_seconds
        weight = min(1.0, dt / self._time_constant) if self._time_constant > 0 else 1.0
        self._baseline = wrap_angle(self._baseline + weight * wrap_angle(yaw - self._baseline))
        return wrap_angle(yaw - self._baseline)


def gaze_on_the_ground(frame: DepthFrame, scene: ScenePipeline) -> np.ndarray | None:
    """
    Where the wearer is looking, on the floor, as (lateral, forward) from the walker.

    Casts the gaze pixel as a ray through the camera and intersects it with the scene's current
    floor plane. None when there is no gaze, no plane yet, or the ray does not reach the floor
    ahead, in which case the planner aims straight ahead.
    """
    plane = scene.previous_plane
    if frame.gaze_pixel is None or plane is None:
        return None
    focal_x, focal_y = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
    principal_x, principal_y = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
    ray = np.array([(frame.gaze_pixel[0] - principal_x) / focal_x, (frame.gaze_pixel[1] - principal_y) / focal_y, 1.0])
    # normal . (t ray) + offset == 0 on the plane.
    descent = ray @ plane.normal
    if descent >= 0:
        # Looking level or up. The ray never meets the floor.
        return None
    distance = -plane.offset_meters / descent
    if distance <= 0:
        return None
    point = ray * distance
    lateral_axis, forward_axis = ground_axes(plane)
    return np.array([point @ lateral_axis, point @ forward_axis])


def run(config: RunConfig) -> int:
    """
    Run the pipeline from the configured source to the configured sink until the source ends.

    :param config: Everything, from the command line.
    :return: 0 on normal completion, 1 after an unexpected error in the worker.
    :rtype: int
    """
    scene = ScenePipeline(config.scene, config.walker)
    planner = PlannerPipeline(config.planner, config.walker)
    meter = WorkMeter(config.usermodel)
    baseline = HeadingBaseline(HEADING_BASELINE_SECONDS)

    def process(frame: DepthFrame) -> FrameResult:
        started = time.perf_counter()
        obstacles = scene.process(frame)
        after_scene = time.perf_counter()
        gaze = gaze_on_the_ground(frame, scene)
        path = planner.plan(obstacles, 0.0, config.goal_mode, gaze)
        after_planner = time.perf_counter()
        observed = baseline.observed(yaw_from_quaternion(frame.pose.orientation), frame.timestamp_seconds)
        meter.observe(path, observed, frame.timestamp_seconds)
        finished = time.perf_counter()
        log.debug(
            "frame %.3f: scene %.1f ms, planner %.1f ms, user model %.1f ms, %d groups, heading %.1f deg%s",
            frame.timestamp_seconds,
            (after_scene - started) * 1000,
            (after_planner - after_scene) * 1000,
            (finished - after_planner) * 1000,
            obstacles.groups_in_view,
            np.degrees(path.first_heading_radians),
            " ALARM" if path.alarm else "",
        )
        field = planner.last_field
        # Built here, on the worker thread, right after the scene ran, so the floor the view
        # names is the one this frame used and not a later frame's.
        floor = scene.previous_plane
        floor_source = scene.last_floor_source
        if floor is None or floor_source is None:
            raise RuntimeError("the scene processed a frame and has no floor to show for it")
        view = DebugView(
            frame=frame,
            obstacles=obstacles,
            floor=floor,
            floor_source=floor_source,
            walking_speed_mps=config.planner.walking_speed_mps,
            body_half_width_meters=config.planner.body_half_width_meters,
        )
        return FrameResult(path=path, field=field if field is not None else np.zeros((1, len(planner.grid))), grid=planner.grid, view=view)

    sink = build_sink(config)
    source = build_source(config)
    worker: NewestFrameWorker[FrameResult] = NewestFrameWorker(process)
    worker.start()
    publisher = NewestResultPublisher(sink)

    exit_code = 0
    frames_in = 0
    try:
        # The sink listens before the source has a frame, so a browser or the phone's path
        # connection finds something to connect to from the start. A port it cannot bind is a
        # ConnectionError, caught below as the OSError it is, and the run ends with the reason.
        sink.start()
        while True:
            try:
                for frame in source.frames():
                    frames_in += 1
                    worker.submit(frame)
                    publisher.publish(worker)
            except ConnectionError as nobody_came_back:
                if not (config.reconnect and frames_in > 0):
                    raise
                # The phone had connected before and did not come back within the accept timeout.
                # Waiting was what --reconnect asked for, so this is the walk ending, not a fault.
                log.info("no further connection, ending the run: %s", nobody_came_back)
                break
            # The source ended. Let the worker finish the frame it holds, then publish it.
            worker.wait_until_idle()
            publisher.publish(worker)

            if not config.reconnect:
                break
            # The same source accepts the next connection, and the same worker keeps its counts.
            log.info("source ended, waiting for the next connection")
    except KeyboardInterrupt:
        log.info("interrupted")
    except (OSError, ValueError) as refused:
        # The source or the sink refused its input: a file that is not there, a port nobody sent
        # to, a phone that is unreachable, a recording in the wrong format. Known and named, and
        # the run cannot go on without it, so it ends with the message and no traceback.
        log.error("%s: %s", type(refused).__name__, refused)
        exit_code = 1
    except Exception as unexpected_error:
        # The worker's stored failure, re-raised by latest_result, or anything else nobody
        # predicted. Logged as such, and the exit code says the run did not finish on its own terms.
        log.error("UNEXPECTED %s in the loop, may need a handler", type(unexpected_error).__name__, exc_info=True)
        exit_code = 1
    finally:
        worker.stop()
        sink.close()
        source.close()
        _report(worker, meter, frames_in, config)

    return exit_code


class NewestResultPublisher:
    """Hands each new result to the sink once, however many source frames arrive while it is new."""

    def __init__(self, sink) -> None:
        self._sink = sink
        self._last: FrameResult | None = None

    def publish(self, worker: NewestFrameWorker) -> None:
        result = worker.latest_result()
        # Identity, not equality. The worker hands out the same object until it has a new one,
        # and a web or phone sink sent the same path six times over per planned frame without this.
        if result is None or result is self._last:
            return
        self._last = result
        if isinstance(self._sink, DebugSink):
            self._sink.publish_debug(result.path, result.field, result.grid, result.view)
        else:
            self._sink.publish(result.path)


def _report(worker: NewestFrameWorker, meter: WorkMeter, frames_in: int, config: RunConfig) -> None:
    log.info(
        "%d frames in, %d processed, %d dropped as stale, %d skipped as unusable",
        frames_in, worker.processed, worker.dropped, worker.skipped,
    )
    episodes = meter.completed_episodes()
    for episode in episodes:
        log.info(
            "avoidance %.1f to %.1f s: %.2f bits of work, turn observed %s s, predicted %.2f s",
            episode.start_seconds, episode.end_seconds, episode.work_bits,
            "none" if episode.observed_turn_seconds is None else f"{episode.observed_turn_seconds:.2f}",
            episode.predicted_turn_seconds,
        )
    if meter.episode_open:
        log.info("one avoidance was still in progress when the run ended")

    if config.tap.log_dir is not None:
        log_dir = Path(config.tap.log_dir)
        if log_dir.is_dir():
            # A frame log does not carry the scene configuration that produced it, and a replay
            # with a different floor gate can refuse every frame the recording run accepted. The
            # configuration goes beside the frames so a replay can be given the same flags.
            write_text_lf(log_dir / RUN_CONFIG_FILENAME, json.dumps(_config_as_json(config), indent=2) + "\n")
        if episodes:
            target = log_dir / EPISODES_FILENAME
            for episode in episodes:
                append_text_lf(target, json.dumps(episode.__dict__, allow_nan=False) + "\n")
            log.info("wrote %d episodes to %s", len(episodes), target)


def _config_as_json(config: RunConfig) -> dict:
    """The run configuration as plain JSON values. Enums by value, the test hook left out."""
    def plain(value):
        if is_dataclass(value):
            return {name: plain(getattr(value, name)) for name in value.__dataclass_fields__ if name != "estimator_factory"}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (tuple, list)):
            # The sinks a run named. Without this the enums inside reach the JSON encoder, which
            # refuses them, and a recording loses the configuration it is supposed to carry.
            return [plain(item) for item in value]
        return value

    return plain(config)
