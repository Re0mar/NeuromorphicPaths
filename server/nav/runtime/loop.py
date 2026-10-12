"""
The frame loop. Source to sink, with the scene, the planner and the user model in between.

This is the one place every layer is named together. It builds them from the configuration,
feeds frames to the worker, publishes the newest result, and closes everything in the right order
when the source ends or the user interrupts.
"""

# Standard library imports
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.config import RunConfig, build_source_and_sink
from nav.planner.alarm import path_red_from_bits
from nav.planner.pipeline import PlannerPipeline
from nav.runtime.textio import append_text_lf, write_text_lf
from nav.runtime.timing import StageDurations, TimingLog, TimingRecorder
from nav.runtime.worker import NewestFrameWorker
from nav.scene.floor import ground_axes
from nav.scene.pipeline import ScenePipeline
from nav.scene.transform import rotation_matrix_from_quaternion_wxyz
from nav.types import DebugSink, DebugView, DepthFrame, FloorSource, PlannedPath
from nav.usermodel.work import WorkMeter

log = logging.getLogger(__name__)

EPISODES_FILENAME = "episodes.jsonl"
RUN_CONFIG_FILENAME = "run_config.json"
# The observed heading is measured against a slowly moving baseline, so a long curve in the
# path does not read as one permanent turn. Five seconds is a few strides.
HEADING_BASELINE_SECONDS = 5.0
CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])
# How long the publisher waits for a result before checking whether it was asked to stop. Short,
# so stopping takes a tenth of a second at most. A new result wakes it at once regardless.
PUBLISHER_POLL_SECONDS = 0.1
# How long the end of a source waits for the last path to reach every display.
PUBLISHER_FLUSH_SECONDS = 5.0


@dataclass(frozen=True)
class FrameResult:
    """What one frame produced, kept together so the debug sink draws a field, a view and their own path."""

    path: PlannedPath
    field: np.ndarray
    grid: np.ndarray
    view: DebugView


def yaw_from_quaternion(orientation_wxyz: np.ndarray) -> float:
    """
    The heading the camera's forward axis has about the vertical, in radians.

    It is the angle of the rotated forward axis in the world's x-z plane, measured from +z toward
    +x. Positive is toward the world's +x. Vertical here is camera up before any rotation, so for
    a device that reports orientation relative to gravity this is the compass-free yaw.

    In a y-down world, the camera's own frame, +x is right and so positive is a right turn. In a
    y-up world, which ARCore and the mounted Neon both report, a right turn reads negative.

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
    timing = TimingRecorder(_timing_log(config))
    # Floor sources over the frames the worker took, None for a frame skipped with no usable floor.
    # This is the live run's floor acceptance figure.
    floor_counts: Counter[FloorSource | None] = Counter()
    # The stretch of video the scene and the planner remember. A source that switches between the
    # glasses and a demo, or starts the demo over, moves it on.
    generation = 0

    def process(frame: DepthFrame) -> FrameResult:
        nonlocal scene, planner, baseline, generation
        if frame.source_generation != generation:
            # A floor, a previous plan and a heading baseline from another stretch of video would
            # steer this one's first frames, so all three start afresh, on the worker, before it plans.
            generation = frame.source_generation
            scene = ScenePipeline(config.scene, config.walker)
            planner = PlannerPipeline(config.planner, config.walker)
            baseline = HeadingBaseline(HEADING_BASELINE_SECONDS)
            log.info("new stretch of video %d, the scene and the planner start afresh", generation)
        timing.frame_taken(frame)
        try:
            result, plan_done_seconds, stages = plan_frame(frame)
        except ValueError:
            # The worker skips the frame on this. It can come from the scene refusing the frame or
            # from any later step, the planner included, so the floor is read off the scene rather
            # than assumed missing. The scene clears its source at the start of every frame, so a
            # refusal reads None here. The line is written anyway, or the floor acceptance read back
            # from the log counts only planned frames and reads perfect however many were refused.
            floor_counts[scene.last_floor_source] += 1
            timing.frame_skipped(frame, scene.last_floor_source)
            raise
        # Recorded here, after everything that can raise, so no frame is both planned and skipped.
        floor_counts[scene.last_floor_source] += 1
        timing.frame_planned(frame, plan_done_seconds, scene.last_floor_source, stages)
        return result

    def plan_frame(frame: DepthFrame) -> tuple[FrameResult, float, StageDurations]:
        started = time.perf_counter()
        obstacles = scene.process(frame)
        after_scene = time.perf_counter()
        gaze = gaze_on_the_ground(frame, scene)
        path = planner.plan(obstacles, 0.0, config.goal_mode, gaze)
        after_planner = time.perf_counter()
        plan_done_seconds = laptop_time_seconds()
        observed = baseline.observed(yaw_from_quaternion(frame.pose.orientation), frame.timestamp_seconds)
        meter.observe(path, observed, frame.timestamp_seconds)
        finished = time.perf_counter()
        log.debug(
            "frame %.3f: scene %.1f ms, planner %.1f ms, user model %.1f ms, %d groups, heading %.1f deg, observed %.1f deg%s%s",
            frame.timestamp_seconds,
            (after_scene - started) * 1000,
            (after_planner - after_scene) * 1000,
            (finished - after_planner) * 1000,
            obstacles.groups_in_view,
            np.degrees(path.lookahead_heading_radians),
            np.degrees(observed),
            _latency_shares(frame, plan_done_seconds),
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
            path_red_from_bits=path_red_from_bits(config.planner),
        )
        result = FrameResult(path=path, field=field if field is not None else np.zeros((1, len(planner.grid))), grid=planner.grid, view=view)
        stages = StageDurations(
            scene_milliseconds=(after_scene - started) * 1000,
            planner_milliseconds=(after_planner - after_scene) * 1000,
            usermodel_milliseconds=(finished - after_planner) * 1000,
        )
        return result, plan_done_seconds, stages

    # Built together, because the glasses' video goes from the source's device to the web sink
    # without passing through here.
    source, sink = build_source_and_sink(config, on_path_sent=timing.path_sent)
    worker: NewestFrameWorker[FrameResult] = NewestFrameWorker(process, on_dropped=timing.frame_dropped)
    worker.start()
    publisher = PublisherThread(sink, worker, on_published=timing.path_published)

    def raise_any_failure() -> None:
        # Nothing else would end the run on a dead worker or publisher. The main thread would keep
        # feeding frames to a worker nobody reads.
        worker.raise_failure()
        publisher.raise_failure()

    exit_code = 0
    frames_in = 0
    try:
        # The sink listens before the source has a frame, so a browser or the phone's path
        # connection finds something to connect to from the start. A port it cannot bind is a
        # ConnectionError, caught below as the OSError it is, and the run ends with the reason.
        sink.start()
        publisher.start()
        while True:
            try:
                for frame in source.frames():
                    frames_in += 1
                    worker.submit(frame)
                    raise_any_failure()
            except ConnectionError as nobody_came_back:
                if not (config.reconnect and frames_in > 0):
                    raise
                # The phone had connected before and did not come back within the accept timeout.
                # Waiting was what --reconnect asked for, so this is the walk ending, not a fault.
                log.info("no further connection, ending the run: %s", nobody_came_back)
                break
            # The source ended. Let the worker finish the frame it holds and the publisher send it.
            # A failure on that last frame is only seen here, since no further submit comes.
            worker.wait_until_idle()
            raise_any_failure()
            if not publisher.flush(PUBLISHER_FLUSH_SECONDS):
                log.warning("the last path didn't reach every display within %.0f s", PUBLISHER_FLUSH_SECONDS)
            raise_any_failure()

            if not config.reconnect:
                break
            # The same source accepts the next connection, and the same worker keeps its counts.
            log.info("source ended, waiting for the next connection")
    except KeyboardInterrupt:
        log.info("interrupted")
        # A failure stored while the source sat quiet would otherwise end the run as a success.
        try:
            raise_any_failure()
        except Exception as stored_failure:
            log.error("UNEXPECTED %s before the interrupt, may need a handler", type(stored_failure).__name__, exc_info=stored_failure)
            exit_code = 1
    except (OSError, ValueError) as refused:
        # The source or the sink refused its input: a file that is not there, a port nobody sent
        # to, a phone that is unreachable, a recording in the wrong format. Known and named, and
        # the run cannot go on without it, so it ends with the message and no traceback.
        log.error("%s: %s", type(refused).__name__, refused)
        exit_code = 1
    except Exception as unexpected_error:
        # The worker's or the publisher's stored failure, or anything else nobody predicted.
        # Logged as such, and the exit code says the run did not finish on its own terms.
        log.error("UNEXPECTED %s in the loop, may need a handler", type(unexpected_error).__name__, exc_info=True)
        exit_code = 1
    finally:
        # The publisher stops first, so nothing is handed to a sink after it closes. The timing
        # log closes last, so every frame's last stamp from the worker, the sinks and the source
        # is queued before its writer stops.
        publisher.stop()
        worker.stop()
        sink.close()
        source.close()
        timing.close()
        _report(worker, meter, frames_in, config, floor_counts)

    return exit_code


class PublisherThread(threading.Thread):
    """
    Hands each new result to the sink once, the moment the worker has it.

    Publishing used to happen on the main thread after the next frame arrived, so a finished path
    sat waiting for a frame that had nothing to do with it. On its own thread it goes out at once.
    A slow sink now delays only the next publish, never reading frames or planning them.
    """

    def __init__(self, sink, worker: NewestFrameWorker, on_published: Callable[[float], None] | None = None) -> None:
        """
        :param sink: Every display, behind one sink. Already started, on the main thread.
        :param worker: Where the results come from.
        :param on_published: Called with the path's `timestamp_seconds` after every display has been handed it.
        """
        super().__init__(name="path-publisher", daemon=True)
        self._sink = sink
        self._worker = worker
        self._on_published = on_published
        self._last: FrameResult | None = None
        self._failure: BaseException | None = None
        self._stopping = threading.Event()
        self._published = threading.Condition()

    def run(self) -> None:
        try:
            while not self._stopping.is_set():
                result = self._worker.wait_for_result(self._last, PUBLISHER_POLL_SECONDS)
                if result is not None:
                    self._publish(result)
        except Exception as unexpected_error:
            # The worker's own failure re-raised, or a sink failing in a way no sink handles. The
            # main loop re-raises it. Logged here as well, because a live source can hold the main
            # loop until the next frame.
            if unexpected_error is self._worker.failure:
                # The worker already logged it with its traceback. This is not a second fault.
                log.info("publisher stopping, the worker failed")
            else:
                log.error("publisher stopped on UNEXPECTED %s, the run ends at the next frame", type(unexpected_error).__name__, exc_info=True)
            self._failure = unexpected_error
        finally:
            # Wakes a flush that would otherwise wait out its timeout on a thread that has ended.
            with self._published:
                self._published.notify_all()

    def raise_failure(self) -> None:
        """:raises: Whatever ended this thread. Returns quietly while it is publishing."""
        if self._failure is not None:
            raise self._failure

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        """
        Block until the worker's newest result has been published, or the timeout passes.

        :param timeout_seconds: How long to wait at most.
        :return: True once published, or once this thread has ended. False when the timeout passed first.
        :rtype: bool
        :raises: The worker's stored failure.
        """
        target = self._worker.latest_result()
        if target is None:
            return True
        with self._published:
            return self._published.wait_for(lambda: self._last is target or not self.is_alive(), timeout_seconds)

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """
        Ask the thread to stop, and wait for it at most `timeout_seconds`.

        A thread still inside a sink after that is left running, and said so, because the sinks
        close next and a sink that won't return is the likely reason.
        """
        self._stopping.set()
        if self.is_alive():
            self.join(timeout_seconds)
        if self.is_alive():
            log.warning("publisher thread still inside a display after %.0f s, closing the displays under it", timeout_seconds)

    def _publish(self, result: FrameResult) -> None:
        if isinstance(self._sink, DebugSink):
            self._sink.publish_debug(result.path, result.field, result.grid, result.view)
        else:
            self._sink.publish(result.path)
        if self._on_published is not None:
            self._on_published(result.path.timestamp_seconds)
        with self._published:
            self._last = result
            self._published.notify_all()


def _timing_log(config: RunConfig) -> TimingLog | None:
    """
    Where this run's timing lines go, or None for a run that keeps none.

    --timing-log names a file, for a run that measures without recording. A recording run keeps
    timing.jsonl beside its frames. The parser refuses the two together.
    """
    if config.timing_log is not None:
        return TimingLog(Path(config.timing_log))
    if config.tap.log_dir is not None:
        return TimingLog.in_directory(Path(config.tap.log_dir))
    return None


def _latency_shares(frame: DepthFrame, plan_done_seconds: float) -> str:
    """The frame's trip so far, in milliseconds, for the --verbose line. Empty without timing."""
    timing = frame.timing
    if timing is None:
        return ""
    shares = []
    if timing.capture_seconds is not None:
        shares.append(f"capture to arrival {(timing.arrival_seconds - timing.capture_seconds) * 1000:.0f} ms")
    if timing.depth_ready_seconds is not None:
        shares.append(f"arrival to depth {(timing.depth_ready_seconds - timing.arrival_seconds) * 1000:.0f} ms")
        shares.append(f"depth to plan {(plan_done_seconds - timing.depth_ready_seconds) * 1000:.0f} ms")
    else:
        shares.append(f"arrival to plan {(plan_done_seconds - timing.arrival_seconds) * 1000:.0f} ms")
    return ", " + ", ".join(shares)


def _report(
    worker: NewestFrameWorker,
    meter: WorkMeter,
    frames_in: int,
    config: RunConfig,
    floor_counts: Counter[FloorSource | None],
) -> None:
    log.info(
        "%d frames in, %d processed, %d dropped as stale, %d skipped as unusable",
        frames_in, worker.processed, worker.dropped, worker.skipped,
    )
    taken = sum(floor_counts.values())
    if taken:
        log.info(
            "floor over %d frames taken: %s",
            taken,
            ", ".join(
                f"{source.value if source is not None else 'none, skipped'} {count} ({100.0 * count / taken:.0f}%)"
                for source, count in sorted(floor_counts.items(), key=lambda item: -item[1])
            ),
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
