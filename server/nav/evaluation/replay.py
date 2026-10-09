"""
Replays a recorded walk through the real scene and planner, and scores the planner's arrow.

Frames are read the way production reads a recording, through `LoggedDepthFrameSource`. One scene and
one planner run across the whole log, as a live run does, and each planned frame gets the same
`plan` call the loop makes. The scene is built from the recording's own `run_config.json`, so a replay
sees the floor the live run saw.

The scene pass is the slow part, minutes per walk, and can be cached. The floor fit is seeded, so a
cold pass repeats. Verdicts still come from several cold passes, because their agreeing is the
evidence that it still does.
"""

# Standard library imports
import dataclasses
import hashlib
import json
import os
import pickle
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np
import open3d

# Local package imports
import nav
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.frames import PlannedFrame
from nav.evaluation.overrides import apply_overrides
from nav.evaluation.planner_numbers import PlannerInput, ReplayedFrame
from nav.evaluation.scoring import (
    LaggedCorrelation,
    TurnScore,
    WalkSummary,
    arrow_in_travel_frame,
    find_sidesteps,
    lagged_correlation,
    score_turn,
    summarize,
)
from nav.evaluation.track import WalkerTrack, walker_track
from nav.evaluation.turns import StraightSpread, Turn, TurnTag, detect_turns, straight_stretch_spread, tag_turn
from nav.planner.alarm import alarm_raised
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.pipeline import PlannerPipeline
from nav.runtime.loop import RUN_CONFIG_FILENAME, gaze_on_the_ground
from nav.scene.config import SceneConfig
from nav.scene.floor import FloorRefusal, ground_axes
from nav.scene.pipeline import CAMERA_FORWARD, ScenePipeline
from nav.scene.transform import camera_to_world_plane, rotation_matrix_from_quaternion_wxyz
from nav.sources.framecodec import INDEX_FILENAME
from nav.sources.logged import LoggedDepthFrameSource
from nav.types import FloorSource, ObstacleSet, Plane
from nav.walker import WalkerConfig

NAV_DIR = Path(nav.__file__).resolve().parent
CLONE_DIR = NAV_DIR.parent.parent
# Bumped whenever the pickled layout changes, so an old cache entry is rebuilt rather than misread.
CACHE_FORMAT = "cache-format-4"
# Beside the server code, not in whatever folder the command happens to run from, so it always lands
# where .gitignore covers it.
DEFAULT_CACHE_DIR = NAV_DIR.parent / ".replay_cache"
# Kept per walk. Beyond this many distinct reasons the rest are counted under one line.
MAX_REFUSAL_REASONS = 5
# A number inside a refusal message, so messages that differ only in their counts group together.
REFUSAL_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# The modules whose code decides what a scene pass produces. Changing any of them changes the key.
SCENE_PASS_SOURCES = (
    "scene/*.py",
    "types.py",
    "walker.py",
    "sources/framecodec.py",
    "sources/logged.py",
    "runtime/loop.py",
    "evaluation/replay.py",
)


class RecordingRefused(Exception):
    """A recording this code can't evaluate faithfully. The message says which recording and why."""


class UnalignedFrames(Enum):
    """What a scene pass does with a frame whose orientation isn't gravity aligned."""

    # The planner's figures place the walker on the floor in the world, which such a frame can't do,
    # so the whole recording is refused rather than scored on a guess.
    REFUSE = "refuse"
    # Run it through the scene as the live run did, which falls back to the picture's own up. For a
    # report that only reads the camera-frame floor, and counts these frames apart.
    PROCESS = "process"


@dataclass(frozen=True)
class PlannedScene:
    """What the scene knew about one frame it accepted, everything a planned frame needs but the arrow."""

    timestamp_seconds: float
    obstacles: ObstacleSet
    # The walker's frame in the world. origin and lateral_axis aren't scored here. They are kept so the
    # local scorecard can place consecutive plans in the world from this same pass.
    origin: np.ndarray | None
    lateral_axis: np.ndarray | None
    forward_axis: np.ndarray | None
    camera_position_world: np.ndarray | None
    camera_rotation_world: np.ndarray | None
    floor_world: Plane | None
    intrinsics: np.ndarray
    image_width_pixels: int
    image_height_pixels: int
    gaze_ground_point: np.ndarray | None
    # The floor as the scene chose it, in the camera frame. floor_world needs a position and the Neon
    # never has one, so the camera's height is read from here instead.
    floor_source: FloorSource
    camera_height_meters: float
    # Why the fit gave nothing, set only when floor_source is PREVIOUS.
    floor_refusal: FloorRefusal | None
    # False means the scene used the picture's own up for this frame, not gravity.
    gravity_aligned: bool

    def planner_input(self) -> PlannerInput:
        """The part of this frame the planner and the whole-walk numbers read."""
        return PlannerInput(
            timestamp_seconds=self.timestamp_seconds,
            obstacles=self.obstacles,
            origin=self.origin,
            lateral_axis=self.lateral_axis,
            forward_axis=self.forward_axis,
            gaze_ground_point=self.gaze_ground_point,
        )


@dataclass(frozen=True)
class ScenePass:
    pose_times_seconds: np.ndarray
    # (n, 3), a row of NaN where the pose had no position.
    pose_positions_world: np.ndarray
    planned: list[PlannedScene]
    refused_frames: int
    # Why the scene refused frames, each reason with how many it refused. A refusal is expected on a
    # frame with no floor, and the reasons are what tells that apart from a defect in the scene.
    refusal_reasons: dict[str, int]
    cache_state: str


@dataclass(frozen=True)
class SegmentTrack:
    start_seconds: float
    end_seconds: float
    track: WalkerTrack
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class PassResult:
    """What one scene pass made of a walk's arrow. Tags read the scene, so they are per pass too."""

    summary: WalkSummary
    tag_counts: dict[TurnTag, int]
    correlations: list[LaggedCorrelation]  # one per kept segment
    phone_offsets_radians: np.ndarray  # every defined sample, all kept segments
    scores: list[TurnScore]


@dataclass(frozen=True)
class WalkResult:
    name: str
    scene_source: str
    goal_mode: GoalMode
    frames: int
    planned_frames: int
    refused_frames: int
    refusal_reasons: dict[str, int]
    all_segments: list[tuple[float, float]]
    segments: list[SegmentTrack]
    spread: StraightSpread
    scored_seconds: float
    passes: list[PassResult]
    cache_states: list[str]


@dataclass(frozen=True)
class PooledSpread:
    """The straight-walking spread pooled over several walks, and whether it can set the threshold."""

    straight_seconds: float
    window_count: int
    sample_count: int
    percentiles_degrees: dict[float, float]
    change_cap_degrees: float
    # Each rule the threshold must pass, and why it failed. Empty when it passed.
    failures: list[str]
    threshold_degrees: float | None


def scene_config_for(log_dir: Path, scene_defaults: bool, scene_overrides: list[str]) -> tuple[SceneConfig, str]:
    """
    The scene configuration the recording's live run used.

    A field the recording doesn't name is a rule its live run didn't have, so it must be set on the
    command line rather than taken from today's default.

    :param log_dir: The recording.
    :param scene_defaults: Use today's defaults, for a recording made before run_config.json existed.
    :param scene_overrides: FIELD=VALUE, applied last.
    :return: The configuration, and one line saying where it came from.
    :rtype: tuple[SceneConfig, str]
    :raises RecordingRefused: On a run_config.json with an unknown scene key, a missing field nothing
        supplies, or none at all without scene_defaults, and on scene_defaults alongside one.
    """
    run_config_path = Path(log_dir) / RUN_CONFIG_FILENAME
    defaults, overridden = apply_overrides(SceneConfig(), scene_overrides)
    set_by_flag = f", set by flag: {', '.join(sorted(overridden))}" if overridden else ""
    if not run_config_path.is_file():
        if not scene_defaults:
            raise RecordingRefused(
                f"{log_dir} has no {RUN_CONFIG_FILENAME}, so its scene settings aren't known. "
                f"Pass --scene-defaults with --scene-set for each setting the live run is known to have used"
            )
        return defaults, f"today's defaults (no {RUN_CONFIG_FILENAME}){set_by_flag}"
    if scene_defaults:
        raise RecordingRefused(f"{log_dir} has a {RUN_CONFIG_FILENAME}, so --scene-defaults would contradict it")

    block = _read_run_config(run_config_path).get("scene")
    if not isinstance(block, dict):
        raise RecordingRefused(f"{run_config_path} has no scene block")
    names = {field.name for field in dataclasses.fields(SceneConfig)}
    unknown = sorted(set(block) - names)
    if unknown:
        raise RecordingRefused(f"{run_config_path} names scene settings this code doesn't have: {', '.join(unknown)}")
    missing = sorted(names - set(block) - overridden)
    if missing:
        raise RecordingRefused(
            f"{run_config_path} predates the scene settings {', '.join(missing)}. Its live run didn't have them, "
            f"so set each with --scene-set to the value that matches that run"
        )
    recorded = SceneConfig(**block)
    config, _ = apply_overrides(recorded, scene_overrides)
    return config, f"{RUN_CONFIG_FILENAME}{set_by_flag}"


def goal_mode_for(log_dir: Path) -> GoalMode:
    """
    The goal mode the live run used, ahead when the recording doesn't say.

    :raises RecordingRefused: On a goal mode this code doesn't know.
    """
    run_config_path = Path(log_dir) / RUN_CONFIG_FILENAME
    if not run_config_path.is_file():
        return GoalMode.AHEAD
    spelling = _read_run_config(run_config_path).get("goal_mode", GoalMode.AHEAD.value)
    try:
        return GoalMode(spelling)
    except ValueError:
        raise RecordingRefused(f"{run_config_path} names goal mode {spelling!r}, which this code doesn't have") from None


def scene_pass(
    log_dir: Path,
    scene_config: SceneConfig,
    walker: WalkerConfig,
    cache_dir: Path | None,
    unaligned_frames: UnalignedFrames = UnalignedFrames.REFUSE,
) -> ScenePass:
    """
    Run every frame of the log through one scene, as a live run does, or load the cached result.

    :param log_dir: The recording.
    :param scene_config: The scene's settings.
    :param walker: The walker's size.
    :param cache_dir: Where cached passes live. None runs cold and neither reads nor writes the cache.
    :param unaligned_frames: What to do with a frame whose orientation isn't gravity aligned.
    :return: The pose of every frame, and what the scene knew about each frame it accepted.
    :rtype: ScenePass
    :raises RecordingRefused: On a frame whose orientation isn't gravity aligned, under REFUSE.
    """
    log_dir = Path(log_dir)
    key = cache_key(log_dir, scene_config, walker, unaligned_frames) if cache_dir is not None else None
    cache_file = None if cache_dir is None else Path(cache_dir) / f"{key}.pkl"
    if cache_file is not None and cache_file.is_file():
        try:
            with open(cache_file, "rb") as handle:
                times, positions, rows, refused, reasons = pickle.load(handle)
            return ScenePass(times, positions, [PlannedScene(*row) for row in rows], refused, reasons, f"hit {key} in {cache_dir}")
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as unreadable:
            # A damaged or foreign entry. Rebuilt below like a miss, rather than failing every
            # cached run until somebody deletes it.
            stale_note = f", an unreadable entry ({type(unreadable).__name__}) was rebuilt"
    else:
        stale_note = ""

    scene = ScenePipeline(scene_config, walker)
    times: list[float] = []
    positions: list[np.ndarray] = []
    planned: list[PlannedScene] = []
    refused = 0
    reasons: dict[str, int] = {}
    for frame in LoggedDepthFrameSource(log_dir).frames():
        if not frame.pose.orientation_is_gravity_aligned and unaligned_frames is UnalignedFrames.REFUSE:
            raise RecordingRefused(
                f"the frame at {frame.timestamp_seconds:.3f} s in {log_dir} isn't gravity aligned, "
                f"so its position can't be put on the floor"
            )
        times.append(frame.timestamp_seconds)
        positions.append(np.asarray(frame.pose.position, dtype=np.float64) if frame.pose.has_position else np.full(3, np.nan))
        try:
            obstacles = scene.process(frame)
        except ValueError as degenerate_error:
            # A frame the scene could not use, such as one with no floor and no previous plane.
            # Expected at the start of a run and on a camera pointed at nothing. Costs this
            # frame and the replay continues, as the live worker does.
            refused += 1
            reason = refusal_reason(degenerate_error)
            if reason in reasons or len(reasons) < MAX_REFUSAL_REASONS:
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                reasons["other reasons"] = reasons.get("other reasons", 0) + 1
            continue
        origin = lateral_axis = forward_axis = camera_position = camera_rotation = floor_world = None
        if frame.pose.has_position and scene.previous_plane is not None:
            # The same construction the scene uses for the walker's axes, in the world frame.
            floor_world = camera_to_world_plane(scene.previous_plane, frame.pose)
            camera_rotation = rotation_matrix_from_quaternion_wxyz(frame.pose.orientation)
            lateral_axis, forward_axis = ground_axes(floor_world, camera_rotation @ CAMERA_FORWARD)
            camera_position = np.asarray(frame.pose.position, dtype=np.float64)
            origin = camera_position
        height, width = frame.depth_meters.shape
        planned.append(
            PlannedScene(
                timestamp_seconds=frame.timestamp_seconds,
                obstacles=obstacles,
                origin=origin,
                lateral_axis=lateral_axis,
                forward_axis=forward_axis,
                camera_position_world=camera_position,
                camera_rotation_world=camera_rotation,
                floor_world=floor_world,
                intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
                image_width_pixels=int(width),
                image_height_pixels=int(height),
                # While the scene's plane is still this frame's, as the loop computes it.
                gaze_ground_point=gaze_on_the_ground(frame, scene),
                floor_source=scene.last_floor_source,
                # process() returned, so the scene chose a plane for this frame and kept it.
                camera_height_meters=float(scene.previous_plane.offset_meters),
                floor_refusal=scene.last_floor_refusal,
                gravity_aligned=bool(frame.pose.orientation_is_gravity_aligned),
            )
        )
    time_array = np.asarray(times, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if cache_file is None:
        return ScenePass(time_array, position_array, planned, refused, reasons, "cold")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the entry and renamed into place, so an interrupted run never leaves a
    # half-written entry under the real name.
    partial = cache_file.with_name(cache_file.name + ".partial")
    with open(partial, "wb") as handle:
        # Plain tuples of plain values, nav.types instances and nav.scene.floor's refusal types, never
        # PlannedScene itself, so the file
        # loads in any process. astuple would also flatten the nav.types instances, so it isn't used.
        rows = [tuple(getattr(row, field.name) for field in dataclasses.fields(PlannedScene)) for row in planned]
        pickle.dump((time_array, position_array, rows, refused, reasons), handle)
    os.replace(partial, cache_file)
    return ScenePass(time_array, position_array, planned, refused, reasons, f"miss {key}, written to {cache_dir}{stale_note}")


def refusal_reason(error: ValueError) -> str:
    """
    The scene's refusal message with its numbers replaced by N.

    The scene says "no floor found in 35 points", and one cause with a different count each frame
    would otherwise fill the report's few reason lines and push everything else into "other".
    """
    return REFUSAL_NUMBER.sub("N", str(error)) or type(error).__name__


def cache_key(
    log_dir: Path,
    scene_config: SceneConfig,
    walker: WalkerConfig,
    unaligned_frames: UnalignedFrames = UnalignedFrames.REFUSE,
) -> str:
    """Hash of everything that decides a scene pass's output: the frames, the settings, the code and its libraries."""
    digest = hashlib.sha256()
    index_path = Path(log_dir) / INDEX_FILENAME
    digest.update(index_path.read_bytes())
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame_path = Path(log_dir) / json.loads(line)["file"]
        stat = frame_path.stat()
        digest.update(f"{frame_path.name}:{stat.st_size}:{stat.st_mtime_ns};".encode())
    digest.update(repr(scene_config).encode())
    digest.update(repr(walker).encode())
    # A pass that processed unaligned frames holds rows a refusing pass would never have, so the
    # two can't share an entry, or a cached PROCESS pass would let a REFUSE call skip its refusal.
    digest.update(unaligned_frames.value.encode())
    sources = sorted({path for pattern in SCENE_PASS_SOURCES for path in NAV_DIR.glob(pattern)}, key=lambda path: path.relative_to(NAV_DIR).as_posix())
    for source in sources:
        digest.update(source.relative_to(NAV_DIR).as_posix().encode())
        digest.update(source.read_bytes())
    digest.update(f"numpy {np.__version__}, open3d {open3d.__version__}, {CACHE_FORMAT}".encode())
    return digest.hexdigest()[:16]


def replayed_frames(
    inputs: Sequence[PlannerInput],
    planner_config: PlannerConfig,
    walker: WalkerConfig,
    goal_mode: GoalMode,
    keep_fields: bool = False,
) -> list[ReplayedFrame]:
    """
    Plan every input with one planner for the whole sequence, the way the loop does.

    This is the only place the replay plans, so the turn scores, the whole-walk numbers and the
    band breakdown can't plan differently.

    :param inputs: In timestamp order.
    :param keep_fields: Also keep each frame's field, for a breakdown by term.
    :return: Each input with its path and the alarm's decision before the hold.
    :rtype: list[ReplayedFrame]
    """
    planner = PlannerPipeline(planner_config, walker)
    frames = []
    for row in inputs:
        path = planner.plan(row.obstacles, 0.0, goal_mode, row.gaze_ground_point)
        field = planner.last_field.copy() if keep_fields else None
        frames.append(ReplayedFrame(input=row, path=path, raise_decision=alarm_raised(row.obstacles, planner_config), field=field))
    return frames


def planner_pass(scene: ScenePass, planner_config: PlannerConfig, walker: WalkerConfig, goal_mode: GoalMode) -> list[PlannedFrame]:
    """
    Plan every frame the scene accepted, with one planner for the whole log, the way the loop does.

    :return: Each frame with the planner's own arrow.
    :rtype: list[PlannedFrame]
    """
    replayed = replayed_frames([row.planner_input() for row in scene.planned], planner_config, walker, goal_mode)
    frames = []
    for row, replayed_frame in zip(scene.planned, replayed, strict=True):
        path = replayed_frame.path
        frames.append(
            PlannedFrame(
                timestamp_seconds=row.timestamp_seconds,
                arrow_radians=float(path.lookahead_heading_radians),
                forward_axis_world=row.forward_axis,
                camera_position_world=row.camera_position_world,
                camera_rotation_world=row.camera_rotation_world,
                floor_world=row.floor_world,
                intrinsics=row.intrinsics,
                image_width_pixels=row.image_width_pixels,
                image_height_pixels=row.image_height_pixels,
                obstacles=row.obstacles,
            )
        )
    return frames


def segment_bounds(timestamps: np.ndarray, config: EvaluationConfig) -> list[tuple[float, float]]:
    """Every segment of the walk, split wherever the pose stream paused longer than the segment gap."""
    times = np.asarray(timestamps, dtype=np.float64)
    if times.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) > config.segment_gap_seconds) + 1
    return [(float(chunk[0]), float(chunk[-1])) for chunk in np.split(times, breaks)]


def long_enough(segment: tuple[float, float], config: EvaluationConfig) -> bool:
    return segment[1] - segment[0] >= config.min_segment_seconds - 1e-9


def segment_tracks(scene: ScenePass, config: EvaluationConfig) -> tuple[list[tuple[float, float]], list[SegmentTrack]]:
    """The walk's segments, and the track and turns of each one long enough to score. Pose only."""
    all_segments = segment_bounds(scene.pose_times_seconds, config)
    tracks = []
    for start, end in all_segments:
        if not long_enough((start, end), config):
            continue
        inside = (scene.pose_times_seconds >= start) & (scene.pose_times_seconds <= end)
        track = walker_track(scene.pose_times_seconds[inside], scene.pose_positions_world[inside], config)
        tracks.append(SegmentTrack(start, end, track, detect_turns(track, config)))
    return all_segments, tracks


def score_pass(segments: list[SegmentTrack], frames: list[PlannedFrame], config: EvaluationConfig) -> PassResult:
    """Score one pass's arrow on every kept segment of a walk."""
    scores: list[TurnScore] = []
    sidesteps = []
    correlations = []
    offsets = []
    tag_counts = {tag: 0 for tag in TurnTag}
    arrow_seconds = 0.0
    for segment in segments:
        inside = [frame for frame in frames if segment.start_seconds <= frame.timestamp_seconds <= segment.end_seconds]
        arrow = arrow_in_travel_frame(inside, segment.track, config)
        arrow_seconds += float(np.count_nonzero(np.isfinite(arrow.travel_frame_radians))) * config.resample_step_seconds
        previous_end = None
        for turn in segment.turns:
            tag = tag_turn(turn, segment.track, inside, config)
            tag_counts[tag] += 1
            scores.append(score_turn(turn, tag, arrow, previous_end, config))
            previous_end = turn.end_seconds
        sidesteps.extend(find_sidesteps(arrow, segment.turns, config))
        correlations.append(lagged_correlation(arrow, segment.track, config))
        offsets.append(arrow.phone_offset_radians[np.isfinite(arrow.phone_offset_radians)])
    return PassResult(
        summary=summarize(scores, tuple(sidesteps), arrow_seconds),
        tag_counts=tag_counts,
        correlations=correlations,
        phone_offsets_radians=np.concatenate(offsets) if offsets else np.zeros(0),
        scores=scores,
    )


def evaluate_walk(
    log_dir: Path,
    scene_config: SceneConfig,
    scene_source: str,
    planner_config: PlannerConfig,
    config: EvaluationConfig,
    passes: int,
    cache_dir: Path | None,
) -> WalkResult:
    """
    Replay a walk and score the planner's arrow on it, over one or more scene passes.

    Turns and the straight spread read only the pose, so they come from the first pass and hold for
    all. Tags and every arrow figure are recomputed per pass.
    """
    walker = WalkerConfig()
    goal_mode = goal_mode_for(log_dir)
    pass_results = []
    cache_states = []
    first: ScenePass | None = None
    all_segments: list[tuple[float, float]] = []
    segments: list[SegmentTrack] = []
    scored_seconds = 0.0
    for _ in range(passes):
        scene = scene_pass(log_dir, scene_config, walker, cache_dir)
        cache_states.append(scene.cache_state)
        if first is None:
            first = scene
            all_segments, segments = segment_tracks(scene, config)
            defined = sum(int(np.count_nonzero(np.isfinite(segment.track.heading_radians))) for segment in segments)
            scored_seconds = defined * config.resample_step_seconds
        frames = planner_pass(scene, planner_config, walker, goal_mode)
        pass_results.append(score_pass(segments, frames, config))
    spreads = [straight_stretch_spread(segment.track, config) for segment in segments]
    return WalkResult(
        name=Path(log_dir).name,
        scene_source=scene_source,
        goal_mode=goal_mode,
        frames=int(first.pose_times_seconds.shape[0]),
        planned_frames=len(first.planned),
        refused_frames=first.refused_frames,
        refusal_reasons=first.refusal_reasons,
        all_segments=all_segments,
        segments=segments,
        spread=_pool_spreads(spreads, config),
        scored_seconds=scored_seconds,
        passes=pass_results,
        cache_states=cache_states,
    )


def straight_spread_only(log_dirs: list[Path], config: EvaluationConfig) -> PooledSpread:
    """
    Pool the straight-walking spread over several walks, reading only their poses, and apply the
    threshold rules. This is how the turn threshold is set, in a form anyone can rerun.

    :param log_dirs: The recordings.
    :param config: The straight rule, the cap and the threshold rules.
    :return: The pooled spread, which rules failed, and the threshold when none did.
    :rtype: PooledSpread
    """
    spreads = []
    for log_dir in log_dirs:
        times, positions = _read_poses(Path(log_dir))
        for start, end in segment_bounds(times, config):
            if not long_enough((start, end), config):
                continue
            inside = (times >= start) & (times <= end)
            spreads.append(straight_stretch_spread(walker_track(times[inside], positions[inside], config), config))
    pooled = _pool_spreads(spreads, config)
    samples = pooled.change_samples_degrees
    percentiles = {level: float(np.percentile(samples, level)) for level in (50.0, 95.0, config.threshold_percentile)} if samples.size else {}
    failures = []
    if pooled.straight_seconds < config.min_straight_seconds:
        failures.append(f"only {pooled.straight_seconds:.1f} s of straight walking, the rule needs {config.min_straight_seconds:.0f} s")
    if samples.size:
        measured = percentiles[config.threshold_percentile]
        allowed = config.max_spread_share_of_cap * config.straight_max_change_degrees
        if measured > allowed:
            failures.append(
                f"the {config.threshold_percentile:g}th percentile is {measured:.2f} deg, over {allowed:.2f}, "
                f"{config.max_spread_share_of_cap:g} of the {config.straight_max_change_degrees:g} deg cap, so the cap set it"
            )
        if measured <= config.arrow_dead_band_degrees:
            failures.append(f"the {config.threshold_percentile:g}th percentile is {measured:.2f} deg, inside the arrow's dead band")
    threshold = float(np.ceil(percentiles[config.threshold_percentile])) if samples.size and not failures else None
    return PooledSpread(
        straight_seconds=pooled.straight_seconds,
        window_count=pooled.window_count,
        sample_count=int(samples.size),
        percentiles_degrees=percentiles,
        change_cap_degrees=config.straight_max_change_degrees,
        failures=failures,
        threshold_degrees=threshold,
    )


def clone_state() -> str:
    """
    The commit, whether tracked files differ from it, whether nav holds untracked files, and a hash
    of every nav source file. The hash ties a number to the code that made it even before the
    branch is committed.
    """
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("status", "--porcelain", "--untracked-files=all", "--", "server/nav")
    digest = hashlib.sha256()
    for source in sorted(NAV_DIR.rglob("*.py"), key=lambda path: path.relative_to(NAV_DIR).as_posix()):
        digest.update(source.relative_to(NAV_DIR).as_posix().encode())
        digest.update(source.read_bytes())
    notes = []
    if tracked.strip():
        notes.append("uncommitted changes")
    if any(line.startswith("??") for line in untracked.splitlines()):
        notes.append("untracked files under server/nav")
    state = f" with {' and '.join(notes)}" if notes else ""
    return f"{commit}{state}, nav code {digest.hexdigest()[:12]}"


def _pool_spreads(spreads: list[StraightSpread], config: EvaluationConfig) -> StraightSpread:
    samples = [spread.change_samples_degrees for spread in spreads]
    return StraightSpread(
        straight_seconds=float(sum(spread.straight_seconds for spread in spreads)),
        window_count=int(sum(spread.window_count for spread in spreads)),
        change_samples_degrees=np.concatenate(samples) if samples else np.zeros(0),
        change_cap_degrees=config.straight_max_change_degrees,
    )


def _read_poses(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    times, positions = [], []
    for frame in LoggedDepthFrameSource(log_dir).frames():
        if not frame.pose.orientation_is_gravity_aligned:
            raise RecordingRefused(f"the frame at {frame.timestamp_seconds:.3f} s in {log_dir} isn't gravity aligned")
        times.append(frame.timestamp_seconds)
        positions.append(np.asarray(frame.pose.position, dtype=np.float64) if frame.pose.has_position else np.full(3, np.nan))
    return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64).reshape(-1, 3)


def _read_run_config(path: Path) -> dict:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as json_error:
        raise RecordingRefused(f"{path} is not valid JSON: {json_error}") from json_error
    if not isinstance(content, dict):
        raise RecordingRefused(f"{path} holds {type(content).__name__}, not an object")
    return content


def _git(*arguments: str) -> str:
    completed = subprocess.run(["git", "-C", str(CLONE_DIR), *arguments], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""
