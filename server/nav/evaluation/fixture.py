"""
Golden slices: a stretch of a recorded walk as the scene handed it to the planner, small enough to commit.

A slice holds each frame's obstacle set, the walker's frame in the world and the gaze point, which is
everything the planner and the whole-walk numbers read. The golden test plans a slice and checks the
numbers, so it tests the planner alone. A scene change can't move it, and it needs no recording.

Cut a slice from a recording, from server/ with its venv:

    python -m nav.evaluation.fixture find frame_logs/pixel_walk_3
    python -m nav.evaluation.fixture cut frame_logs/pixel_walk_3 --start 1537301.2 --end 1537332.9 --out tests/fixtures/golden_pixel_walk_3.json.gz

`find` proposes the shortest window in the segment with enough band and clear-corridor frames for
both shares to be results. `cut` writes it, and refuses a slice short of either count, a fixture set
over the size cap, and rounding that would change any plan.

Exit codes: 0 done, 1 a recording, an override or a slice refused, 2 a usage error such as a window
outside the segment, 3 no window in the segment meets the frame rule.
"""

# Standard library imports
import argparse
import dataclasses
import gzip
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.arguments import add_replay_arguments
from nav.evaluation.config import EvaluationConfig, PlannerNumbersConfig
from nav.evaluation.overrides import OverrideRefused, apply_overrides
from nav.evaluation.planner_numbers import (
    ClearanceBand,
    PlannerInput,
    clearance_band,
    nearest_heading_clearance,
    plan_disagreement,
    plan_forward_distances,
)
from nav.evaluation.replay import RecordingRefused, clone_state, goal_mode_for, replayed_frames, scene_config_for, scene_pass, segment_bounds
from nav.planner.config import GoalMode, PlannerConfig
from nav.sources.framecodec import FrameDecodeError
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

FIXTURE_FORMAT = "planner-golden-1"
# Lengths, speeds and positions to 0.1 mm. That is about 300 times finer than the classroom's near
# noise of 3.4 cm, and on 2026-10-04 it changed no heading and no alarm on either golden window.
# Full precision made the two slices 2.6 MB. The cutter re-checks this on every slice it writes.
LENGTH_DECIMALS = 4
# The noise scale one place finer, because it divides the clearance in his surprise.
NOISE_DECIMALS = 5
AXIS_DECIMALS = 6
# Every golden slice in one folder together, gzipped.
MAX_FIXTURE_SET_BYTES = 1_000_000
FIXTURE_GLOB = "golden_*.json.gz"
POINT_FIELDS = ("lateral_meters", "forward_meters", "group_id", "clearance_meters", "noise_scale_meters", "is_wall", "closing_rate_mps", "velocity_mps")
FRAME_FIELDS = ("t", "groups_in_view", "origin", "lateral_axis", "forward_axis", "gaze_ground_point", "points")
# find may take a window this much longer than the shortest one when the longer window holds more of
# the segment's 90th-percentile pairs. Kept small because the shortest windows already came within
# 43 KB of the size cap, about 4 %.
WINDOW_LENGTH_SLACK = 1.04

EXIT_DONE = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2
EXIT_NO_WINDOW = 3


class FixtureRefused(Exception):
    """A slice this code won't write or read. The message names the file, and the frame and field where there is one."""


class UsageError(Exception):
    """A window or segment the walk doesn't have."""


@dataclass(frozen=True)
class GoldenSlice:
    """A slice as read back: where it was cut from, and the planner inputs in timestamp order."""

    path: Path
    provenance: dict
    inputs: tuple[PlannerInput, ...]

    def require_walker(self, walker: WalkerConfig) -> None:
        """
        Refuse a slice cut with a different walker than the one about to plan it.

        The scene computed every clearance with the walker it was given. A planner with another
        footprint would plan clearances that no longer mean what they say.

        :raises FixtureRefused: When the recorded walker differs.
        """
        recorded = self.provenance.get("walker")
        if recorded != dataclasses.asdict(walker):
            raise FixtureRefused(
                f"{self.path} was cut with walker {recorded}, and the planner now uses {dataclasses.asdict(walker)}. "
                f"Re-cut the slice from its recording"
            )


def rounded_input(row: PlannerInput) -> PlannerInput:
    """The input as a slice stores it, rounded as the module constants say."""

    def length(value: float | None) -> float | None:
        return None if value is None else round(float(value), LENGTH_DECIMALS)

    def vector(value: np.ndarray | None, decimals: int) -> np.ndarray | None:
        return None if value is None else np.round(np.asarray(value, dtype=np.float64), decimals)

    points = tuple(
        ObstaclePoint(
            lateral_meters=length(point.lateral_meters),
            forward_meters=length(point.forward_meters),
            group_id=int(point.group_id),
            clearance_meters=length(point.clearance_meters),
            noise_scale_meters=round(float(point.noise_scale_meters), NOISE_DECIMALS),
            closing_rate_mps=length(point.closing_rate_mps),
            velocity_mps=vector(point.velocity_mps, LENGTH_DECIMALS),
            is_wall=bool(point.is_wall),
            camera_point=np.zeros(3),
        )
        for point in row.obstacles.points
    )
    return PlannerInput(
        timestamp_seconds=float(row.timestamp_seconds),
        obstacles=ObstacleSet(float(row.obstacles.timestamp_seconds), points, int(row.obstacles.groups_in_view)),
        origin=vector(row.origin, LENGTH_DECIMALS),
        lateral_axis=vector(row.lateral_axis, AXIS_DECIMALS),
        forward_axis=vector(row.forward_axis, AXIS_DECIMALS),
        gaze_ground_point=vector(row.gaze_ground_point, LENGTH_DECIMALS),
    )


def write_fixture(path: Path, inputs: Sequence[PlannerInput], provenance: dict) -> int:
    """
    Write a slice, rounded, as gzipped JSON with no timestamp in the gzip header.

    :return: The bytes written.
    :rtype: int
    """
    frames = [_encode_frame(rounded_input(row)) for row in inputs]
    document = {"format": FIXTURE_FORMAT, **provenance, "frames": frames}
    # allow_nan=False, so a non-finite value is refused here rather than written as bare NaN.
    payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        # An empty filename and mtime 0 keep the header free of anything that differs between cuts.
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=9) as packed:
            packed.write(payload)
    return path.stat().st_size


def read_fixture(path: Path) -> GoldenSlice:
    """
    Read a slice back as planner inputs.

    :raises FixtureRefused: On an unknown format, a missing field, a point row of the wrong length,
        a non-finite number, a world frame with only some of its parts, or timestamps that go backwards.
    :raises FileNotFoundError: When the file is missing.
    """
    path = Path(path)
    with gzip.open(path, "rb") as packed:
        try:
            document = json.loads(packed.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile, EOFError) as unreadable:
            raise FixtureRefused(f"{path} is not a readable slice: {unreadable}") from unreadable
    if not isinstance(document, dict) or document.get("format") != FIXTURE_FORMAT:
        found = document.get("format") if isinstance(document, dict) else type(document).__name__
        raise FixtureRefused(f"{path} has format {found!r}, and this code reads {FIXTURE_FORMAT!r}")
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise FixtureRefused(f"{path} has no frames list")
    inputs = []
    previous = -math.inf
    for index, frame in enumerate(frames):
        row = _decode_frame(path, index, frame)
        if row.timestamp_seconds < previous:
            raise FixtureRefused(f"{path}, frame {index}: t goes backwards, {row.timestamp_seconds} after {previous}")
        previous = row.timestamp_seconds
        inputs.append(row)
    provenance = {key: value for key, value in document.items() if key not in ("format", "frames")}
    return GoldenSlice(path, provenance, tuple(inputs))


def frame_bands(inputs: Sequence[PlannerInput], planner_config: PlannerConfig, config: PlannerNumbersConfig) -> list[ClearanceBand]:
    """Each input's clearance band, by the nearest point in the heading's corridor."""
    return [clearance_band(nearest_heading_clearance(row.obstacles, config), planner_config, config) for row in inputs]


def shortest_windows(bands: Sequence[ClearanceBand], needed: int) -> list[tuple[int, int]]:
    """
    For each start, the shortest window from it holding at least `needed` band and `needed` clear frames.

    :return: (start, stop) index pairs, stop exclusive, one per start that has such a window.
    :rtype: list[tuple[int, int]]
    """
    band_count = np.r_[0, np.cumsum([band is ClearanceBand.BAND for band in bands])]
    clear_count = np.r_[0, np.cumsum([band is ClearanceBand.CLEAR for band in bands])]
    windows = []
    stop = 0
    for start in range(len(bands)):
        stop = max(stop, start)
        while stop < len(bands) and (band_count[stop] - band_count[start] < needed or clear_count[stop] - clear_count[start] < needed):
            stop += 1
        if band_count[stop] - band_count[start] >= needed and clear_count[stop] - clear_count[start] >= needed:
            windows.append((start, stop))
    return windows


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        return _run(arguments)
    except (RecordingRefused, OverrideRefused, FileNotFoundError, FrameDecodeError, FixtureRefused) as refused:
        # Known and named, so it ends with the message and no traceback.
        print(f"{type(refused).__name__}: {refused}", file=sys.stderr)
        return EXIT_REFUSED
    except UsageError as usage:
        print(f"usage: {usage}", file=sys.stderr)
        return EXIT_USAGE


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m nav.evaluation.fixture", description="Find and cut golden slices of recorded walks.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    find = subcommands.add_parser("find", help="propose the shortest window with enough band and clear frames")
    cut = subcommands.add_parser("cut", help="write a window as a golden slice")
    for subcommand in (find, cut):
        subcommand.add_argument("log_dir", type=Path, help="a frame log written by --record-to")
        subcommand.add_argument("--segment", type=int, default=-1, help="which segment, counted from zero, negative from the end. The last by default")
        add_replay_arguments(subcommand)
    cut.add_argument("--start", type=float, required=True, help="the first frame's timestamp, on the recording's clock")
    cut.add_argument("--end", type=float, required=True, help="the last frame's timestamp, on the recording's clock")
    cut.add_argument("--out", type=Path, required=True, help="where to write the slice, named golden_<walk>.json.gz")
    return parser.parse_args(argv)


def _run(arguments: argparse.Namespace) -> int:
    if arguments.subcommand == "cut" and not arguments.out.match(FIXTURE_GLOB):
        # The golden test and the size cap both find slices by this name, so a slice named otherwise
        # would be tested by nothing and counted against nothing.
        raise FixtureRefused(f"a golden slice is named {FIXTURE_GLOB}, so the golden test and the size cap find it, got {arguments.out.name}")
    planner_config, _ = apply_overrides(PlannerConfig(), arguments.planner_set)
    config = PlannerNumbersConfig()
    walker = WalkerConfig()
    scene_config, scene_source = scene_config_for(arguments.log_dir, arguments.scene_defaults, arguments.scene_set)
    passed = scene_pass(arguments.log_dir, scene_config, walker, arguments.cache_dir if arguments.cached else None)
    bounds = segment_bounds(passed.pose_times_seconds, EvaluationConfig())
    try:
        segment_start, segment_end = bounds[arguments.segment]
    except IndexError:
        raise UsageError(f"{arguments.log_dir.name} has {len(bounds)} segments, so there is no segment {arguments.segment}") from None
    inputs = [row.planner_input() for row in passed.planned if segment_start <= row.timestamp_seconds <= segment_end]
    goal_mode = goal_mode_for(arguments.log_dir)
    if arguments.subcommand == "find":
        return _find(arguments, inputs, planner_config, config, walker, goal_mode)

    if arguments.start > arguments.end:
        print(f"the window starts at {arguments.start} s, after it ends at {arguments.end} s", file=sys.stderr)
        return 2
    if not segment_start <= arguments.start <= arguments.end <= segment_end:
        raise UsageError(
            f"the window {arguments.start} to {arguments.end} s is not inside segment {arguments.segment} of "
            f"{arguments.log_dir.name}, {segment_start} to {segment_end} s"
        )
    window = [row for row in inputs if arguments.start <= row.timestamp_seconds <= arguments.end]
    bands = frame_bands(window, planner_config, config)
    band_frames = sum(band is ClearanceBand.BAND for band in bands)
    clear_frames = sum(band is ClearanceBand.CLEAR for band in bands)
    needed = config.min_frames_for_a_share
    if band_frames < needed or clear_frames < needed:
        raise FixtureRefused(f"the window holds {band_frames} band and {clear_frames} clear frames, and a slice needs {needed} of each")
    check_rounding(window, planner_config, walker, goal_mode)
    provenance = {
        "walk": arguments.log_dir.name,
        "segment": arguments.segment,
        "slice_seconds": [window[0].timestamp_seconds, window[-1].timestamp_seconds],
        "cut_from": {
            "clone_state": clone_state(),
            "scene_config": dataclasses.asdict(scene_config),
            "scene_config_source": scene_source,
            "goal_mode": goal_mode.value,
        },
        "walker": dataclasses.asdict(walker),
    }
    # Written beside the target first, so a cut refused for its size leaves the folder as it was,
    # including a slice the cut would have replaced.
    candidate = arguments.out.with_name(arguments.out.name + ".partial")
    written = write_fixture(candidate, window, provenance)
    total = check_fixture_set_size(candidate, arguments.out)
    os.replace(candidate, arguments.out)
    print(f"wrote {arguments.out}: {len(window)} frames, {band_frames} band, {clear_frames} clear, {written} bytes, set {total} bytes")
    return EXIT_DONE


def check_fixture_set_size(candidate: Path, target: Path) -> int:
    """
    The size the slice folder would have once the candidate replaces the target.

    Every golden slice in the target's folder counts, except the target itself, which the candidate
    replaces. The candidate counts whatever its name.

    :param candidate: The slice just written, not yet in place.
    :param target: Where it will go. It may already exist, from an earlier cut.
    :return: The total in bytes.
    :rtype: int
    :raises FixtureRefused: Over MAX_FIXTURE_SET_BYTES. The candidate is deleted and the target left as it
        was, so a refused cut changes nothing in the folder.
    """
    candidate, target = Path(candidate), Path(target)
    size = candidate.stat().st_size
    others = sum(path.stat().st_size for path in target.parent.glob(FIXTURE_GLOB) if path.resolve() != target.resolve())
    total = size + others
    if total > MAX_FIXTURE_SET_BYTES:
        candidate.unlink()
        raise FixtureRefused(
            f"the fixture set would be {total} bytes, {size} of them {target.name}, over the cap of "
            f"{MAX_FIXTURE_SET_BYTES}. Nothing in {target.parent} was changed"
        )
    return total


def _find(
    arguments: argparse.Namespace,
    inputs: list[PlannerInput],
    planner_config: PlannerConfig,
    config: PlannerNumbersConfig,
    walker: WalkerConfig,
    goal_mode: GoalMode,
) -> int:
    bands = frame_bands(inputs, planner_config, config)
    windows = shortest_windows(bands, config.min_frames_for_a_share)
    if not windows:
        print(
            f"no window in segment {arguments.segment} of {arguments.log_dir.name} holds {config.min_frames_for_a_share} band "
            f"and {config.min_frames_for_a_share} clear frames: the segment has {bands.count(ClearanceBand.BAND)} and "
            f"{bands.count(ClearanceBand.CLEAR)}",
            file=sys.stderr,
        )
        return EXIT_NO_WINDOW
    # The pairs the flip check is about, so the pinned numbers include them where the length allows.
    frames = replayed_frames(inputs, planner_config, walker, goal_mode)
    forward_distances = plan_forward_distances(planner_config)
    gaps = np.full(len(frames), np.nan)
    for index in range(1, len(frames)):
        if frames[index - 1].input.origin is not None and frames[index].input.origin is not None:
            gap = plan_disagreement(frames[index - 1], frames[index], forward_distances)
            gaps[index] = np.nan if gap is None else gap
    threshold = np.nanpercentile(gaps, 90) if np.isfinite(gaps).any() else np.inf
    high_pairs = np.r_[0, np.cumsum(np.nan_to_num(gaps, nan=-np.inf) >= threshold)]
    alarm_changes = np.r_[0, 0, np.cumsum([frames[i].path.alarm != frames[i - 1].path.alarm for i in range(1, len(frames))])]
    shortest = min(stop - start for start, stop in windows)
    allowed = [window for window in windows if window[1] - window[0] <= shortest * WINDOW_LENGTH_SLACK]
    # Pairs inside a window are counted from its second frame, so a pair straddling the start is left out.
    start, stop = max(allowed, key=lambda window: (high_pairs[window[1]] - high_pairs[window[0] + 1], -(window[1] - window[0])))
    print(f"{arguments.log_dir.name}, segment {arguments.segment}: {len(inputs)} planned frames, shortest qualifying window {shortest} frames")
    print(f"proposed window: frames {start} to {stop - 1}, {stop - start} frames, {inputs[stop - 1].timestamp_seconds - inputs[start].timestamp_seconds:.1f} s")
    print(f"  band {bands[start:stop].count(ClearanceBand.BAND)}, clear {bands[start:stop].count(ClearanceBand.CLEAR)}")
    print(f"  90th-percentile pairs {int(high_pairs[stop] - high_pairs[start + 1])} of {int(high_pairs[-1])}, alarm changes {int(alarm_changes[stop] - alarm_changes[start + 1])}")
    print(f"  --start {inputs[start].timestamp_seconds!r} --end {inputs[stop - 1].timestamp_seconds!r}")
    return EXIT_DONE


def check_rounding(window: Sequence[PlannerInput], planner_config: PlannerConfig, walker: WalkerConfig, goal_mode: GoalMode) -> None:
    """
    Refuse a slice whose rounding changes any plan, so the slice pins what the recording gives.

    A plan is its heading, its alarm and its offsets, which the plan disagreement is measured on.

    :raises FixtureRefused: Naming how many frames changed and the first few timestamps.
    """
    exact = replayed_frames(window, planner_config, walker, goal_mode)
    stored = replayed_frames([rounded_input(row) for row in window], planner_config, walker, goal_mode)
    changed = [
        f"{a.input.timestamp_seconds!r}"
        for a, b in zip(exact, stored, strict=True)
        if a.path.lookahead_heading_radians != b.path.lookahead_heading_radians
        or a.path.alarm != b.path.alarm
        or not np.array_equal(a.path.lateral_offsets_meters, b.path.lateral_offsets_meters)
    ]
    if changed:
        raise FixtureRefused(f"rounding changes the plan on {len(changed)} frames, first at t = {', '.join(changed[:5])}")


def _encode_frame(row: PlannerInput) -> dict:
    def vector(value: np.ndarray | None) -> list[float] | None:
        return None if value is None else [float(x) for x in value]

    return {
        "t": row.timestamp_seconds,
        "groups_in_view": row.obstacles.groups_in_view,
        "origin": vector(row.origin),
        "lateral_axis": vector(row.lateral_axis),
        "forward_axis": vector(row.forward_axis),
        "gaze_ground_point": vector(row.gaze_ground_point),
        "points": [
            [
                point.lateral_meters,
                point.forward_meters,
                point.group_id,
                point.clearance_meters,
                point.noise_scale_meters,
                point.is_wall,
                point.closing_rate_mps,
                vector(point.velocity_mps),
            ]
            for point in row.obstacles.points
        ],
    }


def _decode_frame(path: Path, index: int, frame: object) -> PlannerInput:
    where = f"{path}, frame {index}"
    if not isinstance(frame, dict):
        raise FixtureRefused(f"{where}: a frame is an object, got {type(frame).__name__}")
    missing = [name for name in FRAME_FIELDS if name not in frame]
    if missing:
        raise FixtureRefused(f"{where}: missing {', '.join(missing)}")
    time = _number(where, "t", frame["t"])
    world = [frame[name] for name in ("origin", "lateral_axis", "forward_axis")]
    if any(part is None for part in world) and not all(part is None for part in world):
        raise FixtureRefused(f"{where}: origin, lateral_axis and forward_axis must all be present or all null")
    origin, lateral_axis, forward_axis = (_vector(where, name, frame[name], 3) for name in ("origin", "lateral_axis", "forward_axis"))
    gaze = _vector(where, "gaze_ground_point", frame["gaze_ground_point"], 2)
    if not isinstance(frame["points"], list):
        raise FixtureRefused(f"{where}: points is a list")
    points = tuple(_decode_point(f"{where}, point {number}", row) for number, row in enumerate(frame["points"]))
    groups = frame["groups_in_view"]
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise FixtureRefused(f"{where}: groups_in_view is a whole number, got {groups!r}")
    return PlannerInput(time, ObstacleSet(time, points, groups), origin, lateral_axis, forward_axis, gaze)


def _decode_point(where: str, row: object) -> ObstaclePoint:
    if not isinstance(row, list) or len(row) != len(POINT_FIELDS):
        length = len(row) if isinstance(row, list) else type(row).__name__
        raise FixtureRefused(f"{where}: a point row has {len(POINT_FIELDS)} values ({', '.join(POINT_FIELDS)}), got {length}")
    lateral, forward, group, clearance, noise, is_wall, closing, velocity = row
    if isinstance(group, bool) or not isinstance(group, int):
        raise FixtureRefused(f"{where}: group_id is a whole number, got {group!r}")
    if not isinstance(is_wall, bool):
        raise FixtureRefused(f"{where}: is_wall is true or false, got {is_wall!r}")
    return ObstaclePoint(
        lateral_meters=_number(where, "lateral_meters", lateral),
        forward_meters=_number(where, "forward_meters", forward),
        group_id=group,
        clearance_meters=_number(where, "clearance_meters", clearance),
        noise_scale_meters=_number(where, "noise_scale_meters", noise),
        closing_rate_mps=None if closing is None else _number(where, "closing_rate_mps", closing),
        velocity_mps=_vector(where, "velocity_mps", velocity, 2),
        is_wall=is_wall,
        camera_point=np.zeros(3),
    )


def _number(where: str, name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FixtureRefused(f"{where}: {name} must be a finite number, got {value!r}")
    return float(value)


def _vector(where: str, name: str, value: object, length: int) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != length:
        raise FixtureRefused(f"{where}: {name} is a list of {length} numbers, got {value!r}")
    return np.array([_number(where, name, element) for element in value], dtype=np.float64)


if __name__ == "__main__":
    sys.exit(main())
