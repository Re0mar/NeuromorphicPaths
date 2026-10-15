"""
python -m nav.evaluation.check_planner: the planner's own numbers on recorded walks.

Run from server/ with its venv:

    python -m nav.evaluation.check_planner numbers frame_logs/pixel_walk_3
    python -m nav.evaluation.check_planner numbers frame_logs/pixel_display_run --scene-defaults --scene-set floor_max_tilt_degrees=50
    python -m nav.evaluation.check_planner numbers frame_logs/pixel_walk_3 --set lateral_kinetic_weight=0.055 --set contact_term_enabled=false

    python -m nav.evaluation.check_planner flips frame_logs/contact_walk_1 --percentile 90
    python -m nav.evaluation.check_planner band frame_logs/pixel_walk_3
    python -m nav.evaluation.check_planner floor-lean frame_logs/neon_walk_2_before --capture frame_logs/captures/neon_walk_2 --replay-shift-seconds 197905.21935606003

    python -m nav.evaluation.check_planner floor frame_logs/neon_walk_2_replay
    python -m nav.evaluation.check_planner attention frame_logs/neon_walk_2_attention --tolerances-degrees 2,4,6,8,10 --windows-seconds nearest,0.05,0.1,0.2
    python -m nav.evaluation.check_planner floor frame_logs/a_plugin_run --scene-set floor_max_offset_meters=10

`numbers` prints every whole-walk figure the heading and alarm are judged by, each with the frame
count it was taken over, and a pass or fail against each target. `flips` takes the consecutive plan
pairs at or above a percentile of disagreement and says what each is: a tracker jump, the phone
turning, a new obstacle, a side flip around something both frames saw, or a shift on the same side.
`band` takes every frame with something 3 to 5.32 m ahead and the arrow at its limit, splits its
cost by term, and says which known causes explain it, by re-planning the frame without each one.
`floor` plans nothing. It says where each frame's floor came from, how high the camera sat above it,
and why fits were refused. A median camera height near eye level is the check that a depth source
is in meters. Its second form lifts the floor check's height limit. Without it, floors too far down
are refused and show only as a median under "too far". With it, they're fitted and counted in the
camera heights, so a source whose depth is too large shows up in the median itself.
`attention` counts, over the frames the alarm was up on, how often the wearer had looked at the
thing that raised it, for each credit, tolerance and window asked for. A look is on the thing when it
falls on the part of the picture the thing covers, the thing being every piece chained to the danger
within `--link-meters` on the floor. Beside it a control table pairs each frame's
danger with gaze from a frame `--shift-seconds` away, which is what chance looks like.
`floor-lean` refits a glasses replay's own frames with the pose they were logged with and with the
pose from the capture's IMU at each frame's capture, and prints how far each floor leans from up.

One scene runs across the whole walk, as the live run does, and the chosen segment's frames are
planned with one planner. The floor fit is seeded, so one cold pass is a verdict on one machine.
--cached reuses a scene pass. `floor` takes the whole walk, and keeps frames with no gravity rather
than refusing the recording, counting them apart.

Exit codes: 0 printed, 1 a recording or an override refused, 2 a usage error such as a segment the
walk doesn't have, 3 nothing to measure because the segment has no planned frame, or for `floor`
because no walk had a fitted or supplied floor on a frame with gravity.
"""

# Standard library imports
import argparse
import dataclasses
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.arguments import add_replay_arguments, add_scene_arguments
from nav.evaluation.attention_count import (
    NEAREST,
    AttentionCount,
    AttentionSetting,
    AttentionWindow,
    GazeCarried,
    LookCounts,
    attention_count,
    check_windows,
    gaze_carried,
)
from nav.evaluation.band_attribution import FIXABLE, ON_HOLD, TERM_NAMES, BandAttribution, PinCandidate, band_attribution
from nav.evaluation.config import EvaluationConfig, PlannerNumbersConfig
from nav.evaluation.floor_lean import NothingToCompare, StampsDoNotMatch, floor_lean, format_floor_lean
from nav.evaluation.floor_report import floor_report, format_floor_report, has_heights
from nav.evaluation.overrides import OverrideRefused, apply_overrides
from nav.evaluation.planner_numbers import (
    ClearanceBand,
    DiagnosedPair,
    PairCause,
    PlannerNumbers,
    ReplayedFrame,
    disagreement_pairs,
    share,
    whole_walk_numbers,
)
from nav.evaluation.replay import (
    RecordingRefused,
    UnalignedFrames,
    clone_state,
    goal_switches_for,
    replayed_frames,
    scene_config_for,
    scene_pass,
    segment_bounds,
)
from nav.evaluation.route_replay import recorded_route_goals
from nav.evaluation.track import floor_heading_radians, walker_track, wrap_radians
from nav.planner.attention import tolerance_radians
from nav.planner.config import GoalSwitches, PlannerConfig
from nav.route.route import RouteConfig
from nav.sources.framecodec import FrameDecodeError
from nav.walker import WalkerConfig

EXIT_PRINTED = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2
EXIT_NOTHING_MEASURED = 3

# A share over fewer frames than this moves in steps too large to read as a rate.
ATTENTION_FEW_FRAMES = 30

BAND_LABELS = {
    ClearanceBand.CLOSE: "nearer than {close:.2f} m",
    ClearanceBand.NEAR: "{close:.2f} to {low:.2f} m",
    ClearanceBand.BAND: "{low:.2f} to {reach:.2f} m",
    ClearanceBand.CLEAR: "clear past {reach:.2f} m, or empty",
}


class SegmentMissing(Exception):
    """A segment index the walk doesn't have. A usage error, not a refused recording."""


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m nav.evaluation.check_planner", description="The planner's own numbers on recorded walks.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True)
    numbers = subcommands.add_parser("numbers", help="every whole-walk heading and alarm figure, with pass or fail")
    flips = subcommands.add_parser("flips", help="what the consecutive plan pairs at or above a disagreement percentile are")
    band = subcommands.add_parser("band", help="what holds the arrow at its limit with something 3 to 5.32 m ahead")
    attention = subcommands.add_parser("attention", help="how often the gaze was on the danger while the alarm was up, against chance")
    for subcommand in (numbers, flips, band, attention):
        subcommand.add_argument("log_dirs", type=Path, nargs="+", help="frame logs written by --record-to")
        subcommand.add_argument("--segment", type=int, default=-1, help="which segment to plan, counted from zero, negative from the end. The last by default")
        add_replay_arguments(subcommand)
        # The same spelling `python -m nav.evaluation` uses for its own settings. The yardstick, never
        # the planner: a full swing across a gap longer than swing_max_gap_seconds isn't counted, and
        # the glasses plan 0.5 to 1.4 s apart.
        subcommand.add_argument("--eval-set", action="append", default=[], metavar="FIELD=VALUE", help="override a PlannerNumbersConfig field")
    # The count's own sweep, never the planner's setting, so tuning the planner can't move it.
    attention.add_argument("--tolerances-degrees", type=_tolerances, default=(2.0, 4.0, 6.0, 8.0, 10.0), help="angles to count at, comma separated, 2,4,6,8,10 by default")
    attention.add_argument("--windows-seconds", type=_windows, default=_windows("nearest,0.05,0.1,0.2"), help="windows around each frame's stamp, comma separated, 'nearest' for the one nearest sample, nearest,0.05,0.1,0.2 by default")
    attention.add_argument("--credits-seconds", type=_credits, default=(0.0, 2.0), help="how long a look counts for, comma separated, 0 for the looking frame only, 0,2 by default")
    attention.add_argument("--match-meters", type=float, default=0.5, help="how near a later danger must be to where the looked-at one would be by then, 0.5 by default")
    attention.add_argument("--link-meters", type=float, default=0.4, help="how near neighboring pieces must be on the floor to be one object with the danger, 0.4 by default")
    attention.add_argument("--cap-meters", type=float, default=0.6, help="the furthest a piece may be from the danger and still be part of its object, 0.6 by default")
    attention.add_argument("--shift-seconds", type=float, default=5.0, help="how far the control's gaze is taken from, 5 by default")
    flips.add_argument("--percentile", type=float, default=90.0, help="pairs at or above this percentile of disagreement, 90 by default")
    flips.add_argument("--largest", type=int, default=10, help="how many of the largest pairs to list, 10 by default")
    # The floor is a whole-walk figure and nothing is planned, so it takes no segment and no planner override.
    floor = subcommands.add_parser("floor", help="where each frame's floor came from, and the camera's height above it")
    floor.add_argument("log_dirs", type=Path, nargs="+", help="frame logs written by --record-to")
    add_scene_arguments(floor)
    lean = subcommands.add_parser("floor-lean", help="a glasses replay's floors refit with the logged pose and with the pose at capture")
    lean.add_argument("log_dir", type=Path, help="a frame log written by --record-to during a --neon-replay run")
    lean.add_argument("--capture", type=Path, required=True, help="the capture that run played back")
    lean.add_argument(
        "--replay-shift-seconds",
        type=float,
        required=True,
        help="the shift from that run's 'stamps shifted by' log line, copied in full",
    )
    # The scene settings only. The planner, its overrides and the scene cache play no part in a refit.
    lean.add_argument("--scene-defaults", action="store_true", help="today's scene defaults, for a recording with no run_config.json")
    lean.add_argument("--scene-set", action="append", default=[], metavar="FIELD=VALUE", help="override a SceneConfig field")
    arguments = parser.parse_args(argv)
    if arguments.subcommand == "flips" and not 0.0 <= arguments.percentile <= 100.0:
        parser.error(f"--percentile is from 0 to 100, got {arguments.percentile}")
    if arguments.subcommand == "attention":
        try:
            check_windows(arguments.windows_seconds)
        except ValueError as unanswerable:
            parser.error(f"--windows-seconds: {unanswerable}")
        for degrees in arguments.tolerances_degrees:
            try:
                tolerance_radians(degrees)
            except ValueError as out_of_range:
                parser.error(f"--tolerances-degrees: {out_of_range}")
        if not np.isfinite(arguments.shift_seconds) or arguments.shift_seconds <= 0.0:
            parser.error(f"--shift-seconds is above zero, got {arguments.shift_seconds}")
        if not np.isfinite(arguments.match_meters) or arguments.match_meters <= 0.0:
            parser.error(f"--match-meters is above zero, got {arguments.match_meters}")
        if not np.isfinite(arguments.link_meters) or arguments.link_meters <= 0.0:
            parser.error(f"--link-meters is above zero, got {arguments.link_meters}")
        if not np.isfinite(arguments.cap_meters) or arguments.cap_meters <= 0.0:
            parser.error(f"--cap-meters is above zero, got {arguments.cap_meters}")
        if any(not np.isfinite(credit) or credit < 0.0 for credit in arguments.credits_seconds):
            parser.error(f"--credits-seconds are zero or more, got {arguments.credits_seconds}")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.subcommand == "floor":
            return _run_floor(arguments)
        if arguments.subcommand == "floor-lean":
            return _run_floor_lean(arguments)
        return _run(arguments)
    except (RecordingRefused, OverrideRefused, FileNotFoundError, FrameDecodeError, StampsDoNotMatch, NothingToCompare) as refused:
        # A recording or an override this code can't use. Known and named, so it ends with the
        # message and no traceback. Anything else is a defect and keeps its traceback.
        print(f"{type(refused).__name__}: {refused}", file=sys.stderr)
        return EXIT_REFUSED
    except SegmentMissing as missing:
        print(f"usage: {missing}", file=sys.stderr)
        return EXIT_USAGE


@dataclass(frozen=True)
class SegmentReplay:
    """One segment of a walk, replayed, with the poses the walker's track is built from."""

    frames: list[ReplayedFrame]
    scene_source: str
    cache_state: str
    goal_switches: GoalSwitches
    cell_size_meters: float
    pose_times_seconds: np.ndarray
    pose_positions_world: np.ndarray
    # Planned frames in the segment whose orientation wasn't gravity aligned. 0 unless they were kept.
    unaligned_frames: int = 0


def segment_frames(
    log_dir: Path,
    arguments: argparse.Namespace,
    planner_config: PlannerConfig,
    keep_fields: bool = False,
    unaligned_frames: UnalignedFrames = UnalignedFrames.REFUSE,
) -> SegmentReplay:
    """
    Replay one segment of a walk: one scene pass over the whole log, then the segment's frames planned.

    :param keep_fields: Keep each frame's field, for the band breakdown.
    :param unaligned_frames: Whether a frame without gravity refuses the walk or is kept as the live run kept it.
    :return: The segment's replayed frames and poses, where the scene config came from, and how the cache was used.
    :rtype: SegmentReplay
    :raises SegmentMissing: When the walk has no segment at that index.
    """
    walker = WalkerConfig()
    scene_config, scene_source = scene_config_for(log_dir, arguments.scene_defaults, arguments.scene_set)
    passed = scene_pass(log_dir, scene_config, walker, arguments.cache_dir if arguments.cached else None, unaligned_frames=unaligned_frames)
    bounds = segment_bounds(passed.pose_times_seconds, EvaluationConfig())
    try:
        start, end = bounds[arguments.segment]
    except IndexError:
        raise SegmentMissing(f"{log_dir.name} has {len(bounds)} segments, so there is no segment {arguments.segment}") from None
    rows = [row for row in passed.planned if start <= row.timestamp_seconds <= end]
    route_goals = recorded_route_goals(log_dir, [row.timestamp_seconds for row in rows], RouteConfig(), planner_config.goal_distance_meters)
    inputs = [dataclasses.replace(row.planner_input(), route_goal=goal) for row, goal in zip(rows, route_goals, strict=True)]
    goal_switches = goal_switches_for(log_dir)
    inside = (passed.pose_times_seconds >= start) & (passed.pose_times_seconds <= end)
    return SegmentReplay(
        frames=replayed_frames(inputs, planner_config, walker, goal_switches, keep_fields=keep_fields),
        scene_source=scene_source,
        cache_state=passed.cache_state,
        goal_switches=goal_switches,
        cell_size_meters=scene_config.cell_size_meters,
        pose_times_seconds=passed.pose_times_seconds[inside],
        pose_positions_world=passed.pose_positions_world[inside],
        unaligned_frames=sum(1 for row in rows if not row.gravity_aligned),
    )


def travel_offset_lookup(replay: SegmentReplay) -> Callable[[float], float | None]:
    """
    The walker's direction of travel to the right of the phone's forward at each frame, from the pose track.

    The track's heading and the floor heading of the phone's forward are both positive to the right, so
    the offset is their difference. Unknown where the track has no heading, which is while standing or
    near a break, or where the frame has no world pose.
    """
    config = EvaluationConfig()
    track = walker_track(replay.pose_times_seconds, replay.pose_positions_world, config)
    forward_by_time = {frame.input.timestamp_seconds: frame.input.forward_axis for frame in replay.frames}

    def offset_at(timestamp: float) -> float | None:
        forward = forward_by_time.get(timestamp)
        if forward is None or track.times_seconds.size == 0:
            return None
        index = int(np.argmin(np.abs(track.times_seconds - timestamp)))
        if abs(track.times_seconds[index] - timestamp) > config.resample_step_seconds / 2 or not np.isfinite(track.heading_radians[index]):
            return None
        return float(wrap_radians(track.heading_radians[index] - floor_heading_radians(forward)))

    return offset_at


def format_numbers(name: str, numbers: PlannerNumbers, planner_config: PlannerConfig, config: PlannerNumbersConfig) -> str:
    """Every figure with its frame count, then each target's verdict."""
    edges = {"close": config.close_meters, "low": config.band_low_meters, "reach": numbers.horizon_reach_meters}
    lines = [f"heading pinned within {config.pinned_tolerance_degrees} deg of the {numbers.sidestep_limit_degrees:.1f} deg sidestep limit, by the nearest heading-corridor point's clearance:"]
    for band in ClearanceBand:
        label = BAND_LABELS[band].format(**edges)
        lines.append(f"  {label:<32} pinned {_share_text(numbers.pinned_by_band[band], numbers.frames_by_band[band])}")
    lines.append(f"  {'all planned frames':<32} pinned {_share_text(numbers.pinned_frames, numbers.frames)}")
    restated_label = f"{config.band_low_meters:.2f} to {numbers.horizon_reach_meters:.2f} m, nothing nearer within {config.beside_meters:g} m"
    lines.append(f"restated band, {restated_label}: pinned {_share_text(numbers.restated_band_pinned, numbers.restated_band_frames)}")
    per_minute = 60.0 * numbers.full_swings / numbers.span_seconds if numbers.span_seconds > 0 else 0.0
    lines.append(f"full swings, from the limit on one side to the other between consecutive frames: {numbers.full_swings}, {per_minute:.1f} a minute")
    lines.append(f"distinct headings, to {config.distinct_heading_decimals} decimals of a degree: {numbers.distinct_headings}")
    lines.append("")
    lines.append(f"alarm on {_share_text(numbers.alarm_on_frames, numbers.frames)}, {numbers.alarm_changes} state changes")
    lines.append(f"raise decisions with the nearest alarm-corridor point past {config.close_meters} m: {numbers.raised_past_close}")
    lines.append(f"longest the alarm stayed up after its raise decision was last true: {numbers.longest_hold_seconds:.3f} s, hold {planner_config.alarm_hold_seconds} s")
    if numbers.near_noise_median_meters is not None:
        lines.append(f"noise of heading-corridor points within {config.noise_near_meters} m: median {numbers.near_noise_median_meters:.4f} m over {numbers.near_noise_readings} readings")
    lines.append("")
    if numbers.disagreement_median_meters is not None:
        lines.append(
            f"consecutive plans disagree by median {numbers.disagreement_median_meters:.3f} m, "
            f"90th percentile {numbers.disagreement_p90_meters:.3f} m, over {numbers.disagreement_pairs} pairs"
        )
    lines.append(f"  pairs without a world frame: {numbers.pairs_without_world}, pairs with an equal timestamp: {numbers.duplicate_pairs}")
    lines.append("")
    lines.append("verdicts:")
    # The band's verdict reads the restated band, chosen 2026-10-05. The plain band is printed above.
    for label, pinned, count, target in (
        ("clear-corridor pinned", numbers.pinned_by_band[ClearanceBand.CLEAR], numbers.frames_by_band[ClearanceBand.CLEAR], config.clear_target_share),
        ("restated band pinned", numbers.restated_band_pinned, numbers.restated_band_frames, config.band_target_share),
    ):
        measured = share(pinned, count, config)
        if measured is None:
            # Too few frames for a share to be a result. Not a failed target, and not a met one.
            lines.append(f"  NOT JUDGED  {label}: {count} frames, a share needs {config.min_frames_for_a_share}")
            continue
        lines.append(f"  {_verdict(measured <= target)}  {label} {100.0 * measured:.1f} % of {count}, target at most {100.0 * target:.0f} %")
    lines.append(f"  {_verdict(numbers.distinct_headings > 3)}  distinct headings {numbers.distinct_headings}, target more than 3")
    lines.append(f"  {_verdict(numbers.raised_past_close == 0)}  raise decisions past {config.close_meters} m: {numbers.raised_past_close}, target 0")
    hold_met = numbers.longest_hold_seconds <= planner_config.alarm_hold_seconds + 1e-9
    lines.append(f"  {_verdict(hold_met)}  longest held {numbers.longest_hold_seconds:.3f} s, target at most {planner_config.alarm_hold_seconds} s")
    return f"=== {name}\n" + "\n".join(lines) + "\n"


def _run_floor(arguments: argparse.Namespace) -> int:
    print(f"clone at {clone_state()}")
    measured_any = False
    for log_dir in arguments.log_dirs:
        scene_config, scene_source = scene_config_for(log_dir, arguments.scene_defaults, arguments.scene_set)
        # A frame with no gravity is kept, as the live run kept it. The report counts it apart and
        # leaves it out of the height, which reads the camera-frame floor and needs no world.
        passed = scene_pass(
            log_dir,
            scene_config,
            WalkerConfig(),
            arguments.cache_dir if arguments.cached else None,
            unaligned_frames=UnalignedFrames.PROCESS,
        )
        # Stderr, so a cached run and a cold one print the same stdout.
        print(f"{log_dir.name}: scene cache {passed.cache_state}", file=sys.stderr)
        report = floor_report(passed)
        print()
        print(format_floor_report(log_dir.name, report, scene_source), end="")
        if not has_heights(report):
            print(f"nothing measured: {log_dir.name} has no fitted or supplied floor on a frame with gravity", file=sys.stderr)
            continue
        measured_any = True
    return EXIT_PRINTED if measured_any else EXIT_NOTHING_MEASURED


def _run(arguments: argparse.Namespace) -> int:
    planner_config, overridden = apply_overrides(PlannerConfig(), arguments.planner_set)
    config, evaluation_overridden = apply_overrides(PlannerNumbersConfig(), arguments.eval_set)
    print(f"clone at {clone_state()}")
    for field in dataclasses.fields(PlannerConfig):
        marker = "  <- --set" if field.name in overridden else ""
        print(f"  {field.name} = {getattr(planner_config, field.name)}{marker}")
    # Only what was changed, so a default run's header reads as it always has.
    for name in sorted(evaluation_overridden):
        print(f"  evaluation {name} = {getattr(config, name)}  <- --eval-set")
    measured_any = False
    for log_dir in arguments.log_dirs:
        # The attention test is an angle in the camera's own frame, so it needs no gravity. A walk whose
        # IMU sent only zeros is kept, planned against the picture's own up as the live run planned it.
        unaligned = UnalignedFrames.PROCESS if arguments.subcommand == "attention" else UnalignedFrames.REFUSE
        replay = segment_frames(log_dir, arguments, planner_config, keep_fields=arguments.subcommand == "band", unaligned_frames=unaligned)
        frames, scene_source, cache_state = replay.frames, replay.scene_source, replay.cache_state
        print(f"\n{log_dir.name}, segment {arguments.segment}: {len(frames)} planned frames. Scene settings from {scene_source}. Scene cache {cache_state}")
        if not frames:
            print(f"nothing measured: segment {arguments.segment} of {log_dir.name} has no planned frame", file=sys.stderr)
            continue
        measured_any = True
        if arguments.subcommand == "band":
            attribution = band_attribution(frames, travel_offset_lookup(replay), planner_config, WalkerConfig(), replay.goal_switches, replay.cell_size_meters, config)
            print(format_band(log_dir.name, attribution, config), end="")
        elif arguments.subcommand == "attention":
            windows = _countable_windows(arguments.windows_seconds, gaze_carried(frames))
            counted = attention_count(
                frames,
                planner_config,
                arguments.credits_seconds,
                arguments.tolerances_degrees,
                windows,
                arguments.match_meters,
                arguments.link_meters,
                arguments.cap_meters,
                arguments.shift_seconds,
            )
            if replay.unaligned_frames:
                print(f"{replay.unaligned_frames} of the frames had no gravity and were planned against the picture's own up, as live")
            print(format_attention(log_dir.name, counted), end="")
        elif arguments.subcommand == "flips":
            threshold, pairs = disagreement_pairs(frames, planner_config, config, arguments.percentile)
            print(format_flips(log_dir.name, threshold, pairs, arguments.percentile, arguments.largest), end="")
        else:
            print(format_numbers(log_dir.name, whole_walk_numbers(frames, planner_config, config), planner_config, config), end="")
    return EXIT_PRINTED if measured_any else EXIT_NOTHING_MEASURED


def _run_floor_lean(arguments: argparse.Namespace) -> int:
    scene_config, scene_source = scene_config_for(arguments.log_dir, arguments.scene_defaults, arguments.scene_set)
    print(f"clone at {clone_state()}")
    print(f"{arguments.log_dir.name} against {arguments.capture.name}, shift {arguments.replay_shift_seconds!r} s. Scene settings from {scene_source}")
    result = floor_lean(arguments.log_dir, arguments.capture, arguments.replay_shift_seconds, scene_config)
    print(format_floor_lean(arguments.log_dir.name, result), end="")
    return EXIT_PRINTED


def format_flips(name: str, threshold: float | None, pairs: list[DiagnosedPair], percentile: float, largest: int) -> str:
    """How many pairs sit at or above the percentile, how many of each cause, and the largest few to open."""
    if threshold is None:
        return f"=== {name}\nno pair of consecutive plans could be placed in the world\n"
    lines = [f"the {percentile:g}th percentile of consecutive-plan disagreement is {threshold:.3f} m, and {len(pairs)} pairs sit at or above it:"]
    for cause in PairCause:
        count = sum(1 for pair in pairs if pair.cause is cause)
        lines.append(f"  {cause.value:<16} {count:>5}  {100.0 * count / len(pairs):5.1f} %")
    lines.append(f"the {min(largest, len(pairs))} largest, to open in the recording:")
    for pair in sorted(pairs, key=lambda each: each.disagreement_meters, reverse=True)[:largest]:
        point = "" if pair.deciding_point is None else f", point at {pair.deciding_point[0]:+.2f} m lateral, {pair.deciding_point[1]:.2f} m ahead"
        lines.append(
            f"  {pair.earlier_seconds:.3f} to {pair.later_seconds:.3f} s: {pair.disagreement_meters:.3f} m, {pair.cause.value}, "
            f"heading {pair.earlier_heading_degrees:+.1f} to {pair.later_heading_degrees:+.1f} deg, axis turned {pair.axis_turn_degrees:+.1f} deg{point}"
        )
    return f"=== {name}\n" + "\n".join(lines) + "\n"


def format_band(name: str, attribution: BandAttribution, config: PlannerNumbersConfig) -> str:
    """How many pinned band frames each cause explains, which are on hold, and the ceiling without the fixable ones."""
    lines = [
        f"{attribution.pinned_band_frames} of {attribution.band_frames} band frames have the arrow at its limit",
        f"band frames with any wall point: {attribution.band_frames_with_walls}, holding {attribution.band_wall_points} wall points of {attribution.band_points}",
    ]
    if attribution.pinned_band_frames:
        lines.append("each cause, and how many of those frames it explains (a frame can be explained by several):")
        for candidate in PinCandidate:
            hold = ", on hold" if candidate in ON_HOLD else (", can be fixed here" if candidate in FIXABLE else ", by design")
            unknown = attribution.unknown_by[candidate]
            unknown_text = f", {unknown} untested" if unknown else ""
            lines.append(f"  {candidate.value:<16} {attribution.explained_by[candidate]:>4}{unknown_text}{hold}")
        lines.append(f"explained only by causes on hold: {attribution.only_on_hold}")
        lines.append(f"explained by a cause that can be fixed here: {attribution.fixable}")
        lines.append(f"explained by more than one: {attribution.several}")
        lines.append(f"explained by none: {attribution.unexplained}")
        beside = sum(1 for frame in attribution.frames if frame.near_beside)
        beside_unexplained = sum(1 for frame in attribution.frames if frame.near_beside and not frame.explained and not frame.unknown)
        lines.append(
            f"with something nearer than {config.band_low_meters:g} m within {config.beside_meters:g} m of the line: {beside} of them, "
            f"{beside_unexplained} of the {attribution.unexplained} explained by none"
        )
        offsets = [abs(frame.phone_offset_degrees) for frame in attribution.frames if PinCandidate.PHONE_POINTING in frame.explained]
        if offsets:
            lines.append(f"where the phone pointing explains it, the phone was off the walking direction by median {np.median(offsets):.1f} deg, from {min(offsets):.1f} to {max(offsets):.1f}")
        if attribution.band_frames:
            ceiling = 100.0 * (attribution.pinned_band_frames - attribution.fixable) / attribution.band_frames
            lines.append(f"band share with every fixable cause removed: {ceiling:.1f} %, against {100.0 * attribution.pinned_band_frames / attribution.band_frames:.1f} % now")
        mean_terms = {name: float(np.mean([frame.term_difference[name] for frame in attribution.frames])) for name in TERM_NAMES}
        lines.append("mean cost of the chosen path minus straight on, by term, in bits: " + ", ".join(f"{name} {value:+.2f}" for name, value in mean_terms.items()))
        unexplained = [frame for frame in attribution.frames if not frame.explained and not frame.unknown][:10]
        if unexplained:
            lines.append("unexplained frames, to open in the recording: " + ", ".join(f"{frame.timestamp_seconds:.3f}" for frame in unexplained))
    return f"=== {name}\n" + "\n".join(lines) + "\n"


def format_attention(name: str, counted: AttentionCount) -> str:
    """Both tables, each cell with its numerator, its denominator and its unknowns, and what a log couldn't count."""
    lines = [
        f"{counted.frames} planned frames, alarm up on {counted.alarm_frames} with the hold, raised on {counted.raised_frames} before it",
        f"alarm frames with nothing left in the corridor, so nothing to look at: {counted.held_without_danger}",
    ]
    if counted.gaze is GazeCarried.NONE:
        lines.append("this log has no gaze at all, so nothing was counted")
        return f"=== {name}\n" + "\n".join(lines) + "\n"
    if counted.gaze is GazeCarried.NEAREST_ONLY:
        lines.append("this log was written before frames carried their gaze samples, so only the nearest sample could be counted")
    looked_back = f", {counted.control_looked_back} of them looking back because forward ran off the end" if counted.control_looked_back else ""
    if counted.control_unpaired:
        looked_back += f", {counted.control_unpaired} with no frame half the shift away, so unknown in the control"
    for credit in counted.credits_seconds:
        lines.append("")
        credit_text = "the looking frame only" if credit == 0 else f"a look counts for {credit:g} s, matched within {counted.match_meters:g} m"
        lines.append(f"gaze on the danger's object, pieces chained within {counted.link_meters:g} m and {counted.cap_meters:g} m of it, {credit_text}: noticed / alarm frames with something to go on, then frames with nothing in brackets")
        lines.extend(_attention_table(counted, counted.real, credit))
        lines.append(f"control, the gaze taken from the frame {counted.shift_seconds:g} s away{looked_back}:")
        lines.extend(_attention_table(counted, counted.control, credit))
    lines.append("")
    lines.append(f"a * marks a share over fewer than {ATTENTION_FEW_FRAMES} frames. It's a position on a coarse grid, not a rate")
    return f"=== {name}\n" + "\n".join(lines) + "\n"


def _attention_table(counted: AttentionCount, cells: dict[AttentionSetting, LookCounts], credit: float) -> list[str]:
    width = 24
    rows = ["  tolerance " + "".join(f"{window.label:<{width}}" for window in counted.windows)]
    for degrees in counted.tolerances_degrees:
        entries = [_attention_cell(cells[AttentionSetting(credit, degrees, window)]) for window in counted.windows]
        rows.append(f"  {degrees:>5g} deg " + "".join(f"{entry:<{width}}" for entry in entries))
    return rows


def _attention_cell(counts: LookCounts) -> str:
    if counts.with_a_sample == 0:
        # Never 0 %, which would read as "nobody looked" when nothing was seen at all.
        return f"no sample [{counts.unknown}]"
    marker = "*" if counts.with_a_sample < ATTENTION_FEW_FRAMES else ""
    percent = 100.0 * counts.noticed / counts.with_a_sample
    return f"{counts.noticed}/{counts.with_a_sample} {percent:.0f}%{marker} [{counts.unknown}]"


def _countable_windows(windows: tuple[AttentionWindow, ...], gaze: GazeCarried) -> tuple[AttentionWindow, ...]:
    """Every window asked for, or only the nearest sample when the log holds nothing else."""
    if gaze is GazeCarried.NEAREST_ONLY:
        return (NEAREST,)
    return windows


def _tolerances(text: str) -> tuple[float, ...]:
    try:
        # Each once, in the order given, so a repeat doesn't print a row twice.
        return tuple(dict.fromkeys(float(part) for part in text.split(",")))
    except ValueError as malformed:
        raise argparse.ArgumentTypeError(f"tolerances are numbers in degrees, comma separated, got {text!r}") from malformed


def _credits(text: str) -> tuple[float, ...]:
    try:
        return tuple(dict.fromkeys(float(part) for part in text.split(",")))
    except ValueError as malformed:
        raise argparse.ArgumentTypeError(f"credits are numbers in seconds, comma separated, got {text!r}") from malformed


def _windows(text: str) -> tuple[AttentionWindow, ...]:
    windows = []
    for part in text.split(","):
        if part.strip() == "nearest":
            windows.append(NEAREST)
            continue
        try:
            windows.append(AttentionWindow(float(part)))
        except ValueError as malformed:
            raise argparse.ArgumentTypeError(f"windows are seconds or 'nearest', comma separated, got {part!r}") from malformed
    return tuple(dict.fromkeys(windows))


def _share_text(count: int, total: int) -> str:
    if total == 0:
        return "no frames"
    return f"{100.0 * count / total:5.1f} % of {total}"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


if __name__ == "__main__":
    sys.exit(main())
