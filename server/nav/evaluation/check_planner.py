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
    python -m nav.evaluation.check_planner floor frame_logs/a_plugin_run --scene-set floor_max_offset_meters=10

`numbers` prints every whole-walk figure the heading and alarm are judged by, each with the frame
count it was taken over, and a pass or fail against each target. `flips` takes the consecutive plan
pairs at or above a percentile of disagreement and says what each is: a tracker jump, the phone
turning, a new obstacle, a side flip around something both frames saw, or a shift on the same side.
`band` takes every frame with something 3 to 5.32 m ahead and the arrow at its limit, splits its
cost by term, and says which known causes explain it, by re-planning the frame without each one.
`floor` plans nothing. It says where each frame's floor came from, how high the camera sat above it,
and why fits were refused. A median camera height near eye level is the check that a depth source
is in meters. Its second form lifts the floor check's height limit, so a source whose depth is too
large still shows how much too large instead of being cut off at the limit.
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
from nav.evaluation.band_attribution import FIXABLE, ON_HOLD, TERM_NAMES, BandAttribution, PinCandidate, band_attribution
from nav.evaluation.config import EvaluationConfig, PlannerNumbersConfig
from nav.evaluation.floor_report import floor_report, format_floor_report, has_heights
from nav.evaluation.floor_lean import NothingToCompare, StampsDoNotMatch, floor_lean, format_floor_lean
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
    goal_mode_for,
    replayed_frames,
    scene_config_for,
    scene_pass,
    segment_bounds,
)
from nav.evaluation.track import floor_heading_radians, walker_track, wrap_radians
from nav.planner.config import GoalMode, PlannerConfig
from nav.sources.framecodec import FrameDecodeError
from nav.walker import WalkerConfig

EXIT_PRINTED = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2
EXIT_NOTHING_MEASURED = 3

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
    for subcommand in (numbers, flips, band):
        subcommand.add_argument("log_dirs", type=Path, nargs="+", help="frame logs written by --record-to")
        subcommand.add_argument("--segment", type=int, default=-1, help="which segment to plan, counted from zero, negative from the end. The last by default")
        add_replay_arguments(subcommand)
        # The same spelling `python -m nav.evaluation` uses for its own settings. The yardstick, never
        # the planner: a full swing across a gap longer than swing_max_gap_seconds isn't counted, and
        # the glasses plan 0.5 to 1.4 s apart.
        subcommand.add_argument("--eval-set", action="append", default=[], metavar="FIELD=VALUE", help="override a PlannerNumbersConfig field")
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
    goal_mode: GoalMode
    cell_size_meters: float
    pose_times_seconds: np.ndarray
    pose_positions_world: np.ndarray


def segment_frames(log_dir: Path, arguments: argparse.Namespace, planner_config: PlannerConfig, keep_fields: bool = False) -> SegmentReplay:
    """
    Replay one segment of a walk: one scene pass over the whole log, then the segment's frames planned.

    :param keep_fields: Keep each frame's field, for the band breakdown.
    :return: The segment's replayed frames and poses, where the scene config came from, and how the cache was used.
    :rtype: SegmentReplay
    :raises SegmentMissing: When the walk has no segment at that index.
    """
    walker = WalkerConfig()
    scene_config, scene_source = scene_config_for(log_dir, arguments.scene_defaults, arguments.scene_set)
    passed = scene_pass(log_dir, scene_config, walker, arguments.cache_dir if arguments.cached else None)
    bounds = segment_bounds(passed.pose_times_seconds, EvaluationConfig())
    try:
        start, end = bounds[arguments.segment]
    except IndexError:
        raise SegmentMissing(f"{log_dir.name} has {len(bounds)} segments, so there is no segment {arguments.segment}") from None
    inputs = [row.planner_input() for row in passed.planned if start <= row.timestamp_seconds <= end]
    goal_mode = goal_mode_for(log_dir)
    inside = (passed.pose_times_seconds >= start) & (passed.pose_times_seconds <= end)
    return SegmentReplay(
        frames=replayed_frames(inputs, planner_config, walker, goal_mode, keep_fields=keep_fields),
        scene_source=scene_source,
        cache_state=passed.cache_state,
        goal_mode=goal_mode,
        cell_size_meters=scene_config.cell_size_meters,
        pose_times_seconds=passed.pose_times_seconds[inside],
        pose_positions_world=passed.pose_positions_world[inside],
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
        replay = segment_frames(log_dir, arguments, planner_config, keep_fields=arguments.subcommand == "band")
        frames, scene_source, cache_state = replay.frames, replay.scene_source, replay.cache_state
        print(f"\n{log_dir.name}, segment {arguments.segment}: {len(frames)} planned frames. Scene settings from {scene_source}. Scene cache {cache_state}")
        if not frames:
            print(f"nothing measured: segment {arguments.segment} of {log_dir.name} has no planned frame", file=sys.stderr)
            continue
        measured_any = True
        if arguments.subcommand == "band":
            attribution = band_attribution(frames, travel_offset_lookup(replay), planner_config, WalkerConfig(), replay.goal_mode, replay.cell_size_meters, config)
            print(format_band(log_dir.name, attribution, config), end="")
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


def _share_text(count: int, total: int) -> str:
    if total == 0:
        return "no frames"
    return f"{100.0 * count / total:5.1f} % of {total}"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


if __name__ == "__main__":
    sys.exit(main())
