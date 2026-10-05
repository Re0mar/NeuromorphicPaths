"""
python -m nav.evaluation.check_planner: the planner's own numbers on recorded walks.

Run from server/ with its venv:

    python -m nav.evaluation.check_planner numbers frame_logs/pixel_walk_3
    python -m nav.evaluation.check_planner numbers frame_logs/pixel_display_run --scene-defaults --scene-set floor_max_tilt_degrees=50
    python -m nav.evaluation.check_planner numbers frame_logs/pixel_walk_3 --set lateral_kinetic_weight=0.055 --set contact_term_enabled=false

    python -m nav.evaluation.check_planner flips frame_logs/contact_walk_1 --percentile 90

`numbers` prints every whole-walk figure the heading and alarm are judged by, each with the frame
count it was taken over, and a pass or fail against each target. `flips` takes the consecutive plan
pairs at or above a percentile of disagreement and says what each is: a tracker jump, the phone
turning, a new obstacle, a side flip around something both frames saw, or a shift on the same side. One scene runs across the whole
walk, as the live run does, and the chosen segment's frames are planned with one planner. The floor
fit is seeded, so one cold pass is a verdict on one machine. --cached reuses a scene pass.

Exit codes: 0 printed, 1 a recording or an override refused, 2 a usage error such as a segment the
walk doesn't have, 3 nothing to measure because the segment has no planned frame.
"""

# Standard library imports
import argparse
import dataclasses
import sys
from pathlib import Path

# Local package imports
from nav.evaluation.arguments import add_replay_arguments
from nav.evaluation.config import EvaluationConfig, PlannerNumbersConfig
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
from nav.evaluation.replay import RecordingRefused, clone_state, goal_mode_for, replayed_frames, scene_config_for, scene_pass, segment_bounds
from nav.planner.config import PlannerConfig
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
    for subcommand in (numbers, flips):
        subcommand.add_argument("log_dirs", type=Path, nargs="+", help="frame logs written by --record-to")
        subcommand.add_argument("--segment", type=int, default=-1, help="which segment to plan, counted from zero, negative from the end. The last by default")
        add_replay_arguments(subcommand)
    flips.add_argument("--percentile", type=float, default=90.0, help="pairs at or above this percentile of disagreement, 90 by default")
    flips.add_argument("--largest", type=int, default=10, help="how many of the largest pairs to list, 10 by default")
    arguments = parser.parse_args(argv)
    if arguments.subcommand == "flips" and not 0.0 <= arguments.percentile <= 100.0:
        parser.error(f"--percentile is from 0 to 100, got {arguments.percentile}")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        return _run(arguments)
    except (RecordingRefused, OverrideRefused, FileNotFoundError, FrameDecodeError) as refused:
        # A recording or an override this code can't use. Known and named, so it ends with the
        # message and no traceback. Anything else is a defect and keeps its traceback.
        print(f"{type(refused).__name__}: {refused}", file=sys.stderr)
        return EXIT_REFUSED
    except SegmentMissing as missing:
        print(f"usage: {missing}", file=sys.stderr)
        return EXIT_USAGE


def segment_frames(log_dir: Path, arguments: argparse.Namespace, planner_config: PlannerConfig) -> tuple[list[ReplayedFrame], str, str]:
    """
    Replay one segment of a walk: one scene pass over the whole log, then the segment's frames planned.

    :return: The segment's replayed frames, where the scene config came from, and how the cache was used.
    :rtype: tuple[list[ReplayedFrame], str, str]
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
    return replayed_frames(inputs, planner_config, walker, goal_mode_for(log_dir)), scene_source, passed.cache_state


def format_numbers(name: str, numbers: PlannerNumbers, planner_config: PlannerConfig, config: PlannerNumbersConfig) -> str:
    """Every figure with its frame count, then each target's verdict."""
    edges = {"close": config.close_meters, "low": config.band_low_meters, "reach": numbers.horizon_reach_meters}
    lines = [f"heading pinned within {config.pinned_tolerance_degrees} deg of the {numbers.sidestep_limit_degrees:.1f} deg sidestep limit, by the nearest heading-corridor point's clearance:"]
    for band in ClearanceBand:
        label = BAND_LABELS[band].format(**edges)
        lines.append(f"  {label:<32} pinned {_share_text(numbers.pinned_by_band[band], numbers.frames_by_band[band])}")
    lines.append(f"  {'all planned frames':<32} pinned {_share_text(numbers.pinned_frames, numbers.frames)}")
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
    for label, band, target in (
        ("clear-corridor pinned", ClearanceBand.CLEAR, config.clear_target_share),
        (f"{config.band_low_meters:.0f} to {numbers.horizon_reach_meters:.2f} m pinned", ClearanceBand.BAND, config.band_target_share),
    ):
        count = numbers.frames_by_band[band]
        measured = share(numbers.pinned_by_band[band], count, config)
        if measured is None:
            lines.append(f"  FAIL  {label}: too few frames, {count}")
            continue
        lines.append(f"  {_verdict(measured <= target)}  {label} {100.0 * measured:.1f} % of {count}, target at most {100.0 * target:.0f} %")
    lines.append(f"  {_verdict(numbers.distinct_headings > 3)}  distinct headings {numbers.distinct_headings}, target more than 3")
    lines.append(f"  {_verdict(numbers.raised_past_close == 0)}  raise decisions past {config.close_meters} m: {numbers.raised_past_close}, target 0")
    hold_met = numbers.longest_hold_seconds <= planner_config.alarm_hold_seconds + 1e-9
    lines.append(f"  {_verdict(hold_met)}  longest held {numbers.longest_hold_seconds:.3f} s, target at most {planner_config.alarm_hold_seconds} s")
    return f"=== {name}\n" + "\n".join(lines) + "\n"


def _run(arguments: argparse.Namespace) -> int:
    planner_config, overridden = apply_overrides(PlannerConfig(), arguments.planner_set)
    config = PlannerNumbersConfig()
    print(f"clone at {clone_state()}")
    for field in dataclasses.fields(PlannerConfig):
        marker = "  <- --set" if field.name in overridden else ""
        print(f"  {field.name} = {getattr(planner_config, field.name)}{marker}")
    measured_any = False
    for log_dir in arguments.log_dirs:
        frames, scene_source, cache_state = segment_frames(log_dir, arguments, planner_config)
        print(f"\n{log_dir.name}, segment {arguments.segment}: {len(frames)} planned frames. Scene settings from {scene_source}. Scene cache {cache_state}")
        if not frames:
            print(f"nothing measured: segment {arguments.segment} of {log_dir.name} has no planned frame", file=sys.stderr)
            continue
        measured_any = True
        if arguments.subcommand == "flips":
            threshold, pairs = disagreement_pairs(frames, planner_config, config, arguments.percentile)
            print(format_flips(log_dir.name, threshold, pairs, arguments.percentile, arguments.largest), end="")
        else:
            print(format_numbers(log_dir.name, whole_walk_numbers(frames, planner_config, config), planner_config, config), end="")
    return EXIT_PRINTED if measured_any else EXIT_NOTHING_MEASURED


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


def _share_text(count: int, total: int) -> str:
    if total == 0:
        return "no frames"
    return f"{100.0 * count / total:5.1f} % of {total}"


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


if __name__ == "__main__":
    sys.exit(main())
