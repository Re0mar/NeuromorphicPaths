"""
Formats replay results as Markdown, which reads fine in a terminal and pastes into a document.

Every share is printed with its count, every unknown with its own count, and every figure that varies
between scene passes as min to max across them. A lead time never appears without the onset's known
bias beside it.
"""

# Standard library imports
import dataclasses
from collections.abc import Iterable

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.replay import PooledSpread, WalkResult
from nav.evaluation.turns import TurnTag
from nav.planner.config import PlannerConfig

# Printed only, never used to drop anything. It sizes how much the phone's own pointing can be
# lending to the agreement figures.
LARGE_PHONE_OFFSET_DEGREES = 30.0
ONSET_BIAS_NOTE = (
    "Onset is where the turn's middle, extended back at its own rate, meets the heading before it. "
    "Exact for a turn at a steady rate. A turn that speeds up gets an early onset, so its lead reads "
    "short, and one that slows down reads long."
)


def format_report(
    results: list[WalkResult],
    clone: str,
    planner_config: PlannerConfig,
    planner_overrides: frozenset[str],
    config: EvaluationConfig,
    evaluation_overrides: frozenset[str],
) -> str:
    """
    The whole report: what produced it, then one section per walk.

    :return: Markdown.
    :rtype: str
    """
    lines = ["# The planner's arrow against the walker's turns", ""]
    lines.append(f"- Code: {clone}")
    passes = len(results[0].passes) if results else 0
    lines.append(f"- Scene passes per walk: {passes}, cache: {_cache_summary(results)}")
    lines.append(f"- Planner: {_config_line(planner_config, planner_overrides)}")
    lines.append(f"- Evaluation: {_config_line(config, evaluation_overrides)}")
    lines.append("")
    for result in results:
        lines.extend(_walk_section(result, config))
    return "\n".join(lines) + "\n"


def format_spread(pooled: PooledSpread, log_names: list[str], config: EvaluationConfig) -> str:
    """The pooled straight-walking spread and the threshold it sets, or why it can't set one."""
    lines = ["# Straight-walking spread", ""]
    lines.append(f"- Walks: {', '.join(log_names)}")
    lines.append(
        f"- Straight walking: {pooled.straight_seconds:.1f} s over {pooled.window_count} windows, "
        f"{pooled.sample_count} change samples over {config.turn_window_seconds:g} s"
    )
    lines.append(f"- Change cap: {pooled.change_cap_degrees:g} deg. No sample can exceed it.")
    for level, value in sorted(pooled.percentiles_degrees.items()):
        lines.append(f"- {level:g}th percentile: {value:.2f} deg")
    if pooled.failures:
        lines.append("- Threshold: not set")
        lines.extend(f"  - {failure}" for failure in pooled.failures)
    else:
        lines.append(
            f"- Threshold: {pooled.threshold_degrees:g} deg, the {config.threshold_percentile:g}th percentile rounded up"
        )
    return "\n".join(lines) + "\n"


def _walk_section(result: WalkResult, config: EvaluationConfig) -> list[str]:
    lines = [f"## {result.name}", ""]
    lines.append(f"- Scene settings: {result.scene_source}. Goal mode: {result.goal_mode.value}")
    lines.append(f"- Frames: {result.frames}, planned {result.planned_frames}, refused by the scene {result.refused_frames}")
    kept = [segment for segment in result.segments]
    kept_lengths = ", ".join(f"{segment.end_seconds - segment.start_seconds:.1f}" for segment in kept) or "none"
    all_lengths = ", ".join(f"{end - start:.1f}" for start, end in result.all_segments) or "none"
    lines.append(f"- Segments: {len(result.all_segments)} ({all_lengths} s), {len(kept)} kept at {config.min_segment_seconds:g} s or more ({kept_lengths} s)")
    tracks = [segment.track for segment in kept]
    lines.append(
        f"- Track: {result.scored_seconds:.1f} s of walking with a known heading. "
        f"Breaks for tracker jumps {_total(track.breaks_for_jumps for track in tracks)}, "
        f"for gaps {_total(track.breaks_for_gaps for track in tracks)}, "
        f"poses without a position {_total(track.poses_without_position for track in tracks)}, "
        f"duplicate timestamps {_total(track.duplicates_dropped for track in tracks)}, "
        f"slow samples {_total(track.samples_too_slow for track in tracks)}"
    )
    spread = result.spread
    if spread.change_samples_degrees.size:
        percentiles = ", ".join(
            f"{level:g}th {np.percentile(spread.change_samples_degrees, level):.2f}" for level in (50, 95, 99)
        )
        lines.append(f"- Straight walking: {spread.straight_seconds:.1f} s, change over {config.turn_window_seconds:g} s in deg: {percentiles}, cap {spread.change_cap_degrees:g}")
    else:
        lines.append(f"- Straight walking: none found, cap {spread.change_cap_degrees:g}")
    turns = [turn for segment in kept for turn in segment.turns]
    sides = ", ".join(f"{turn.side.value} {np.degrees(turn.change_radians):+.0f}" for turn in turns) or "none"
    lines.append(f"- Turns: {len(turns)} at a {config.turn_threshold_degrees:g} deg threshold ({sides})")
    lines.append("")

    lines.append("| Tag | Turns | Arrow read | Agreed | Wrong side | Carry on | Not read | Median lead s (known, at limit) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tag in TurnTag:
        summaries = [result_pass.summary.by_tag[tag] for result_pass in result.passes]
        lead = _span([summary.median_lead_seconds for summary in summaries], "{:.2f}")
        known = _span([summary.known_leads for summary in summaries], "{}")
        at_limit = _span([summary.leads_at_limit for summary in summaries], "{}")
        lines.append(
            f"| {tag.value} | {_span([result_pass.tag_counts[tag] for result_pass in result.passes], '{}')} "
            f"| {_span([summary.arrow_read for summary in summaries], '{}')} "
            f"| {_span([summary.agreed for summary in summaries], '{}')} "
            f"| {_span([summary.wrong_side for summary in summaries], '{}')} "
            f"| {_span([summary.carry_on for summary in summaries], '{}')} "
            f"| {_span([summary.not_read for summary in summaries], '{}')} "
            f"| {lead} ({known}, {at_limit}) |"
        )
    lines.append("")
    lines.append(f"Lead times: {ONSET_BIAS_NOTE}")
    lines.append("")
    rates = [result_pass.summary.sidesteps_per_minute for result_pass in result.passes]
    counts = [result_pass.summary.sidesteps for result_pass in result.passes]
    lines.append(f"- Sidesteps while walking straight: {_span(counts, '{}')}, per minute {_span(rates, '{:.2f}')}")
    for index, segment in enumerate(kept):
        correlations = [result_pass.correlations[index] for result_pass in result.passes]
        peak_lags = [correlation.peak_lag_seconds for correlation in correlations]
        peaks = [correlation.peak_correlation for correlation in correlations]
        pairs = [int(correlation.pair_counts[0]) for correlation in correlations]
        lines.append(
            f"- Segment at {segment.start_seconds:.1f} s, arrow against turn rate: peak lag {_span(peak_lags, '{:.1f}')} s, "
            f"correlation {_span(peaks, '{:.2f}')}, over {_span(pairs, '{}')} pairs at lag 0"
        )
    offsets = [np.degrees(result_pass.phone_offsets_radians) for result_pass in result.passes]
    medians = [float(np.median(values)) if values.size else None for values in offsets]
    spreads = [float(np.subtract(*np.percentile(values, [75, 25]))) if values.size else None for values in offsets]
    large = [float(np.mean(np.abs(values) > LARGE_PHONE_OFFSET_DEGREES)) * 100.0 if values.size else None for values in offsets]
    lines.append(
        f"- Phone pointing off the direction of travel: median {_span(medians, '{:+.1f}')} deg, "
        f"interquartile range {_span(spreads, '{:.1f}')} deg, over {LARGE_PHONE_OFFSET_DEGREES:g} deg on "
        f"{_span(large, '{:.1f}')} % of samples"
    )
    lines.append("")
    return lines


def _span(values: Iterable[float | int | None], template: str) -> str:
    """Min to max across passes, one value when they agree, unknown counted rather than dropped."""
    values = list(values)
    known = [value for value in values if value is not None]
    unknown = len(values) - len(known)
    if not known:
        return "unknown"
    low, high = min(known), max(known)
    text = template.format(low) if template.format(low) == template.format(high) else f"{template.format(low)} to {template.format(high)}"
    return f"{text} ({unknown} of {len(values)} passes unknown)" if unknown else text


def _total(values: Iterable[int]) -> int:
    return int(sum(values))


def _config_line(config: object, overridden: frozenset[str]) -> str:
    parts = []
    for field in dataclasses.fields(config):
        marker = " (set by flag)" if field.name in overridden else ""
        parts.append(f"{field.name}={getattr(config, field.name)}{marker}")
    return ", ".join(parts)


def _cache_summary(results: list[WalkResult]) -> str:
    states = {state.split(" ")[0] for result in results for state in result.cache_states}
    return ", ".join(sorted(states)) or "none"
