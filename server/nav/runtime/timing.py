"""
Where each frame's time went, written per frame during a run and summarized afterwards.

One line per frame the worker took, in timing.jsonl beside the frame log. Every time is on the
laptop clock from nav.clock. The line says when the frame was captured, when it reached the
laptop, when its depth was ready, when its plan was done, and where its floor came from, so the
floor acceptance rate and every latency share can be read back from the file alone.

Nothing here knows which sensor the frames came from. A source that fills in a capture time gets
the capture shares, and one that does not gets the laptop's shares only.
"""

# Standard library imports
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.runtime.textio import append_text_lf

TIMING_FILENAME = "timing.jsonl"
MILLISECONDS_PER_SECOND = 1000.0
NO_FLOOR = "none"  # How a skipped frame's missing floor is counted in a summary.


@dataclass(frozen=True)
class TimingRecord:
    """One frame's line. Any time can be None when the frame never reached that point or never had it."""

    timestamp_seconds: float
    capture_seconds: float | None
    arrival_seconds: float | None
    depth_ready_seconds: float | None
    plan_done_seconds: float | None
    floor_source: str | None


@dataclass(frozen=True)
class ShareStatistics:
    """One latency share over the frames that had both of its ends, in milliseconds."""

    frame_count: int
    median_milliseconds: float
    percentile_95_milliseconds: float
    worst_milliseconds: float


@dataclass(frozen=True)
class TimingSummary:
    """What a run's timing log says, after the cold start is left out."""

    frames_measured: int
    frames_excluded_as_cold: int
    capture_to_arrival: ShareStatistics | None
    arrival_to_depth_ready: ShareStatistics | None
    depth_ready_to_plan_done: ShareStatistics | None
    arrival_to_plan_done: ShareStatistics | None
    capture_to_plan_done: ShareStatistics | None
    planned_frames_per_second: float | None
    longest_gap_between_plans_seconds: float | None
    floor_source_counts: dict[str, int]
    # A capture stamped after its own arrival means the clock offset is off by at least that much.
    capture_after_arrival_count: int


class TimingLog:
    """Appends one line per frame. Opens nothing until the first line, so an unused log leaves no file."""

    def __init__(self, log_dir: Path) -> None:
        self._path = Path(log_dir) / TIMING_FILENAME

    def append(self, record: TimingRecord) -> None:
        append_text_lf(self._path, json.dumps(asdict(record), allow_nan=False) + "\n")


def read_timing_log(path: Path) -> list[TimingRecord]:
    """
    Read a timing.jsonl back.

    :param path: The file, or the record directory holding it.
    :return: One record per line, in file order.
    :rtype: list[TimingRecord]
    :raises FileNotFoundError: When there is no timing log there.
    """
    path = Path(path)
    if path.is_dir():
        path = path / TIMING_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no timing log at {path}. Only a run with --record-to writes one")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(TimingRecord(**json.loads(line)))
    return records


def summarize(records: list[TimingRecord], exclude_first_seconds: float) -> TimingSummary:
    """
    Turn a run's timing lines into the figures worth writing down.

    Durations and their spread, not ratios over a handful of frames. The first stretch of the run
    is left out, measured from the first arrival, because the network, the GPU and the phone are
    all still settling then.

    :param records: The run's lines, in order.
    :param exclude_first_seconds: How much of the start to leave out.
    :return: The summary.
    :rtype: TimingSummary
    """
    starts = [record.arrival_seconds for record in records if record.arrival_seconds is not None]
    if starts:
        cold_until = min(starts) + exclude_first_seconds
        measured = [record for record in records if record.arrival_seconds is None or record.arrival_seconds >= cold_until]
    else:
        measured = list(records)

    plan_times = sorted(record.plan_done_seconds for record in measured if record.plan_done_seconds is not None)
    frames_per_second = None
    longest_gap = None
    if len(plan_times) >= 2:
        span = plan_times[-1] - plan_times[0]
        frames_per_second = (len(plan_times) - 1) / span if span > 0 else None
        longest_gap = float(np.max(np.diff(plan_times)))

    floor_counts = Counter(record.floor_source if record.floor_source is not None else NO_FLOOR for record in measured)

    return TimingSummary(
        frames_measured=len(measured),
        frames_excluded_as_cold=len(records) - len(measured),
        capture_to_arrival=_share(measured, "capture_seconds", "arrival_seconds"),
        arrival_to_depth_ready=_share(measured, "arrival_seconds", "depth_ready_seconds"),
        depth_ready_to_plan_done=_share(measured, "depth_ready_seconds", "plan_done_seconds"),
        arrival_to_plan_done=_share(measured, "arrival_seconds", "plan_done_seconds"),
        capture_to_plan_done=_share(measured, "capture_seconds", "plan_done_seconds"),
        planned_frames_per_second=frames_per_second,
        longest_gap_between_plans_seconds=longest_gap,
        floor_source_counts=dict(floor_counts),
        capture_after_arrival_count=sum(
            1
            for record in measured
            if record.capture_seconds is not None
            and record.arrival_seconds is not None
            and record.capture_seconds > record.arrival_seconds
        ),
    )


def format_summary(summary: TimingSummary) -> str:
    """The summary as the lines a person copies into the README, one figure per line."""
    lines = [
        f"frames measured            {summary.frames_measured}  ({summary.frames_excluded_as_cold} left out as the cold start)",
        f"planned frames per second  {_or_dash(summary.planned_frames_per_second, '{:.2f}')}",
        f"longest gap between plans  {_or_dash(summary.longest_gap_between_plans_seconds, '{:.3f} s')}",
        "latency, ms                 median     p95   worst   frames",
    ]
    for label, share in (
        ("capture to arrival", summary.capture_to_arrival),
        ("arrival to depth ready", summary.arrival_to_depth_ready),
        ("depth ready to plan done", summary.depth_ready_to_plan_done),
        ("arrival to plan done", summary.arrival_to_plan_done),
        ("capture to plan done", summary.capture_to_plan_done),
    ):
        if share is None:
            lines.append(f"  {label:<26} no frame had both ends")
        else:
            lines.append(
                f"  {label:<26}{share.median_milliseconds:8.1f}{share.percentile_95_milliseconds:8.1f}"
                f"{share.worst_milliseconds:8.1f}{share.frame_count:9d}"
            )
    total = sum(summary.floor_source_counts.values())
    lines.append("floor source                count   share")
    for source, count in sorted(summary.floor_source_counts.items()):
        lines.append(f"  {source:<26}{count:6d}  {100.0 * count / total:5.1f}%")
    if summary.capture_after_arrival_count:
        lines.append(
            f"WARNING {summary.capture_after_arrival_count} frames were captured after they arrived, "
            "so the clock offset is off and the capture shares are not to be trusted"
        )
    return "\n".join(lines)


def _share(records: list[TimingRecord], start_field: str, end_field: str) -> ShareStatistics | None:
    durations = [
        getattr(record, end_field) - getattr(record, start_field)
        for record in records
        if getattr(record, start_field) is not None and getattr(record, end_field) is not None
    ]
    if not durations:
        return None
    milliseconds = np.asarray(durations) * MILLISECONDS_PER_SECOND
    return ShareStatistics(
        frame_count=len(durations),
        median_milliseconds=float(np.median(milliseconds)),
        percentile_95_milliseconds=float(np.percentile(milliseconds, 95)),
        worst_milliseconds=float(np.max(milliseconds)),
    )


def _or_dash(value: float | None, template: str) -> str:
    return "-" if value is None else template.format(value)
