"""
Joins the Pixel's timing log with the laptop's, frame by frame, into the frame-to-arrow delay.

The phone stamps when it handled a frame, sent it, got its path back and first drew the arrow,
all on its own clock. The laptop stamps when the frame arrived and when the phone sink wrote the
path, on the laptop's clock. The two clocks never meet, and they don't need to. Each share is a
duration on one clock, and durations subtract across machines:

    total    = drawn - handled                                       phone clock
    phone    = sent - handled                                        phone clock
    laptop   = path sent - arrival                                   laptop clock
    network  = (received - sent) on the phone, minus the laptop      both, as durations
    display  = drawn - received                                      phone clock

The network figure is both hops together. Splitting them would need the clocks to agree.

The frames are matched on the ARCore frame timestamp, which the laptop hands back in every path.
The phone writes it in nanoseconds and the laptop in seconds, so the join converts with
`frame_ns_from_seconds` and never compares floats.
"""

# Standard library imports
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.runtime.textio import write_text_lf
from nav.runtime.timing import (
    LAPTOP_SHARE_SECONDS,
    PHONE_AT_BYTES,
    PHONE_COUNT,
    PHONE_DRAWN_NS,
    PHONE_FRAME_NS,
    PHONE_HANDLED_NS,
    PHONE_RECEIVED_NS,
    PHONE_SCHEMA,
    PHONE_SCHEMA_VERSION,
    PHONE_SENT_NS,
    PHONE_TYPE,
    PHONE_TYPE_DRAWN,
    PHONE_TYPE_DROPPED,
    PHONE_TYPE_FRAME,
    PHONE_TYPE_LOST,
    PHONE_TYPE_RECEIVED,
    PHONE_TYPE_SENT,
    PHONE_TYPE_SESSION,
    PHONE_TYPE_TRUNCATED,
    FrameOutcome,
    Share,
    ShareStatistics,
    TimingRecord,
    format_share_row,
    frame_ns_from_seconds,
    share_statistics,
)

NANOSECONDS_PER_SECOND = 1e9
# Fewer joined paths than this and a 95th percentile rests on five frames or fewer.
MINIMUM_JOINED_PATHS = 100
# The clock check needs this many frames before it says anything about the phone's clocks.
MINIMUM_FRAMES_FOR_CLOCK_CHECK = 100
# A handled stamp this soon after its own sensor stamp, every time, is the same clock seen twice.
# Two different bases differ by the phone's whole time asleep, hours after a few days of uptime.
# The first Pixel 8 walk, 2026-10-06, read 80.9 to 254.7 ms, median 137.1, over 5,757 frames:
# camera, ARCore and depth before the app sees the frame. 200 ms, set before any real gaps were
# seen, was under that walk's worst, and 500 ms keeps a wide margin below any sleep offset.
SAME_BASE_MAXIMUM_GAP_SECONDS = 0.5
# And steady. That walk spread 57 ms from the 5th to the 95th percentile. A clock drifting from the
# sensor's would spread further and further over a walk.
SAME_BASE_MAXIMUM_SPREAD_SECONDS = 0.1
# ARCore stamps a frame 0 before it has a clock reading for it, at the start of a session. That
# frame says nothing about the clocks, so it is counted and left out of the check.
ARCORE_NO_TIMESTAMP_NS = 0
NEGATIVE_EXAMPLES_SHOWN = 5
# The network share builds up and drains over seconds, which one median over a walk hides. Ten
# seconds is long enough for a few hundred paths and short enough to show a queue filling.
NETWORK_SLICE_SECONDS = 10.0
MINIMUM_RUNS_TO_COMPARE = 3
# The order shares are printed and saved in. Every key of Share appears once.
SHARE_ORDER = (
    Share.TOTAL,
    Share.PHONE,
    Share.NETWORK,
    Share.LAPTOP,
    Share.LAPTOP_QUEUE_WAIT,
    Share.LAPTOP_PROCESSING,
    Share.LAPTOP_PUBLISH_WAIT,
    Share.DISPLAY,
    Share.SENSOR_TO_HANDLED,
    Share.TOTAL_FROM_SENSOR,
)
SHARE_LABELS = {
    Share.TOTAL: "total, handled to drawn",
    Share.PHONE: "phone, handled to sent",
    Share.NETWORK: "network, both hops",
    Share.LAPTOP: "laptop, arrival to sent",
    Share.LAPTOP_QUEUE_WAIT: "  queue wait",
    Share.LAPTOP_PROCESSING: "  processing",
    Share.LAPTOP_PUBLISH_WAIT: "  publish wait",
    Share.DISPLAY: "display, received to drawn",
    Share.SENSOR_TO_HANDLED: "sensor to handled",
    Share.TOTAL_FROM_SENSOR: "total, sensor to drawn",
}


class ClockVerdict(Enum):
    """Whether ARCore's frame timestamps and the phone's elapsed-realtime clock count from one base."""

    SAME_BASE = "same base"
    DIFFERENT_BASE = "different base"
    NOT_ENOUGH_FRAMES = "not enough frames"


class Orphan(Enum):
    """A frame seen on one side, or at one stage, without what should follow it."""

    HANDLED_NEVER_SENT = "phone handled, never sent or dropped"
    SENT_NO_LAPTOP_LINE = "phone sent, no laptop line"
    PUBLISHED_NEVER_RECEIVED = "laptop sent to the phone, phone never received"
    RECEIVED_NO_LAPTOP_LINE = "phone received, no laptop line"
    RECEIVED_NEVER_DRAWN = "phone received, never drawn"
    DRAWN_NEVER_RECEIVED = "phone drawn, never received"


class Count(Enum):
    """What the report counts on the way to the join. The values are the printed and saved names."""

    PHONE_FRAMES_HANDLED = "phone frames handled"
    PHONE_FRAMES_WITH_NO_ARCORE_STAMP = "phone frames with no ARCore stamp"
    PHONE_FRAMES_SENT = "phone frames sent"
    PHONE_FRAMES_DROPPED = "phone frames dropped"
    LAPTOP_FRAMES_RECEIVED = "laptop frames received"
    LAPTOP_FRAMES_PUBLISHED = "laptop frames published"
    LAPTOP_FRAMES_SUPERSEDED = "laptop frames superseded"
    LAPTOP_FRAMES_DROPPED = "laptop frames dropped"
    LAPTOP_FRAMES_SKIPPED = "laptop frames skipped"
    LAPTOP_FRAMES_IN_FLIGHT = "laptop frames in flight"
    PATHS_RECEIVED = "paths received"
    PATHS_DRAWN = "paths drawn"
    PHONE_RECORDS_LOST = "phone records lost"
    DUPLICATE_HANDLED_LINES = "duplicate handled lines"
    DUPLICATE_SENT_LINES = "duplicate sent lines"
    DUPLICATE_RECEIVED_LINES = "duplicate received lines"
    DUPLICATE_DRAWN_LINES = "duplicate drawn lines"
    DUPLICATE_LAPTOP_LINES = "duplicate laptop lines"


# Each laptop outcome's count. Every outcome has one, and a test holds that.
LAPTOP_OUTCOME_COUNTS: dict[FrameOutcome, Count] = {
    FrameOutcome.PUBLISHED: Count.LAPTOP_FRAMES_PUBLISHED,
    FrameOutcome.SUPERSEDED: Count.LAPTOP_FRAMES_SUPERSEDED,
    FrameOutcome.DROPPED: Count.LAPTOP_FRAMES_DROPPED,
    FrameOutcome.SKIPPED: Count.LAPTOP_FRAMES_SKIPPED,
    FrameOutcome.IN_FLIGHT: Count.LAPTOP_FRAMES_IN_FLIGHT,
}


@dataclass
class PhoneFrame:
    """Everything the phone logged about one frame. Each stamp is None until its line is read."""

    handled_ns: int | None = None
    sent_ns: int | None = None
    dropped: bool = False
    received_ns: int | None = None
    drawn_ns: int | None = None


@dataclass
class PhoneLog:
    """A phone timing log, read and checked."""

    device: str
    build_type: str
    started_wall: str
    frames: dict[int, PhoneFrame] = field(default_factory=dict)
    lost_records: int = 0
    truncated_at_bytes: int | None = None
    duplicate_handled: int = 0
    duplicate_sent: int = 0
    duplicate_received: int = 0
    duplicate_drawn: int = 0


@dataclass(frozen=True)
class NetworkSlice:
    """The network share's median over one stretch of a walk."""

    start_seconds: float  # From the first joined path's handling on the phone.
    paths: int
    median_milliseconds: float


@dataclass(frozen=True)
class JoinedReport:
    """The frame-to-arrow delay of one run, with every count it rests on."""

    shares: dict[Share, ShareStatistics | None]
    joined_paths: int
    counts: dict[Count, int]
    orphans: dict[Orphan, int]
    negative_examples: dict[Share, list[float]]
    clock_verdict: ClockVerdict
    clock_reason: str
    excluded_as_cold: int
    phone_log_truncated: bool
    network_by_slice: list[NetworkSlice] = field(default_factory=list)
    # How many of each share came out negative and were left out. Only the first few are examples.
    negative_counts: dict[Share, int] = field(default_factory=dict)


# *******************************************
# Reading the phone log
# *******************************************


def read_phone_log(path: Path) -> PhoneLog:
    """
    Read and check a phone timing log, pulled off the Pixel with `adb pull`.

    :param path: The `timing_<start>.jsonl` file.
    :return: The log, per frame.
    :rtype: PhoneLog
    :raises FileNotFoundError: When there is no file there.
    :raises ValueError: When the log has no session line, a schema this reader does not know, or a
        line no phone writes. The message names the file and the line.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no phone timing log at {path}. Pull it with adb pull, see pixel_app/README.md")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()]
    phone_log: PhoneLog | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path} line {line_number}"
        raw = _json_object(line, where)
        kind = _string(raw, PHONE_TYPE, where)
        if phone_log is None:
            phone_log = _session(raw, kind, where)
            continue
        if phone_log.truncated_at_bytes is not None:
            raise ValueError(f"{where} follows the truncated line, and a phone writes nothing after it")
        _read_phone_line(phone_log, raw, kind, where)
    if phone_log is None:
        raise ValueError(f"{path} is empty, and a phone log starts with its session line")
    return phone_log


def _session(raw: dict, kind: str, where: str) -> PhoneLog:
    if kind != PHONE_TYPE_SESSION:
        raise ValueError(f"{where} is a {kind!r} line, and a phone log starts with its session line")
    schema = _integer(raw, PHONE_SCHEMA, where)
    if schema != PHONE_SCHEMA_VERSION:
        raise ValueError(f"{where} has schema {schema}, and this reader knows schema {PHONE_SCHEMA_VERSION}")
    return PhoneLog(
        device=_string(raw, "device", where),
        build_type=_string(raw, "build_type", where),
        started_wall=_string(raw, "started_wall", where),
    )


def _read_phone_line(phone_log: PhoneLog, raw: dict, kind: str, where: str) -> None:
    if kind == PHONE_TYPE_LOST:
        phone_log.lost_records += _integer(raw, PHONE_COUNT, where)
        return
    if kind == PHONE_TYPE_TRUNCATED:
        phone_log.truncated_at_bytes = _integer(raw, PHONE_AT_BYTES, where)
        return
    if kind == PHONE_TYPE_SESSION:
        raise ValueError(f"{where} is a second session line, and one log is one session")
    frame = phone_log.frames.setdefault(_integer(raw, PHONE_FRAME_NS, where), PhoneFrame())
    # A phone that reconnects can hand back an older clock, and a frame timestamp can repeat. Every
    # line type keeps its first value and counts the repeat, so one frame's stamps never mix with
    # another's, and one stray line doesn't refuse a whole walk.
    if kind == PHONE_TYPE_FRAME:
        handled_ns = _integer(raw, PHONE_HANDLED_NS, where)
        if frame.handled_ns is None:
            frame.handled_ns = handled_ns
        else:
            phone_log.duplicate_handled += 1
    elif kind == PHONE_TYPE_SENT:
        sent_ns = _integer(raw, PHONE_SENT_NS, where)
        if frame.sent_ns is None:
            frame.sent_ns = sent_ns
        else:
            phone_log.duplicate_sent += 1
    elif kind == PHONE_TYPE_DROPPED:
        frame.dropped = True
    elif kind == PHONE_TYPE_RECEIVED:
        received_ns = _integer(raw, PHONE_RECEIVED_NS, where)
        if frame.received_ns is None:
            frame.received_ns = received_ns
        else:
            phone_log.duplicate_received += 1
    elif kind == PHONE_TYPE_DRAWN:
        drawn_ns = _integer(raw, PHONE_DRAWN_NS, where)
        if frame.drawn_ns is None:
            frame.drawn_ns = drawn_ns
        else:
            phone_log.duplicate_drawn += 1
    else:
        raise ValueError(f"{where} has type {kind!r}, which no phone writes")


def _json_object(line: str, where: str) -> dict:
    try:
        raw = json.loads(line)
    except ValueError as json_error:
        # A phone killed mid-write can leave a truncated last line.
        raise ValueError(f"{where} is not valid JSON: {json_error}") from json_error
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a JSON object, got {type(raw).__name__}")
    return raw


def _integer(raw: dict, key: str, where: str) -> int:
    if key not in raw:
        raise ValueError(f"{where} is missing {key}")
    value = raw[key]
    # bool is an int to Python, and a true here is a damaged line, not one nanosecond.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} has {key} {value!r}, which is not an integer")
    return value


def _string(raw: dict, key: str, where: str) -> str:
    if key not in raw:
        raise ValueError(f"{where} is missing {key}")
    value = raw[key]
    if not isinstance(value, str):
        raise ValueError(f"{where} has {key} {value!r}, which is not a string")
    return value


# *******************************************
# Joining
# *******************************************


def join(phone_log: PhoneLog, laptop_records: list[TimingRecord], exclude_first_seconds: float) -> JoinedReport:
    """
    The frame-to-arrow delay of one run, from the two logs.

    The first stretch of the run is left out of both sides, measured from the first frame's
    arrival on the laptop, or for a phone frame the laptop never saw, from the first frame the
    phone handled. The network and the phone are still settling then.

    :param phone_log: The phone's log.
    :param laptop_records: The laptop's lines.
    :param exclude_first_seconds: How much of the start to leave out.
    :return: The report.
    :rtype: JoinedReport
    """
    laptop: dict[int, TimingRecord] = {}
    duplicate_laptop_lines = 0
    for record in laptop_records:
        # The first line for a key is kept and a repeat counted, as on the phone side.
        frame_ns = frame_ns_from_seconds(record.timestamp_seconds)
        if frame_ns in laptop:
            duplicate_laptop_lines += 1
        else:
            laptop[frame_ns] = record
    cold = _cold_frames(phone_log, laptop, exclude_first_seconds)
    # ARCore's 0 stamp is the same key for every frame that has it, so its lines can belong to
    # different frames. It's counted, and joined with nothing.
    excluded = cold | {ARCORE_NO_TIMESTAMP_NS}
    phone_frames = {frame_ns: frame for frame_ns, frame in phone_log.frames.items() if frame_ns not in excluded}
    laptop = {frame_ns: record for frame_ns, record in laptop.items() if frame_ns not in excluded}

    verdict, reason = _clock_check(phone_log.frames)
    durations: dict[Share, list[float]] = {share: [] for share in Share}
    negatives: dict[Share, list[float]] = {share: [] for share in Share}
    network_when: list[tuple[int, float]] = []
    joined_paths = 0
    for frame_ns, frame in phone_frames.items():
        record = laptop.get(frame_ns)
        if record is None or frame.received_ns is None or frame.drawn_ns is None or frame.handled_ns is None or frame.sent_ns is None:
            continue
        laptop_seconds = LAPTOP_SHARE_SECONDS[Share.LAPTOP](record)
        if laptop_seconds is None:
            continue
        joined_paths += 1
        per_frame = {
            Share.TOTAL: (frame.drawn_ns - frame.handled_ns) / NANOSECONDS_PER_SECOND,
            Share.PHONE: (frame.sent_ns - frame.handled_ns) / NANOSECONDS_PER_SECOND,
            Share.NETWORK: (frame.received_ns - frame.sent_ns) / NANOSECONDS_PER_SECOND - laptop_seconds,
            Share.DISPLAY: (frame.drawn_ns - frame.received_ns) / NANOSECONDS_PER_SECOND,
            **{share: seconds_of(record) for share, seconds_of in LAPTOP_SHARE_SECONDS.items()},
        }
        if verdict is ClockVerdict.SAME_BASE:
            per_frame[Share.SENSOR_TO_HANDLED] = (frame.handled_ns - frame_ns) / NANOSECONDS_PER_SECOND
            per_frame[Share.TOTAL_FROM_SENSOR] = (frame.drawn_ns - frame_ns) / NANOSECONDS_PER_SECOND
        for share, seconds in per_frame.items():
            if seconds is None:
                continue
            # A negative duration is a stamp taken in the wrong place, not a fast frame.
            (negatives if seconds < 0 else durations)[share].append(seconds)
        if per_frame[Share.NETWORK] >= 0:
            network_when.append((frame.handled_ns, per_frame[Share.NETWORK]))

    return JoinedReport(
        shares={share: share_statistics(durations[share]) for share in Share},
        joined_paths=joined_paths,
        counts=_counts(phone_log, phone_frames, laptop, duplicate_laptop_lines),
        orphans=_orphans(phone_frames, laptop),
        negative_examples={share: values[:NEGATIVE_EXAMPLES_SHOWN] for share, values in negatives.items() if values},
        clock_verdict=verdict,
        clock_reason=reason,
        excluded_as_cold=len(cold),
        phone_log_truncated=phone_log.truncated_at_bytes is not None,
        network_by_slice=_network_by_slice(network_when),
        negative_counts={share: len(values) for share, values in negatives.items() if values},
    )


def _network_by_slice(network_when: list[tuple[int, float]]) -> list[NetworkSlice]:
    """
    The network share's median per stretch of the walk, in order. Stretches with no path are left out.

    :param network_when: (phone handled stamp in ns, network share in seconds) per joined path.
    """
    if not network_when:
        return []
    first_ns = min(handled_ns for handled_ns, _ in network_when)
    by_slice: dict[int, list[float]] = {}
    for handled_ns, seconds in network_when:
        index = int((handled_ns - first_ns) / NANOSECONDS_PER_SECOND // NETWORK_SLICE_SECONDS)
        by_slice.setdefault(index, []).append(seconds)
    return [
        NetworkSlice(
            start_seconds=index * NETWORK_SLICE_SECONDS,
            paths=len(values),
            median_milliseconds=float(np.median(values)) * 1000,
        )
        for index, values in sorted(by_slice.items())
    ]


def _cold_frames(phone_log: PhoneLog, laptop: dict[int, TimingRecord], exclude_first_seconds: float) -> set[int]:
    excluded: set[int] = set()
    arrivals = [record.arrival_seconds for record in laptop.values() if record.arrival_seconds is not None]
    if arrivals:
        cold_until = min(arrivals) + exclude_first_seconds
        excluded.update(frame_ns for frame_ns, record in laptop.items() if record.arrival_seconds is not None and record.arrival_seconds < cold_until)
    handled = [frame.handled_ns for frame in phone_log.frames.values() if frame.handled_ns is not None]
    if handled:
        phone_cold_until = min(handled) + exclude_first_seconds * NANOSECONDS_PER_SECOND
        excluded.update(
            frame_ns
            for frame_ns, frame in phone_log.frames.items()
            if frame_ns not in laptop and frame.handled_ns is not None and frame.handled_ns < phone_cold_until
        )
    return excluded


def _clock_check(frames: dict[int, PhoneFrame]) -> tuple[ClockVerdict, str]:
    gaps = [
        (frame.handled_ns - frame_ns) / NANOSECONDS_PER_SECOND
        for frame_ns, frame in frames.items()
        if frame.handled_ns is not None and frame_ns != ARCORE_NO_TIMESTAMP_NS
    ]
    if len(gaps) < MINIMUM_FRAMES_FOR_CLOCK_CHECK:
        return ClockVerdict.NOT_ENOUGH_FRAMES, f"{len(gaps)} handled frames, {MINIMUM_FRAMES_FOR_CLOCK_CHECK} needed"
    smallest, largest = min(gaps), max(gaps)
    if smallest < 0 or largest > SAME_BASE_MAXIMUM_GAP_SECONDS:
        return (
            ClockVerdict.DIFFERENT_BASE,
            f"handled minus sensor runs {smallest * 1000:.1f} to {largest * 1000:.1f} ms, outside 0 to {SAME_BASE_MAXIMUM_GAP_SECONDS * 1000:.0f} ms",
        )
    spread = float(np.percentile(gaps, 95) - np.percentile(gaps, 5))
    if spread > SAME_BASE_MAXIMUM_SPREAD_SECONDS:
        return (
            ClockVerdict.DIFFERENT_BASE,
            f"handled minus sensor spreads {spread * 1000:.1f} ms from the 5th to the 95th percentile, over {SAME_BASE_MAXIMUM_SPREAD_SECONDS * 1000:.0f} ms",
        )
    return ClockVerdict.SAME_BASE, f"handled minus sensor {smallest * 1000:.1f} to {largest * 1000:.1f} ms over {len(gaps)} frames"


def _counts(
    phone_log: PhoneLog,
    phone_frames: dict[int, PhoneFrame],
    laptop: dict[int, TimingRecord],
    duplicate_laptop_lines: int,
) -> dict[Count, int]:
    outcomes = {LAPTOP_OUTCOME_COUNTS[outcome]: sum(record.outcome is outcome for record in laptop.values()) for outcome in FrameOutcome}
    return {
        Count.PHONE_FRAMES_HANDLED: sum(frame.handled_ns is not None for frame in phone_frames.values()),
        # From the whole log, since the join leaves these frames out.
        Count.PHONE_FRAMES_WITH_NO_ARCORE_STAMP: int(ARCORE_NO_TIMESTAMP_NS in phone_log.frames),
        Count.PHONE_FRAMES_SENT: sum(frame.sent_ns is not None for frame in phone_frames.values()),
        Count.PHONE_FRAMES_DROPPED: sum(frame.dropped for frame in phone_frames.values()),
        Count.LAPTOP_FRAMES_RECEIVED: len(laptop),
        **outcomes,
        Count.PATHS_RECEIVED: sum(frame.received_ns is not None for frame in phone_frames.values()),
        Count.PATHS_DRAWN: sum(frame.drawn_ns is not None for frame in phone_frames.values()),
        Count.PHONE_RECORDS_LOST: phone_log.lost_records,
        Count.DUPLICATE_HANDLED_LINES: phone_log.duplicate_handled,
        Count.DUPLICATE_SENT_LINES: phone_log.duplicate_sent,
        Count.DUPLICATE_RECEIVED_LINES: phone_log.duplicate_received,
        Count.DUPLICATE_DRAWN_LINES: phone_log.duplicate_drawn,
        Count.DUPLICATE_LAPTOP_LINES: duplicate_laptop_lines,
    }


def _orphans(phone_frames: dict[int, PhoneFrame], laptop: dict[int, TimingRecord]) -> dict[Orphan, int]:
    sent_to_phone = {frame_ns for frame_ns, record in laptop.items() if record.outcome is FrameOutcome.PUBLISHED and record.sent_seconds is not None}
    return {
        Orphan.HANDLED_NEVER_SENT: sum(
            frame.handled_ns is not None and frame.sent_ns is None and not frame.dropped for frame in phone_frames.values()
        ),
        Orphan.SENT_NO_LAPTOP_LINE: sum(frame.sent_ns is not None and frame_ns not in laptop for frame_ns, frame in phone_frames.items()),
        Orphan.PUBLISHED_NEVER_RECEIVED: sum(
            frame_ns not in phone_frames or phone_frames[frame_ns].received_ns is None for frame_ns in sent_to_phone
        ),
        Orphan.RECEIVED_NO_LAPTOP_LINE: sum(
            frame.received_ns is not None and frame_ns not in laptop for frame_ns, frame in phone_frames.items()
        ),
        Orphan.RECEIVED_NEVER_DRAWN: sum(frame.received_ns is not None and frame.drawn_ns is None for frame in phone_frames.values()),
        Orphan.DRAWN_NEVER_RECEIVED: sum(frame.drawn_ns is not None and frame.received_ns is None for frame in phone_frames.values()),
    }


# *******************************************
# Output
# *******************************************


def format_joined(report: JoinedReport) -> str:
    """The report as text: a warning if thin, the shares, the clock verdict, then every count."""
    lines = []
    if report.joined_paths < MINIMUM_JOINED_PATHS:
        lines.append(f"WARNING only {report.joined_paths} joined paths, under {MINIMUM_JOINED_PATHS}. These figures are too thin to report")
    if report.phone_log_truncated:
        lines.append("WARNING the phone log hit its size cap, so the end of the walk is missing from it")
    lines.append(f"joined paths               {report.joined_paths}  ({report.excluded_as_cold} frames left out as the cold start)")
    lines.append("frame to arrow, ms           median     p95   worst   frames   (percentiles linear between ranks)")
    for share in SHARE_ORDER:
        if share in (Share.SENSOR_TO_HANDLED, Share.TOTAL_FROM_SENSOR) and report.clock_verdict is not ClockVerdict.SAME_BASE:
            continue
        lines.append(format_share_row(SHARE_LABELS[share], report.shares[share]))
    lines.append(f"phone clocks               {report.clock_verdict.value}: {report.clock_reason}")
    if report.clock_verdict is not ClockVerdict.SAME_BASE:
        lines.append("                           so the totals start at the phone handling the frame, not at the sensor")
    for share, values in report.negative_examples.items():
        shown = ", ".join(f"{value * 1000:.1f}" for value in values)
        count = report.negative_counts.get(share, len(values))
        lines.append(f"WARNING {SHARE_LABELS[share].strip()} came out negative {count} times, left out. First: {shown} ms")
    if report.network_by_slice:
        lines.append(f"network median per {NETWORK_SLICE_SECONDS:.0f} s of the walk, ms (paths)")
        lines.append(
            "  "
            + ", ".join(f"{piece.start_seconds:.0f} s: {piece.median_milliseconds:.0f} ({piece.paths})" for piece in report.network_by_slice)
        )
    lines.append("counts")
    lines.extend(f"  {name.value:<34}{count:8d}" for name, count in report.counts.items())
    lines.append("not joined")
    lines.extend(f"  {orphan.value:<46}{count:8d}" for orphan, count in report.orphans.items())
    return "\n".join(lines)


def report_as_json(shares: dict[Share, ShareStatistics | None], extra: dict) -> dict:
    """What `--json` saves, and what `--compare` reads back. Share keys are `Share` values."""
    return {
        "shares": {
            share.value: None if statistics is None else {
                "frame_count": statistics.frame_count,
                "median_milliseconds": statistics.median_milliseconds,
                "percentile_95_milliseconds": statistics.percentile_95_milliseconds,
                "worst_milliseconds": statistics.worst_milliseconds,
            }
            for share, statistics in shares.items()
        },
        **extra,
    }


def joined_as_json(report: JoinedReport, laptop_log_closed: bool | None = None) -> dict:
    """
    The joined report as saved by `--json`.

    :param laptop_log_closed: Whether the laptop's log ended with its closing line, when the caller
        checked. Saved so `--compare` can say a run may be only part of a walk.
    """
    return report_as_json(
        report.shares,
        {
            "joined_paths": report.joined_paths,
            "excluded_as_cold": report.excluded_as_cold,
            "clock_verdict": report.clock_verdict.value,
            "clock_reason": report.clock_reason,
            "phone_log_truncated": report.phone_log_truncated,
            "laptop_log_closed": laptop_log_closed,
            "negative_counts": {share.value: count for share, count in report.negative_counts.items()},
            "counts": {name.value: count for name, count in report.counts.items()},
            "orphans": {orphan.value: count for orphan, count in report.orphans.items()},
            "network_by_slice": [
                {"start_seconds": piece.start_seconds, "paths": piece.paths, "median_milliseconds": piece.median_milliseconds}
                for piece in report.network_by_slice
            ],
        },
    )


def write_json(path: Path, content: dict) -> None:
    write_text_lf(Path(path), json.dumps(content, indent=2, sort_keys=True) + "\n")


# *******************************************
# Comparing runs
# *******************************************


def compare(before: list[dict], after: list[dict]) -> str:
    """
    Each share's median before and after, with the run-to-run spread on each side.

    A share whose two spreads overlap is flagged. A difference that fits inside how much the same
    code varies from one run to the next is not a result, and replays with fitted floors don't
    repeat exactly.

    :param before: Saved reports from runs before the change, at least three.
    :param after: Saved reports from runs after it, at least three.
    :return: One line per share.
    :rtype: str
    :raises ValueError: With fewer than three runs on a side, because a spread needs runs.
    """
    for side, runs in (("before", before), ("after", after)):
        if len(runs) < MINIMUM_RUNS_TO_COMPARE:
            raise ValueError(f"{len(runs)} runs {side}, and a spread needs at least {MINIMUM_RUNS_TO_COMPARE}")
    lines = [
        f"median of run medians, ms   before   (min to max)         after   (min to max)      runs {len(before)} and {len(after)}",
    ]
    for side, runs in (("before", before), ("after", after)):
        # A run cut short still has medians, and they read like any other run's.
        truncated = sum(run.get("phone_log_truncated") is True for run in runs)
        unclosed = sum(run.get("laptop_log_closed") is False for run in runs)
        if truncated:
            lines.append(f"WARNING {truncated} of the {side} runs had a phone log cut at its size cap")
        if unclosed:
            lines.append(f"WARNING {unclosed} of the {side} runs had a laptop log with no closing line, so possibly only part of the run")
    for share in SHARE_ORDER:
        before_medians = _run_medians(before, share)
        after_medians = _run_medians(after, share)
        if not before_medians and not after_medians:
            continue
        if len(before_medians) < MINIMUM_RUNS_TO_COMPARE or len(after_medians) < MINIMUM_RUNS_TO_COMPARE:
            # A run without this share, such as one whose clocks had different bases, gives it no
            # median. Fewer than three left on a side is no spread at all.
            lines.append(
                f"  {SHARE_LABELS[share].strip():<26}in {len(before_medians)} runs before and {len(after_medians)} after, "
                f"under {MINIMUM_RUNS_TO_COMPARE} a side, not compared"
            )
            continue
        overlap = min(before_medians) <= max(after_medians) and min(after_medians) <= max(before_medians)
        lines.append(
            f"  {SHARE_LABELS[share].strip():<26}{np.median(before_medians):7.1f} ({min(before_medians):6.1f} to {max(before_medians):6.1f})"
            f"   {np.median(after_medians):7.1f} ({min(after_medians):6.1f} to {max(after_medians):6.1f})"
            f"{'   spreads overlap, not a result' if overlap else ''}"
        )
    return "\n".join(lines)


def _run_medians(runs: list[dict], share: Share) -> list[float]:
    medians = []
    for run in runs:
        statistics = run.get("shares", {}).get(share.value)
        if statistics is not None:
            medians.append(statistics["median_milliseconds"])
    return medians
