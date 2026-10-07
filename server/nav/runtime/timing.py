"""
Where each frame's time went, written per frame during a run and summarized afterwards.

One line per frame the laptop received, in timing.jsonl beside the frame log or wherever
--timing-log points. Every time is on the laptop clock from nav.clock. The line says when the
frame was captured, when it reached the laptop, when the worker started on it, when its depth was
ready, when its plan was done, when the phone was sent its path, where its floor came from, and
what became of it. So the floor acceptance rate and every latency share can be read back from
the file alone.

Nothing here knows which sensor the frames came from. A source that fills in a capture time gets
the capture shares, and one that does not gets the laptop's shares only.

A line is written when its frame's fate is settled, so the file is in that order, not in frame
order. A planned frame waits for its path to go out while a later frame can be refused at once.
Sort by `timestamp_seconds` for frame order. Nothing in the summary depends on line order.

The phone writes a log of its own, and the field names it uses are declared here too, so the
report that joins the two reads both from one place.
"""

# Standard library imports
import json
import logging
import math
import queue
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.clock import laptop_time_seconds
from nav.runtime.textio import append_text_lf
from nav.types import DepthFrame, FloorSource

log = logging.getLogger(__name__)

TIMING_FILENAME = "timing.jsonl"
MILLISECONDS_PER_SECOND = 1000.0
NANOSECONDS_PER_SECOND = 1e9
NO_FLOOR = "none"  # How a skipped frame's missing floor is printed in a summary.
# A replayed frame log carries the walk's arrival beside the replay's own plan time, so the two
# are hours or days apart. The longest live delay recorded so far was about 12 s. A minute is
# well clear of both, so a gap past it means the line mixes two runs.
MIXED_RUN_GAP_SECONDS = 60.0
# A frame stays open from the worker taking it until its path is sent or a newer one replaces it,
# so a handful are open at once. A thousand is minutes of frames that never finished, which only
# a stuck publisher produces, and the bound keeps that from growing without end.
DEFAULT_MAX_OPEN_RECORDS = 1000
DEFAULT_CLOSE_TIMEOUT_SECONDS = 5.0
# The key of the line a recorder writes last on a clean close. No frame line has it.
CLOSING_KEY = "log_closed"


class FrameOutcome(Enum):
    """What became of a frame the laptop received. Exactly one per line."""

    PUBLISHED = "published"  # Planned, and its path handed to the displays.
    SUPERSEDED = "superseded"  # Planned, and a newer plan went out before it did.
    DROPPED = "dropped"  # Replaced by a newer frame before the worker took it.
    SKIPPED = "skipped"  # Taken, and refused by the scene or the planner.
    IN_FLIGHT = "in_flight"  # Still unfinished when the run ended, or pushed out by the open-record bound.


class Share(Enum):
    """
    One slice of the frame-to-arrow delay, or of the laptop's part of it.

    The values are the keys of the report's JSON, which runs are compared by, so they are a
    contract with every saved report.
    """

    LAPTOP_QUEUE_WAIT = "laptop_queue_wait"  # Arrival to the worker taking the frame. Includes decode.
    LAPTOP_PROCESSING = "laptop_processing"  # Scene and planner, up to the plan being done.
    LAPTOP_PUBLISH_WAIT = "laptop_publish_wait"  # Plan done to the phone sink's write. Includes the user model.
    LAPTOP = "laptop"  # Arrival to the phone sink's write. The laptop's whole share.
    PHONE = "phone"  # Frame handled on the phone to its depth sent.
    NETWORK = "network"  # Both hops, the phone's sent-to-received minus the laptop's whole share.
    DISPLAY = "display"  # Path received to the arrow first drawn with it.
    TOTAL = "total"  # Frame handled to arrow drawn.
    SENSOR_TO_HANDLED = "sensor_to_handled"  # ARCore's frame stamp to the phone handling it. Same clock base only.
    TOTAL_FROM_SENSOR = "total_from_sensor"  # Sensor to arrow drawn. Same clock base only.


def _difference(end: float | None, start: float | None) -> float | None:
    return None if end is None or start is None else end - start


def _processing_seconds(record: "TimingRecord") -> float | None:
    # The user model runs after the plan is done, so its time is already inside the publish wait.
    # Counting it here too made the three parts add up to more than the laptop's share.
    stages = (record.scene_milliseconds, record.planner_milliseconds)
    if any(stage is None for stage in stages):
        return None
    return sum(stages) / MILLISECONDS_PER_SECOND


# The laptop's shares of one line, in seconds, or None when the line doesn't have both ends. The one
# definition the summary and the phone join both use, so a laptop share means the same in each.
LAPTOP_SHARE_SECONDS: dict[Share, Callable[["TimingRecord"], float | None]] = {
    Share.LAPTOP_QUEUE_WAIT: lambda record: _difference(record.started_seconds, record.arrival_seconds),
    Share.LAPTOP_PROCESSING: _processing_seconds,
    Share.LAPTOP_PUBLISH_WAIT: lambda record: _difference(record.sent_seconds, record.plan_done_seconds),
    Share.LAPTOP: lambda record: _difference(record.sent_seconds, record.arrival_seconds),
}


@dataclass(frozen=True)
class StageDurations:
    """How long the worker spent in each stage of one frame, on the performance counter."""

    scene_milliseconds: float
    planner_milliseconds: float
    usermodel_milliseconds: float


@dataclass(frozen=True)
class TimingRecord:
    """
    One frame's line. Any time can be None when the frame never reached that point or never had it.

    The first six fields are on every line ever written. The rest came later and are None on a
    line from before them, which the summary reads as a frame the worker took.
    """

    timestamp_seconds: float
    capture_seconds: float | None
    arrival_seconds: float | None
    depth_ready_seconds: float | None
    plan_done_seconds: float | None
    floor_source: FloorSource | None
    outcome: FrameOutcome | None = None
    started_seconds: float | None = None
    scene_milliseconds: float | None = None
    planner_milliseconds: float | None = None
    usermodel_milliseconds: float | None = None
    sent_seconds: float | None = None


# Every key a line must carry, old or new. A line from before the outcome was written has only these.
_RECORD_KEYS = ("timestamp_seconds", "capture_seconds", "arrival_seconds", "depth_ready_seconds", "plan_done_seconds", "floor_source")
_OPTIONAL_TIME_KEYS = ("capture_seconds", "arrival_seconds", "depth_ready_seconds", "plan_done_seconds")
# Read when present. Every line this module writes carries them, null where there is no value.
_STAGE_KEYS = ("scene_milliseconds", "planner_milliseconds", "usermodel_milliseconds")
_EXTENDED_NUMBER_KEYS = ("started_seconds", "sent_seconds", *_STAGE_KEYS)
_KNOWN_KEYS = (*_RECORD_KEYS, "outcome", *_EXTENDED_NUMBER_KEYS)

# What each outcome must have a value for. A line that lacks one was written by something other
# than this module, and its shares would silently go missing from the summary.
REQUIRED_VALUES_BY_OUTCOME: dict[FrameOutcome, tuple[str, ...]] = {
    FrameOutcome.PUBLISHED: ("started_seconds", "plan_done_seconds", *_STAGE_KEYS),
    FrameOutcome.SUPERSEDED: ("started_seconds", "plan_done_seconds", *_STAGE_KEYS),
    FrameOutcome.SKIPPED: ("started_seconds",),
    FrameOutcome.DROPPED: (),
    FrameOutcome.IN_FLIGHT: (),
}
# The frames the worker took. Only these have a floor to count or a plan in the rate.
_PROCESSED_OUTCOMES = frozenset({FrameOutcome.PUBLISHED, FrameOutcome.SUPERSEDED, FrameOutcome.SKIPPED})

# *******************************************
# Phone log
# *******************************************

# The Pixel app's timing log, written by pixel_app's timing/TimingRecord.kt. These names must
# match that file's constants string for string. The fixture the app's test writes,
# tests/fixtures/pixel_app_timing.jsonl, is what checks that they do.
PHONE_SCHEMA_VERSION = 1
PHONE_TYPE = "type"
PHONE_SCHEMA = "schema"
PHONE_STARTED_WALL = "started_wall"
PHONE_DEVICE = "device"
PHONE_BUILD_TYPE = "build_type"
PHONE_FRAME_NS = "frame_ns"
PHONE_HANDLED_NS = "handled_ns"
PHONE_SENT_NS = "sent_ns"
PHONE_RECEIVED_NS = "received_ns"
PHONE_DRAWN_NS = "drawn_ns"
PHONE_COUNT = "count"
PHONE_AT_BYTES = "at_bytes"
PHONE_TYPE_SESSION = "session"
PHONE_TYPE_FRAME = "frame"
PHONE_TYPE_SENT = "sent"
PHONE_TYPE_DROPPED = "dropped"
PHONE_TYPE_RECEIVED = "received"
PHONE_TYPE_DRAWN = "drawn"
PHONE_TYPE_LOST = "lost"
PHONE_TYPE_TRUNCATED = "truncated"


def frame_ns_from_seconds(timestamp_seconds: float) -> int:
    """
    The phone's ARCore frame timestamp in nanoseconds, back from the seconds it travels in.

    The app divides the integer by 1e9 to fill `timestamp_seconds`. Past about 48 days of uptime a
    double can't hold that to the nanosecond, so this isn't always the original integer. It's a
    key, and the app derives every one of its keys through the same seconds and the same rounding,
    so the two logs still match. The app's `TimingRecord.frameNanosFromSeconds` is this function's
    twin.

    Rounds half up like Java's `Math.round`. Python's `round` goes half to even, and past 48 days
    exact halves are common enough to split the join.

    example: 123456.789012345 -> 123456789012345
    """
    return math.floor(timestamp_seconds * NANOSECONDS_PER_SECOND + 0.5)


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
    floor_source_counts: dict[FloorSource | None, int]
    # A capture stamped after its own arrival means the clock offset is off by at least that much.
    capture_after_arrival_count: int
    # Frames planned more than MIXED_RUN_GAP_SECONDS after they arrived. Any at all means a replay.
    mixed_run_count: int
    # The laptop's shares toward the phone. None on a log from before they were written, and then
    # the summary prints exactly what it printed before.
    laptop_shares: dict[Share, "ShareStatistics | None"] | None = None


class TimingLog:
    """Appends one line per frame. Opens nothing until the first line, so an unused log leaves no file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @classmethod
    def in_directory(cls, log_dir: Path) -> "TimingLog":
        """The log a recording run keeps beside its frames, `timing.jsonl` in the record directory."""
        return cls(Path(log_dir) / TIMING_FILENAME)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: TimingRecord) -> None:
        line = asdict(record)
        # The enums stop here. On disk each is its word, or null.
        line["floor_source"] = None if record.floor_source is None else record.floor_source.value
        line["outcome"] = None if record.outcome is None else record.outcome.value
        append_text_lf(self._path, json.dumps(line, allow_nan=False) + "\n")

    def append_closing_line(self, lines_written: int) -> None:
        """The last line, written only when the recorder closed cleanly. A log without it was cut short."""
        append_text_lf(self._path, json.dumps({CLOSING_KEY: True, "lines_written": lines_written}) + "\n")


def timing_log_closed(path: Path) -> bool:
    """
    Whether a timing log ends with its closing line.

    A log without one was stopped before it closed: the run was killed, the writer failed, or the
    log predates the closing line. Its lines are then possibly only the first part of the run.

    :param path: The file, or the record directory holding it.
    :rtype: bool
    """
    path = Path(path)
    if path.is_dir():
        path = path / TIMING_FILENAME
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    try:
        last = json.loads(lines[-1])
    except ValueError:
        # A last line cut off mid-write is the clearest sign of all that the log didn't close.
        return False
    return isinstance(last, dict) and last.get(CLOSING_KEY) is True


_CLOSE = object()  # The writer's last event. Everything still open is written in flight after it.


class TimingRecorder:
    """
    Collects one frame's stamps from the threads that see it, and writes its line once its fate is known.

    The source stamps arrival on the frame. The worker stamps when it took the frame and when the
    plan was done, the phone sink stamps the send, and the publisher says the path went out. None
    of those threads may wait on a file, because each is inside the delay being measured. So every
    call reads the clock on the caller's thread and puts one event on a queue. One writer thread of
    its own keeps the open records and writes each line.

    A recorder without a log does nothing and starts no thread, so the loop never checks for one.
    """

    def __init__(
        self,
        timing_log: TimingLog | None,
        clock: Callable[[], float] = laptop_time_seconds,
        max_open_records: int = DEFAULT_MAX_OPEN_RECORDS,
        close_timeout_seconds: float = DEFAULT_CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        self._timing_log = timing_log
        self._clock = clock
        self._max_open_records = max_open_records
        self._close_timeout_seconds = close_timeout_seconds
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._closing = threading.Event()
        self._writer: threading.Thread | None = None
        # Writer-thread state only. Keyed by frame_ns, in the order the worker took the frames.
        self._open: OrderedDict[int, dict] = OrderedDict()
        self._evicted = 0
        self._lines_written = 0
        self._writing_failed = False
        if timing_log is not None:
            self._writer = threading.Thread(target=self._write_lines, name="timing-writer", daemon=True)
            self._writer.start()

    def frame_taken(self, frame: DepthFrame) -> None:
        """The worker has started on this frame. Called on the worker thread."""
        self._put(("taken", frame, self._clock()))

    def frame_planned(self, frame: DepthFrame, plan_done_seconds: float, floor_source: FloorSource | None, stages: StageDurations) -> None:
        """The plan for this frame is done. Its line waits until the path goes out or a newer one does."""
        self._put(("planned", frame, plan_done_seconds, floor_source, stages))

    def frame_skipped(self, frame: DepthFrame, floor_source: FloorSource | None) -> None:
        """The scene or the planner refused this frame. Its floor, if the scene found one, is kept."""
        self._put(("skipped", frame, floor_source))

    def frame_dropped(self, frame: DepthFrame) -> None:
        """A newer frame replaced this one before the worker took it. Called on the loop's thread."""
        self._put(("dropped", frame))

    def path_sent(self, timestamp_seconds: float) -> None:
        """The phone sink has written this frame's path to the phone's socket. Called on the sink's thread."""
        self._put(("sent", timestamp_seconds, self._clock()))

    def path_published(self, timestamp_seconds: float) -> None:
        """Every display has been handed this frame's path. Ends its record."""
        self._put(("published", timestamp_seconds))

    def close(self) -> None:
        """Write what is queued and every frame still open as in flight, waiting at most the close timeout."""
        if self._writer is None or self._closing.is_set():
            return
        self._closing.set()
        self._events.put(_CLOSE)
        self._writer.join(self._close_timeout_seconds)
        if self._writer.is_alive():
            # The disk or the log is stuck. The run is ending, and waiting longer would hold up
            # the exit for a measurement, so the daemon thread is left to the process.
            log.warning("timing writer did not finish within %.1f s, its last lines may be missing", self._close_timeout_seconds)
        if self._evicted:
            log.warning("%d frames were still open when the bound of %d was reached and were written in flight", self._evicted, self._max_open_records)

    def _put(self, event: tuple) -> None:
        # A stamp after close needs no check here. The writer stops at the close marker, and
        # nothing queued behind it is ever written.
        if self._writer is None:
            return
        self._events.put(event)

    def _write_lines(self) -> None:
        while True:
            event = self._events.get()
            try:
                if event is _CLOSE:
                    self._close_open_records()
                    return
                self._handle(event)
            except OSError as write_error:
                # A full disk or a vanished directory. The run goes on without its timing, and says
                # so once, because nothing is written after the first failure.
                log.error("timing log stopped writing to %s: %s", self._timing_log.path, write_error)
                self._writing_failed = True
            except Exception as unexpected_error:
                # Nobody predicted this one. The measurement is lost, the walk must not be.
                log.error("UNEXPECTED %s in the timing writer, may need a handler", type(unexpected_error).__name__, exc_info=True)
                self._writing_failed = True
            if event is _CLOSE:
                return

    def _close_open_records(self) -> None:
        """Write every frame still open as in flight, then the closing line if nothing failed."""
        for frame_ns in list(self._open):
            self._emit(self._open.pop(frame_ns), FrameOutcome.IN_FLIGHT)
        # A run that saw no frame leaves no file, as before, so its --timing-log path stays usable.
        if not self._writing_failed and self._lines_written > 0:
            self._timing_log.append_closing_line(self._lines_written)

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "taken":
            _, frame, started_seconds = event
            fields = _frame_fields(frame)
            fields["started_seconds"] = started_seconds
            self._open[frame_ns_from_seconds(frame.timestamp_seconds)] = fields
            if len(self._open) > self._max_open_records:
                _, oldest = self._open.popitem(last=False)
                self._evicted += 1
                self._emit(oldest, FrameOutcome.IN_FLIGHT)
        elif kind == "planned":
            _, frame, plan_done_seconds, floor_source, stages = event
            fields = self._open.setdefault(frame_ns_from_seconds(frame.timestamp_seconds), _frame_fields(frame))
            fields.update(plan_done_seconds=plan_done_seconds, floor_source=floor_source, **asdict(stages))
        elif kind == "skipped":
            _, frame, floor_source = event
            fields = self._open.pop(frame_ns_from_seconds(frame.timestamp_seconds), None) or _frame_fields(frame)
            fields["floor_source"] = floor_source
            self._emit(fields, FrameOutcome.SKIPPED)
        elif kind == "dropped":
            _, frame = event
            self._emit(_frame_fields(frame), FrameOutcome.DROPPED)
        elif kind == "sent":
            _, timestamp_seconds, sent_seconds = event
            fields = self._open.get(frame_ns_from_seconds(timestamp_seconds))
            if fields is not None:
                fields["sent_seconds"] = sent_seconds
        elif kind == "published":
            self._close_published(frame_ns_from_seconds(event[1]))

    def _close_published(self, frame_ns: int) -> None:
        published = self._open.pop(frame_ns, None)
        if published is None:
            # A path for a frame this recorder never saw taken, such as one already pushed out by
            # the bound. There is no line to finish.
            return
        # Only the newest result is ever sent, so anything planned no later than this one and
        # still open never will be. Plan order, not frame timestamps, because a phone that
        # reconnects starts its clock again lower.
        plan_done = published["plan_done_seconds"]
        if plan_done is not None:
            for open_ns in [key for key, fields in self._open.items() if fields["plan_done_seconds"] is not None and fields["plan_done_seconds"] <= plan_done]:
                self._emit(self._open.pop(open_ns), FrameOutcome.SUPERSEDED)
        self._emit(published, FrameOutcome.PUBLISHED)

    def _emit(self, fields: dict, outcome: FrameOutcome) -> None:
        if self._writing_failed:
            return
        self._timing_log.append(TimingRecord(outcome=outcome, **fields))
        self._lines_written += 1


def _frame_fields(frame: DepthFrame) -> dict:
    """Every field of a line, filled from the frame as it arrived and None for what has not happened."""
    timing = frame.timing
    return {
        "timestamp_seconds": frame.timestamp_seconds,
        "capture_seconds": None if timing is None else timing.capture_seconds,
        "arrival_seconds": None if timing is None else timing.arrival_seconds,
        "depth_ready_seconds": None if timing is None else timing.depth_ready_seconds,
        "plan_done_seconds": None,
        "floor_source": None,
        "started_seconds": None,
        "scene_milliseconds": None,
        "planner_milliseconds": None,
        "usermodel_milliseconds": None,
        "sent_seconds": None,
    }


def read_timing_log(path: Path) -> list[TimingRecord]:
    """
    Read a timing.jsonl back.

    A key the reader does not know is ignored, so a field added later does not break this reader.
    It is logged once per file.

    :param path: The file, or the record directory holding it.
    :return: One record per line, in file order.
    :rtype: list[TimingRecord]
    :raises FileNotFoundError: When there is no timing log there.
    :raises ValueError: When a line is not JSON, misses a key, or holds a value no writer writes.
        The message names the file and the line.
    """
    path = Path(path)
    if path.is_dir():
        path = path / TIMING_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no timing log at {path}. Only a run with --record-to or --timing-log writes one")
    records = []
    unknown_keys: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path} line {line_number}"
        try:
            raw = json.loads(line)
        except ValueError as json_error:
            # A run killed mid-write leaves a truncated last line, and this is what it looks like.
            # JSONDecodeError is a ValueError, and so is json's refusal of a 4300-digit integer.
            raise ValueError(f"{where} is not valid JSON: {json_error}") from json_error
        except RecursionError as nesting_error:
            raise ValueError(f"{where} nests too deeply to be a timing line") from nesting_error
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a JSON object, got {type(raw).__name__}")
        if CLOSING_KEY in raw:
            # The closing line carries no frame. `timing_log_closed` is what reads it.
            continue
        missing = [key for key in _RECORD_KEYS if key not in raw]
        if missing:
            raise ValueError(f"{where} is missing {', '.join(missing)}")
        unknown_keys.update(key for key in raw if key not in _KNOWN_KEYS)
        records.append(_record_from_line(raw, where))
    if unknown_keys:
        log.info("ignored keys this reader does not know in %s: %s", path, ", ".join(sorted(unknown_keys)))
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
    # The cold start is timed from the run's first frame, taken or not. The figures cover only the
    # frames the worker took, because a dropped frame has no floor and no plan, and a line from
    # before outcomes were written was always one the worker took.
    processed = [record for record in records if record.outcome is None or record.outcome in _PROCESSED_OUTCOMES]
    starts = [record.arrival_seconds for record in records if record.arrival_seconds is not None]
    if starts:
        cold_until = min(starts) + exclude_first_seconds
        measured = [record for record in processed if record.arrival_seconds is None or record.arrival_seconds >= cold_until]
    else:
        measured = list(processed)

    plan_times = sorted(record.plan_done_seconds for record in measured if record.plan_done_seconds is not None)
    frames_per_second = None
    longest_gap = None
    if len(plan_times) >= 2:
        span = plan_times[-1] - plan_times[0]
        frames_per_second = (len(plan_times) - 1) / span if span > 0 else None
        longest_gap = float(np.max(np.diff(plan_times)))

    floor_counts = Counter(record.floor_source for record in measured)

    return TimingSummary(
        frames_measured=len(measured),
        frames_excluded_as_cold=len(processed) - len(measured),
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
        mixed_run_count=sum(
            1
            for record in measured
            if record.arrival_seconds is not None
            and record.plan_done_seconds is not None
            and record.plan_done_seconds - record.arrival_seconds > MIXED_RUN_GAP_SECONDS
        ),
        laptop_shares=_laptop_shares(measured) if any(record.outcome is not None for record in measured) else None,
    )


def _laptop_shares(records: list[TimingRecord]) -> dict[Share, ShareStatistics | None]:
    return {
        share: share_statistics([seconds for seconds in map(seconds_of, records) if seconds is not None])
        for share, seconds_of in LAPTOP_SHARE_SECONDS.items()
    }


def share_statistics(durations_seconds: list[float]) -> ShareStatistics | None:
    """
    Median, 95th percentile and worst of some durations, in milliseconds. None when there are none.

    numpy's default percentile, linear between the two nearest ranks, for every share in every
    report, so two reports never differ in method.
    """
    if not durations_seconds:
        return None
    milliseconds = np.asarray(durations_seconds) * MILLISECONDS_PER_SECOND
    return ShareStatistics(
        frame_count=len(durations_seconds),
        median_milliseconds=float(np.median(milliseconds)),
        percentile_95_milliseconds=float(np.percentile(milliseconds, 95)),
        worst_milliseconds=float(np.max(milliseconds)),
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
    floor_rows = sorted(
        (NO_FLOOR if source is None else source.value, count) for source, count in summary.floor_source_counts.items()
    )
    for label, count in floor_rows:
        lines.append(f"  {label:<26}{count:6d}  {100.0 * count / total:5.1f}%")
    if summary.laptop_shares is not None:
        lines.append("laptop toward the phone, ms  median     p95   worst   frames")
        lines.extend(format_share_row(LAPTOP_SHARE_LABELS[share], summary.laptop_shares[share]) for share in LAPTOP_SHARE_SECONDS)
    if summary.capture_after_arrival_count:
        lines.append(
            f"WARNING {summary.capture_after_arrival_count} frames were captured after they arrived, "
            "so the clock offset is off and the capture shares are not to be trusted"
        )
    if summary.mixed_run_count:
        lines.append(
            f"WARNING {summary.mixed_run_count} frames were planned more than {MIXED_RUN_GAP_SECONDS:.0f} s after "
            "they arrived, so this log mixes two runs and the shares are not latencies. "
            "A --source logged replay does this"
        )
    return "\n".join(lines)


def _record_from_line(raw: dict, where: str) -> TimingRecord:
    # The one place the floor source is a string. A value no writer writes is refused here, not
    # counted later as a floor source of its own.
    raw_floor = raw["floor_source"]
    if raw_floor is None:
        floor_source = None
    else:
        try:
            floor_source = FloorSource(raw_floor)
        except ValueError as unknown_floor_error:
            allowed = ", ".join(member.value for member in FloorSource)
            raise ValueError(f"{where} has floor_source {raw_floor!r}, allowed values are {allowed} or null") from unknown_floor_error
    times = {key: _optional_time(raw[key], key, where) for key in _OPTIONAL_TIME_KEYS}
    timestamp_seconds = _optional_time(raw["timestamp_seconds"], "timestamp_seconds", where)
    if timestamp_seconds is None:
        raise ValueError(f"{where} has timestamp_seconds null, and every frame has one")
    extended = {key: _optional_time(raw.get(key), key, where) for key in _EXTENDED_NUMBER_KEYS}
    outcome = _outcome(raw.get("outcome"), where)
    if outcome is not None:
        values = {**times, **extended}
        for key in REQUIRED_VALUES_BY_OUTCOME[outcome]:
            if values[key] is None:
                raise ValueError(f"{where} is {outcome.value} and has no {key}")
    return TimingRecord(timestamp_seconds=timestamp_seconds, floor_source=floor_source, outcome=outcome, **times, **extended)


def _outcome(raw_outcome: object, where: str) -> FrameOutcome | None:
    # Absent or null on a line from before outcomes were written. Anything else must be one of ours.
    if raw_outcome is None:
        return None
    try:
        return FrameOutcome(raw_outcome)
    except ValueError as unknown_outcome_error:
        allowed = ", ".join(member.value for member in FrameOutcome)
        raise ValueError(f"{where} has outcome {raw_outcome!r}, allowed values are {allowed} or null") from unknown_outcome_error


def _optional_time(value: object, key: str, where: str) -> float | None:
    if value is None:
        return None
    # bool is an int to Python, and a true here is a damaged line, not one second.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} has {key} {value!r}, which is not a number or null")
    try:
        number = float(value)
    except OverflowError as overflow_error:
        raise ValueError(f"{where} has {key} as an integer too large for a float") from overflow_error
    if not np.isfinite(number):
        raise ValueError(f"{where} has {key} {value!r}, which is not finite")
    return number


def _share(records: list[TimingRecord], start_field: str, end_field: str) -> ShareStatistics | None:
    durations = [
        getattr(record, end_field) - getattr(record, start_field)
        for record in records
        if getattr(record, start_field) is not None and getattr(record, end_field) is not None
    ]
    return share_statistics(durations)


LAPTOP_SHARE_LABELS = {
    Share.LAPTOP_QUEUE_WAIT: "queue wait",
    Share.LAPTOP_PROCESSING: "processing",
    Share.LAPTOP_PUBLISH_WAIT: "publish wait",
    Share.LAPTOP: "arrival to sent",
}


def format_share_row(label: str, share: ShareStatistics | None) -> str:
    """One share under the median, p95, worst and frames columns, the layout the summary uses."""
    if share is None:
        return f"  {label:<26} no frame had both ends"
    return (
        f"  {label:<26}{share.median_milliseconds:8.1f}{share.percentile_95_milliseconds:8.1f}"
        f"{share.worst_milliseconds:8.1f}{share.frame_count:9d}"
    )


def _or_dash(value: float | None, template: str) -> str:
    return "-" if value is None else template.format(value)
