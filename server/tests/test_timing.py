"""
Covers the timing log and its summary, against lines whose answers were worked out by hand.

The summary reports durations, percentiles and maxima rather than ratios, and leaves out the cold
start. Both are what make a latency figure from one walk comparable with the next.
"""

# Standard library imports
import dataclasses
import importlib.util
import json
import logging
import re
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.runtime.timing import (
    TIMING_FILENAME,
    ShareStatistics,
    TimingLog,
    TimingRecord,
    TimingSummary,
    format_summary,
    read_timing_log,
    summarize,
)
from nav.types import FloorSource

EXAMPLES = Path(__file__).parent.parent / "examples"
SECONDS_PER_DAY = 86_400.0


def _ten_frames() -> list[TimingRecord]:
    """Ten frames a tenth of a second apart. Capture 50 ms before arrival, depth 100 ms after.

    The plan follows depth by 20 ms on nine frames and by 120 ms on the last one.
    """
    records = []
    for index in range(10):
        arrival = 100.0 + index * 0.1
        depth_ready = arrival + 0.1
        plan_done = depth_ready + (0.12 if index == 9 else 0.02)
        records.append(
            TimingRecord(
                timestamp_seconds=float(index),
                capture_seconds=arrival - 0.05,
                arrival_seconds=arrival,
                depth_ready_seconds=depth_ready,
                plan_done_seconds=plan_done,
                floor_source=FloorSource.FITTED,
            )
        )
    return records


def _write_log(directory: Path, records: list[TimingRecord]) -> Path:
    """Through the writer the loop uses, so the reader is tested against what a run produces."""
    timing_log = TimingLog(directory)
    for record in records:
        timing_log.append(record)
    return directory / TIMING_FILENAME


def _timing_report_main():
    # examples/ is not a package, so the script is loaded by path, the way a person runs it.
    spec = importlib.util.spec_from_file_location("timing_report", EXAMPLES / "timing_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


# *******************************************
# Reading the log
# *******************************************


def test_reading_a_directory_with_no_timing_log_says_where_it_looked(tmp_path: Path) -> None:
    """A person pointing the report at the wrong directory needs to see which one it read."""
    with pytest.raises(FileNotFoundError, match=f"no timing log at {re.escape(str(tmp_path / TIMING_FILENAME))}"):
        read_timing_log(tmp_path)


def test_timing_report_exits_1_without_a_timing_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No log is a person's mistake, so it gets a message and an exit code, not a traceback."""
    exit_code = _timing_report_main()([str(tmp_path)])

    assert exit_code == 1
    assert "no timing log" in capsys.readouterr().out


def test_timing_report_exits_1_naming_the_line_when_the_last_line_is_truncated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A run killed mid-write leaves half a line. The report says which, rather than dying in json."""
    log_path = _write_log(tmp_path, _ten_frames()[:3])
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"timestamp_seconds": 3.0, "capt')

    exit_code = _timing_report_main()([str(tmp_path)])

    printed = capsys.readouterr().out
    assert exit_code == 1
    assert f"{log_path} line 4 is not valid JSON" in printed


def test_timing_report_exits_1_naming_the_key_a_line_is_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A line without a field cannot be summarized, and a TypeError from the constructor named neither."""
    log_path = _write_log(tmp_path, _ten_frames()[:1])
    line = json.loads(log_path.read_text(encoding="utf-8"))
    del line["floor_source"]
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(line) + "\n")

    exit_code = _timing_report_main()([str(tmp_path)])

    printed = capsys.readouterr().out
    assert exit_code == 1
    assert f"{log_path} line 2 is missing floor_source" in printed


def test_timing_report_reads_a_line_with_a_key_it_does_not_know(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A field added by a later writer must not stop an older reader from summarizing the rest."""
    log_path = tmp_path / TIMING_FILENAME
    lines = []
    for record in _ten_frames()[:3]:
        line = {**dataclasses.asdict(record), "floor_source": "fitted", "gpu_temperature_celsius": 61}
        lines.append(json.dumps(line) + "\n")
    log_path.write_bytes("".join(lines).encode("utf-8"))

    with caplog.at_level(logging.INFO, logger="nav.runtime.timing"):
        exit_code = _timing_report_main()([str(tmp_path), "--exclude-first-seconds", "0"])

    assert exit_code == 0
    assert "frames measured            3  (0 left out as the cold start)" in capsys.readouterr().out
    # Said once for the file, not once per line.
    notes = [record.message for record in caplog.records if "gpu_temperature_celsius" in record.message]
    assert len(notes) == 1
    assert read_timing_log(tmp_path) == _ten_frames()[:3]


def test_an_unknown_floor_source_is_refused_naming_the_line(tmp_path: Path) -> None:
    """A misspelled floor source would otherwise be counted as a floor source of its own."""
    log_path = tmp_path / TIMING_FILENAME
    line = {**dataclasses.asdict(_ten_frames()[0]), "floor_source": "guessed"}
    log_path.write_bytes((json.dumps(line) + "\n").encode("utf-8"))

    with pytest.raises(ValueError, match="line 1 has floor_source 'guessed', allowed values are supplied, fitted, previous or null"):
        read_timing_log(tmp_path)


@pytest.mark.parametrize(
    ("key", "bad_value", "expected"),
    [
        ("arrival_seconds", "100.0", "line 1 has arrival_seconds '100.0', which is not a number or null"),
        ("plan_done_seconds", True, "line 1 has plan_done_seconds True, which is not a number or null"),
        ("timestamp_seconds", None, "line 1 has timestamp_seconds null"),
    ],
)
def test_a_time_no_writer_writes_is_refused_naming_the_line(tmp_path: Path, key: str, bad_value: object, expected: str) -> None:
    """A string or a bool in a time would reach the summary's subtraction and fail there with no line number."""
    line = {**dataclasses.asdict(_ten_frames()[0]), "floor_source": "fitted", key: bad_value}
    (tmp_path / TIMING_FILENAME).write_bytes((json.dumps(line) + "\n").encode("utf-8"))

    with pytest.raises(ValueError, match=re.escape(expected)):
        read_timing_log(tmp_path)


def test_timing_log_lines_end_in_lf_only_and_read_back_equal(tmp_path: Path) -> None:
    """The floor source is an enum in memory and its word on disk, so old logs and new readers agree."""
    records = _ten_frames()[:3]
    records[2] = dataclasses.replace(records[2], floor_source=None, plan_done_seconds=None)
    log_path = _write_log(tmp_path, records)

    raw = log_path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["floor_source"] == "fitted"
    assert json.loads(lines[2])["floor_source"] is None
    read_back = read_timing_log(tmp_path)
    assert read_back == records
    assert read_back[0].floor_source is FloorSource.FITTED


def test_an_unused_timing_log_leaves_no_file(tmp_path: Path) -> None:
    """A run that planned nothing should not leave an empty log that reads as zero frames measured."""
    TimingLog(tmp_path)

    assert not (tmp_path / TIMING_FILENAME).exists()


# *******************************************
# Summarizing
# *******************************************


def test_summary_of_no_records_is_empty_and_still_formats() -> None:
    """A run that took no frames still gets a report rather than an exception."""
    summary = summarize([], exclude_first_seconds=10.0)

    assert summary.frames_measured == 0
    assert summary.frames_excluded_as_cold == 0
    assert summary.arrival_to_plan_done is None
    assert summary.planned_frames_per_second is None
    assert summary.longest_gap_between_plans_seconds is None
    assert summary.floor_source_counts == {}
    assert "frames measured            0  (0 left out as the cold start)" in format_summary(summary)


def test_summary_with_every_record_in_the_cold_window_is_empty_and_still_formats() -> None:
    """A short run inside the default 10 s window leaves nothing to measure, which is a result, not a crash."""
    summary = summarize(_ten_frames(), exclude_first_seconds=100.0)

    assert summary.frames_measured == 0
    assert summary.frames_excluded_as_cold == 10
    assert summary.planned_frames_per_second is None
    assert "frames measured            0  (10 left out as the cold start)" in format_summary(summary)


def test_summary_of_one_planned_frame_has_no_rate_and_no_gap() -> None:
    """A rate and a gap need two plans. One plan has neither."""
    summary = summarize(_ten_frames()[:1], exclude_first_seconds=0.0)

    assert summary.planned_frames_per_second is None
    assert summary.longest_gap_between_plans_seconds is None
    assert summary.arrival_to_plan_done.frame_count == 1
    assert "planned frames per second  -" in format_summary(summary)


def test_summary_of_frames_with_no_arrival_measures_all_of_them() -> None:
    """A frame log older than the timing key replays with no arrivals, and there is no cold start to measure from."""
    records = [
        dataclasses.replace(record, capture_seconds=None, arrival_seconds=None, depth_ready_seconds=None)
        for record in _ten_frames()
    ]

    summary = summarize(records, exclude_first_seconds=10.0)

    assert summary.frames_measured == 10
    assert summary.frames_excluded_as_cold == 0
    assert summary.arrival_to_plan_done is None
    assert summary.planned_frames_per_second == pytest.approx(9.0)


def test_summary_reports_median_p95_and_worst_for_each_share() -> None:
    """Each share comes from its own pair of stamps. These were worked out by hand from _ten_frames."""
    summary = summarize(_ten_frames(), exclude_first_seconds=0.0)

    assert summary.capture_to_arrival.median_milliseconds == pytest.approx(50.0)
    assert summary.capture_to_arrival.worst_milliseconds == pytest.approx(50.0)
    assert summary.arrival_to_depth_ready.median_milliseconds == pytest.approx(100.0)
    # Nine at 20 ms and one at 120. The 95th percentile interpolates 55 percent of the way between
    # the ninth and tenth, which is 75.
    assert summary.depth_ready_to_plan_done.median_milliseconds == pytest.approx(20.0)
    assert summary.depth_ready_to_plan_done.percentile_95_milliseconds == pytest.approx(75.0)
    assert summary.depth_ready_to_plan_done.worst_milliseconds == pytest.approx(120.0)
    # Arrival to plan is depth's 100 plus the plan's 20 or 120. The only end-to-end share a source
    # with no capture time gets, so it is stated rather than left to the others.
    assert summary.arrival_to_plan_done.median_milliseconds == pytest.approx(120.0)
    assert summary.arrival_to_plan_done.percentile_95_milliseconds == pytest.approx(175.0)
    assert summary.arrival_to_plan_done.worst_milliseconds == pytest.approx(220.0)
    assert summary.capture_to_plan_done.worst_milliseconds == pytest.approx(270.0)
    assert summary.depth_ready_to_plan_done.frame_count == 10
    # Plans from 100.12 to 101.12, ten of them, so nine intervals over one second.
    assert summary.planned_frames_per_second == pytest.approx(9.0)
    assert summary.longest_gap_between_plans_seconds == pytest.approx(0.2)


def test_summary_rate_and_gap_do_not_depend_on_file_order() -> None:
    """A writer thread can put lines in an order other than plan order. The rate is over plan times."""
    in_order = summarize(_ten_frames(), exclude_first_seconds=0.0)
    reversed_order = summarize(list(reversed(_ten_frames())), exclude_first_seconds=0.0)

    assert reversed_order.planned_frames_per_second == pytest.approx(9.0)
    assert reversed_order.longest_gap_between_plans_seconds == pytest.approx(in_order.longest_gap_between_plans_seconds)


def test_summary_leaves_out_the_cold_window_and_counts_what_it_left_out() -> None:
    """The record exactly on the boundary is kept. 100.0 plus five steps of 0.1 is exactly 100.5."""
    # Arrivals at 100.0 to 100.9. Leaving out the first half second keeps 100.5 onward.
    summary = summarize(_ten_frames(), exclude_first_seconds=0.5)

    assert summary.frames_measured == 5
    assert summary.frames_excluded_as_cold == 5


def test_summary_counts_floor_sources_including_none() -> None:
    """A skipped frame counts, or the floor acceptance reads perfect however many were refused."""
    records = _ten_frames()
    records[3] = dataclasses.replace(records[3], floor_source=None, plan_done_seconds=None)
    records[4] = dataclasses.replace(records[4], floor_source=FloorSource.PREVIOUS)

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.floor_source_counts == {FloorSource.FITTED: 8, None: 1, FloorSource.PREVIOUS: 1}


def test_summary_of_frames_without_capture_reports_only_the_shares_it_can() -> None:
    """A source with no clock offset has no capture time, and a share without both ends is absent, not zero."""
    records = [dataclasses.replace(record, capture_seconds=None) for record in _ten_frames()]

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.capture_to_arrival is None
    assert summary.capture_to_plan_done is None
    assert summary.arrival_to_plan_done is not None


def test_summary_counts_capture_after_arrival_as_an_offset_warning() -> None:
    """A capture after its own arrival can only come from a clock offset that is off."""
    records = _ten_frames()
    records[0] = dataclasses.replace(records[0], capture_seconds=records[0].arrival_seconds + 0.01)

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.capture_after_arrival_count == 1


def test_a_live_run_with_a_slow_plan_is_not_flagged_as_mixed() -> None:
    """A live frame 12 s late was measured on the first glasses walk, and it is a real latency."""
    records = [dataclasses.replace(record, plan_done_seconds=record.arrival_seconds + 12.0) for record in _ten_frames()]

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.mixed_run_count == 0
    assert "mixes two runs" not in format_summary(summary)


def test_a_replayed_log_is_flagged_as_mixing_two_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    A logged replay keeps the walk's arrival and stamps its own plan time days later.

    Its shares would print as ordinary figures of a few hundred million milliseconds.
    """
    replayed = [
        dataclasses.replace(record, plan_done_seconds=record.plan_done_seconds + 3 * SECONDS_PER_DAY)
        for record in _ten_frames()
    ]
    _write_log(tmp_path, replayed)

    summary = summarize(replayed, exclude_first_seconds=0.0)
    exit_code = _timing_report_main()([str(tmp_path), "--exclude-first-seconds", "0"])

    assert summary.mixed_run_count == 10
    assert exit_code == 0
    assert "WARNING 10 frames were planned more than 60 s after they arrived, so this log mixes two runs" in capsys.readouterr().out


# *******************************************
# Formatting
# *******************************************


def _literal_summary(**overrides) -> TimingSummary:
    """Figures chosen so that no two columns share a value, so a swapped column shows."""
    fields = {
        "frames_measured": 10,
        "frames_excluded_as_cold": 2,
        "capture_to_arrival": None,
        "arrival_to_depth_ready": ShareStatistics(frame_count=10, median_milliseconds=100.0, percentile_95_milliseconds=104.5, worst_milliseconds=110.0),
        "depth_ready_to_plan_done": ShareStatistics(frame_count=9, median_milliseconds=20.0, percentile_95_milliseconds=75.0, worst_milliseconds=120.0),
        "arrival_to_plan_done": ShareStatistics(frame_count=8, median_milliseconds=120.0, percentile_95_milliseconds=175.0, worst_milliseconds=220.0),
        "capture_to_plan_done": None,
        "planned_frames_per_second": 9.0,
        "longest_gap_between_plans_seconds": 0.2,
        "floor_source_counts": {FloorSource.FITTED: 8, None: 2},
        "capture_after_arrival_count": 0,
        "mixed_run_count": 0,
    }
    fields.update(overrides)
    return TimingSummary(**fields)


def test_format_summary_puts_each_figure_under_its_column() -> None:
    """The README's figures are copied from this text, so a figure under the wrong column is a wrong figure."""
    expected = [
        "frames measured            10  (2 left out as the cold start)",
        "planned frames per second  9.00",
        "longest gap between plans  0.200 s",
        "latency, ms                 median     p95   worst   frames",
        "  capture to arrival" + " " * 9 + "no frame had both ends",
        "  arrival to depth ready" + " " * 4 + "   100.0   104.5   110.0       10",
        "  depth ready to plan done" + " " * 2 + "    20.0    75.0   120.0        9",
        "  arrival to plan done" + " " * 6 + "   120.0   175.0   220.0        8",
        "  capture to plan done" + " " * 7 + "no frame had both ends",
        "floor source                count   share",
        "  fitted" + " " * 20 + "     8   80.0%",
        "  none" + " " * 22 + "     2   20.0%",
    ]

    assert format_summary(_literal_summary()).splitlines() == expected


def test_format_summary_ends_with_the_clock_offset_warning_when_a_capture_came_after_arrival() -> None:
    """Capture shares from a wrong offset look like ordinary figures unless the report says otherwise."""
    lines = format_summary(_literal_summary(capture_after_arrival_count=3)).splitlines()

    assert lines[-1] == (
        "WARNING 3 frames were captured after they arrived, "
        "so the clock offset is off and the capture shares are not to be trusted"
    )


def test_format_summary_ends_with_the_mixed_run_warning_when_a_log_mixes_two_runs() -> None:
    """Without this line a replay's shares read as latencies of a few days."""
    lines = format_summary(_literal_summary(mixed_run_count=4)).splitlines()

    assert lines[-1] == (
        "WARNING 4 frames were planned more than 60 s after they arrived, so this log mixes two runs "
        "and the shares are not latencies. A --source logged replay does this"
    )


def test_timing_report_prints_the_summary_for_a_record_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The flag reaches the summary. With the default 10 s window these ten frames would all be left out."""
    _write_log(tmp_path, _ten_frames())

    exit_code = _timing_report_main()([str(tmp_path), "--exclude-first-seconds", "0"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "frames measured            10  (0 left out as the cold start)" in printed
    assert "  arrival to plan done" + " " * 6 + "   120.0   175.0   220.0       10" in printed
    assert "  fitted" + " " * 20 + "    10  100.0%" in printed
