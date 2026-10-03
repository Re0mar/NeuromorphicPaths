"""
Covers the timing log and its summary, against lines whose answers were worked out by hand.

The summary reports durations, percentiles and maxima rather than ratios, and leaves out the cold
start. Both are what make a latency figure from one walk comparable with the next.
"""

# Standard library imports
import importlib.util
import json
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.runtime.timing import TIMING_FILENAME, TimingLog, TimingRecord, read_timing_log, summarize

EXAMPLES = Path(__file__).parent.parent / "examples"


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
                floor_source="fitted",
            )
        )
    return records


def test_summary_reports_median_p95_and_worst_for_each_share() -> None:
    summary = summarize(_ten_frames(), exclude_first_seconds=0.0)

    assert summary.capture_to_arrival.median_milliseconds == pytest.approx(50.0)
    assert summary.capture_to_arrival.worst_milliseconds == pytest.approx(50.0)
    assert summary.arrival_to_depth_ready.median_milliseconds == pytest.approx(100.0)
    # Nine at 20 ms and one at 120. The 95th percentile interpolates 55 percent of the way between
    # the ninth and tenth, which is 75.
    assert summary.depth_ready_to_plan_done.median_milliseconds == pytest.approx(20.0)
    assert summary.depth_ready_to_plan_done.percentile_95_milliseconds == pytest.approx(75.0)
    assert summary.depth_ready_to_plan_done.worst_milliseconds == pytest.approx(120.0)
    assert summary.capture_to_plan_done.worst_milliseconds == pytest.approx(270.0)
    assert summary.depth_ready_to_plan_done.frame_count == 10
    # Plans from 100.12 to 101.12, ten of them, so nine intervals over one second.
    assert summary.planned_frames_per_second == pytest.approx(9.0)
    assert summary.longest_gap_between_plans_seconds == pytest.approx(0.2)


def test_summary_leaves_out_the_cold_window_and_counts_what_it_left_out() -> None:
    # Arrivals at 100.0 to 100.9. Leaving out the first half second keeps 100.5 onward.
    summary = summarize(_ten_frames(), exclude_first_seconds=0.5)

    assert summary.frames_measured == 5
    assert summary.frames_excluded_as_cold == 5


def test_summary_counts_floor_sources_including_none() -> None:
    records = _ten_frames()
    records[3] = TimingRecord(**{**records[3].__dict__, "floor_source": None, "plan_done_seconds": None})
    records[4] = TimingRecord(**{**records[4].__dict__, "floor_source": "previous"})

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.floor_source_counts == {"fitted": 8, "none": 1, "previous": 1}


def test_summary_of_frames_without_capture_reports_only_the_shares_it_can() -> None:
    records = [TimingRecord(**{**record.__dict__, "capture_seconds": None}) for record in _ten_frames()]

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.capture_to_arrival is None
    assert summary.capture_to_plan_done is None
    assert summary.arrival_to_plan_done is not None


def test_summary_counts_capture_after_arrival_as_an_offset_warning() -> None:
    records = _ten_frames()
    records[0] = TimingRecord(**{**records[0].__dict__, "capture_seconds": records[0].arrival_seconds + 0.01})

    summary = summarize(records, exclude_first_seconds=0.0)

    assert summary.capture_after_arrival_count == 1


def test_timing_log_lines_end_in_lf_only_and_read_back_equal(tmp_path: Path) -> None:
    timing_log = TimingLog(tmp_path)
    for record in _ten_frames()[:3]:
        timing_log.append(record)

    raw = (tmp_path / TIMING_FILENAME).read_bytes()
    assert b"\r" not in raw
    assert len(raw.splitlines()) == 3
    assert read_timing_log(tmp_path) == _ten_frames()[:3]


def test_an_unused_timing_log_leaves_no_file(tmp_path: Path) -> None:
    TimingLog(tmp_path)

    assert not (tmp_path / TIMING_FILENAME).exists()


def test_reading_a_directory_with_no_timing_log_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no timing log"):
        read_timing_log(tmp_path)


def _timing_report_main():
    # examples/ is not a package, so the script is loaded by path, the way a person runs it.
    spec = importlib.util.spec_from_file_location("timing_report", EXAMPLES / "timing_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_timing_report_prints_the_summary_for_a_record_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with (tmp_path / TIMING_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
        for record in _ten_frames():
            handle.write(json.dumps(record.__dict__) + "\n")

    exit_code = _timing_report_main()([str(tmp_path), "--exclude-first-seconds", "0"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "frames measured            10" in printed
    assert "capture to arrival" in printed
    assert "fitted" in printed


def test_timing_report_exits_1_without_a_timing_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = _timing_report_main()([str(tmp_path)])

    assert exit_code == 1
    assert "no timing log" in capsys.readouterr().out
