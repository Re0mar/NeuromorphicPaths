"""
Covers joining the phone's timing log with the laptop's into the frame-to-arrow delay.

The phone side of the contract is tests/fixtures/pixel_app_timing.jsonl, written by the Pixel
app's own test. A log that only round trips through this file's code would prove nothing about
the app, so the fixture is what the join is tested against. The laptop side,
tests/fixtures/laptop_timing.jsonl, is written by the laptop's real writer from the values below.
Regenerate it with `python tests/test_timing_join.py` after a change to the laptop's line.
"""

# Standard library imports
import importlib.util
import json
import sys
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.runtime.timing import (
    FrameOutcome,
    Share,
    TimingLog,
    TimingRecord,
    frame_ns_from_seconds,
    read_timing_log,
    summarize,
)
from nav.runtime.timing_join import (
    MINIMUM_JOINED_PATHS,
    ClockVerdict,
    Orphan,
    PhoneFrame,
    PhoneLog,
    compare,
    format_joined,
    join,
    joined_as_json,
    read_phone_log,
)
from nav.types import FloorSource

FIXTURES = Path(__file__).parent / "fixtures"
PHONE_FIXTURE = FIXTURES / "pixel_app_timing.jsonl"
LAPTOP_FIXTURE = FIXTURES / "laptop_timing.jsonl"
EXAMPLES = Path(__file__).parent.parent / "examples"

# The phone fixture's frames, as its own test wrote them.
PUBLISHED_FRAME_NS = 123_456_789_012_345
DROPPED_FRAME_NS = 123_456_822_345_678
NEVER_SENT_FRAME_NS = 123_456_855_679_011
# A frame the laptop planned and sent that the phone's fixture never mentions.
UNKNOWN_TO_THE_PHONE_FRAME_NS = 123_456_888_888_888
ARRIVAL_SECONDS = 1_759_500_000.0
# Wall-clock seconds near 1.76e9 resolve about 0.24 us in a double, so a share computed from them
# lands within a microsecond of its hand-worked value, not on it.
MICROSECOND_IN_MILLISECONDS = 0.001


def _laptop_fixture_records() -> list[TimingRecord]:
    """
    The laptop's lines for the phone fixture's walk, worked out by hand.

    The published frame arrives at 1759500000.0, is taken 3 ms later, planned at 50 ms and sent at
    52 ms, so the laptop's whole share is 52 ms. The phone fixture has it handled at 0, sent at 20 ms,
    received at 80 ms and drawn at 100 ms on its own clock. So the phone takes 20 ms, the two hops
    60 - 52 = 8 ms, the display 20 ms, and all four make the 100 ms total.
    """
    published = TimingRecord(
        timestamp_seconds=PUBLISHED_FRAME_NS / 1e9,
        capture_seconds=None,
        arrival_seconds=ARRIVAL_SECONDS,
        depth_ready_seconds=None,
        plan_done_seconds=ARRIVAL_SECONDS + 0.050,
        floor_source=FloorSource.SUPPLIED,
        outcome=FrameOutcome.PUBLISHED,
        started_seconds=ARRIVAL_SECONDS + 0.003,
        scene_milliseconds=40.0,
        planner_milliseconds=6.0,
        usermodel_milliseconds=1.0,
        sent_seconds=ARRIVAL_SECONDS + 0.052,
    )
    unknown_to_the_phone = TimingRecord(
        timestamp_seconds=UNKNOWN_TO_THE_PHONE_FRAME_NS / 1e9,
        capture_seconds=None,
        arrival_seconds=ARRIVAL_SECONDS + 0.100,
        depth_ready_seconds=None,
        plan_done_seconds=ARRIVAL_SECONDS + 0.150,
        floor_source=FloorSource.SUPPLIED,
        outcome=FrameOutcome.PUBLISHED,
        started_seconds=ARRIVAL_SECONDS + 0.103,
        scene_milliseconds=40.0,
        planner_milliseconds=6.0,
        usermodel_milliseconds=1.0,
        sent_seconds=ARRIVAL_SECONDS + 0.152,
    )
    return [published, unknown_to_the_phone]


def _write_laptop_fixture(target: Path) -> None:
    if target.exists():
        target.unlink()
    log = TimingLog(target)
    for record in _laptop_fixture_records():
        log.append(record)


def _timing_report_main():
    spec = importlib.util.spec_from_file_location("timing_report", EXAMPLES / "timing_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _phone_log(frames: dict[int, PhoneFrame]) -> PhoneLog:
    return PhoneLog(device="Pixel 8", build_type="release", started_wall="2026-10-06T10:00:00Z", frames=frames)


def _write_lines(path: Path, lines: list[dict]) -> Path:
    path.write_bytes("".join(json.dumps(line) + "\n" for line in lines).encode("utf-8"))
    return path


SESSION = {"type": "session", "schema": 1, "started_wall": "2026-10-06T10:00:00Z", "device": "Pixel 8", "build_type": "release"}


# *******************************************
# The contract between the two languages
# *******************************************


def test_the_committed_laptop_fixture_is_what_the_writer_writes(tmp_path: Path) -> None:
    fresh = tmp_path / "laptop_timing.jsonl"
    _write_laptop_fixture(fresh)

    assert LAPTOP_FIXTURE.read_bytes() == fresh.read_bytes(), "regenerate it with python tests/test_timing_join.py"


def test_the_laptop_fixture_keys_join_the_phone_fixture() -> None:
    """Built from the phone's integers, divided the way the app divides them, and recovered by rounding."""
    keys = [frame_ns_from_seconds(record.timestamp_seconds) for record in read_timing_log(LAPTOP_FIXTURE)]

    assert keys == [PUBLISHED_FRAME_NS, UNKNOWN_TO_THE_PHONE_FRAME_NS]


def test_the_kotlin_written_phone_log_joins_with_a_laptop_log() -> None:
    report = join(read_phone_log(PHONE_FIXTURE), read_timing_log(LAPTOP_FIXTURE), exclude_first_seconds=0.0)

    assert report.joined_paths == 1
    medians = {share: statistics.median_milliseconds for share, statistics in report.shares.items() if statistics is not None}
    assert medians[Share.TOTAL] == pytest.approx(100.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.PHONE] == pytest.approx(20.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.NETWORK] == pytest.approx(8.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.LAPTOP] == pytest.approx(52.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.LAPTOP_QUEUE_WAIT] == pytest.approx(3.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.LAPTOP_PROCESSING] == pytest.approx(47.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.LAPTOP_PUBLISH_WAIT] == pytest.approx(2.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert medians[Share.DISPLAY] == pytest.approx(20.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert report.counts["phone frames handled"] == 3
    assert report.counts["phone frames dropped"] == 1
    assert report.orphans[Orphan.HANDLED_NEVER_SENT] == 1
    assert report.orphans[Orphan.PUBLISHED_NEVER_RECEIVED] == 1
    # Three frames is far short of what the clock check needs.
    assert report.clock_verdict is ClockVerdict.NOT_ENOUGH_FRAMES


def test_shares_add_up() -> None:
    """Phone, network, laptop and display tile the total exactly. The laptop share cancels out of the sum."""
    report = join(read_phone_log(PHONE_FIXTURE), read_timing_log(LAPTOP_FIXTURE), exclude_first_seconds=0.0)
    median = {share: report.shares[share].median_milliseconds for share in (Share.PHONE, Share.NETWORK, Share.LAPTOP, Share.DISPLAY, Share.TOTAL)}

    assert median[Share.PHONE] + median[Share.NETWORK] + median[Share.LAPTOP] + median[Share.DISPLAY] == pytest.approx(median[Share.TOTAL], abs=1e-3)


# *******************************************
# The phone's clocks
# *******************************************


def _steady_phone(frame_count: int, handled_after_seconds: float, spread_seconds: float = 0.0) -> PhoneLog:
    frames = {}
    for index in range(frame_count):
        frame_ns = 100_000_000_000_000 + index * 33_333_333
        wobble = spread_seconds * (index % 2)
        frames[frame_ns] = PhoneFrame(handled_ns=frame_ns + round((handled_after_seconds + wobble) * 1e9))
    return _phone_log(frames)


def test_clock_base_same_when_all_gaps_are_small_and_positive() -> None:
    report = join(_steady_phone(120, handled_after_seconds=0.011), [], exclude_first_seconds=0.0)

    assert report.clock_verdict is ClockVerdict.SAME_BASE


def test_clock_base_different_when_a_gap_is_negative() -> None:
    report = join(_steady_phone(120, handled_after_seconds=-0.005), [], exclude_first_seconds=0.0)

    assert report.clock_verdict is ClockVerdict.DIFFERENT_BASE
    assert "outside 0 to 200 ms" in report.clock_reason


def test_clock_base_different_when_the_spread_is_too_wide() -> None:
    """Inside the 0 to 200 ms window throughout, and still not one clock."""
    report = join(_steady_phone(120, handled_after_seconds=0.010, spread_seconds=0.120), [], exclude_first_seconds=0.0)

    assert report.clock_verdict is ClockVerdict.DIFFERENT_BASE
    assert "spreads" in report.clock_reason


def test_clock_base_not_enough_frames_under_100() -> None:
    report = join(_steady_phone(99, handled_after_seconds=0.011), [], exclude_first_seconds=0.0)

    assert report.clock_verdict is ClockVerdict.NOT_ENOUGH_FRAMES


def test_the_sensor_slice_is_left_out_unless_the_clocks_share_a_base() -> None:
    report = join(_steady_phone(120, handled_after_seconds=-0.005), [], exclude_first_seconds=0.0)

    assert "sensor to handled" not in format_joined(report)
    assert "not at the sensor" in format_joined(report)


# *******************************************
# Counting what didn't join
# *******************************************


def _published(frame_ns: int, arrival: float, sent_after: float = 0.052) -> TimingRecord:
    return TimingRecord(
        timestamp_seconds=frame_ns / 1e9,
        capture_seconds=None,
        arrival_seconds=arrival,
        depth_ready_seconds=None,
        plan_done_seconds=arrival + 0.050,
        floor_source=FloorSource.SUPPLIED,
        outcome=FrameOutcome.PUBLISHED,
        started_seconds=arrival + 0.003,
        scene_milliseconds=40.0,
        planner_milliseconds=6.0,
        usermodel_milliseconds=1.0,
        sent_seconds=arrival + sent_after,
    )


def test_orphans_are_counted_by_kind_not_dropped() -> None:
    sent_only, received_only, drawn_only, received_not_drawn = 10, 20, 30, 40
    phone = _phone_log(
        {
            sent_only: PhoneFrame(handled_ns=1, sent_ns=2),
            received_only: PhoneFrame(handled_ns=1, sent_ns=2, received_ns=3),
            drawn_only: PhoneFrame(handled_ns=1, sent_ns=2, drawn_ns=4),
            received_not_drawn: PhoneFrame(handled_ns=1, sent_ns=2, received_ns=3),
        }
    )
    laptop = [_published(received_not_drawn, 1000.0)]

    report = join(phone, laptop, exclude_first_seconds=0.0)

    assert report.joined_paths == 0
    assert report.orphans[Orphan.SENT_NO_LAPTOP_LINE] == 3
    assert report.orphans[Orphan.RECEIVED_NO_LAPTOP_LINE] == 1
    assert report.orphans[Orphan.RECEIVED_NEVER_DRAWN] == 2
    assert report.orphans[Orphan.DRAWN_NEVER_RECEIVED] == 1


def test_negative_durations_are_counted_and_excluded() -> None:
    """A path received before its frame was sent is a stamp in the wrong place, not a fast network."""
    phone = _phone_log(
        {
            1_000: PhoneFrame(handled_ns=0, sent_ns=20_000_000, received_ns=80_000_000, drawn_ns=100_000_000),
            2_000: PhoneFrame(handled_ns=0, sent_ns=90_000_000, received_ns=80_000_000, drawn_ns=100_000_000),
        }
    )
    laptop = [_published(1_000, 1000.0), _published(2_000, 1000.1)]

    report = join(phone, laptop, exclude_first_seconds=0.0)

    assert report.shares[Share.NETWORK].frame_count == 1
    assert Share.NETWORK in report.negative_examples
    assert "network, both hops came out negative" in format_joined(report)


def test_a_duplicate_received_row_is_counted_and_the_first_used(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "phone.jsonl",
        [
            SESSION,
            {"type": "received", "frame_ns": 5, "received_ns": 100},
            {"type": "received", "frame_ns": 5, "received_ns": 900},
            {"type": "drawn", "frame_ns": 5, "drawn_ns": 110},
            {"type": "drawn", "frame_ns": 5, "drawn_ns": 990},
        ],
    )

    phone = read_phone_log(path)

    assert phone.frames[5].received_ns == 100
    assert phone.frames[5].drawn_ns == 110
    assert phone.duplicate_received == 1 and phone.duplicate_drawn == 1


def test_lost_and_truncated_lines_are_reported_not_refused(tmp_path: Path) -> None:
    path = _write_lines(
        tmp_path / "phone.jsonl",
        [
            SESSION,
            {"type": "lost", "count": 3},
            {"type": "frame", "frame_ns": 5, "handled_ns": 100},
            {"type": "lost", "count": 2},
            {"type": "truncated", "at_bytes": 20_000_000},
        ],
    )

    phone = read_phone_log(path)
    report = join(phone, [], exclude_first_seconds=0.0)

    assert phone.lost_records == 5
    assert report.counts["phone records lost"] == 5
    assert "hit its size cap" in format_joined(report)


def test_exclude_first_seconds_removes_the_same_frames_from_both_logs() -> None:
    # The early frame without a laptop line is timed from the phone's first frame, not the laptop's.
    early, early_never_reached_the_laptop, late = 1_000, 1_500, 2_000
    phone = _phone_log(
        {
            early: PhoneFrame(handled_ns=0, sent_ns=20_000_000, received_ns=80_000_000, drawn_ns=100_000_000),
            early_never_reached_the_laptop: PhoneFrame(handled_ns=1_000_000_000, sent_ns=1_020_000_000),
            late: PhoneFrame(handled_ns=20_000_000_000, sent_ns=20_020_000_000, received_ns=20_080_000_000, drawn_ns=20_100_000_000),
        }
    )
    laptop = [_published(early, 1000.0), _published(late, 1020.0)]

    report = join(phone, laptop, exclude_first_seconds=10.0)

    assert report.joined_paths == 1
    assert report.excluded_as_cold == 2
    assert report.counts["phone frames handled"] == 1
    assert report.counts["laptop frames received"] == 1
    assert report.orphans[Orphan.SENT_NO_LAPTOP_LINE] == 0


def test_fewer_than_100_joined_paths_prints_the_warning() -> None:
    report = join(read_phone_log(PHONE_FIXTURE), read_timing_log(LAPTOP_FIXTURE), exclude_first_seconds=0.0)

    assert report.joined_paths < MINIMUM_JOINED_PATHS
    assert format_joined(report).splitlines()[0].startswith("WARNING only 1 joined paths, under 100")


# *******************************************
# Refusals
# *******************************************


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([dict(SESSION, schema=2)], "has schema 2, and this reader knows schema 1"),
        ([SESSION, {"type": "frame", "frame_ns": "123", "handled_ns": 1}], "line 2 has frame_ns '123', which is not an integer"),
        ([SESSION, {"type": "frame", "frame_ns": True, "handled_ns": 1}], "line 2 has frame_ns True, which is not an integer"),
        ([{"type": "frame", "frame_ns": 1, "handled_ns": 1}], "line 1 is a 'frame' line, and a phone log starts with its session line"),
        ([], "is empty, and a phone log starts with its session line"),
        ([SESSION, {"type": "teleported", "frame_ns": 1}], "line 2 has type 'teleported', which no phone writes"),
        ([SESSION, SESSION], "line 2 is a second session line"),
        ([SESSION, {"type": "truncated", "at_bytes": 10}, {"type": "frame", "frame_ns": 1, "handled_ns": 1}], "line 3 follows the truncated line"),
        ([SESSION, {"type": "frame", "frame_ns": 1}], "line 2 is missing handled_ns"),
    ],
)
def test_a_phone_log_no_phone_writes_is_refused_naming_the_line(tmp_path: Path, lines: list[dict], expected: str) -> None:
    path = _write_lines(tmp_path / "phone.jsonl", lines)

    with pytest.raises(ValueError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        read_phone_log(path)


def test_the_report_exits_1_naming_the_line_of_a_damaged_phone_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    phone = _write_lines(tmp_path / "phone.jsonl", [dict(SESSION, schema=7)])

    exit_code = _timing_report_main()([str(LAPTOP_FIXTURE), "--phone", str(phone)])

    assert exit_code == 1
    assert "has schema 7" in capsys.readouterr().out


def test_the_report_exits_1_on_a_laptop_line_missing_a_value_its_outcome_needs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    line = json.loads(LAPTOP_FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    line["plan_done_seconds"] = None
    laptop = _write_lines(tmp_path / "timing.jsonl", [line])

    exit_code = _timing_report_main()([str(laptop), "--phone", str(PHONE_FIXTURE)])

    assert exit_code == 1
    assert "is published and has no plan_done_seconds" in capsys.readouterr().out


# *******************************************
# The command line
# *******************************************


def test_a_directory_and_a_file_are_both_accepted_as_laptop_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record_dir = tmp_path / "walk"
    record_dir.mkdir()
    (record_dir / "timing.jsonl").write_bytes(LAPTOP_FIXTURE.read_bytes())

    assert _timing_report_main()([str(record_dir), "--exclude-first-seconds", "0"]) == 0
    from_directory = capsys.readouterr().out
    assert _timing_report_main()([str(LAPTOP_FIXTURE), "--exclude-first-seconds", "0"]) == 0

    assert capsys.readouterr().out == from_directory


def test_laptop_only_prints_laptop_shares_and_names_what_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
    assert _timing_report_main()([str(LAPTOP_FIXTURE), "--exclude-first-seconds", "0"]) == 0

    printed = capsys.readouterr().out
    assert "  publish wait" + " " * 14 + "     2.0     2.0     2.0        2" in printed
    assert "need the phone's own log, given with --phone" in printed


def test_summarize_reports_queue_and_publish_wait() -> None:
    summary = summarize(read_timing_log(LAPTOP_FIXTURE), exclude_first_seconds=0.0)

    assert summary.laptop_shares[Share.LAPTOP_QUEUE_WAIT].median_milliseconds == pytest.approx(3.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert summary.laptop_shares[Share.LAPTOP_PUBLISH_WAIT].median_milliseconds == pytest.approx(2.0, abs=MICROSECOND_IN_MILLISECONDS)
    assert summary.laptop_shares[Share.LAPTOP].median_milliseconds == pytest.approx(52.0, abs=MICROSECOND_IN_MILLISECONDS)


def test_the_saved_report_round_trips_into_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    saved = [tmp_path / f"run_{index}.json" for index in range(3)]
    for path in saved:
        assert _timing_report_main()([str(LAPTOP_FIXTURE), "--phone", str(PHONE_FIXTURE), "--exclude-first-seconds", "0", "--json", str(path)]) == 0
    capsys.readouterr()

    assert _timing_report_main()(["--compare", "--before", *map(str, saved), "--after", *map(str, saved)]) == 0
    assert "spreads overlap, not a result" in capsys.readouterr().out
    assert b"\r" not in saved[0].read_bytes()


def _run_json(**medians: float) -> dict:
    return {"shares": {name: {"frame_count": 100, "median_milliseconds": value, "percentile_95_milliseconds": value, "worst_milliseconds": value} for name, value in medians.items()}}


def test_compare_flags_overlapping_spreads() -> None:
    before = [_run_json(laptop_publish_wait=value, laptop=60.0 + value) for value in (40.0, 45.0, 50.0)]
    after = [_run_json(laptop_publish_wait=value, laptop=100.0 + value) for value in (2.0, 3.0, 4.0)]

    lines = compare(before, after).splitlines()

    publish = next(line for line in lines if "publish wait" in line)
    laptop = next(line for line in lines if line.strip().startswith("laptop, arrival to sent"))
    assert "overlap" not in publish
    assert "spreads overlap, not a result" in laptop


def test_compare_with_fewer_than_three_runs_a_side_is_refused() -> None:
    runs = [_run_json(laptop=1.0)] * 3

    with pytest.raises(ValueError, match="2 runs after, and a spread needs at least 3"):
        compare(runs, runs[:2])


def test_compare_refuses_a_log_given_with_it(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _timing_report_main()([str(LAPTOP_FIXTURE), "--compare", "--before", "a", "b", "c", "--after", "d", "e", "f"])

    assert "--compare reads saved reports" in capsys.readouterr().err


if __name__ == "__main__":
    _write_laptop_fixture(LAPTOP_FIXTURE)
    print(f"wrote {LAPTOP_FIXTURE}")
    sys.exit(0)
