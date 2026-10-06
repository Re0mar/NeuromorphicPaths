"""
Prints the latency and floor figures from a run's timing log, and joins the phone's log to it.

Run it on the directory a live run was recorded to with --record-to, or on a --timing-log file:

    python examples/timing_report.py frame_logs/neon_walk_1
    python examples/timing_report.py frame_logs/neon_walk_1 --exclude-first-seconds 30

With the Pixel's own log, pulled off the phone, it prints the whole frame-to-arrow delay and
its shares, and --json saves the figures for --compare:

    python examples/timing_report.py frame_logs/walk --phone timing_from_phone/timing_2026-10-06T10-12-03Z.jsonl --json frame_logs/walk/report.json
    python examples/timing_report.py --compare --before b1.json b2.json b3.json --after a1.json a2.json a3.json

Only a live run's log, or a run fed by fake_arcore_sender.py or --neon-replay, means anything here.
A --source logged replay re-records the walk's capture, arrival and depth times beside its own
plan times, so its shares would mix two runs. The summary ends with a warning when it sees that.
"""

# Standard library imports
import argparse
import json
import sys
from pathlib import Path

# Local package imports
from nav.runtime.timing import format_summary, read_timing_log, summarize
from nav.runtime.timing_join import compare, format_joined, join, joined_as_json, read_phone_log, report_as_json, write_json

# The network, the GPU and the phone are all still settling for the first several seconds.
DEFAULT_EXCLUDE_FIRST_SECONDS = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a run's timing log, alone or joined with the phone's.")
    parser.add_argument("record_dir", nargs="?", help="a directory a run was recorded to with --record-to, or a --timing-log file")
    parser.add_argument("--phone", help="the phone's timing log for the same walk, to report the whole frame-to-arrow delay")
    parser.add_argument(
        "--exclude-first-seconds",
        type=float,
        default=DEFAULT_EXCLUDE_FIRST_SECONDS,
        help="how much of the start to leave out, counted from the first frame's arrival",
    )
    parser.add_argument("--json", help="also save the figures to this file, for --compare")
    parser.add_argument("--compare", action="store_true", help="compare saved reports instead of reading a log")
    parser.add_argument("--before", nargs="+", default=[], help="with --compare, reports from before the change")
    parser.add_argument("--after", nargs="+", default=[], help="with --compare, reports from after it")
    arguments = parser.parse_args(argv)

    if arguments.compare:
        if arguments.record_dir is not None or arguments.phone is not None:
            parser.error("--compare reads saved reports, give them with --before and --after rather than a log")
        return _compare(arguments.before, arguments.after)
    if arguments.record_dir is None:
        parser.error("give a timing log, or --compare with --before and --after")
    if arguments.before or arguments.after:
        parser.error("--before and --after only apply to --compare")

    try:
        records = read_timing_log(Path(arguments.record_dir))
        phone_log = None if arguments.phone is None else read_phone_log(Path(arguments.phone))
    except FileNotFoundError as missing:
        print(missing)
        return 1
    except ValueError as damaged:
        # A truncated or hand-edited line. The message names the file and the line to look at.
        print(damaged)
        return 1

    summary = summarize(records, arguments.exclude_first_seconds)
    print(format_summary(summary))
    if phone_log is None:
        if summary.laptop_shares is not None:
            print("the phone, network, display and total shares need the phone's own log, given with --phone")
        saved = report_as_json(summary.laptop_shares or {}, {"frames_measured": summary.frames_measured})
    else:
        joined = join(phone_log, records, arguments.exclude_first_seconds)
        print()
        print(format_joined(joined))
        saved = joined_as_json(joined)
    if arguments.json is not None:
        write_json(Path(arguments.json), saved)
    return 0


def _compare(before_paths: list[str], after_paths: list[str]) -> int:
    try:
        before = [json.loads(Path(path).read_text(encoding="utf-8")) for path in before_paths]
        after = [json.loads(Path(path).read_text(encoding="utf-8")) for path in after_paths]
        print(compare(before, after))
    except FileNotFoundError as missing:
        print(missing)
        return 1
    except ValueError as refused:
        # Too few runs to have a spread, or a saved report that is not JSON.
        print(refused)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
