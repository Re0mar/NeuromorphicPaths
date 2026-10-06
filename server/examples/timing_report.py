"""
Prints the latency and floor figures from a recorded run's timing log.

Run it on the directory a live run was recorded to with --record-to:

    python examples/timing_report.py frame_logs/neon_walk_1
    python examples/timing_report.py frame_logs/neon_walk_1 --exclude-first-seconds 30

Only the live run's log means anything here. A replay re-records the walk's capture, arrival and
depth times beside its own plan times, so its shares would mix two runs. The summary ends with a
warning when it sees that.
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

# Local package imports
from nav.runtime.timing import format_summary, read_timing_log, summarize

# The network, the GPU and the phone are all still settling for the first several seconds.
DEFAULT_EXCLUDE_FIRST_SECONDS = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a recorded run's timing log.")
    parser.add_argument("record_dir", help="the directory a run was recorded to with --record-to")
    parser.add_argument(
        "--exclude-first-seconds",
        type=float,
        default=DEFAULT_EXCLUDE_FIRST_SECONDS,
        help="how much of the start to leave out, counted from the first frame's arrival",
    )
    arguments = parser.parse_args(argv)

    try:
        records = read_timing_log(Path(arguments.record_dir))
    except FileNotFoundError as missing:
        print(missing)
        return 1
    except ValueError as damaged:
        # A truncated or hand-edited line. The message names the file and the line to look at.
        print(damaged)
        return 1

    print(format_summary(summarize(records, arguments.exclude_first_seconds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
