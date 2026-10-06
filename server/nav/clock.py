"""
The laptop's clock for latency stamps: wall time that never runs backwards, to well under a millisecond.

Read once at import, then advanced by the performance counter. It agrees with time.time() at
start-up, which is the base the Neon's Time Echo offset is measured against, so a Neon timestamp
plus that offset lands on this clock. And when Windows corrects the wall clock in the middle of a
walk, this one does not jump, so a duration measured between two stamps cannot come out negative.

Not time.monotonic(). Measured on Python 3.12 on Windows, it moves in 15.6 ms steps, about the
size of the latency shares being measured. The performance counter is monotonic too and resolved
to well under a microsecond on the same machine.

Sources and the runtime both stamp with it, so it lives at the package root rather than in either.
"""

# Standard library imports
import time

# How long to wait for the wall clock to tick at start-up. Measured on Python 3.12 on Windows it
# ticks every 15.6 ms, so this catches one there, and it bounds the wait where it never moves.
ANCHOR_WAIT_SECONDS = 0.05


def _anchor() -> tuple[float, float]:
    """
    A wall time and the performance counter at the same instant.

    The wall clock is read just as it ticks over, because a reading taken anywhere between two
    ticks is up to a whole tick stale, and that error would sit in every stamp for the whole run.
    """
    first = time.time()
    deadline = time.perf_counter() + ANCHOR_WAIT_SECONDS
    while time.perf_counter() < deadline:
        now = time.time()
        if now != first:
            return now, time.perf_counter()
    return time.time(), time.perf_counter()


_WALL_AT_START_SECONDS, _COUNTER_AT_START_SECONDS = _anchor()


def laptop_time_seconds() -> float:
    """
    Seconds since the Unix epoch on the laptop, as of start-up, advanced by the performance counter.

    :return: Wall-clock seconds that only ever increase.
    :rtype: float
    """
    return _WALL_AT_START_SECONDS + (time.perf_counter() - _COUNTER_AT_START_SECONDS)
