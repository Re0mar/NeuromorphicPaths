"""
Covers the laptop clock the latency stamps use.

It has to agree with the wall clock at start-up, which is what the Neon's Time Echo offset is
measured against, and it must never run backwards when the wall clock is corrected mid-walk.
"""

# Standard library imports
import time

# Third party imports
import pytest

# Local package imports
from nav import clock

# The laptop clock is float seconds since 1970, about 1.7e9. A float that size rounds to about
# 0.24 microseconds, so two durations computed from it can differ by a few of those.
EPOCH_ROUNDING_SECONDS = 1e-6
BUSY_WAIT_SECONDS = 0.003


def test_laptop_time_never_runs_backwards_when_the_wall_clock_steps_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows corrects its wall clock mid-walk, and a share measured across that must not go negative."""
    before = clock.laptop_time_seconds()
    # Windows time sync pulling the wall clock back an hour. The laptop clock read its wall time
    # once at import, so this must not move it.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 3600.0)

    after = clock.laptop_time_seconds()

    assert after >= before


def test_the_anchor_is_taken_as_the_wall_clock_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A wall time read between two ticks is up to a tick stale, 15.6 ms on Windows.

    That error would sit in every stamp of the run, the size of the shares being measured.
    """
    tick_start = 1_700_000_000.0
    next_tick = tick_start + 0.0156
    # Two reads inside one tick, then the tick, then the clock holds still.
    readings = iter([tick_start, tick_start, next_tick])
    monkeypatch.setattr(time, "time", lambda: next(readings, next_tick))

    wall_seconds, counter_seconds = clock._anchor()

    assert wall_seconds == next_tick
    assert counter_seconds == pytest.approx(time.perf_counter(), abs=0.05)


def test_laptop_time_starts_at_the_wall_clock() -> None:
    """
    The Neon's clock offset is measured against time.time(), so any gap here lands in every capture share.

    A few ticks of the wall clock, not a second. The performance counter drifts from the wall clock
    by parts per million, which is milliseconds over a test session.
    """
    assert clock.laptop_time_seconds() == pytest.approx(time.time(), abs=0.05)


def test_laptop_time_resolves_a_few_milliseconds() -> None:
    """
    time.monotonic() on Windows moves in 15.6 ms steps, so a 3 ms wait reads as 0 or 15.6 there.

    The latency shares are tens of milliseconds, and a clock that coarse would quantize them.
    """
    # Bracketed by the performance counter rather than held to a fixed ceiling, so a test machine
    # under load that pauses this thread cannot fail it. A coarse clock still fails one of the two
    # bounds, and three tries make a lucky pause lining up with a tick unlikely every time.
    for _ in range(3):
        counter_before = time.perf_counter()
        start = clock.laptop_time_seconds()
        deadline = time.perf_counter() + BUSY_WAIT_SECONDS
        while time.perf_counter() < deadline:
            pass
        elapsed = clock.laptop_time_seconds() - start
        bracket = time.perf_counter() - counter_before

        assert BUSY_WAIT_SECONDS - EPOCH_ROUNDING_SECONDS <= elapsed <= bracket + EPOCH_ROUNDING_SECONDS
