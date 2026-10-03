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


def test_laptop_time_starts_at_the_wall_clock() -> None:
    assert clock.laptop_time_seconds() == pytest.approx(time.time(), abs=1.0)


def test_laptop_time_never_runs_backwards_when_the_wall_clock_steps_back(monkeypatch: pytest.MonkeyPatch) -> None:
    before = clock.laptop_time_seconds()
    # Windows time sync pulling the wall clock back an hour. The laptop clock read its wall time
    # once at import, so this must not move it.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 3600.0)

    after = clock.laptop_time_seconds()

    assert after >= before


def test_laptop_time_resolves_a_few_milliseconds() -> None:
    # time.monotonic() on Windows moves in 15.6 ms steps, so a 3 ms wait reads as 0 or 15.6 there.
    # The latency shares are tens of milliseconds, and a clock that coarse would quantize them.
    start = clock.laptop_time_seconds()
    deadline = time.perf_counter() + 0.003
    while time.perf_counter() < deadline:
        pass

    elapsed = clock.laptop_time_seconds() - start

    assert 0.002 < elapsed < 0.012
